# SPDX-License-Identifier: Apache-2.0
"""Stable-prefix incremental tokenization for streaming text input.

BPE prefix instability: ``encode(prefix)`` is not guaranteed to be a prefix
of ``encode(full_string)`` — the token at the boundary can merge with
characters that arrive later. The streaming-TTS protocol has no undo (an
injected token cannot be retracted), so incremental input must only release
tokens that can no longer change.

Strategy (design doc 20260717 streaming_tts_v1, serving item 5):

- Released text is FROZEN. The unreleased character buffer is tokenized on
  its own, so ``released_tokens + encode(buffer)`` is always a valid,
  self-consistent tokenization of the text seen so far.
- A release cut is only made at a HARD BOUNDARY, where GPT-style byte-BPE
  pretokenizers never merge across:
    * before a whitespace character (the space attaches to the FOLLOWING
      word, so everything strictly before it is a closed segment);
    * after a CJK/fullwidth punctuation character that is already followed
      by another character (the punctuation run is closed).
  At such cuts the released tokenization is identical to the canonical
  full-string tokenization.
- Fallback: if the buffer grows beyond ``max_hold_chars`` without a hard
  boundary (e.g. a long unpunctuated CJK run), release all but the last
  ``fallback_hold_chars`` characters. The cut is at a character position,
  so the released chunk is still valid text; the tokenization may be
  non-canonical at the cut, which the model tolerates far better than a
  retraction it cannot express.
- ``flush()`` releases the whole buffer (end of upstream text or an
  explicit client flush).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Sequence

# Fullwidth/CJK punctuation that closes a pretokenizer segment. ASCII
# sentence punctuation is intentionally absent: in Latin text it is always
# followed by whitespace (the dominant boundary), and mid-token ASCII
# punctuation ("don't", "3.14", "e.g.") must not be treated as a cut point.
_CJK_BOUNDARY_CHARS = frozenset("。，！？；：、）】》〉」』…—")

_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_DERIVED_HOLD_CACHE: dict[int, "HoldParams"] = {}


def _is_cjk(ch: str) -> bool:
    return "㐀" <= ch <= "鿿"


class HoldParams:
    """Vocab-derived holdback windows for stable-prefix release."""

    def __init__(
        self, max_hold_chars: int, fallback_hold_chars: int, cjk_hold_chars: int
    ) -> None:
        self.max_hold_chars = max_hold_chars
        self.fallback_hold_chars = fallback_hold_chars
        self.cjk_hold_chars = cjk_hold_chars

    def __eq__(self, other: Any) -> bool:  # test convenience
        return isinstance(other, HoldParams) and vars(self) == vars(other)

    def __repr__(self) -> str:
        return f"HoldParams({vars(self)})"


def derive_hold_params(
    tokenizer: Any,
    *,
    min_hold: int = 8,
    max_hold_cap: int = 64,
) -> HoldParams:
    """Derive holdback windows from the MODEL's vocabulary.

    A future character can only merge into a token that overlaps it, and a
    token spans at most ``max token length`` characters — so a cut that
    holds back at least the longest vocab token keeps every released token
    out of reach of later merges. Two windows are derived:

    - ``fallback_hold_chars``: max length over letter-bearing tokens
      (whitespace/punct runs are already handled by hard boundaries);
    - ``cjk_hold_chars``: max length over CJK-bearing tokens — much shorter
      (7 for Qwen3), enabling CONTINUOUS release inside unpunctuated CJK
      runs instead of waiting for the next punctuation mark.

    One O(vocab) scan (~0.6 s for Qwen3's 152k vocab), cached per tokenizer.
    """
    key = id(tokenizer)
    cached = _DERIVED_HOLD_CACHE.get(key)
    if cached is not None:
        return cached
    longest = 0
    longest_cjk = 0
    vocab = tokenizer.get_vocab()
    for token_id in vocab.values():
        piece = tokenizer.decode([token_id])
        if len(piece) > longest and _LETTER_RE.search(piece):
            longest = len(piece)
        if len(piece) > longest_cjk and any(_is_cjk(c) for c in piece):
            longest_cjk = len(piece)
    fallback = max(min_hold, min(longest, max_hold_cap))
    cjk_hold = max(2, min(longest_cjk or fallback, max_hold_cap))
    params = HoldParams(fallback * 3, fallback, cjk_hold)
    _DERIVED_HOLD_CACHE[key] = params
    return params


def _last_hard_boundary(buffer: str) -> int:
    """Return the largest safe cut position in ``buffer``, or 0 if none.

    A cut at position ``i`` means ``buffer[:i]`` is a closed pretokenizer
    region: appended characters can never merge into it.
    """
    best = 0
    for i, ch in enumerate(buffer):
        if ch.isspace():
            # the whitespace attaches to what FOLLOWS it -> cut before it
            best = max(best, i)
        elif ch in _CJK_BOUNDARY_CHARS and i + 1 < len(buffer):
            # punctuation run is closed once another char follows
            if buffer[i + 1] not in _CJK_BOUNDARY_CHARS:
                best = max(best, i + 1)
    return best


class StablePrefixTokenizer:
    """Incrementally tokenize streaming text, releasing only stable tokens.

    ``encode`` must map text to token ids without adding special tokens
    (e.g. ``lambda s: hf_tokenizer.encode(s, add_special_tokens=False)``).
    """

    def __init__(
        self,
        encode: Callable[[str], Sequence[int]],
        *,
        max_hold_chars: int = 32,
        fallback_hold_chars: int = 4,  # prefer derive_hold_params(tokenizer)
        cjk_hold_chars: int | None = None,
        encode_full: (
            Callable[[str], tuple[Sequence[int], Sequence[tuple[int, int]]]] | None
        ) = None,
    ) -> None:
        if max_hold_chars <= fallback_hold_chars:
            raise ValueError("max_hold_chars must exceed fallback_hold_chars")
        self._encode = encode
        self._max_hold_chars = int(max_hold_chars)
        self._fallback_hold_chars = int(fallback_hold_chars)
        # continuous CJK release: inside an unpunctuated CJK run, tokens whose
        # span ends >= cjk_hold_chars behind the tip are out of reach of any
        # future merge (CJK vocab tokens are short — 7 chars for Qwen3), so
        # they release immediately instead of waiting for punctuation. The
        # released ids are the token-PREFIX of the buffer's own tokenization
        # (via encode_full = ids+offsets), so the seam re-merges nothing.
        self._cjk_hold_chars = int(cjk_hold_chars) if cjk_hold_chars else None
        self._encode_full = encode_full
        self._buffer = ""
        self._released_tokens = 0
        self._released_text = ""

    @property
    def pending_text(self) -> str:
        """Characters buffered but not yet released."""
        return self._buffer

    @property
    def released_text(self) -> str:
        """Concatenation of all released chunks."""
        return self._released_text

    @property
    def released_tokens(self) -> int:
        return self._released_tokens

    def push(self, text: str) -> list[int]:
        """Append ``text``; return newly released (stable) token ids."""
        self._buffer += text
        hard = _last_hard_boundary(self._buffer)
        soft_cut, soft_ids = self._cjk_soft_cut()
        if soft_cut > hard:
            return self._release(soft_cut, precomputed_ids=soft_ids)
        cut = hard
        if cut == 0 and len(self._buffer) > self._max_hold_chars:
            # no boundary in an overlong run: force a character-position cut
            cut = len(self._buffer) - self._fallback_hold_chars
        if cut == 0:
            return []
        return self._release(cut)

    def _cjk_soft_cut(self) -> tuple[int, list[int] | None]:
        """Largest token-aligned cut inside a trailing CJK run, keeping
        ``cjk_hold_chars`` characters of holdback behind the tip.

        Returns ``(cut, token_ids_up_to_cut)`` — the ids are the prefix of
        the CURRENT buffer's tokenization, released verbatim so the seam is
        a genuine token boundary of the text the model will have seen.
        ``(0, None)`` when no such cut exists.
        """
        hold = self._cjk_hold_chars
        if hold is None or self._encode_full is None:
            return 0, None
        b = self._buffer
        if len(b) <= hold:
            return 0, None
        limit = len(b) - hold
        # both sides of the cut must be CJK so the holdback bound applies
        # (CJK vocab tokens only) — scan for the trailing CJK run
        start = len(b)
        while start > 0 and _is_cjk(b[start - 1]):
            start -= 1
        if start >= limit:  # run too short to clear the holdback
            return 0, None
        ids, offsets = self._encode_full(b)
        best = 0
        best_k = 0
        for k, (_s, e) in enumerate(offsets):
            if e > limit:
                break
            # cut at a token boundary strictly inside the CJK run
            if e > start and e > best and _is_cjk(b[e - 1]) and _is_cjk(b[e]):
                best = e
                best_k = k + 1
        if best == 0:
            return 0, None
        return best, [int(t) for t in ids[:best_k]]

    def flush(self) -> list[int]:
        """Release everything still buffered (end of text / explicit flush)."""
        if not self._buffer:
            return []
        return self._release(len(self._buffer))

    def _release(self, cut: int, precomputed_ids: list[int] | None = None) -> list[int]:
        chunk, self._buffer = self._buffer[:cut], self._buffer[cut:]
        tokens = (
            precomputed_ids
            if precomputed_ids is not None
            else list(self._encode(chunk))
        )
        self._released_tokens += len(tokens)
        self._released_text += chunk
        return tokens


__all__ = ["StablePrefixTokenizer", "derive_hold_params"]
