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

from typing import Callable, Sequence

# Fullwidth/CJK punctuation that closes a pretokenizer segment. ASCII
# sentence punctuation is intentionally absent: in Latin text it is always
# followed by whitespace (the dominant boundary), and mid-token ASCII
# punctuation ("don't", "3.14", "e.g.") must not be treated as a cut point.
_CJK_BOUNDARY_CHARS = frozenset("。，！？；：、）】》〉」』…—")


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
        fallback_hold_chars: int = 4,
    ) -> None:
        if max_hold_chars <= fallback_hold_chars:
            raise ValueError("max_hold_chars must exceed fallback_hold_chars")
        self._encode = encode
        self._max_hold_chars = int(max_hold_chars)
        self._fallback_hold_chars = int(fallback_hold_chars)
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
        cut = _last_hard_boundary(self._buffer)
        if cut == 0 and len(self._buffer) > self._max_hold_chars:
            # no boundary in an overlong run: force a character-position cut
            cut = len(self._buffer) - self._fallback_hold_chars
        if cut == 0:
            return []
        return self._release(cut)

    def flush(self) -> list[int]:
        """Release everything still buffered (end of text / explicit flush)."""
        if not self._buffer:
            return []
        return self._release(len(self._buffer))

    def _release(self, cut: int) -> list[int]:
        chunk, self._buffer = self._buffer[:cut], self._buffer[cut:]
        tokens = list(self._encode(chunk))
        self._released_tokens += len(tokens)
        self._released_text += chunk
        return tokens


__all__ = ["StablePrefixTokenizer"]
