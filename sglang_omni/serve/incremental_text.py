# SPDX-License-Identifier: Apache-2.0
"""Canonical safe-prefix incremental tokenization for streaming text input.

BPE prefix instability: ``encode(prefix)`` is not guaranteed to be a prefix
of ``encode(full_string)`` — tokens at the boundary can merge with
characters that arrive later. The streaming-TTS protocol has no undo (an
injected token cannot be retracted), and the model is trained exclusively
on canonical tokenizations, so incremental input must release ids that
(a) never change later and (b) match the canonical full-text tokenization.

Three layers, each with an explicit guarantee:

1. PRE-TOKEN FREEZING (exact). BPE merges never cross pre-tokenizer
   boundaries, and appending text can only reshape the LAST pre-token of
   the buffer — so once a pre-token is followed by another, its canonical
   tokenization is final and it releases immediately. The model's own
   pre-tokenizer regex decides the boundaries, so this is script-agnostic:
   Latin words, CJK punctuation runs, kana and Thai all behave identically.

2. SAFE-PREFIX COMMIT (canonical, empirically bounded) inside the trailing
   unfinished pre-token — e.g. an unpunctuated CJK run, which the regex
   keeps as one long pre-token. The trailing pre-token is re-tokenized on
   every push; tokens that agree with the previous tokenization (longest
   common prefix) AND sit at least ``guard_tokens`` behind the tip are
   committed. Measured merge rollback for this vocab is <= 3 tokens per
   appended character; the default guard of 4 covers it.

3. BREACH FALLBACK (bounded degradation, never corruption). Every step
   verifies that already-committed ids are still a prefix of the current
   tokenization. If a merge ever reaches past the guard, the committed
   text is sealed and the remainder re-tokenized fresh: one non-canonical
   seam instead of a corrupted stream. ``guard_breaches`` counts these
   (expected 0 on natural text).
"""

from __future__ import annotations

from typing import Callable, Sequence

EncodeFull = Callable[[str], tuple[Sequence[int], Sequence[tuple[int, int]]]]
PreTokenize = Callable[[str], Sequence[tuple[int, int]]]

DEFAULT_GUARD_TOKENS = 4
DEFAULT_MAX_BUFFER_CHARS = 8192


class StablePrefixTokenizer:
    """Incrementally tokenize streaming text, releasing only stable tokens.

    ``encode_full`` maps text to ``(token_ids, char_offsets)`` without
    special tokens; ``pre_tokenize`` maps text to the char spans of the
    model pre-tokenizer's pieces (contiguous, covering the string).
    """

    def __init__(
        self,
        encode_full: EncodeFull,
        pre_tokenize: PreTokenize,
        *,
        guard_tokens: int = DEFAULT_GUARD_TOKENS,
        max_buffer_chars: int = DEFAULT_MAX_BUFFER_CHARS,
    ) -> None:
        if guard_tokens < 1:
            raise ValueError("guard_tokens must be >= 1")
        self._encode_full = encode_full
        self._pre_tokenize = pre_tokenize
        self._guard = int(guard_tokens)
        self._max_buffer_chars = int(max_buffer_chars)
        # _buffer holds the trailing pre-token plus any newer text; its
        # first _committed_chars characters are already released (their ids
        # are _committed_ids) but must stay in the buffer so the pre-token
        # keeps re-tokenizing as one canonical unit.
        self._buffer = ""
        self._committed_ids: list[int] = []
        self._committed_chars = 0
        self._prev_ids: list[int] = []
        self._released_text = ""
        self._released_tokens = 0
        self.guard_breaches = 0
        self.forced_reseals = 0

    @property
    def pending_text(self) -> str:
        """Characters buffered but not yet released."""
        return self._buffer[self._committed_chars :]

    @property
    def released_text(self) -> str:
        """Concatenation of all released chunks."""
        return self._released_text

    @property
    def released_tokens(self) -> int:
        return self._released_tokens

    def push(self, text: str) -> list[int]:
        """Append ``text``; return newly released (stable) token ids."""
        if not text:
            return []
        self._buffer += text
        return self._process()

    def flush(self) -> list[int]:
        """Release everything still buffered (end of text / explicit flush)."""
        if not self._buffer:
            return []
        ids = [int(t) for t in self._encode_full(self._buffer)[0]]
        k = len(self._committed_ids)
        if ids[:k] != self._committed_ids:
            self.guard_breaches += 1
            self._seal()
            k = 0
            ids = (
                [int(t) for t in self._encode_full(self._buffer)[0]]
                if self._buffer
                else []
            )
        out = ids[k:]
        self._released_tokens += len(out)
        self._released_text += self._buffer[self._committed_chars :]
        self._buffer = ""
        self._committed_ids = []
        self._committed_chars = 0
        self._prev_ids = []
        return out

    def _process(self) -> list[int]:
        out: list[int] = []
        # a seal restarts processing on the remaining buffer; after a seal
        # the committed state is empty so a second breach cannot occur —
        # the loop bound is a hard backstop, not an expected path
        for _ in range(4):
            if not self._buffer:
                break
            spans = self._pre_tokenize(self._buffer)
            if len(spans) > 1 and not self._release_frozen(spans, out):
                continue  # breach: buffer was resealed, reprocess remainder
            if not self._buffer:
                break
            if self._soft_commit(out):
                break
        return out

    def _release_frozen(self, spans: Sequence[tuple[int, int]], out: list[int]) -> bool:
        """Release every complete pre-token (all spans but the last).

        Their canonical tokenization is final (layer 1). Returns False on a
        committed-prefix breach, after sealing the buffer.
        """
        frozen_end = spans[-1][0]
        ids_frozen: list[int] = []
        for start, end in spans[:-1]:
            ids_frozen.extend(
                int(t) for t in self._encode_full(self._buffer[start:end])[0]
            )
        k = len(self._committed_ids)
        if ids_frozen[:k] != self._committed_ids:
            self.guard_breaches += 1
            self._seal()
            return False
        out.extend(ids_frozen[k:])
        self._released_tokens += len(ids_frozen) - k
        self._released_text += self._buffer[self._committed_chars : frozen_end]
        self._buffer = self._buffer[frozen_end:]
        self._committed_ids = []
        self._committed_chars = 0
        self._prev_ids = []
        return True

    def _soft_commit(self, out: list[int]) -> bool:
        """Commit stable tokens inside the trailing pre-token (layer 2).

        Returns False when the buffer was sealed and needs reprocessing.
        """
        ids_raw, offsets = self._encode_full(self._buffer)
        ids = [int(t) for t in ids_raw]
        k = len(self._committed_ids)
        if ids[:k] != self._committed_ids:
            self.guard_breaches += 1
            self._seal()
            return False
        target = min(self._lcp(self._prev_ids, ids), len(ids) - self._guard)
        # byte-level BPE can split one character across two tokens; never
        # cut between tokens that share a character
        while k < target < len(ids) and offsets[target - 1][1] > offsets[target][0]:
            target -= 1
        if target > k:
            cut = offsets[target - 1][1]
            out.extend(ids[k:target])
            self._released_tokens += target - k
            self._released_text += self._buffer[self._committed_chars : cut]
            self._committed_ids = ids[:target]
            self._committed_chars = cut
        self._prev_ids = ids
        if len(self._buffer) > self._max_buffer_chars and self._committed_chars > 0:
            # cap the O(len) re-tokenization cost on degenerate
            # boundary-free runs; the seam is non-canonical (counted)
            self.forced_reseals += 1
            self._seal()
        return True

    def _seal(self) -> None:
        """Freeze the committed prefix as released text and restart the
        buffer from the remainder (which re-tokenizes fresh — one
        non-canonical seam)."""
        self._buffer = self._buffer[self._committed_chars :]
        self._committed_ids = []
        self._committed_chars = 0
        self._prev_ids = []

    @staticmethod
    def _lcp(a: Sequence[int], b: Sequence[int]) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i


__all__ = ["StablePrefixTokenizer", "DEFAULT_GUARD_TOKENS"]
