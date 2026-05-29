# SPDX-License-Identifier: Apache-2.0
"""Language-agnostic TTS text chunker for Higgs TTS (ported from ``chunk.py``).

Each chunk = one sentence; adjacent sentences are never merged (preserves
prosody boundaries). An oversized sentence (synthesis time > ``max_seconds``)
sub-splits through 5 tiers — sentence punct → clause punct → close brackets/
quotes → whitespace → per-char — re-packed within the sentence only.

:class:`HiggsTextChunker` offers batch :meth:`chunk` and incremental
:meth:`add_text`/:meth:`flush` (WS stream-in). Boundary rule (vLLM-Omni): ASCII
``.!?`` only split when followed by whitespace (so ``3.14``/``U.S.A`` survive);
non-ASCII terminators (``。！？``…) split immediately. Sizing uses a CPS budget
(default 10); callers may pass a per-request ``cps`` from the reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import regex  # PCRE-style Unicode \p{...} property classes

DEFAULT_MAX_SECONDS = 8.0
DEFAULT_CPS = 10.0  # ~10 chars/s sits between Latin (~15-20) and CJK (~3-5)


class TextChunker(Protocol):
    """Splits text into sentence-bounded TTS chunks. Batch :meth:`chunk`, or
    incremental :meth:`add_text`/:meth:`flush` (WS stream-in); a 1-element
    ``chunk`` return means "no chunking needed"."""

    def chunk(self, text: str, *, cps: float | None = None) -> list[str]: ...

    def add_text(self, text: str) -> list[str]: ...

    def flush(self) -> list[str]: ...


@dataclass(frozen=True)
class ChunkerOptions:
    """Launch-time chunker knobs (from the model's :class:`PipelineConfig`)."""

    split_granularity: str = "sentence"
    max_seconds: float = DEFAULT_MAX_SECONDS  # per-chunk synthesis-time budget
    # Codec Hz — lets the orchestrator get ref duration from ``vq_codes`` rows
    # for CPS. ``None`` → skip that source (no audio I/O), use the default CPS.
    codec_frame_rate: float | None = None


# Unicode classes (regex \p{}): STerm = sentence terminals (.!? 。！？…); the
# Term−STerm difference = clause stops (, ; : 、); Pe/Pf = close brackets +
# final quotes.
_CLAUSE_CLS = r"[\p{Term}--\p{STerm}]"
_CLOSE_PUNCT_CLS = r"[\p{Pe}\p{Pf}\"']"
# Non-ASCII terminators — cut immediately (no 3.14/U.S.A ambiguity).
_IMMEDIATE_SENT_CLS = r"[\p{STerm}…--[\x00-\x7f]]"
_IMMEDIATE_CLAUSE_CLS = r"[\p{Term}…--[\x00-\x7f]]"

# ASCII .!? are boundaries only before whitespace/end; non-ASCII run greedily.
# Shared by batch + streaming so behaviour matches.
_SENT_BOUNDARY_RE = regex.compile(
    rf"(?:{_IMMEDIATE_SENT_CLS}+|[.!?](?=\s|$))",
    regex.V1,
)
# Streaming variant: ASCII needs trailing whitespace (no end-of-input shortcut;
# ``flush`` handles end-of-input).
_CONFIRMED_SENT_RE = regex.compile(
    rf"(?:{_IMMEDIATE_SENT_CLS}|[.!?](?=\s))",
    regex.V1,
)
_CONFIRMED_CLAUSE_RE = regex.compile(
    rf"(?:{_IMMEDIATE_CLAUSE_CLS}|[.!?](?=\s))",
    regex.V1,
)

_WORD_RE = regex.compile(r"\S+\s*|\s+")


def _spoken_len(text: str) -> int:
    """Non-whitespace char count (whitespace is silent in TTS)."""
    return sum(1 for c in text if not c.isspace())


def estimate_seconds(text: str, cps: float = DEFAULT_CPS) -> float:
    """Estimated synthesis time (s) at ``cps`` chars/sec."""
    return _spoken_len(text) / cps if cps > 0 else 0.0


@lru_cache(maxsize=None)
def _split_re(cls: str):
    """Match ``<anything><delim+>`` runs plus a trailing tail (no delim)."""
    return regex.compile(rf".*?{cls}+|.+", regex.V1 | regex.DOTALL)


_CLAUSE_RE = _split_re(_CLAUSE_CLS)
_CLOSE_RE = _split_re(_CLOSE_PUNCT_CLS)


def _split_keep(text: str, compiled) -> list[str]:
    """Split keeping each match verbatim (whitespace preserved)."""
    return [m.group(0) for m in compiled.finditer(text) if m.group(0)]


def _budget_split_chars(text: str, max_seconds: float, cps: float) -> list[str]:
    """Tier 5: greedy-pack chars into pieces of ≤ ``max_seconds`` (whitespace
    flows through; pieces re-concatenate to ``text``)."""
    out: list[str] = []
    buf: list[str] = []
    t = 0.0
    for c in text:
        ct = 0.0 if c.isspace() else 1.0 / cps
        if buf and t + ct > max_seconds:
            out.append("".join(buf))
            buf, t = [c], ct
        else:
            buf.append(c)
            t += ct
    if buf:
        out.append("".join(buf))
    return out


def _refine(piece: str, max_seconds: float, cps: float) -> list[str]:
    """Apply tiers 2→5 to one piece until under budget."""
    if estimate_seconds(piece, cps) <= max_seconds:
        return [piece]
    parts = _split_keep(piece, _CLAUSE_RE)  # tier 2: clause
    if len(parts) > 1:
        out: list[str] = []
        for p in parts:
            out.extend(_refine_tier3(p, max_seconds, cps))
        return out
    return _refine_tier3(piece, max_seconds, cps)


def _refine_tier3(piece: str, max_seconds: float, cps: float) -> list[str]:
    if estimate_seconds(piece, cps) <= max_seconds:
        return [piece]
    parts = _split_keep(piece, _CLOSE_RE)
    if len(parts) > 1:
        out: list[str] = []
        for p in parts:
            out.extend(_refine_tier4(p, max_seconds, cps))
        return out
    return _refine_tier4(piece, max_seconds, cps)


def _refine_tier4(piece: str, max_seconds: float, cps: float) -> list[str]:
    if estimate_seconds(piece, cps) <= max_seconds:
        return [piece]
    parts = _split_keep(piece, _WORD_RE)
    if len(parts) > 1:
        out: list[str] = []
        for p in parts:
            if estimate_seconds(p, cps) <= max_seconds:
                out.append(p)
            else:
                out.extend(_budget_split_chars(p, max_seconds, cps))
        return out
    return _budget_split_chars(piece, max_seconds, cps)


def _pack(parts: list[str], max_seconds: float, cps: float) -> list[str]:
    """Greedy-merge consecutive fragments under budget (join with ``''``)."""
    out: list[str] = []
    buf = ""
    t = 0.0
    for p in parts:
        pt = estimate_seconds(p, cps)
        if not buf:
            buf, t = p, pt
            continue
        if t + pt <= max_seconds:
            buf, t = buf + p, t + pt
        else:
            out.append(buf)
            buf, t = p, pt
    if buf:
        out.append(buf)
    return out


def _chunk(
    text: str,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    cps: float = DEFAULT_CPS,
) -> list[str]:
    """Batch algorithm (public entry point is :meth:`HiggsTextChunker.chunk`).
    ``[]`` for empty input; ``[text]`` (1 chunk) when it fits one budget, so
    callers can fast-path to single-shot."""
    text = (text or "").strip()
    if not text:
        return []

    # Pass 1: cut at sentence boundaries (trailing tail = its own piece).
    pieces: list[str] = []
    last = 0
    for m in _SENT_BOUNDARY_RE.finditer(text):
        piece = text[last : m.end()]
        if piece:
            pieces.append(piece)
        last = m.end()
    if last < len(text):
        pieces.append(text[last:])

    # Pass 2: refine + repack oversized pieces within their own sentence.
    out: list[str] = []
    for p in pieces:
        if estimate_seconds(p, cps) > max_seconds:
            out.extend(_pack(_refine(p, max_seconds, cps), max_seconds, cps))
        else:
            out.append(p)
    return out


def _last_confirmed_cut(buffer: str, granularity: str) -> int:
    """Offset after the last confirmed sentence/clause boundary (trailing
    whitespace consumed); 0 if none yet confirmed."""
    pattern = (
        _CONFIRMED_SENT_RE if granularity == "sentence" else _CONFIRMED_CLAUSE_RE
    )
    last = None
    for m in pattern.finditer(buffer):
        last = m
    if last is None:
        return 0
    end = last.end()
    while end < len(buffer) and buffer[end].isspace():
        end += 1
    return end


class HiggsTextChunker:
    """:class:`TextChunker` for Higgs TTS. Stateless batch :meth:`chunk`;
    streaming :meth:`add_text`/:meth:`flush` hold buffer state, so create one
    instance per WS session."""

    def __init__(self, options: ChunkerOptions | None = None) -> None:
        opts = options or ChunkerOptions()
        self._max_seconds = opts.max_seconds
        self._granularity = opts.split_granularity
        # Read by the orchestrator for CPS (see ChunkerOptions.codec_frame_rate).
        self.codec_frame_rate: float | None = opts.codec_frame_rate
        self._buffer: str = ""

    def chunk(self, text: str, *, cps: float | None = None) -> list[str]:
        return _chunk(
            text,
            max_seconds=self._max_seconds,
            cps=cps if cps is not None else DEFAULT_CPS,
        )

    def add_text(self, text: str) -> list[str]:
        """Buffer a fragment; emit confirmed-complete sentences (the unconfirmed
        tail is held for the next ``add_text``/``flush``)."""
        if not text:
            return []
        self._buffer += text
        cut = _last_confirmed_cut(self._buffer, self._granularity)
        if cut == 0:
            return []
        confirmed = self._buffer[:cut]
        self._buffer = self._buffer[cut:]
        return _chunk(confirmed, max_seconds=self._max_seconds, cps=DEFAULT_CPS)

    def flush(self) -> list[str]:
        """Drain whatever remains at end-of-input."""
        if not self._buffer.strip():
            self._buffer = ""
            return []
        out = _chunk(self._buffer, max_seconds=self._max_seconds, cps=DEFAULT_CPS)
        self._buffer = ""
        return out


__all__ = [
    "ChunkerOptions",
    "DEFAULT_CPS",
    "DEFAULT_MAX_SECONDS",
    "HiggsTextChunker",
    "TextChunker",
    "estimate_seconds",
]
