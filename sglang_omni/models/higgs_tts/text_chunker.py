# SPDX-License-Identifier: Apache-2.0
"""Language-agnostic, TTS-ready text chunker — Higgs TTS streaming preprocessing.

This is Higgs TTS's streaming text-input strategy: ``HiggsTtsPipelineConfig``
declares it via ``create_streaming_text_splitter`` and the serve layer receives
it through dependency injection (see
:mod:`sglang_omni.utils.streaming_text`). Ported from the reference ``chunk.py``
chunker and adapted for incremental WebSocket input. Two layers:

- A batch chunker (:func:`chunk_text`) that splits a complete string into
  TTS-ready chunks. Each chunk is one sentence; adjacent sentences are never
  merged (preserves prosody boundaries). An oversized sentence is sub-split
  through tiers 2-5 (clause → close-punct → whitespace → per-character) until
  every piece fits a per-chunk time budget.
- An incremental wrapper (:class:`StreamingTextChunker`) that buffers text
  fragments arriving over the wire and emits completed sentences as soon as
  their boundary is *confirmed*, holding the unconfirmed tail until more text
  (or :meth:`StreamingTextChunker.flush`) arrives.

Sentence detection is language-agnostic via the third-party ``regex`` package's
Unicode property classes (``\\p{STerm}`` etc.), covering Latin, CJK, Arabic,
Devanagari, Armenian, and more. To stay safe for *incremental* input we follow
the vLLM-Omni streaming rule: ASCII ``.``/``!``/``?`` are sentence boundaries
only when followed by whitespace (so ``3.14`` / ``U.S.A`` / a mid-token period
are not premature cuts), while unambiguous non-ASCII terminators (``。！？``,
Arabic ``؟``, Devanagari ``।`` …) cut immediately. With
``split_granularity="clause"`` non-ASCII clause terminators (``，；、`` …) also
become top-level boundaries.

Time estimation uses a single chars-per-second (CPS) rate; the streaming path
uses a fixed conservative :data:`DEFAULT_CPS` (no per-session audio I/O — the
CPS only sets the oversized-sentence sub-split threshold, so precision is
low-value), while :func:`estimate_cps` can derive it from a known
``(text, duration)`` pair for offline/batch callers. Markup tags (``<|...|>``)
are tag-aware: state
tags (emotion/style/prosody) force a boundary and prefix every following chunk;
``<|clear|>`` resets that state; transient tags (pause/sfx) stay inline and cost
no synthesis time.
"""

from __future__ import annotations

from functools import lru_cache

import regex  # third-party `regex`: PCRE-style Unicode property classes

DEFAULT_MAX_SECONDS = 8.0
DEFAULT_CPS = 10.0  # fallback when no reference is available

# Unicode property classes (see chunk.py for the full rationale):
#   STerm  — Sentence_Terminal (.!? 。！？ ؟ ։ … across scripts)
#   Term   — Terminal_Punctuation; (Term − STerm) = clause-internal stops
#   Pe/Pf  — close brackets / final quotes
_SENT_END_CLS = r"[\p{STerm}…]"
_CLAUSE_CLS = r"[\p{Term}--\p{STerm}]"
_CLOSE_PUNCT_CLS = r"[\p{Pe}\p{Pf}\"']"

# Top-level (immediate) boundary terminators per granularity. ASCII .!? are
# handled separately (they need trailing whitespace), so the immediate set is
# the *non-ASCII* terminators only.
_IMMEDIATE_SENT_CLS = r"[\p{STerm}…--[\x00-\x7f]]"
_IMMEDIATE_CLAUSE_CLS = r"[\p{Term}…--[\x00-\x7f]]"

_TAG_RE = regex.compile(r"<\|[^|]+\|>")
# State tags whose effect persists over the following speech (emotion/style/
# prosody/clear). A run of consecutive ones forces a chunk boundary.
_STATE_TAG_RUN_RE = regex.compile(
    r"(?:<\|(?:clear"
    r"|emotion:[^|]+"
    r"|style:[^|]+"
    r"|prosody:(?:expressive_[^|]+|speed_[^|]+|pitch_[^|]+))\|>)+"
)
_CLEAR_TAG = "<|clear|>"
_TAG_SENTINEL = "￾"  # Unicode noncharacter: never valid text, not whitespace

# A trailing, not-yet-closed tag at the end of a buffer ("<|emo" or a lone "<").
_INCOMPLETE_TAG_TAIL_RE = regex.compile(r"<(?:\|(?:(?!\|>).)*)?$", regex.DOTALL)

_WORD_RE = regex.compile(r"\S+\s*|\s+")


# ---------------------------------------------------------------------------
# Time estimation
# ---------------------------------------------------------------------------
def _spoken_len(text: str) -> int:
    """Count synthesised characters: excludes whitespace, tags, and sentinel."""
    stripped = _TAG_RE.sub("", text).replace(_TAG_SENTINEL, "")
    return sum(1 for c in stripped if not c.isspace())


def estimate_seconds(text: str, cps: float = DEFAULT_CPS) -> float:
    """Estimated TTS synthesis time in seconds at ``cps`` chars/sec."""
    return _spoken_len(text) / cps if cps > 0 else 0.0


def estimate_cps(ref_text: str | None, ref_audio_dur_s: float | None) -> float:
    """Chars/sec measured from a reference ``(text, audio_duration)`` pair.

    Counts spoken (non-whitespace, non-tag) characters only. Falls back to
    :data:`DEFAULT_CPS` when the reference is missing or unusable.
    """
    if not ref_text or not ref_audio_dur_s or ref_audio_dur_s <= 0:
        return DEFAULT_CPS
    n = _spoken_len(ref_text)
    if n == 0:
        return DEFAULT_CPS
    return n / ref_audio_dur_s


# ---------------------------------------------------------------------------
# Tag handling
# ---------------------------------------------------------------------------
def _classify_tag(tag: str) -> str:
    """Return one of ``{'state', 'clear', 'transient'}`` for a ``<|...|>`` tag."""
    if tag == _CLEAR_TAG:
        return "clear"
    if _STATE_TAG_RUN_RE.fullmatch(tag):
        return "state"
    return "transient"


def _iter_tokens(text: str):
    """Lex ``text`` into ``(kind, text)`` tokens.

    ``kind`` is ``'text'``, ``'state'``, ``'clear'``, or ``'transient'``.
    """
    last = 0
    for m in _TAG_RE.finditer(text):
        if m.start() > last:
            yield ("text", text[last : m.start()])
        yield (_classify_tag(m.group(0)), m.group(0))
        last = m.end()
    if last < len(text):
        yield ("text", text[last:])


# ---------------------------------------------------------------------------
# Tier 2-5 sub-splitting (verbatim port of chunk.py, sentinel/tag aware)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _split_re(cls: str):
    return regex.compile(rf".*?{cls}+|.+", regex.V1 | regex.DOTALL)


def _split_keep(text: str, compiled) -> list[str]:
    """Split by ``compiled`` keeping each match verbatim (whitespace preserved)."""
    return [m.group(0) for m in compiled.finditer(text) if m.group(0)]


def _split_words(text: str) -> list[str]:
    return _split_keep(text, _WORD_RE)


def _budget_split_chars(text: str, max_seconds: float, cps: float) -> list[str]:
    """Tier 5: greedy-pack characters into pieces of ≤ ``max_seconds``."""
    out: list[str] = []
    buf: list[str] = []
    t = 0.0
    for c in text:
        ct = 0.0 if (c.isspace() or c == _TAG_SENTINEL) else 1.0 / cps
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
    parts = _split_keep(piece, _split_re(_CLAUSE_CLS))
    if len(parts) > 1:
        out: list[str] = []
        for p in parts:
            out.extend(_refine_tier3(p, max_seconds, cps))
        return out
    return _refine_tier3(piece, max_seconds, cps)


def _refine_tier3(piece: str, max_seconds: float, cps: float) -> list[str]:
    if estimate_seconds(piece, cps) <= max_seconds:
        return [piece]
    parts = _split_keep(piece, _split_re(_CLOSE_PUNCT_CLS))
    if len(parts) > 1:
        out: list[str] = []
        for p in parts:
            out.extend(_refine_tier4(p, max_seconds, cps))
        return out
    return _refine_tier4(piece, max_seconds, cps)


def _refine_tier4(piece: str, max_seconds: float, cps: float) -> list[str]:
    if estimate_seconds(piece, cps) <= max_seconds:
        return [piece]
    parts = _split_words(piece)
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
    """Greedy-merge consecutive fragments while staying under budget."""
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


def _refine_oversized(body: str, max_seconds: float, cps: float) -> list[str]:
    """Tier-2..5 refine + repack on an oversized sentence ``body``.

    Inline ``<|...|>`` tags are sentinel-substituted for the duration of the
    split so their inner ``:`` / ``.`` cannot break a tag, then restored.
    """
    tags: list[str] = []

    def _sub(m):
        tags.append(m.group(0))
        return _TAG_SENTINEL

    stripped = _TAG_RE.sub(_sub, body)
    parts = _pack(_refine(stripped, max_seconds, cps), max_seconds, cps)
    out: list[str] = []
    idx = 0
    for p in parts:
        while _TAG_SENTINEL in p:
            p = p.replace(_TAG_SENTINEL, tags[idx], 1)
            idx += 1
        out.append(p)
    assert idx == len(tags), f"tag/sentinel mismatch: consumed {idx} of {len(tags)}"
    return out


# ---------------------------------------------------------------------------
# Sentence boundary detection (vLLM-Omni streaming rule, language-agnostic)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _boundary_re(granularity: str):
    immediate = (
        _IMMEDIATE_CLAUSE_CLS if granularity == "clause" else _IMMEDIATE_SENT_CLS
    )
    # Non-ASCII terminators cut immediately; ASCII .!? only before whitespace.
    return regex.compile(rf"{immediate}+|[.!?]+(?=\s)", regex.V1 | regex.DOTALL)


def _split_confirmed(text: str, granularity: str) -> tuple[list[str], str]:
    """Split a text run into confirmed sentence pieces + an unconfirmed tail."""
    boundary = _boundary_re(granularity)
    pieces: list[str] = []
    last = 0
    for m in boundary.finditer(text):
        pieces.append(text[last : m.end()])
        last = m.end()
    return pieces, text[last:]


def _has_spoken(text: str) -> bool:
    return _spoken_len(text) > 0


def _emit_chunks(body: str, state: str, max_seconds: float, cps: float) -> list[str]:
    """Turn one confirmed sentence ``body`` into one-or-more emitted chunks.

    Prepends the active ``state`` tag run to each chunk and strips surrounding
    whitespace. Chunks with no spoken content are dropped.
    """
    if not _has_spoken(body):
        return []
    if estimate_seconds(body, cps) <= max_seconds:
        pieces = [body]
    else:
        pieces = _refine_oversized(body, max_seconds, cps)
    out: list[str] = []
    for piece in pieces:
        cleaned = piece.strip()
        if not _has_spoken(cleaned):
            continue
        out.append(state + cleaned)
    return out


def _segment(
    text: str,
    *,
    state: str,
    state_used: bool,
    granularity: str,
    max_seconds: float,
    cps: float,
    final: bool,
) -> tuple[list[str], str, str, bool]:
    """Causal walker. Returns ``(chunks, leftover, state, state_used)``.

    ``leftover`` is the raw, not-yet-confirmed trailing text (empty when
    ``final``). State carries across calls so a state tag set in one feed
    prefixes chunks produced by later feeds.
    """
    out: list[str] = []
    cur = ""  # raw in-progress text (no state prefix), incl. inline transient tags

    def finalize() -> None:
        nonlocal cur, state, state_used
        body, cur = cur, ""
        chunks = _emit_chunks(body, state, max_seconds, cps)
        if chunks:
            out.extend(chunks)
            state_used = True

    for kind, tok in _iter_tokens(text):
        if kind == "state":
            finalize()
            if not state:
                state = tok  # first state ever (or after a clear)
            elif not state_used:
                state += tok  # accumulate consecutive state tags
            else:
                state = tok  # replace: prior state already produced a chunk
                state_used = False
        elif kind == "clear":
            finalize()
            state = ""
            state_used = False
        elif kind == "transient":
            cur += tok  # inline, no cut, zero spoken cost
        else:  # text
            pieces, remainder = _split_confirmed(tok, granularity)
            for piece in pieces:
                cur += piece
                finalize()
            cur += remainder

    if final:
        finalize()
        return out, "", state, state_used
    return out, cur, state, state_used


def chunk_text(
    text: str,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    cps: float = DEFAULT_CPS,
    *,
    split_granularity: str = "sentence",
) -> list[str]:
    """Split a complete ``text`` into TTS-ready chunks under ``max_seconds`` each.

    Empty / whitespace-only input → ``[]``. State tags are propagated, so each
    chunk is self-contained for TTS dispatch; the ``''.join(chunks)`` invariant
    is intentionally not preserved.
    """
    text = (text or "").replace(_TAG_SENTINEL, "").strip()
    if not text:
        return []
    chunks, _, _, _ = _segment(
        text,
        state="",
        state_used=False,
        granularity=split_granularity,
        max_seconds=max_seconds,
        cps=cps,
        final=True,
    )
    return chunks


class StreamingTextChunker:
    """Incremental, tag-aware sentence chunker for streaming text input.

    Buffers fragments and returns completed sentences from :meth:`add_text` as
    soon as their boundary is confirmed; :meth:`flush` drains the remainder.
    Carries state-tag and split state across calls.
    """

    def __init__(
        self,
        split_granularity: str = "sentence",
        *,
        max_seconds: float = DEFAULT_MAX_SECONDS,
        cps: float = DEFAULT_CPS,
        max_buffer_chars: int | None = None,
    ) -> None:
        self._granularity = "clause" if split_granularity == "clause" else "sentence"
        self._max_seconds = max_seconds
        self._cps = cps if cps and cps > 0 else DEFAULT_CPS
        self._max_buffer_chars = max_buffer_chars
        self._buffer = ""
        self._state = ""
        self._state_used = False

    def add_text(self, text: str) -> list[str]:
        if not text:
            return []
        self._buffer += text.replace(_TAG_SENTINEL, "")
        if (
            self._max_buffer_chars is not None
            and len(self._buffer) > self._max_buffer_chars
        ):
            raise ValueError(
                "Text buffer exceeded maximum size "
                f"({self._max_buffer_chars} chars). "
                "Consider adding sentence-ending punctuation to your input."
            )

        # Hold back a trailing, not-yet-closed tag so we never split inside it.
        m = _INCOMPLETE_TAG_TAIL_RE.search(self._buffer)
        if m:
            region, partial_tag = self._buffer[: m.start()], self._buffer[m.start() :]
        else:
            region, partial_tag = self._buffer, ""

        chunks, leftover, self._state, self._state_used = _segment(
            region,
            state=self._state,
            state_used=self._state_used,
            granularity=self._granularity,
            max_seconds=self._max_seconds,
            cps=self._cps,
            final=False,
        )
        self._buffer = leftover + partial_tag
        return chunks

    def flush(self) -> list[str]:
        region = self._buffer
        self._buffer = ""
        if not region.strip():
            self._state = ""
            self._state_used = False
            return []
        chunks, _, self._state, self._state_used = _segment(
            region,
            state=self._state,
            state_used=self._state_used,
            granularity=self._granularity,
            max_seconds=self._max_seconds,
            cps=self._cps,
            final=True,
        )
        return chunks


__all__ = [
    "DEFAULT_CPS",
    "DEFAULT_MAX_SECONDS",
    "StreamingTextChunker",
    "chunk_text",
    "estimate_cps",
    "estimate_seconds",
]
