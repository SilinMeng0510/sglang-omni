# SPDX-License-Identifier: Apache-2.0
"""Language-agnostic, tag-aware TTS text chunker for Higgs TTS.

Each chunk is one sentence (adjacent sentences are never merged, to keep prosody
boundaries). A sentence whose synthesis time exceeds ``max_seconds`` is
sub-split through tiers — clause punct → close brackets/quotes → whitespace →
per-char — re-packed within that sentence only.

Control tags ``<|...|>`` are honoured (the model and users emit them). *State*
tags (``emotion``/``style``/``prosody:speed_*|pitch_*|expressive_*``) force a
boundary and are propagated as a prefix onto every following chunk until
``<|clear|>`` (which resets and is dropped) or the next state tag. *Transient*
tags (``prosody:pause``/``sfx:*`` …) stay inline and cost no budget.

:class:`HiggsTextChunker` offers batch :meth:`chunk` and incremental
:meth:`add_text`/:meth:`flush` (WS stream-in); both share one core walk. ASCII
``.!?`` only split before whitespace (so ``3.14``/``U.S.A`` survive); non-ASCII
terminators (``。！？``…) split immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import regex  # PCRE-style Unicode \p{...} property classes


class TextChunker(Protocol):
    """Batch :meth:`chunk`, or incremental :meth:`add_text`/:meth:`flush`
    (WS stream-in); a 1-element ``chunk`` return means "no chunking needed"."""

    def chunk(self, text: str, *, cps: float | None = None) -> list[str]: ...

    def add_text(self, text: str) -> list[str]: ...

    def flush(self) -> list[str]: ...

    def rearm_fastout(self) -> None: ...

    def reset(self) -> None: ...


@dataclass(frozen=True)
class ChunkerOptions:
    """Launch-time chunker knobs (from the model's :class:`PipelineConfig`)."""

    max_seconds: float
    cps: float
    fastout: bool = False
    codec_frame_rate: float | None = None


# Sentence / clause boundary matchers.
_SENT_RE, _CLAUSE_RE = (
    regex.compile(rf"(?:[{cls}…--[\x00-\x7f]]|[{asc}](?=\s))", regex.V1)
    for cls, asc in ((r"\p{STerm}", r".!?"), (r"\p{Term}", r".!?,;:"))
)

_REFINE_TIERS = tuple(
    regex.compile(rf".*?{cls}+|.+", regex.V1 | regex.DOTALL)
    for cls in (_CLAUSE_RE.pattern, r"[\p{Pe}\p{Pf}\"']", r"\s")
)

_TAG_RE = regex.compile(r"<\|[^|]+\|>")

_STATE_TAG_RUN_RE = regex.compile(
    r"(?:<\|(?:clear"
    r"|emotion:[^|]+"
    r"|style:[^|]+"
    r"|prosody:(?:expressive_[^|]+|speed_[^|]+|pitch_[^|]+))\|>)+"
)

_CLEAR_TAG = "<|clear|>"  # resets state; forces a cut, then is dropped

_ATOM_RE = regex.compile(
    r"(?:https?://|www\.)\S+"  # URL
    r"|[^\s@]+@[^\s@]+\.[^\s@]+"  # email
    r"|\d[\d.,:/-]*\d|\d"  # number / phone / time / date / IP
)


def _classify_tag(tag: str) -> str:
    """One of {'state', 'clear', 'transient'} for a ``<|...|>`` tag."""
    if tag == _CLEAR_TAG:
        return "clear"
    if _STATE_TAG_RUN_RE.fullmatch(tag):
        return "state"
    return "transient"


def _iter_tokens(text: str):
    """Lex ``text`` into ``(kind, text)`` tokens; kind in
    text/state/clear/transient."""
    last = 0
    for m in _TAG_RE.finditer(text):
        if m.start() > last:
            yield ("text", text[last : m.start()])
        yield (_classify_tag(m.group(0)), m.group(0))
        last = m.end()
    if last < len(text):
        yield ("text", text[last:])


def estimate_seconds(text: str, cps: float) -> float:
    """Estimated synthesis time (s); whitespace is silent so it's excluded."""
    spoken = sum(1 for c in text if not c.isspace())
    return spoken / cps if cps > 0 else 0.0


def _pack(parts: list[str], max_seconds: float, cps: float) -> list[str]:
    """Greedy-merge consecutive fragments under budget (join with ``''``). A
    whole-tag fragment is a control directive, not audio, so it costs nothing;
    an atom never gets split, so an over-budget atom stays whole on its own."""
    out: list[str] = []
    buf = ""
    t = 0.0
    for p in parts:
        pt = 0.0 if _TAG_RE.fullmatch(p) else estimate_seconds(p, cps)
        if buf and t + pt > max_seconds:
            out.append(buf)
            buf, t = p, pt
        else:
            buf, t = buf + p, t + pt
    if buf:
        out.append(buf)
    return out


def _char_atoms(piece: str) -> list[str]:
    """Atomize a piece for the per-character budget split: each char is its own
    atom, except a structured token (:data:`_ATOM_RE`) stays whole."""
    atoms: list[str] = []
    last = 0
    for m in _ATOM_RE.finditer(piece):
        atoms.extend(piece[last : m.start()])
        atoms.append(m.group(0))
        last = m.end()
    atoms.extend(piece[last:])
    return atoms


def _refine(piece: str, max_seconds: float, cps: float, tier: int = 0) -> list[str]:
    """Recursively split an oversized PURE-TEXT piece to fit the budget: try
    each tier, then a per-character split as the last resort."""
    if estimate_seconds(piece, cps) <= max_seconds:
        return [piece]
    if tier >= len(_REFINE_TIERS):
        return _pack(_char_atoms(piece), max_seconds, cps)
    parts = [m.group(0) for m in _REFINE_TIERS[tier].finditer(piece) if m.group(0)]
    if len(parts) <= 1:  # this tier didn't divide it — drop to the next
        return _refine(piece, max_seconds, cps, tier + 1)
    out: list[str] = []
    for p in parts:
        out.extend(_refine(p, max_seconds, cps, tier + 1))
    return out


def _refine_oversized(body: str, max_seconds: float, cps: float) -> list[str]:
    """Refine an oversized sentence that may carry inline tags. Tags never enter
    the splittable range: ``_TAG_RE`` lexes them out so only pure-text spans are
    split, each tag rides along as an atomic cost-free unit, and ``_pack``
    re-interleaves them in order — no tag is ever cut at its inner ``:``."""
    atoms: list[str] = []
    last = 0
    for m in _TAG_RE.finditer(body):
        if m.start() > last:
            atoms.extend(_refine(body[last : m.start()], max_seconds, cps))
        atoms.append(m.group(0))  # tag: atomic, never split
        last = m.end()
    if last < len(body):
        atoms.extend(_refine(body[last:], max_seconds, cps))
    return _pack(atoms, max_seconds, cps)


def _split_safe_tag(buffer: str) -> tuple[str, str]:
    """Hold back a trailing INCOMPLETE tag (a ``<|`` with no closing ``|>``) so
    a tag arriving across two ``add_text`` fragments isn't lexed mid-token.
    Returns ``(safe, held)``."""
    i = buffer.rfind("<|")
    if i != -1 and buffer.find("|>", i + 2) == -1:
        return buffer[:i], buffer[i:]
    return buffer, ""


def _walk(
    text: str,
    *,
    state: str,
    state_used: bool,
    emitted_first: bool,
    fastout: bool,
    max_seconds: float,
    cps: float,
    final: bool,
) -> tuple[list[str], str, str, bool, bool]:
    """Causal, tag-aware walk over a tag-safe slice.

    State tags force a cut and propagate as a prefix; ``<|clear|>`` resets;
    transient tags stay inline. Text cuts at confirmed sentence boundaries (the
    first chunk at a clause boundary when ``fastout`` and nothing emitted yet).
    When ``final`` the trailing fragment is emitted; otherwise it's returned as
    ``leftover`` to re-buffer.

    Returns ``(chunks, leftover, state, state_used, emitted_first)``.
    """
    chunks: list[str] = []
    body = ""  # spoken text + inline transient tags for the chunk-in-progress

    def emit_seg(seg: str) -> None:
        nonlocal emitted_first, state_used
        if not emitted_first:
            seg = seg.lstrip()  # trim the very first chunk's leading whitespace
        if not seg.strip():
            return
        if estimate_seconds(seg, cps) > max_seconds:
            chunks.extend(state + p for p in _refine_oversized(seg, max_seconds, cps))
        else:
            chunks.append(state + seg)
        state_used = emitted_first = True

    def flush_body() -> None:
        """Emit ``body`` under the current state (forced cut at a tag, or EOF)."""
        nonlocal body
        if not body:
            return
        if _TAG_RE.sub("", body).strip():  # real spoken text
            emit_seg(body)
        elif chunks:  # transient/whitespace filler → attach to the prior chunk
            chunks[-1] += body
        elif body.strip():  # standalone transient with no prior chunk (rare)
            emit_seg(body)
        body = ""

    for kind, tok in _iter_tokens(text):
        if kind == "state":
            flush_body()
            if not state:
                state = tok  # first state (or after a clear)
            elif not state_used:
                state += tok  # accumulate consecutive state tags
            else:
                state, state_used = tok, False  # replace: prior state was used
        elif kind == "clear":
            flush_body()
            state, state_used = "", False
        elif kind == "transient":
            body += tok  # inline, no cut, no budget cost
        else:  # text
            body += tok
            while body:
                rx = _CLAUSE_RE if (fastout and not emitted_first) else _SENT_RE
                m = rx.search(body)
                if not m:
                    break
                seg, body = body[: m.end()], body[m.end() :]
                emit_seg(seg)

    if final:
        flush_body()
    return chunks, ("" if final else body), state, state_used, emitted_first


class HiggsTextChunker:
    """:class:`TextChunker` for Higgs TTS. Batch :meth:`chunk` is stateless;
    streaming :meth:`add_text`/:meth:`flush` hold buffer + tag state, so use one
    instance per WS session."""

    def __init__(self, options: ChunkerOptions) -> None:
        self._max_seconds = options.max_seconds
        self._cps = options.cps
        self._fastout = options.fastout
        self.codec_frame_rate = options.codec_frame_rate  # read by orchestrator
        self._buffer = ""
        self._state = ""  # propagated state-tag prefix run
        self._state_used = False
        self._emitted_first = False

    def chunk(self, text: str, *, cps: float | None = None) -> list[str]:
        """Batch (stateless, no ``fastout``): ``[]`` for empty input; a 1-element
        list when it fits one budget, so callers can fast-path to single-shot."""
        text = (text or "").strip()
        if not text:
            return []
        chunks, *_ = _walk(
            text,
            state="",
            state_used=False,
            emitted_first=False,
            fastout=False,
            max_seconds=self._max_seconds,
            cps=self._cps if cps is None else cps,
            final=True,
        )
        return chunks

    def _drive(self, text: str, *, final: bool) -> tuple[list[str], str]:
        """Run the shared walk with this instance's tag state + fastout flag."""
        chunks, leftover, self._state, self._state_used, self._emitted_first = _walk(
            text,
            state=self._state,
            state_used=self._state_used,
            emitted_first=self._emitted_first,
            fastout=self._fastout,
            max_seconds=self._max_seconds,
            cps=self._cps,
            final=final,
        )
        return chunks, leftover

    def add_text(self, text: str) -> list[str]:
        """Buffer a fragment; emit confirmed-complete chunks. The unconfirmed
        tail (and a tag split across fragments) is held for the next call. With
        ``fastout`` the first chunk is released at the earliest clause boundary;
        state tags force a cut and propagate as a prefix across chunks."""
        if not text:
            return []
        self._buffer += text
        safe, held = _split_safe_tag(self._buffer)
        if not safe:
            return []  # only a partial tag so far — wait for the rest
        chunks, leftover = self._drive(safe, final=False)
        self._buffer = leftover + held
        return chunks

    def flush(self) -> list[str]:
        """Drain whatever remains at end-of-input."""
        text, self._buffer = self._buffer.strip(), ""
        if not text:
            return []
        return self._drive(text, final=True)[0]

    def rearm_fastout(self) -> None:
        """Re-arm ``fastout`` so the next chunk is again cut at the earliest
        clause boundary — the WS handler calls this on ``input.wait`` (agent
        paused) so every turn opens fast. Tag ``state`` is left intact."""
        self._emitted_first = False

    def reset(self) -> None:
        """Drop all buffered text and tag state — used on ``input.stop``
        (barge-in): the interrupted turn's unspoken remainder is abandoned and
        the next turn starts clean. ``fastout`` is re-armed (``_emitted_first``
        cleared) so the resumed turn opens fast."""
        self._buffer = ""
        self._state = ""
        self._state_used = False
        self._emitted_first = False


__all__ = [
    "ChunkerOptions",
    "HiggsTextChunker",
    "TextChunker",
    "estimate_seconds",
]
