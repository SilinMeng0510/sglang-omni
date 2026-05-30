# SPDX-License-Identifier: Apache-2.0
"""Engine-side cross-chunk continuity store for Higgs TTS.

Each chunk after the first is conditioned on the prior chunks' generated audio
(delay-pattern codes fed back like the voice reference). This lives engine-side
so the codes never surface to serve and the growing prompt prefix reuses radix
KV (per-session ``extra_key``); serve only splits text and tags each chunk with
a shared ``session_id``. No ``sglang`` import → unit-testable without a GPU.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field


def session_extra_key(session_id: str) -> str:
    """Radix-cache namespace: stable per session (prefix KV reuse), distinct
    across sessions (no KV bleed between differing overlaid embeddings)."""
    return "sess-" + hashlib.blake2b(
        session_id.encode("utf-8"), digest_size=12
    ).hexdigest()


@dataclass
class Segment:
    """One committed prior chunk: its text tokens + delayed audio codes, tagged
    with the chunk's ``index`` (serve's per-session sentence index) so a barge-in
    can roll history back to a specific chunk via :meth:`SessionStore.truncate_after`."""

    text_token_ids: list[int]
    codes_delayed: list[list[int]]
    index: int = -1  # serve sentence_index; -1 = unknown (pre-index caller)

    @property
    def num_audio_rows(self) -> int:
        return len(self.codes_delayed)


@dataclass
class SessionState:
    """Committed segments for one session. The voice reference is not stored —
    it rides in fresh (and deterministically) on every chunk's request."""

    segments: list[Segment] = field(default_factory=list)
    last_used_s: float = field(default_factory=time.monotonic)

    def history_for_prompt(
        self, cap: int
    ) -> tuple[list[tuple[list[int], int]], list[list[int]]]:
        """Sliding window as ``(prompt_history, overlay_codes)`` — the
        ``(text_ids, num_audio_rows)`` list for ``build_prompt_from_ids`` and
        those segments' codes concatenated for the runner's ``-100`` overlay.
        Produced together so they can't drift."""
        window = self.segments[-cap:] if cap > 0 else []
        prompt_history = [(s.text_token_ids, s.num_audio_rows) for s in window]
        overlay_codes: list[list[int]] = []
        for s in window:
            overlay_codes.extend(s.codes_delayed)
        return prompt_history, overlay_codes


class SessionStore:
    """Thread-safe per-``session_id`` store. Bounded three ways so a vanished
    client can't leak: explicit :meth:`evict`, idle TTL, and an LRU cap."""

    def __init__(
        self,
        *,
        max_history_chunks: int,
        max_sessions: int = 256,
        idle_ttl_s: float = 600.0,
    ) -> None:
        self.max_history_chunks = max(int(max_history_chunks), 0)
        self._max_sessions = max(int(max_sessions), 1)
        self._idle_ttl_s = float(idle_ttl_s)
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def history_for(
        self, session_id: str
    ) -> tuple[list[tuple[list[int], int]], list[list[int]]]:
        """Sliding-window history for ``session_id``'s next prompt; ``([], [])``
        for a new session (its first chunk is a plain single-shot prompt)."""
        with self._lock:
            self._gc_locked()
            state = self._sessions.get(session_id)
            if state is None:
                return [], []
            state.last_used_s = time.monotonic()
            return state.history_for_prompt(self.max_history_chunks)

    def commit(
        self,
        session_id: str,
        text_token_ids: list[int],
        codes_delayed: list[list[int]] | None,
        index: int = -1,
    ) -> None:
        """Append a generated chunk to the session. A chunk with no codes is
        dropped; only the most-recent ``max_history_chunks`` are kept. ``index``
        is the chunk's serve-side sentence index, recorded for :meth:`truncate_after`."""
        if not codes_delayed:
            return
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = SessionState()
                self._sessions[session_id] = state
            state.segments.append(
                Segment(
                    text_token_ids=list(text_token_ids),
                    codes_delayed=codes_delayed,
                    index=index,
                )
            )
            if self.max_history_chunks > 0:
                excess = len(state.segments) - self.max_history_chunks
                if excess > 0:
                    del state.segments[:excess]
            state.last_used_s = time.monotonic()
            self._enforce_capacity_locked()

    def truncate_after(self, session_id: str, last_kept_index: int) -> None:
        """Barge-in rollback: drop every committed segment whose chunk index is
        greater than ``last_kept_index`` (chunks generated ahead of playback but
        never heard). Segments with an unknown index (``-1``) are kept; no-op for
        an unknown session."""
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return
            state.segments = [
                s for s in state.segments if s.index < 0 or s.index <= last_kept_index
            ]
            state.last_used_s = time.monotonic()

    def evict(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _gc_locked(self) -> None:
        if self._idle_ttl_s <= 0:
            return
        now = time.monotonic()
        stale = [
            sid
            for sid, st in self._sessions.items()
            if now - st.last_used_s > self._idle_ttl_s
        ]
        for sid in stale:
            self._sessions.pop(sid, None)

    def _enforce_capacity_locked(self) -> None:
        while len(self._sessions) > self._max_sessions:
            oldest = min(
                self._sessions, key=lambda sid: self._sessions[sid].last_used_s
            )
            self._sessions.pop(oldest, None)


__all__ = [
    "Segment",
    "SessionState",
    "SessionStore",
    "session_extra_key",
]
