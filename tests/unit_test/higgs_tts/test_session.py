# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the engine-side cross-chunk continuity store.

Pure-Python — no sglang / GPU / torch. Covers the per-session radix key, the
sliding window, and the store's eviction policies.
"""

from __future__ import annotations

from sglang_omni.models.higgs_tts.session import (
    Segment,
    SessionState,
    SessionStore,
    session_extra_key,
)


# ---------------------------------------------------------------------------
# session_extra_key
# ---------------------------------------------------------------------------


def test_extra_key_stable_and_distinct() -> None:
    assert session_extra_key("abc") == session_extra_key("abc")
    assert session_extra_key("abc") != session_extra_key("abd")
    assert session_extra_key("abc").startswith("sess-")


# ---------------------------------------------------------------------------
# SessionState.history_for_prompt
# ---------------------------------------------------------------------------


def test_history_for_prompt_aligns_rows_and_overlay() -> None:
    state = SessionState()
    state.segments = [
        Segment([10, 11], [[1, 1], [2, 2], [3, 3]]),  # 3 audio rows
        Segment([12], [[4, 4], [5, 5]]),  # 2 audio rows
    ]
    prompt_history, overlay = state.history_for_prompt(cap=4)
    assert prompt_history == [([10, 11], 3), ([12], 2)]
    # Overlay codes concatenate in order; total rows == sum of declared counts.
    assert overlay == [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
    assert len(overlay) == sum(n for _, n in prompt_history)


def test_history_for_prompt_applies_window_cap() -> None:
    state = SessionState()
    state.segments = [Segment([i], [[i]]) for i in range(6)]
    prompt_history, overlay = state.history_for_prompt(cap=4)
    assert [t for t, _ in prompt_history] == [[2], [3], [4], [5]]
    assert overlay == [[2], [3], [4], [5]]


def test_history_for_prompt_cap_zero_disables_history() -> None:
    state = SessionState()
    state.segments = [Segment([1], [[1]])]
    assert state.history_for_prompt(cap=0) == ([], [])


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------


def test_store_commit_and_history_roundtrip() -> None:
    store = SessionStore(max_history_chunks=4)
    assert store.history_for("s") == ([], [])  # new session → empty
    store.commit("s", [10], [[1, 1]])
    store.commit("s", [11], [[2, 2]])
    prompt_history, overlay = store.history_for("s")
    assert prompt_history == [([10], 1), ([11], 1)]
    assert overlay == [[1, 1], [2, 2]]


def test_store_sliding_window_drops_oldest() -> None:
    store = SessionStore(max_history_chunks=3)
    for i in range(5):
        store.commit("s", [i], [[i]])
    prompt_history, _ = store.history_for("s")
    assert [t for t, _ in prompt_history] == [[2], [3], [4]]


def test_store_drops_chunk_without_codes() -> None:
    store = SessionStore(max_history_chunks=4)
    store.commit("s", [1], None)
    store.commit("s", [2], [])
    assert store.history_for("s") == ([], [])


def test_store_evict() -> None:
    store = SessionStore(max_history_chunks=4)
    store.commit("s", [1], [[1]])
    assert len(store) == 1
    store.evict("s")
    assert len(store) == 0
    assert store.history_for("s") == ([], [])


def test_store_idle_ttl_gc() -> None:
    # idle_ttl_s <= 0 disables GC, so the entry survives.
    store = SessionStore(max_history_chunks=4, idle_ttl_s=0.0)
    store.commit("s", [1], [[1]])
    assert len(store) == 1

    store2 = SessionStore(max_history_chunks=4, idle_ttl_s=1e-6)
    store2.commit("s", [1], [[1]])
    import time as _t

    _t.sleep(0.005)
    # A read triggers GC of the stale entry.
    assert store2.history_for("s") == ([], [])


def test_store_lru_capacity_cap() -> None:
    store = SessionStore(max_history_chunks=4, max_sessions=2)
    store.commit("a", [1], [[1]])
    store.commit("b", [1], [[1]])
    store.commit("c", [1], [[1]])  # evicts LRU ("a")
    assert len(store) == 2
    assert store.history_for("a") == ([], [])
    assert store.history_for("b") != ([], [])
    assert store.history_for("c") != ([], [])


# ---------------------------------------------------------------------------
# SessionStore.truncate_after (barge-in rollback)
# ---------------------------------------------------------------------------


def test_commit_records_index() -> None:
    store = SessionStore(max_history_chunks=8)
    store.commit("s", [0], [[0]], index=7)
    assert store._sessions["s"].segments[0].index == 7


def test_truncate_after_drops_chunks_past_index() -> None:
    # Server ran ahead of playback: chunks 0..3 committed, user barged in while
    # chunk 1 was playing → keep 0,1; drop 2,3 (generated but never spoken).
    store = SessionStore(max_history_chunks=8)
    for i in range(4):
        store.commit("s", [i], [[i]], index=i)
    store.truncate_after("s", 1)
    prompt_history, overlay = store.history_for("s")
    assert [t for t, _ in prompt_history] == [[0], [1]]
    assert overlay == [[0], [1]]


def test_truncate_after_to_latest_is_noop() -> None:
    store = SessionStore(max_history_chunks=8)
    for i in range(3):
        store.commit("s", [i], [[i]], index=i)
    store.truncate_after("s", 2)  # last committed chunk → nothing to drop
    prompt_history, _ = store.history_for("s")
    assert [t for t, _ in prompt_history] == [[0], [1], [2]]


def test_truncate_after_unknown_session_is_noop() -> None:
    store = SessionStore(max_history_chunks=8)
    store.truncate_after("missing", 0)  # must not raise
    assert store.history_for("missing") == ([], [])


def test_truncate_after_keeps_unindexed_segments() -> None:
    # A segment committed without an index (-1) can't be located, so it's kept;
    # only positively-identified later chunks are dropped.
    store = SessionStore(max_history_chunks=8)
    store.commit("s", [0], [[0]])  # index defaults to -1
    store.commit("s", [1], [[1]], index=1)
    store.truncate_after("s", 0)
    prompt_history, _ = store.history_for("s")
    assert [t for t, _ in prompt_history] == [[0]]
