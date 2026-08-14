# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the streaming-TTS protocol state machine.

Drives :class:`StreamingProtocolState` and a mini-simulator transcribed
directly from the reference loop structure in higgs-mm
``higgs_mm/eval/tts/streaming.py`` (``generate_streaming_tts``) with the same
scripted model behavior (decision outcomes + EOC position), and asserts both
produce the same protocol event sequence and row counts.
"""

from __future__ import annotations

import pytest

from sglang_omni.models.higgs_tts.streaming_protocol import (
    StepRole,
    StreamingProtocolConfig,
    StreamingProtocolState,
)

TEXT_ID = 151672
AUDIO_ID = 151670
TEXT_END_ID = 151673
EXTRAS = 7  # num_codebooks - 1
K = 4
WINDDOWN = 6  # num_codebooks - 2 rows after the EOC row


def _cfg(**kw) -> StreamingProtocolConfig:
    defaults = dict(
        text_token_id=TEXT_ID,
        audio_token_id=AUDIO_ID,
        text_end_token_id=TEXT_END_ID,
        num_extra_tokens=EXTRAS,
        frames_per_block=K,
        max_opening_waits=16,
        max_mid_waits=16,
        max_frames=200,
    )
    defaults.update(kw)
    return StreamingProtocolConfig(**defaults)


def _reference_trace(
    text_ids: list[int],
    decisions: list[bool],
    *,
    eoc_row: int | None,
    cfg: StreamingProtocolConfig,
) -> tuple[list, int]:
    """Transcription of generate_streaming_tts's loop structure.

    ``decisions`` yields the constrained decision at every sample_decision
    call (True = <|audio|>). ``eoc_row`` is the 0-based row index at which
    cb0 first samples EOC (stream then winds down ``WINDDOWN`` more rows);
    None = never (length fuse).
    Returns (trace, rows_sampled) with ('audio', n) runs merged like the
    reference emit_rows trace.
    """
    decisions = iter(decisions)
    trace: list = [("inject", text_ids[0])]
    injected = 1
    n = len(text_ids)
    rows = 0
    opening_waits = mid_waits = 0
    text_end_sent = False
    done = False

    def emit_rows(n_rows: int, stop_at_eoc: bool) -> bool:
        nonlocal rows
        emitted = 0
        hit = False
        end_idx = None
        while emitted < n_rows:
            # sample one row
            row_idx = rows
            rows += 1
            emitted += 1
            if end_idx is None and eoc_row is not None and row_idx == eoc_row:
                end_idx = emitted - 1
                if not stop_at_eoc:
                    hit = True
                    break
            if end_idx is not None and (emitted - 1) >= end_idx + WINDDOWN:
                hit = True
                break
        trace.append(("audio", emitted))
        if end_idx is not None:
            trace.append(("eoc",))
        return hit

    # opening
    while True:
        choice_audio = next(decisions)
        if choice_audio or text_end_sent:
            break
        opening_waits += 1
        if opening_waits > cfg.max_opening_waits:
            break
        trace.append(("wait",))
        if injected < n:
            trace.append(("inject", text_ids[injected]))
            injected += 1
        else:
            trace.append(("text_end",))
            text_end_sent = True

    n_first = (
        (cfg.max_frames + cfg.num_extra_tokens)
        if text_end_sent
        else (cfg.num_extra_tokens + cfg.frames_per_block)
    )
    done = emit_rows(n_first, stop_at_eoc=text_end_sent)

    while not done and not text_end_sent:
        if injected < n:
            trace.append(("inject", text_ids[injected]))
            injected += 1
        else:
            trace.append(("text_end",))
            text_end_sent = True
        while True:
            choice_audio = next(decisions)
            if choice_audio or text_end_sent:
                break
            mid_waits += 1
            if mid_waits > cfg.max_mid_waits:
                break
            trace.append(("wait",))
            if injected < n:
                trace.append(("inject", text_ids[injected]))
                injected += 1
            else:
                trace.append(("text_end",))
                text_end_sent = True
                break
        if text_end_sent:
            done = emit_rows(
                cfg.max_frames + cfg.num_extra_tokens - rows, stop_at_eoc=True
            )
        else:
            done = emit_rows(cfg.frames_per_block, stop_at_eoc=False)
        if rows >= cfg.max_frames + cfg.num_extra_tokens:
            done = True

    return trace, rows


def _machine_trace(
    text_ids: list[int],
    decisions: list[bool],
    *,
    eoc_row: int | None,
    cfg: StreamingProtocolConfig,
) -> tuple[list, int, StreamingProtocolState]:
    """Drive StreamingProtocolState the way the model runner does."""
    state = StreamingProtocolState(cfg=cfg, inject_text_ids=list(text_ids[1:]))
    state.trace.append(("inject", text_ids[0]))
    decisions = iter(decisions)
    rows = 0
    generation_done = False
    eoc_abort = False
    winddown_left: int | None = None
    for _ in range(100_000):
        role = state.role
        decision = None
        if role is StepRole.DECISION:
            decision = next(decisions)
        elif role is StepRole.AUDIO:
            # emulate the delay/EOC sampler for this sampled row, mirroring
            # the runner: non-tail EOC aborts, tail EOC winds down N-2 rows
            row_idx = rows
            rows += 1
            if winddown_left is not None:
                winddown_left -= 1
                if winddown_left <= 0:
                    generation_done = True
            elif eoc_row is not None and row_idx == eoc_row:
                if state.in_tail_flush:
                    winddown_left = WINDDOWN
                else:
                    eoc_abort = True
        plan = state.on_step_output(decision)
        # invariants: a step whose output is an audio row must advance the
        # sampler; every other step must freeze it. Text-space inputs must
        # carry a token id (starved plans carry none by definition).
        assert plan.audio_advance == (plan.role is StepRole.AUDIO)
        if not plan.starved:
            assert (plan.input_token_id is not None) == plan.input_is_text
        if generation_done or eoc_abort or plan.hard_stop:
            break
    else:  # pragma: no cover
        raise AssertionError("machine did not terminate")
    if eoc_row is not None and rows > eoc_row:
        state._flush_audio_trace()
        state.trace.append(("eoc",))
    return state.finalize_trace(), rows, state


def _assert_parity(text_ids, decisions, *, eoc_row, cfg=None):
    cfg = cfg or _cfg()
    ref_trace, ref_rows = _reference_trace(
        text_ids, list(decisions), eoc_row=eoc_row, cfg=cfg
    )
    got_trace, got_rows, _ = _machine_trace(
        text_ids, list(decisions), eoc_row=eoc_row, cfg=cfg
    )
    assert got_trace == ref_trace
    assert got_rows == ref_rows


TEXT = list(range(1000, 1012))  # 12 fake text token ids


def test_immediate_audio_short_utterance():
    # audio decision straight away at every boundary; EOC mid-way
    decisions = [True] * 50
    _assert_parity(TEXT, decisions, eoc_row=40)


def test_opening_waits_then_audio():
    decisions = [False, False, True] + [True] * 50
    _assert_parity(TEXT, decisions, eoc_row=45)


def test_mid_stream_wait_honored():
    # audio open, then one mid-stream wait at the second boundary
    decisions = [True, False, True] + [True] * 50
    _assert_parity(TEXT, decisions, eoc_row=45)


def test_text_exhausted_tail_flush():
    # long audio: all 12 tokens injected, then tail flush to EOC
    decisions = [True] * 60
    _assert_parity(TEXT, decisions, eoc_row=70)


def test_whole_text_fits_in_opening():
    # model keeps waiting until text_end during the opening
    decisions = [False] * 20 + [True] * 5
    _assert_parity(TEXT, decisions, eoc_row=30)


def test_opening_fuse_trips():
    cfg = _cfg(max_opening_waits=3)
    decisions = [False] * 10 + [True] * 50
    _assert_parity(TEXT, decisions, eoc_row=35, cfg=cfg)
    _, _, state = _machine_trace(TEXT, [False] * 10 + [True] * 50, eoc_row=35, cfg=cfg)
    assert state.fuse_tripped


def test_no_eoc_hits_length_fuse():
    cfg = _cfg(max_frames=40)
    decisions = [True] * 100
    ref_trace, ref_rows = _reference_trace(TEXT, list(decisions), eoc_row=None, cfg=cfg)
    got_trace, got_rows, state = _machine_trace(
        TEXT, list(decisions), eoc_row=None, cfg=cfg
    )
    assert state.fuse_tripped
    assert got_rows == ref_rows == cfg.max_frames + cfg.num_extra_tokens


def test_eoc_immediately_in_first_block():
    decisions = [True] * 20
    _assert_parity(TEXT, decisions, eoc_row=2)


def test_stats_counters():
    cfg = _cfg()
    decisions = [False, False, True] + [True, False, True] + [True] * 50
    _, _, state = _machine_trace(TEXT, list(decisions), eoc_row=60, cfg=cfg)
    assert state.opening_waits == 2
    assert state.mid_waits == 1
    assert state.blocks >= 2


def _machine_trace_incremental(
    text_ids: list[int],
    decisions: list[bool],
    *,
    eoc_row: int | None,
    cfg: StreamingProtocolConfig,
    feed_chunks: list[list[int]],
) -> tuple[list, int]:
    """Drive the machine with an initially-open queue fed chunk by chunk;
    every starvation consumes the next chunk (the last one closes the queue).
    """
    state = StreamingProtocolState(
        cfg=cfg, inject_text_ids=list(text_ids[1:]), text_done=False
    )
    state.trace.append(("inject", text_ids[0]))
    decisions = iter(decisions)
    chunks = iter(feed_chunks)
    rows = 0
    generation_done = False
    eoc_abort = False
    winddown_left: int | None = None
    starvations = 0
    for _ in range(100_000):
        role = state.role
        decision = None
        if role is StepRole.DECISION:
            decision = next(decisions)
        elif role is StepRole.AUDIO:
            row_idx = rows
            rows += 1
            if winddown_left is not None:
                winddown_left -= 1
                if winddown_left <= 0:
                    generation_done = True
            elif eoc_row is not None and row_idx == eoc_row:
                if state.in_tail_flush:
                    winddown_left = WINDDOWN
                else:
                    eoc_abort = True
        plan = state.on_step_output(decision)
        while plan.starved:
            starvations += 1
            try:
                chunk = next(chunks)
            except StopIteration:
                state.append_text([], done=True)
            else:
                is_last = False
                try:
                    peek = next(chunks)
                    chunks = iter([peek, *chunks])
                except StopIteration:
                    is_last = True
                    chunks = iter([])
                state.append_text(chunk, done=is_last)
            plan = state.resume_plan()
        if generation_done or eoc_abort or plan.hard_stop:
            break
    else:  # pragma: no cover
        raise AssertionError("machine did not terminate")
    assert starvations > 0, "incremental scenario never starved"
    if eoc_row is not None and rows > eoc_row:
        state._flush_audio_trace()
        state.trace.append(("eoc",))
    return state.finalize_trace(), rows


def test_incremental_feed_matches_full_text_trace():
    """Starvation + append/resume must produce the same protocol trace as
    handing the machine the full text upfront."""
    cfg = _cfg()
    decisions = [False, True] + [True] * 80
    full_trace, full_rows, _ = _machine_trace(
        TEXT, list(decisions), eoc_row=70, cfg=cfg
    )
    # start with ONLY T0 available; feed the rest in small chunks on demand
    rest = TEXT[1:]
    inc_trace, inc_rows = _machine_trace_incremental(
        [TEXT[0]],
        list(decisions),
        eoc_row=70,
        cfg=cfg,
        feed_chunks=[rest[i : i + 2] for i in range(0, len(rest), 2)],
    )
    assert inc_trace == full_trace
    assert inc_rows == full_rows


def test_starved_plan_freezes_state():
    cfg = _cfg()
    state = StreamingProtocolState(cfg=cfg, inject_text_ids=[], text_done=False)
    # opening decision: wait -> WAIT_FEED needs a token -> starves
    state.on_step_output(False)  # DECISION(wait) -> WAIT_FEED plan
    plan = state.on_step_output(None)  # WAIT_FEED output discarded -> feed
    assert plan.starved and plan.role is StepRole.WAIT_FEED
    before = (state.text_pos, state.role, state.opening_waits)
    plan2 = state.resume_plan()  # still empty: starves again, no state change
    assert plan2.starved
    assert (state.text_pos, state.role, state.opening_waits) == before
    state.append_text([1234])
    plan3 = state.resume_plan()
    assert not plan3.starved
    assert plan3.input_token_id == 1234
    assert state.role is StepRole.DECISION


def test_closed_empty_queue_sends_text_end_not_starved():
    cfg = _cfg()
    state = StreamingProtocolState(cfg=cfg, inject_text_ids=[], text_done=False)
    state.on_step_output(False)
    plan = state.on_step_output(None)
    assert plan.starved
    state.append_text([], done=True)
    plan = state.resume_plan()
    assert not plan.starved
    assert plan.input_token_id == TEXT_END_ID
    assert state.text_end_sent


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
