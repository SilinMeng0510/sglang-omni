# SPDX-License-Identifier: Apache-2.0
"""Streaming-protocol rollout capture: audio-row logprobs aligned with codes,
trainable opening decisions (first forced <|text|> excluded), layout in
non_action_outputs."""

from dataclasses import dataclass, field
from typing import Any

import pytest
import torch

from sglang_omni.models.higgs_tts.model_runner import HiggsTTSModelRunner
from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.models.higgs_tts.request_builders import (
    apply_higgs_result,
    build_sglang_higgs_request,
)
from sglang_omni.models.higgs_tts.streaming_protocol import (
    StepRole,
    StreamingProtocolConfig,
    StreamingProtocolState,
)

TEXT_TOKEN = 900
AUDIO_TOKEN = 901
TEXT_END = 902
N = 8


class _FakeReq:
    def __init__(self) -> None:
        self.finished_reason = None

    def finished(self) -> bool:
        return self.finished_reason is not None


@dataclass
class _FakeData:
    protocol_state: StreamingProtocolState
    req: _FakeReq = field(default_factory=_FakeReq)
    streaming_plan: Any = None
    streaming_inflight: list = field(default_factory=list)
    streaming_launch_token: Any = None
    input_starved: bool = False
    output_codes: list = field(default_factory=list)
    output_logprobs: list = field(default_factory=list)
    opening_decision_actions: list = field(default_factory=list)
    opening_decision_logprobs: list = field(default_factory=list)
    watchdog_recent_rows: list = field(default_factory=list)
    watchdog_repeat_rows: int = 0
    generation_done: bool = False
    stream_metadata: Any = None
    return_omni_rollout: bool = True
    return_logprob: bool = True
    num_codebooks: int = N
    codebook_size: int = 1026
    input_ids: list = field(default_factory=lambda: list(range(5)))


@dataclass
class _FakeSchedReq:
    data: _FakeData
    request_id: str = "req-sroll"


def _drive_streaming_request(decisions, text_ids, max_frames=40):
    """Run one streaming request (sync ordering) with rollout capture on."""
    runner = object.__new__(HiggsTTSModelRunner)
    runner._outbox = None
    proto = StreamingProtocolState(
        cfg=StreamingProtocolConfig(
            text_token_id=TEXT_TOKEN,
            audio_token_id=AUDIO_TOKEN,
            text_end_token_id=TEXT_END,
            num_extra_tokens=7,
            frames_per_block=4,
            max_frames=max_frames,
        ),
        inject_text_ids=list(text_ids),
    )
    sched = _FakeSchedReq(_FakeData(protocol_state=proto))
    data = sched.data
    decision_iter = iter(decisions)
    step = 0
    while not data.generation_done and step < 400:
        role = (
            data.streaming_plan.role if data.streaming_plan is not None else proto.role
        )
        opening = not proto.first_block_emitted
        runner._advance_streaming_plans_at_launch([sched])
        codes = (
            torch.arange(N, dtype=torch.long) * 3 + step
            if role is StepRole.AUDIO
            else None
        )
        lp = torch.randn(N) - 2.0 if role is StepRole.AUDIO else None
        if data.streaming_inflight:
            runner._consume_streaming_step(sched, codes, False, lp)
        else:
            dec = {}
            if opening and role is StepRole.DECISION:
                dec = {0: (next(decision_iter), -0.5, -1.5)}
            runner._advance_streaming_request(sched, dec, 0, codes, False, lp)
        step += 1
    return sched


def test_audio_logprobs_align_and_opening_decisions_recorded():
    # decisions: d1 wait (forced layout, excluded), d2 wait, d3 audio
    sched = _drive_streaming_request([False, False, True], list(range(100, 106)))
    data = sched.data
    assert len(data.output_logprobs) == len(data.output_codes) > 0
    assert data.opening_decision_actions == [0, 1]
    assert data.opening_decision_logprobs == [-0.5, -1.5]

    state = HiggsTtsState(num_codebooks=N, codebook_size=1026)
    apply_higgs_result(state, data)
    ro = state.omni_rollout
    names = [s["name"] for s in ro["action_streams"]]
    assert names == ["higgs_codes", "opening_decisions"]
    codes_stream, dec_stream = ro["action_streams"]
    assert codes_stream["logprobs"] is not None
    assert dec_stream["actions"] == [0, 1]
    assert dec_stream["logprobs"] == [-0.5, -1.5]
    assert dec_stream["deterministic_mask"] == [1, 1]
    mask_count = sum(v for row in codes_stream["action_mask"] for v in row)
    assert ro["total_action_count"] == mask_count + 2
    layout = ro["non_action_outputs"][0]
    assert layout["name"] == "streaming_protocol_layout"
    assert layout["trace"][0] == ["wait"]
    assert ["inject", 100] in layout["trace"]
    assert layout["frames_per_block"] == 4


def test_immediate_audio_records_no_decisions():
    # d1 audio: the only decision is at the forced-first position -> nothing
    sched = _drive_streaming_request([True], list(range(100, 106)))
    data = sched.data
    assert data.opening_decision_actions == []
    state = HiggsTtsState(num_codebooks=N, codebook_size=1026)
    apply_higgs_result(state, data)
    names = [s["name"] for s in state.omni_rollout["action_streams"]]
    assert names == ["higgs_codes"]  # no opening stream, layout still present
    assert state.omni_rollout["non_action_outputs"][0]["name"] == (
        "streaming_protocol_layout"
    )


def test_overrun_launch_rolled_back_on_finish():
    """Lookahead ordering: the launch after the finishing step advances the
    machine; the finish must roll it back so trace/rows match reality."""
    for finish_at in (3, 4, 12, 13, 14, 15):  # spans AUDIO/BLOCK_END/DECISION phantoms
        runner = object.__new__(HiggsTTSModelRunner)
        runner._outbox = None
        proto = StreamingProtocolState(
            cfg=StreamingProtocolConfig(
                text_token_id=TEXT_TOKEN,
                audio_token_id=AUDIO_TOKEN,
                text_end_token_id=TEXT_END,
                num_extra_tokens=7,
                frames_per_block=4,
                max_frames=40,
            ),
            inject_text_ids=list(range(100, 110)),
        )
        sched = _FakeSchedReq(_FakeData(protocol_state=proto))
        data = sched.data
        pending = []
        step = 0
        while not data.generation_done and step < 100:
            role = (
                data.streaming_plan.role
                if data.streaming_plan is not None
                else proto.role
            )
            opening = not proto.first_block_emitted
            runner._advance_streaming_plans_at_launch([sched])
            codes = (
                torch.arange(N, dtype=torch.long) + step
                if role is StepRole.AUDIO
                else None
            )
            lp = torch.zeros(N) - 1.0 if role is StepRole.AUDIO else None
            pending.append((role, codes, lp, opening))
            lag = 1 if not opening else 0
            while len(pending) > lag:
                r, c, l, was_opening = pending.pop(0)
                if data.req.finished():
                    continue
                gen_done = step >= finish_at and r is StepRole.AUDIO
                if data.streaming_inflight:
                    runner._consume_streaming_step(sched, c, gen_done, l)
                else:
                    dec = (
                        {0: (True, -0.5, -1.5)}
                        if (was_opening and r is StepRole.DECISION)
                        else {}
                    )
                    runner._advance_streaming_request(sched, dec, 0, c, gen_done, l)
            step += 1
        assert data.generation_done
        trace = proto.finalize_trace()
        n_audio = sum(e[1] for e in trace if e[0] == "audio")
        assert n_audio == len(data.output_codes) == proto.rows_emitted, (
            f"finish_at={finish_at}: trace audio {n_audio} vs "
            f"appended {len(data.output_codes)} vs rows {proto.rows_emitted}"
        )
        n_inject = sum(1 for e in trace if e[0] == "inject")
        assert n_inject == proto.text_pos
        assert len(data.output_logprobs) == len(data.output_codes)


def test_request_build_gating():
    common = dict(
        prompt_token_ids=[1, 2, 3],
        streaming_protocol=True,
        inject_text_ids=[10, 11],
        streaming_text_token_id=TEXT_TOKEN,
        streaming_audio_token_id=AUDIO_TOKEN,
        streaming_text_end_token_id=TEXT_END,
        return_omni_rollout=True,
        return_logprob=True,
    )
    # full-text streaming protocol + rollout: now allowed
    data = build_sglang_higgs_request(HiggsTtsState(**common))
    assert data.protocol_state is not None and data.return_omni_rollout
    # incremental (WebSocket) input stays rejected
    with pytest.raises(ValueError, match="incremental"):
        build_sglang_higgs_request(HiggsTtsState(**common, streaming_incremental=True))
