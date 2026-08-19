# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the plan-at-launch split in HiggsTTSModelRunner.

The post-opening protocol advance moved from collect time
(``_advance_streaming_request``) to launch time
(``_advance_streaming_plans_at_launch``), with collect consuming recorded
``(plan_for_step, plan_after)`` pairs (``_consume_streaming_step``). These
tests drive both paths over identical model-output sequences and require
byte-identical traces and published-token sequences, in sync ordering
(consume right after launch) and lookahead ordering (consume lagging one
launch, including the overrun launch after a finish).
"""

from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.higgs_tts.model_runner import HiggsTTSModelRunner
from sglang_omni.models.higgs_tts.streaming_protocol import (
    StepRole,
    StreamingProtocolConfig,
    StreamingProtocolState,
)

TEXT_TOKEN = 900
AUDIO_TOKEN = 901
TEXT_END = 902
NUM_EXTRA = 7
FRAMES_PER_BLOCK = 4


def make_cfg(max_frames: int = 60) -> StreamingProtocolConfig:
    return StreamingProtocolConfig(
        text_token_id=TEXT_TOKEN,
        audio_token_id=AUDIO_TOKEN,
        text_end_token_id=TEXT_END,
        num_extra_tokens=NUM_EXTRA,
        frames_per_block=FRAMES_PER_BLOCK,
        max_frames=max_frames,
    )


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
    input_starved: bool = False
    return_omni_rollout: bool = False
    return_logprob: bool = False
    output_logprobs: list = field(default_factory=list)
    opening_decision_actions: list = field(default_factory=list)
    opening_decision_logprobs: list = field(default_factory=list)
    output_codes: list = field(default_factory=list)
    watchdog_recent_rows: list = field(default_factory=list)
    watchdog_repeat_rows: int = 0
    generation_done: bool = False
    stream_metadata: Any = None


@dataclass
class _FakeSchedReq:
    data: _FakeData
    request_id: str = "req-parity"


def make_runner() -> HiggsTTSModelRunner:
    runner = object.__new__(HiggsTTSModelRunner)
    runner._outbox = None
    return runner


def audio_row(step_idx: int) -> torch.Tensor:
    # distinct rows so the watchdog never trips
    return torch.arange(8, dtype=torch.long) * 7 + step_idx


def _drive(text_ids, decisions, mode, max_frames=60, max_steps=500):
    """Run one full request in the given execution mode.

    - ``legacy``: pre-split semantics, collect-time advance every step.
    - ``sync``: plan-at-launch with immediate consume (Phase-1 sync decode).
    - ``lookahead``: plan-at-launch with consume lagging one launch for
      post-opening steps (async decode), including the overrun launch that
      happens after the finishing step but before its resolve.
    """
    runner = make_runner()
    proto = StreamingProtocolState(
        cfg=make_cfg(max_frames), inject_text_ids=list(text_ids)
    )
    sched = _FakeSchedReq(_FakeData(protocol_state=proto))
    data = sched.data
    published: list[int] = []
    decision_iter = iter(decisions)
    pending: list[tuple[StepRole, Any, Any]] = []

    def consume(role, codes, opening_decision):
        # mirrors _decode_collect_host: rows already finished skip consumption
        if data.req.finished():
            return
        if data.streaming_inflight:
            published.append(runner._consume_streaming_step(sched, codes, False))
        else:
            dec = (
                {0: (opening_decision, -0.5, -1.5)}
                if opening_decision is not None
                else {}
            )
            published.append(
                runner._advance_streaming_request(sched, dec, 0, codes, False)
            )

    for step in range(max_steps):
        if data.generation_done or data.input_starved:
            break
        role = proto.role if mode == "legacy" else None
        if mode == "legacy":
            codes = audio_row(step) if role is StepRole.AUDIO else None
            dec = {}
            if role is StepRole.DECISION and not proto.first_block_emitted:
                dec = {0: (next(decision_iter), -0.5, -1.5)}
            published.append(
                runner._advance_streaming_request(sched, dec, 0, codes, False)
            )
            continue
        role = (
            data.streaming_plan.role if data.streaming_plan is not None else proto.role
        )
        opening = not proto.first_block_emitted
        runner._advance_streaming_plans_at_launch([sched])
        codes = audio_row(step) if role is StepRole.AUDIO else None
        opening_decision = None
        if opening and role is StepRole.DECISION:
            opening_decision = next(decision_iter)
        pending.append((role, codes, opening_decision))
        lag = 1 if (mode == "lookahead" and not opening) else 0
        while len(pending) > lag:
            consume(*pending.pop(0))
    while pending:
        consume(*pending.pop(0))
    return proto.finalize_trace(), published, data


def assert_parity(text_ids, decisions, max_frames=60):
    ref_trace, ref_pub, ref_data = _drive(text_ids, decisions, "legacy", max_frames)
    for mode in ("sync", "lookahead"):
        trace, pub, split_data = _drive(text_ids, decisions, mode, max_frames)
        assert trace == ref_trace, f"trace mismatch ({mode})"
        assert pub == ref_pub, f"published tokens mismatch ({mode})"
        assert split_data.generation_done == ref_data.generation_done
        assert not split_data.streaming_inflight
    return ref_data


def test_parity_no_waits():
    data = assert_parity(list(range(100, 112)), [True])
    assert data.protocol_state.fuse_tripped  # ends via the length fuse here


def test_parity_with_opening_waits():
    assert_parity(list(range(100, 112)), [False, False, False, True])


def test_parity_short_text_tail_flush():
    assert_parity([100, 101], [False, True])


def test_parity_tiny_max_frames_hard_stop():
    data = assert_parity(list(range(100, 140)), [True], max_frames=16)
    assert data.protocol_state.fuse_tripped


class _FakeSamplingParams:
    repetition_penalty = 1.0
    frequency_penalty = 0.0
    presence_penalty = 0.0
    min_new_tokens = 0


class _FakeBatchReq:
    def __init__(self, data=None):
        self._omni_data = data
        self.sampling_params = _FakeSamplingParams()


class _FakeBatch:
    def __init__(self, reqs):
        self.reqs = reqs


def _post_opening_data(text_done=True, queue_left=5):
    proto = StreamingProtocolState(
        cfg=make_cfg(),
        inject_text_ids=list(range(100, 100 + queue_left)),
        text_done=text_done,
    )
    data = _FakeData(protocol_state=proto)
    runner = make_runner()
    sched = _FakeSchedReq(data)
    # drive through the opening (immediate audio decision) into the first block
    runner._advance_streaming_request(sched, {0: (True, -0.5, -1.5)}, 0, None, False)
    return data, runner, sched


def test_lookahead_eligible_gates():
    runner = make_runner()

    # offline-only batch: eligible
    assert runner.lookahead_eligible(_FakeBatch([_FakeBatchReq()]))

    # opening-phase streaming request: sync
    proto = StreamingProtocolState(cfg=make_cfg(), inject_text_ids=[100, 101])
    opening = _FakeData(protocol_state=proto)
    assert not runner.lookahead_eligible(_FakeBatch([_FakeBatchReq(opening)]))

    # post-opening with closed queue: eligible (mixed with offline)
    data, _, _ = _post_opening_data()
    batch = _FakeBatch([_FakeBatchReq(data), _FakeBatchReq()])
    assert runner.lookahead_eligible(batch)

    # starved: sync
    data.input_starved = True
    assert not runner.lookahead_eligible(batch)
    data.input_starved = False

    # hard-stop plan: sync
    saved = data.streaming_plan
    data.streaming_plan = type(saved)(
        input_token_id=None,
        input_is_text=False,
        audio_advance=False,
        role=StepRole.BLOCK_END,
        hard_stop=True,
    )
    assert not runner.lookahead_eligible(batch)
    data.streaming_plan = saved


def test_lookahead_eligible_open_queue_starve_risk():
    data, runner, sched = _post_opening_data(text_done=False, queue_left=1)
    batch = _FakeBatch([_FakeBatchReq(data)])
    proto = data.protocol_state
    # walk to the injection point (BLOCK_END plan) with an empty open queue
    step = 0
    while not (data.streaming_plan.role is StepRole.BLOCK_END and proto.queue_empty):
        runner._advance_streaming_plans_at_launch([sched])
        codes = (
            audio_row(step)
            if data.streaming_inflight
            and data.streaming_inflight[0][0].role is StepRole.AUDIO
            else None
        )
        runner._consume_streaming_step(sched, codes, False)
        step += 1
        assert step < 100
    assert not runner.lookahead_eligible(batch)
    proto.append_text([200], done=False)
    assert runner.lookahead_eligible(batch)


def test_starve_flagged_at_launch():
    """Open queue runs dry post-opening: the launch advance flags
    input_starved and the collect publishes a placeholder."""
    _, _, data = _drive([100, 101], [True], "sync")
    proto = data.protocol_state
    assert not data.input_starved  # closed queue never starves

    proto2 = StreamingProtocolState(
        cfg=make_cfg(), inject_text_ids=[100, 101], text_done=False
    )
    runner = make_runner()
    sched = _FakeSchedReq(_FakeData(protocol_state=proto2))
    d = sched.data
    step = 0
    while not d.input_starved and step < 100:
        role = d.streaming_plan.role if d.streaming_plan is not None else proto2.role
        opening = not proto2.first_block_emitted
        runner._advance_streaming_plans_at_launch([sched])
        codes = audio_row(step) if role is StepRole.AUDIO else None
        if d.streaming_inflight:
            runner._consume_streaming_step(sched, codes, False)
        else:
            dec = (
                {0: (True, -0.5, -1.5)}
                if (opening and role is StepRole.DECISION)
                else {}
            )
            runner._advance_streaming_request(sched, dec, 0, codes, False)
        step += 1
    assert d.input_starved
    assert not d.streaming_inflight
    assert not d.generation_done
    # refill + resume repairs the machine exactly as the scheduler sweep does
    proto2.append_text([102, 103], done=True)
    resume = proto2.resume_plan()
    assert not resume.starved and resume.role is StepRole.DECISION
