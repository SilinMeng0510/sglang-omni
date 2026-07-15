# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.higgs_tts import request_builders
from sglang_omni.models.higgs_tts.audio.utils import TAIL_TRIM_FRAMES
from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.proto import OmniRequest, StagePayload


def test_higgs_scheduler_adapters_clamp_cap_and_record_engine_time(
    monkeypatch,
) -> None:
    ticks = iter([10.0, 12.5])
    reset_calls: list[str] = []
    monkeypatch.setattr(
        request_builders,
        "time",
        SimpleNamespace(perf_counter=lambda: next(ticks)),
    )
    request_builder, result_adapter = request_builders.make_higgs_scheduler_adapters(
        SimpleNamespace(reset_request=reset_calls.append),
        max_new_tokens_cap=2048,
    )
    state = HiggsTtsState(
        prompt_token_ids=[1, 2, 3],
        max_new_tokens=4096,
    )
    payload = StagePayload(
        request_id="req-higgs",
        request=OmniRequest(inputs={}),
        data=state.to_dict(),
    )

    data = request_builder(payload)
    data.output_codes.append(torch.tensor([1, 2, 3], dtype=torch.long))
    result = result_adapter(data)

    assert data.max_new_tokens == 2048
    assert data.req.sampling_params.max_new_tokens == 2048
    assert result.data["completion_tokens"] == 1
    assert result.data["engine_time_s"] == 2.5
    assert reset_calls == ["req-higgs"]


def test_apply_higgs_result_trims_tail_wind_down_frame() -> None:
    # The last data frame before EOC is completed during the delay-pattern
    # wind-down and decodes into an audible click; apply_higgs_result must drop
    # it from the codes fed to the vocoder / committed to the session, while
    # completion_tokens still counts every generated row.
    state = HiggsTtsState(prompt_token_ids=[1, 2, 3])
    data = request_builders.HiggsSGLangRequestData(
        input_ids=torch.tensor([1, 2, 3], dtype=torch.long)
    )
    data.output_codes = [
        torch.tensor([row, row, row], dtype=torch.long) for row in range(5)
    ]

    request_builders.apply_higgs_result(state, data)

    assert state.completion_tokens == 5
    assert len(state.output_codes_delayed) == 5 - TAIL_TRIM_FRAMES
    assert state.output_codes_delayed[-1] == [3, 3, 3]


def test_apply_higgs_result_trimmed_to_empty_yields_no_codes() -> None:
    state = HiggsTtsState(prompt_token_ids=[1])
    data = request_builders.HiggsSGLangRequestData(
        input_ids=torch.tensor([1], dtype=torch.long)
    )
    data.output_codes = [torch.tensor([7, 7, 7], dtype=torch.long)]

    request_builders.apply_higgs_result(state, data)

    assert state.completion_tokens == 1
    assert state.output_codes_delayed is None


def test_seed_maps_to_sglang_sampling_seed() -> None:
    # sglang's SamplingParams uses ``sampling_seed`` (not ``seed``); the request
    # builder must map the user-facing ``seed`` onto it. Passing ``seed`` directly
    # raises TypeError and 500s the request.
    state = HiggsTtsState(prompt_token_ids=[1, 2, 3], seed=1234)
    data = request_builders.build_sglang_higgs_request(state)
    assert data.req.sampling_params.sampling_seed == 1234


def test_lora_id_is_attached_to_sglang_request() -> None:
    state = HiggsTtsState(prompt_token_ids=[1, 2, 3], lora_id="stable-adapter-id")

    data = request_builders.build_sglang_higgs_request(state)

    assert data.req.lora_id == "stable-adapter-id"
    # Req includes the adapter id in the radix-cache namespace, preventing a
    # base-model prefix from being reused for a LoRA adapter (or vice versa).
    assert "stable-adapter-id" in data.req.extra_key


def test_first_session_chunk_reuses_fixed_voice_cache_namespace() -> None:
    class Adapter:
        def build_prompt_from_ids(self, target_ids, **_kwargs):
            return list(target_ids)

    class EmptySessionStore:
        def history_for(self, _session_id):
            return [], []

    request_builder, _ = request_builders.make_higgs_scheduler_adapters(
        SimpleNamespace(reset_request=lambda _rid: None),
        adapter=Adapter(),
        session_store=EmptySessionStore(),
    )
    state = HiggsTtsState(
        prompt_token_ids=[1, 2, 3],
        reference_codes_delayed=[[1, 2], [3, 4]],
        target_text_token_ids=[2, 3],
        session_id="unique-request-session",
    )
    payload = StagePayload(
        request_id="first-chunk",
        request=OmniRequest(inputs={}),
        data=state.to_dict(),
    )

    data = request_builder(payload)

    assert not data.req.extra_key.startswith("sess-")
