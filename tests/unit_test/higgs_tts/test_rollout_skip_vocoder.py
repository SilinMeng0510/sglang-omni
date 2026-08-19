# SPDX-License-Identifier: Apache-2.0
"""output_modalities without "audio" skips vocoder synthesis (RL rollout)."""

from __future__ import annotations

import numpy as np
import torch

from sglang_omni.models.higgs_tts import stages
from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.proto import OmniRequest, StagePayload


def _make_payload(request_id: str, state: HiggsTtsState) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs=""),
        data=state.to_dict(),
    )


def _fake_codec_fixtures(monkeypatch):
    decode_batch_sizes: list[int] = []

    class FakeCodec:
        SAMPLE_RATE = 24_000

        def decode(self, codes_TN):
            return torch.zeros(codes_TN.shape[0], dtype=torch.float32)

        def decode_batch(self, codes_list):
            decode_batch_sizes.append(len(codes_list))
            return [torch.arange(c.shape[0], dtype=torch.float32) for c in codes_list]

        def decode_masked_batch(self, codes_list, counts_list):
            del counts_list
            return [torch.arange(c.shape[0], dtype=torch.float32) for c in codes_list]

    monkeypatch.setattr(stages, "resolve_checkpoint", lambda p: p)
    monkeypatch.setattr(stages, "get_or_load_codec", lambda *a, **kw: FakeCodec())
    return decode_batch_sizes


def test_output_audio_roundtrips():
    assert "output_audio" not in HiggsTtsState().to_dict()
    data = HiggsTtsState(output_audio=False).to_dict()
    assert data["output_audio"] is False
    assert HiggsTtsState.from_dict(data).output_audio is False
    assert HiggsTtsState.from_dict({}).output_audio is True


def test_vocoder_skips_decode_without_audio_modality(monkeypatch) -> None:
    decode_batch_sizes = _fake_codec_fixtures(monkeypatch)
    scheduler = stages.create_vocoder_executor(
        "fake-model", vocoder_decode_batch_size=4, max_batch_wait_ms=2
    )
    rollout = {"action_streams": [{"name": "higgs_codes"}], "total_action_count": 80}
    skip = _make_payload(
        "r-rollout",
        HiggsTtsState(
            output_codes_delayed=[[i % 100] * 8 for i in range(10)],
            output_audio=False,
            omni_rollout=rollout,
            prompt_tokens=5,
            completion_tokens=10,
        ),
    )
    synth = _make_payload(
        "r-audio",
        HiggsTtsState(output_codes_delayed=[[i % 100] * 8 for i in range(12)]),
    )

    results = scheduler._batch_fn([skip, synth])

    assert decode_batch_sizes == [1]  # only the audio request reached the codec
    skipped, synthesized = results
    assert np.frombuffer(skipped.data["audio_waveform"], dtype=np.float32).size == 0
    assert skipped.data["omni_rollout"] == rollout
    assert skipped.data["usage"]["prompt_tokens"] == 5
    assert np.frombuffer(synthesized.data["audio_waveform"], dtype=np.float32).size > 0


def test_preprocessing_reads_output_modalities(monkeypatch) -> None:
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda model_path: model_path)
    monkeypatch.setattr(stages.Tokenizer, "from_file", lambda _path: object())
    monkeypatch.setattr(
        stages, "PreTrainedTokenizerFast", lambda tokenizer_object: object()
    )

    class FakeAdapter:
        def __init__(self, _tokenizer) -> None:
            pass

        def build_prompt(self, text, *, num_ref_tokens, reference_text):
            return [1, 2, 3]

    monkeypatch.setattr(stages, "HiggsTokenizerAdapter", FakeAdapter)

    scheduler = stages.create_preprocessing_executor("ckpt", num_codebooks=2)
    preprocess = scheduler._fn

    def run(metadata):
        payload = StagePayload(
            request_id="r",
            request=OmniRequest(inputs={"text": "hello"}, params={}, metadata=metadata),
            data={},
        )
        return HiggsTtsState.from_dict(preprocess(payload).data)

    assert run(None).output_audio is True
    assert run({"output_modalities": ["text", "audio"]}).output_audio is True
    assert run({"output_modalities": ["text"]}).output_audio is False
