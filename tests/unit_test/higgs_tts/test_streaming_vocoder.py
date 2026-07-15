# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage

BOC_ID = 1024
EOC_ID = 1025


def _apply_delay_pattern(codes_TN: torch.Tensor) -> torch.Tensor:
    T, N = codes_TN.shape
    out = torch.full((T + N - 1, N), EOC_ID, dtype=codes_TN.dtype)
    for codebook in range(N):
        out[:codebook, codebook] = BOC_ID
        out[codebook : codebook + T, codebook] = codes_TN[:, codebook]
    return out


class _FakeHiggsCodec:
    SAMPLE_RATE = 24000

    def __init__(self, *, frame_length: int = 2, codebook_size: int = 1024) -> None:
        self.model = SimpleNamespace(
            config=SimpleNamespace(
                hop_length=frame_length,
                codebook_size=codebook_size,
            )
        )
        self.decode_calls: list[torch.Tensor] = []
        self.masked_decode_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def decode(self, codes_TN: torch.Tensor) -> torch.Tensor:
        self.decode_calls.append(codes_TN.detach().clone())
        frame_values = codes_TN.to(torch.float32).sum(dim=1)
        offsets = torch.arange(
            self.model.config.hop_length,
            dtype=torch.float32,
            device=frame_values.device,
        )
        return (frame_values[:, None] * 10.0 + offsets[None, :]).reshape(-1)

    def decode_batch(self, codes_BTN: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.decode(codes) for codes in codes_BTN])

    def decode_masked_batch(
        self, codes_BTN: torch.Tensor, counts_BT: torch.Tensor
    ) -> torch.Tensor:
        self.masked_decode_calls.append(
            (codes_BTN.detach().clone(), counts_BT.detach().clone())
        )
        outputs = []
        for codes, counts in zip(codes_BTN, counts_BT):
            mask = torch.arange(codes.shape[1])[None, :] < counts[:, None]
            values = torch.where(mask, codes, 0).to(torch.float32).sum(dim=1)
            offsets = torch.arange(self.model.config.hop_length, dtype=torch.float32)
            outputs.append((values[:, None] * 10 + offsets[None, :]).reshape(-1))
        return torch.stack(outputs)


def _payload(
    request_id: str,
    *,
    stream: bool = True,
    stage_params: dict | None = None,
) -> StagePayload:
    state = HiggsTtsState(
        output_codes_delayed=_apply_delay_pattern(
            torch.arange(8 * 3, dtype=torch.long).reshape(8, 3)
        ).tolist(),
        num_codebooks=3,
        codebook_size=1026,
        prompt_tokens=5,
        completion_tokens=8,
        engine_time_s=0.25,
    )
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(
            inputs="hello",
            params={
                "stream": stream,
                **({"stage_params": stage_params} if stage_params is not None else {}),
            },
        ),
        data=state.to_dict(),
    )


def _row(row: torch.Tensor) -> IncomingMessage:
    return IncomingMessage("req", "stream_chunk", row.reshape(1, -1))


def test_vllm_style_first_chunk_waits_for_chunk_plus_delay_rows() -> None:
    from sglang_omni.models.higgs_tts.streaming_vocoder import (
        _HiggsStreamState,
        build_higgs_stream_chunk,
    )

    codec = _FakeHiggsCodec()
    state = _HiggsStreamState()
    raw_codes = torch.arange(10 * 3, dtype=torch.long).reshape(10, 3)
    delayed = _apply_delay_pattern(raw_codes)
    outputs = []

    for idx, row in enumerate(delayed[:6]):
        output = build_higgs_stream_chunk(
            state,
            row.reshape(1, 3),
            codec=codec,
            num_codebooks=3,
            audio_chunk_size=4,
            audio_chunk_overlap_size=4,
        )
        outputs.append(output)
        if idx < 5:
            assert output is None

    first = outputs[-1]
    assert first is not None
    assert first["modality"] == "audio"
    assert first["sample_rate"] == 24000
    assert len(first["audio_data"]) == 4
    assert state.is_first_chunk is False
    assert state.delayed_tokens_cache.shape == (4, 3)


def test_higgs_tts_default_streaming_chunk_size_targets_realtime_start() -> None:
    from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig

    cfg = HiggsTtsPipelineConfig(model_path="test-model")
    vocoder = next(stage for stage in cfg.stages if stage.name == "vocoder")

    assert vocoder.factory_args["streaming"] is True
    assert vocoder.factory_args["audio_chunk_size"] == 16
    assert vocoder.factory_args["audio_chunk_overlap_size"] == 16


def test_vllm_style_followup_chunk_keeps_overlap_and_crossfades_tail() -> None:
    from sglang_omni.models.higgs_tts.streaming_vocoder import (
        _HiggsStreamState,
        build_higgs_stream_chunk,
    )

    codec = _FakeHiggsCodec()
    state = _HiggsStreamState()
    raw_codes = torch.arange(12 * 3, dtype=torch.long).reshape(12, 3)
    delayed = _apply_delay_pattern(raw_codes)
    chunks = []

    for row in delayed[:10]:
        output = build_higgs_stream_chunk(
            state,
            row.reshape(1, 3),
            codec=codec,
            num_codebooks=3,
            audio_chunk_size=4,
            audio_chunk_overlap_size=4,
        )
        if output is not None:
            chunks.append(torch.tensor(output["audio_data"]))

    assert len(chunks) == 2
    assert codec.decode_calls[0].shape == (4, 3)
    assert codec.decode_calls[1].shape == (6, 3)
    assert state.delayed_tokens_cache.shape == (4, 3)
    assert state.fade_out_audio is not None

    full_second_decode = codec.decode_calls[1]
    raw_second_audio = codec.decode(full_second_decode)[:8]
    assert chunks[1].shape == raw_second_audio.shape
    assert chunks[1][0] != raw_second_audio[0]


def test_higgs_stream_flush_emits_remaining_cache_and_clears_tail() -> None:
    from sglang_omni.models.higgs_tts.streaming_vocoder import (
        _HiggsStreamState,
        build_higgs_stream_chunk,
        flush_higgs_stream_chunk,
    )

    codec = _FakeHiggsCodec()
    state = _HiggsStreamState()
    raw_codes = torch.arange(8 * 3, dtype=torch.long).reshape(8, 3)
    delayed = _apply_delay_pattern(raw_codes)

    for row in delayed[:6]:
        build_higgs_stream_chunk(
            state,
            row.reshape(1, 3),
            codec=codec,
            num_codebooks=3,
            audio_chunk_size=4,
            audio_chunk_overlap_size=4,
        )

    flush = flush_higgs_stream_chunk(
        state,
        codec=codec,
        num_codebooks=3,
        audio_chunk_size=4,
    )

    assert flush is not None
    assert flush["modality"] == "audio"
    assert state.delayed_tokens_cache.numel() == 0
    assert state.fade_out_audio is None
    assert state.is_first_chunk is True


def test_higgs_stream_flush_trims_tail_frame_and_fades_final_audio() -> None:
    from sglang_omni.models.higgs_tts.streaming_vocoder import (
        _HiggsStreamState,
        build_higgs_stream_chunk,
        flush_higgs_stream_chunk,
    )

    codec = _FakeHiggsCodec()
    state = _HiggsStreamState()
    raw_codes = torch.arange(8 * 3, dtype=torch.long).reshape(8, 3)
    delayed = _apply_delay_pattern(raw_codes)

    for row in delayed[:6]:
        build_higgs_stream_chunk(
            state,
            row.reshape(1, 3),
            codec=codec,
            num_codebooks=3,
            audio_chunk_size=4,
            audio_chunk_overlap_size=4,
        )

    flush = flush_higgs_stream_chunk(
        state,
        codec=codec,
        num_codebooks=3,
        audio_chunk_size=4,
    )

    # Cache held 4 delayed rows; the click-prone wind-down frame is dropped, so
    # the final decode sees 3 rows -> 1 data frame (2 without the trim).
    assert codec.decode_calls[-1].shape == (1, 3)
    assert flush is not None
    assert flush["audio_data"][-1] == 0.0  # faded to silence


def test_higgs_stream_flush_emits_retained_tail_when_cache_not_decodable() -> None:
    from sglang_omni.models.higgs_tts.streaming_vocoder import (
        _HiggsStreamState,
        flush_higgs_stream_chunk,
    )

    # After the tail trim the cache is too short to reverse the delay pattern;
    # the decoded-but-unreleased tail must still be emitted (it used to be
    # silently dropped), faded out.
    state = _HiggsStreamState()
    state.delayed_tokens_cache = torch.ones((2, 3), dtype=torch.long)
    state.fade_out_audio = torch.tensor([0.5, 0.5, 0.5, 0.5])

    flush = flush_higgs_stream_chunk(
        state,
        codec=_FakeHiggsCodec(),
        num_codebooks=3,
        audio_chunk_size=4,
    )

    assert flush is not None
    assert flush["audio_data"][0] == 0.5
    assert flush["audio_data"][-1] == 0.0
    assert state.delayed_tokens_cache.numel() == 0
    assert state.fade_out_audio is None
    assert state.is_first_chunk is True


def test_higgs_non_streaming_full_decode_fades_tail() -> None:
    from sglang_omni.models.higgs_tts.streaming_vocoder import HiggsVocoderScheduler

    codec = _FakeHiggsCodec()
    scheduler = HiggsVocoderScheduler(
        codec,
        device="cpu",
        num_codebooks=3,
        max_batch_wait_ms=1,
    )
    thread = threading.Thread(target=scheduler.start, daemon=True)
    try:
        thread.start()
        scheduler.inbox.put(
            IncomingMessage("req", "new_request", _payload("req", stream=False))
        )
        final = scheduler.outbox.get(timeout=2.0)
        assert final.type == "result"
        assert final.data.data["audio_data"][-1] == 0.0
    finally:
        scheduler.stop()
        thread.join(timeout=2.0)


def test_higgs_streaming_scheduler_emits_audio_before_terminal_result() -> None:
    from sglang_omni.models.higgs_tts.streaming_vocoder import HiggsVocoderScheduler

    codec = _FakeHiggsCodec()
    scheduler = HiggsVocoderScheduler(
        codec,
        device="cpu",
        num_codebooks=3,
        audio_chunk_size=4,
        audio_chunk_overlap_size=4,
        max_batch_wait_ms=1,
    )
    thread = threading.Thread(target=scheduler.start, daemon=True)
    raw_codes = torch.arange(8 * 3, dtype=torch.long).reshape(8, 3)
    delayed = _apply_delay_pattern(raw_codes)
    try:
        thread.start()
        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload("req")))
        for row in delayed[:6]:
            scheduler.inbox.put(_row(row))
        stream = scheduler.outbox.get(timeout=2.0)
        assert stream.type == "stream"

        scheduler.inbox.put(IncomingMessage("req", "stream_done"))
        flush = scheduler.outbox.get(timeout=2.0)
        final = scheduler.outbox.get(timeout=2.0)
        assert flush.type == "stream"
        assert final.type == "result"
        assert final.data.data["modality"] == "audio"
        assert final.data.data["audio_data"] == []
        assert final.data.data["usage"]["prompt_tokens"] == 5
    finally:
        scheduler.stop()
        thread.join(timeout=2.0)


def test_higgs_streaming_scheduler_uses_request_chunk_size_override() -> None:
    from sglang_omni.models.higgs_tts.streaming_vocoder import HiggsVocoderScheduler

    codec = _FakeHiggsCodec()
    scheduler = HiggsVocoderScheduler(
        codec,
        device="cpu",
        num_codebooks=3,
        audio_chunk_size=5,
        audio_chunk_overlap_size=5,
        max_batch_wait_ms=1,
    )
    thread = threading.Thread(target=scheduler.start, daemon=True)
    raw_codes = torch.arange(8 * 3, dtype=torch.long).reshape(8, 3)
    delayed = _apply_delay_pattern(raw_codes)
    try:
        thread.start()
        scheduler.inbox.put(
            IncomingMessage(
                "req",
                "new_request",
                _payload(
                    "req",
                    stage_params={
                        "vocoder": {
                            "audio_chunk_size": 4,
                            "audio_chunk_overlap_size": 4,
                        },
                    },
                ),
            )
        )
        for row in delayed[:6]:
            scheduler.inbox.put(_row(row))

        stream = scheduler.outbox.get(timeout=2.0)

        assert stream.type == "stream"
        assert len(stream.data["audio_data"]) == 4
    finally:
        scheduler.stop()
        thread.join(timeout=2.0)


def test_higgs_streaming_scheduler_uses_pre_payload_stream_metadata_override() -> None:
    from sglang_omni.models.higgs_tts.streaming_vocoder import HiggsVocoderScheduler
    from sglang_omni.pipeline.stage.stream_queue import StreamItem

    codec = _FakeHiggsCodec()
    scheduler = HiggsVocoderScheduler(
        codec,
        device="cpu",
        num_codebooks=3,
        audio_chunk_size=5,
        audio_chunk_overlap_size=5,
        max_batch_wait_ms=1,
    )
    thread = threading.Thread(target=scheduler.start, daemon=True)
    raw_codes = torch.arange(8 * 3, dtype=torch.long).reshape(8, 3)
    delayed = _apply_delay_pattern(raw_codes)
    try:
        thread.start()
        for index, row in enumerate(delayed[:6]):
            scheduler.inbox.put(
                IncomingMessage(
                    "req",
                    "stream_chunk",
                    StreamItem(
                        chunk_id=index,
                        data=row.reshape(1, 3),
                        from_stage="tts_engine",
                        metadata={
                            "request_params": {
                                "stream": True,
                                "stage_params": {
                                    "vocoder": {
                                        "audio_chunk_size": 4,
                                        "audio_chunk_overlap_size": 4,
                                    },
                                },
                            }
                        },
                    ),
                )
            )

        stream = scheduler.outbox.get(timeout=2.0)

        assert stream.type == "stream"
        assert len(stream.data["audio_data"]) == 4
    finally:
        scheduler.stop()
        thread.join(timeout=2.0)


def test_higgs_request_chunk_size_override_defaults_overlap_to_chunk_size() -> None:
    from sglang_omni.models.higgs_tts.streaming_vocoder import HiggsVocoderScheduler

    scheduler = HiggsVocoderScheduler(
        _FakeHiggsCodec(),
        device="cpu",
        num_codebooks=3,
        audio_chunk_size=5,
        audio_chunk_overlap_size=5,
    )
    scheduler._request_params["req"] = {
        "stage_params": {"vocoder": {"audio_chunk_size": 4}}
    }

    assert scheduler._stream_sizes_for_request("req") == (4, 4)


def test_higgs_non_streaming_scheduler_uses_full_decode() -> None:
    from sglang_omni.models.higgs_tts.streaming_vocoder import HiggsVocoderScheduler

    codec = _FakeHiggsCodec()
    scheduler = HiggsVocoderScheduler(
        codec,
        device="cpu",
        num_codebooks=3,
        max_batch_wait_ms=1,
    )
    thread = threading.Thread(target=scheduler.start, daemon=True)
    try:
        thread.start()
        scheduler.inbox.put(
            IncomingMessage("req", "new_request", _payload("req", stream=False))
        )
        final = scheduler.outbox.get(timeout=2.0)
        assert final.type == "result"
        assert final.data.data["modality"] == "audio"
        assert len(final.data.data["audio_data"]) > 0
    finally:
        scheduler.stop()
        thread.join(timeout=2.0)


def test_full_context_streaming_emits_only_center_from_complete_codebooks() -> None:
    from sglang_omni.models.higgs_tts.streaming_vocoder import HiggsVocoderScheduler

    codec = _FakeHiggsCodec()
    scheduler = HiggsVocoderScheduler(
        codec,
        num_codebooks=3,
        audio_chunk_size=4,
        audio_chunk_overlap_size=4,
        full_context_streaming=True,
        context_frames=2,
        startup_full_chunk_frames=2,
        startup_full_chunk_count=1,
    )
    raw = torch.arange(1, 1 + 12 * 3, dtype=torch.long).reshape(12, 3)
    delayed = _apply_delay_pattern(raw)

    # Two target frames + two real future frames + the two delay rows.
    for row in delayed[:5]:
        scheduler._on_chunk_batch([_row(row)])
    assert scheduler.outbox.empty()
    scheduler._on_chunk_batch([_row(delayed[5])])

    first = scheduler.outbox.get_nowait()
    hop = codec.model.config.hop_length
    assert len(first.data["audio_data"]) == 2 * hop
    assert codec.masked_decode_calls == []
    assert codec.decode_calls[-1].tolist() == raw[:4].tolist()
    expected = codec.decode(raw[:2]).tolist()
    assert first.data["audio_data"] == expected

    # The next window has two real history, four target, and two future frames.
    for row in delayed[6:10]:
        scheduler._on_chunk_batch([_row(row)])
    second = scheduler.outbox.get_nowait()
    assert codec.decode_calls[-1].tolist() == raw[:8].tolist()
    expected = codec.decode(raw[2:6]).tolist()
    assert second.data["audio_data"] == expected


def test_full_context_streaming_limits_reduced_context_to_startup() -> None:
    from sglang_omni.models.higgs_tts.streaming_vocoder import HiggsVocoderScheduler

    codec = _FakeHiggsCodec()
    scheduler = HiggsVocoderScheduler(
        codec,
        num_codebooks=3,
        audio_chunk_size=4,
        audio_chunk_overlap_size=4,
        full_context_streaming=True,
        context_frames=2,
        startup_reduced_context_frames=1,
        startup_reduced_context_until_frames=4,
        startup_full_chunk_frames=2,
        startup_full_chunk_count=1,
    )
    raw = torch.arange(1, 1 + 12 * 3, dtype=torch.long).reshape(12, 3)
    delayed = _apply_delay_pattern(raw)

    # Reduced startup: two targets, one future frame, and two delay rows.
    for row in delayed[:5]:
        scheduler._on_chunk_batch([_row(row)])
    first = scheduler.outbox.get_nowait()
    assert len(first.data["audio_data"]) == 2 * codec.model.config.hop_length
    assert codec.decode_calls[-1].tolist() == raw[:3].tolist()

    for row in delayed[5:7]:
        scheduler._on_chunk_batch([_row(row)])
    second = scheduler.outbox.get_nowait()
    assert len(second.data["audio_data"]) == 2 * codec.model.config.hop_length
    assert codec.decode_calls[-1].tolist() == raw[:5].tolist()

    # Once four frames are emitted, the configured two-frame context returns.
    for row in delayed[7:10]:
        scheduler._on_chunk_batch([_row(row)])
    third = scheduler.outbox.get_nowait()
    assert len(third.data["audio_data"]) == 2 * codec.model.config.hop_length
    assert codec.decode_calls[-1].tolist() == raw[2:8].tolist()


def test_full_context_streaming_masks_only_unready_startup_residuals() -> None:
    from sglang_omni.models.higgs_tts.streaming_vocoder import HiggsVocoderScheduler

    codec = _FakeHiggsCodec()
    scheduler = HiggsVocoderScheduler(
        codec,
        num_codebooks=4,
        audio_chunk_size=4,
        audio_chunk_overlap_size=4,
        full_context_streaming=True,
        context_frames=2,
        startup_masked_delay_rows=4,
        startup_masked_emit_frames=2,
        startup_masked_until_frames=4,
        startup_full_chunk_frames=2,
        startup_full_chunk_count=1,
    )
    raw = torch.arange(1, 1 + 16 * 4, dtype=torch.long).reshape(16, 4)
    delayed = _apply_delay_pattern(raw)

    for row in delayed[:4]:
        scheduler._on_chunk_batch([_row(row)])
    first = scheduler.outbox.get_nowait()
    assert len(first.data["audio_data"]) == 2 * codec.model.config.hop_length
    codes, counts = codec.masked_decode_calls[-1]
    assert counts.tolist() == [[4, 3, 2, 1]]
    assert codes[0, 0].tolist() == raw[0].tolist()

    for row in delayed[4:6]:
        scheduler._on_chunk_batch([_row(row)])
    second = scheduler.outbox.get_nowait()
    assert len(second.data["audio_data"]) == 2 * codec.model.config.hop_length
    _, counts = codec.masked_decode_calls[-1]
    assert counts.tolist() == [[4, 4, 4, 3, 2, 1]]

    # Once the masked region is emitted, wait for complete RVQ frames plus
    # real right context and return to the ordinary unmasked decoder.
    for row in delayed[6:12]:
        scheduler._on_chunk_batch([_row(row)])
    third = scheduler.outbox.get_nowait()
    assert len(third.data["audio_data"]) == 2 * codec.model.config.hop_length
    assert codec.decode_calls[-1].tolist() == raw[2:8].tolist()


def test_higgs_scheduler_done_before_payload_finalizes_after_new_request() -> None:
    from sglang_omni.models.higgs_tts.streaming_vocoder import HiggsVocoderScheduler

    codec = _FakeHiggsCodec()
    scheduler = HiggsVocoderScheduler(
        codec,
        device="cpu",
        num_codebooks=3,
        audio_chunk_size=4,
        audio_chunk_overlap_size=4,
        max_batch_wait_ms=1,
    )
    thread = threading.Thread(target=scheduler.start, daemon=True)
    raw_codes = torch.arange(8 * 3, dtype=torch.long).reshape(8, 3)
    delayed = _apply_delay_pattern(raw_codes)
    try:
        thread.start()
        for row in delayed[:6]:
            scheduler.inbox.put(_row(row))
        scheduler.inbox.put(IncomingMessage("req", "stream_done"))
        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload("req")))
        stream = scheduler.outbox.get(timeout=2.0)
        flush = scheduler.outbox.get(timeout=2.0)
        final = scheduler.outbox.get(timeout=2.0)
        assert stream.type == "stream"
        assert flush.type == "stream"
        assert final.type == "result"
        with pytest.raises(queue.Empty):
            scheduler.outbox.get(timeout=0.2)
    finally:
        scheduler.stop()
        thread.join(timeout=2.0)
