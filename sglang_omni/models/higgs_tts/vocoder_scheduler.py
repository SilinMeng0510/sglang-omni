# SPDX-License-Identifier: Apache-2.0
"""Streaming vocoder scheduler for Higgs TTS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from sglang_omni.models.higgs_tts.audio_codec import HiggsAudioCodec
from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.codec_delay import reverse_delay_pattern


@dataclass
class _HiggsStreamState:
    delayed_rows: list[torch.Tensor] = field(default_factory=list)
    emitted_raw_frames: int = 0
    next_decode_rows: int = 0
    num_codebooks: int | None = None
    codebook_size: int | None = None
    next_emit_frame: int = 0
    startup_full_chunks_emitted: int = 0
    # word onsets from the engine (word_align.py) not yet sent to the client
    words_pending: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _ContextDecodeTask:
    request_id: str
    state: _HiggsStreamState
    codes_TN: torch.Tensor
    counts_T: torch.Tensor
    audio_offset_samples: int
    emit_frames: int
    phase: str


@dataclass
class _ContextStepPlan:
    tasks: list[_ContextDecodeTask]
    masked: bool


class HiggsStreamingVocoderScheduler(
    StreamingVocoderBase[_HiggsStreamState, _ContextStepPlan]
):
    """Decode Higgs codec rows incrementally, with batched final decode.

    Finalization omits the EOC-adjacent final raw codec frame and clips decoded
    audio to the exact frame boundary. Listening A/B tests selected this over
    retaining the noisy final frame.
    """

    def __init__(
        self,
        codec: HiggsAudioCodec,
        *,
        stream_stride: int = 75,
        stream_overlap_tokens: int = 8,
        stream_holdback_tokens: int = 4,
        max_batch_size: int = 4,
        max_batch_wait_ms: int = 2,
        startup_masked: bool = True,
        # >= the DAC decoder's receptive field (+-10.4 frames at 25 Hz),
        # so steady-phase chunk size never affects decode quality
        context_frames: int = 11,
        startup_full_chunk_frames: int = 8,
        startup_full_chunk_count: int = 3,
        startup_reduced_context_frames: int = 6,
        startup_reduced_left_context_frames: int = 9,
        startup_reduced_context_until_frames: int = 11,
        startup_masked_delay_rows: int = 8,
        startup_masked_emit_frames: int = 3,
        startup_masked_until_frames: int = 8,
    ) -> None:
        if stream_stride <= 0:
            raise ValueError("stream_stride must be > 0")
        if stream_overlap_tokens < 0:
            raise ValueError("stream_overlap_tokens must be >= 0")
        if stream_holdback_tokens < 0:
            raise ValueError("stream_holdback_tokens must be >= 0")
        if context_frames < 0:
            raise ValueError("context_frames must be >= 0")
        if startup_masked_delay_rows <= 0:
            raise ValueError("startup_masked_delay_rows must be > 0")
        if not 1 <= startup_masked_emit_frames <= startup_masked_delay_rows:
            raise ValueError(
                "startup_masked_emit_frames must be within masked delay rows"
            )
        if startup_masked_until_frames < startup_masked_emit_frames:
            raise ValueError(
                "startup_masked_until_frames must be >= startup masked emit frames"
            )

        self._codec = codec
        self._stream_stride = int(stream_stride)
        self._stream_overlap_tokens = int(stream_overlap_tokens)
        self._stream_holdback_tokens = int(stream_holdback_tokens)
        self._samples_per_frame = self._resolve_samples_per_frame(codec)
        self._startup_masked = bool(startup_masked)
        self._context_frames = int(context_frames)
        self._startup_full_chunk_frames = int(startup_full_chunk_frames)
        self._startup_full_chunk_count = int(startup_full_chunk_count)
        self._startup_reduced_context_frames = int(startup_reduced_context_frames)
        self._startup_reduced_left_context_frames = int(
            startup_reduced_left_context_frames
        )
        self._startup_reduced_context_until_frames = int(
            startup_reduced_context_until_frames
        )
        self._startup_masked_delay_rows = int(startup_masked_delay_rows)
        self._startup_masked_emit_frames = int(startup_masked_emit_frames)
        self._startup_masked_until_frames = int(startup_masked_until_frames)
        if not hasattr(codec, "decode_masked_batch"):
            raise TypeError("Higgs streaming requires decode_masked_batch")
        self._can_batch_stream_chunks = True

        super().__init__(
            self._vocode_payload,
            batch_compute_fn=self._vocode_payloads,
            sample_rate=HiggsAudioCodec.SAMPLE_RATE,
            stream_source_hint="Higgs",
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )

    def create_stream_state(self, request_id: str) -> _HiggsStreamState:
        del request_id
        return _HiggsStreamState()

    def latch_stream_contract(
        self,
        request_id: str,
        state: _HiggsStreamState,
        source: StagePayload | Mapping[str, Any],
        *,
        origin: str,
    ) -> None:
        if origin == "payload":
            payload = source
            if not isinstance(payload.data, dict):
                raise TypeError(
                    f"Higgs streaming payload for {request_id!r} must be a dict, "
                    f"got {type(payload.data).__name__}"
                )
            missing = [
                key
                for key in ("num_codebooks", "codebook_size")
                if key not in payload.data
            ]
            if missing and (state.num_codebooks is None or state.codebook_size is None):
                raise RuntimeError(
                    f"Higgs streaming payload for {request_id!r} is missing fields: "
                    f"{', '.join(missing)}"
                )
            self._latch_contract_values(
                request_id,
                state,
                num_codebooks=payload.data.get("num_codebooks", state.num_codebooks),
                codebook_size=payload.data.get("codebook_size", state.codebook_size),
                source=origin,
            )
            return
        metadata: Mapping[str, Any] = source
        missing = [
            key for key in ("num_codebooks", "codebook_size") if key not in metadata
        ]
        if missing and (state.num_codebooks is None or state.codebook_size is None):
            raise RuntimeError(
                f"Higgs stream chunk for {request_id!r} is missing metadata fields: "
                f"{', '.join(missing)}"
            )
        if "num_codebooks" in metadata and "codebook_size" in metadata:
            self._latch_contract_values(
                request_id,
                state,
                num_codebooks=metadata["num_codebooks"],
                codebook_size=metadata["codebook_size"],
                source=origin,
            )

    def validate_chunk(
        self, request_id: str, state: _HiggsStreamState, codes: torch.Tensor
    ) -> torch.Tensor:
        row = codes.to(dtype=torch.long)
        if row.ndim != 1:
            raise ValueError(
                f"Higgs stream chunk must be 1-D [N], got {tuple(row.shape)}"
            )
        num_codebooks = self._require_stream_contract(state, request_id)[0]
        if int(row.shape[0]) != num_codebooks:
            raise ValueError(
                f"Higgs stream chunk has {int(row.shape[0])} codebooks, "
                f"expected {num_codebooks}"
            )
        return row

    def ingest(
        self, request_id: str, state: _HiggsStreamState, codes: torch.Tensor
    ) -> None:
        del request_id
        state.delayed_rows.append(codes)

    def _ingest_stream_item(self, request_id: str, item: Any) -> _HiggsStreamState | None:
        state = super()._ingest_stream_item(request_id, item)
        if state is not None and isinstance(item.metadata, dict):
            state.words_pending.extend(item.metadata.get("words") or ())
        return state

    def decode_delta(
        self, request_id: str, state: _HiggsStreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        delayed_count = len(state.delayed_rows)
        if delayed_count == 0:
            return None
        num_codebooks, codebook_size = self._require_stream_contract(state, "<stream>")
        if delayed_count < num_codebooks:
            return None
        raw_total = delayed_count - num_codebooks + 1

        next_decode_rows = state.next_decode_rows or max(
            num_codebooks, self._stream_stride
        )
        if not is_final and delayed_count < next_decode_rows:
            state.next_decode_rows = next_decode_rows
            return None

        if is_final and raw_total > 0:
            emit_until_raw = raw_total - 1
            if state.emitted_raw_frames > emit_until_raw:
                raise RuntimeError(
                    "Higgs final frame was emitted before the final-frame holdback"
                )
        else:
            emit_until_raw = raw_total
        if not is_final:
            emit_until_raw = max(
                0,
                raw_total - max(1, self._stream_holdback_tokens),
            )
        can_flush_codec_tail = is_final and self._samples_per_frame is not None
        if emit_until_raw < state.emitted_raw_frames or (
            emit_until_raw == state.emitted_raw_frames and not can_flush_codec_tail
        ):
            state.next_decode_rows = delayed_count + self._stream_stride
            return None

        window_start_raw = max(
            0, state.emitted_raw_frames - self._stream_overlap_tokens
        )
        rows_end = emit_until_raw + num_codebooks - 1
        rows = state.delayed_rows[window_start_raw:rows_end]
        audio = self._decode_delayed_rows(
            rows,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
        )

        decoded_raw_frames = emit_until_raw - window_start_raw
        samples_per_frame = self._samples_per_frame or max(
            int(audio.shape[-1]) // max(decoded_raw_frames, 1), 1
        )
        trim_frames = state.emitted_raw_frames - window_start_raw
        trim_samples = min(int(trim_frames * samples_per_frame), int(audio.shape[-1]))
        if self._samples_per_frame is not None:
            new_frames = emit_until_raw - state.emitted_raw_frames
            emit_samples = int(new_frames * samples_per_frame)
            delta = audio[trim_samples : trim_samples + emit_samples].contiguous()
        else:
            delta = audio[trim_samples:].contiguous()
        if delta.numel() == 0:
            state.next_decode_rows = delayed_count + self._stream_stride
            return None

        state.emitted_raw_frames = emit_until_raw
        state.next_decode_rows = delayed_count + self._stream_stride
        return delta

    def select_step_participants(self) -> list[tuple[str, _HiggsStreamState]]:
        tasks = [
            task
            for request_id, state in self._stream_state_items()
            if (task := self._prepare_context_task(request_id, state)) is not None
        ]
        if not tasks:
            return []
        tasks.sort(key=lambda task: task.state.next_emit_frame)
        lead_key = self._context_task_key(tasks[0])
        selected = [task for task in tasks if self._context_task_key(task) == lead_key]
        return [
            (task.request_id, task.state) for task in selected[: self._max_batch_size]
        ]

    def build_step_plan(
        self, participants: list[tuple[str, _HiggsStreamState]]
    ) -> _ContextStepPlan:
        tasks = []
        for request_id, state in participants:
            task = self._prepare_context_task(request_id, state)
            if task is None:
                raise RuntimeError(f"Higgs stream {request_id!r} became unready")
            tasks.append(task)
        return _ContextStepPlan(
            tasks=tasks, masked=tasks[0].phase == "partial_masked_delay"
        )

    def run_step(
        self,
        participants: list[tuple[str, _HiggsStreamState]],
        plan: _ContextStepPlan,
    ) -> dict[str, torch.Tensor]:
        del participants
        codes = [task.codes_TN for task in plan.tasks]
        audios = (
            self._codec.decode_masked_batch(
                codes, [task.counts_T for task in plan.tasks]
            )
            if plan.masked
            else self._codec.decode_batch(codes)
        )
        out: dict[str, torch.Tensor] = {}
        for task, audio in zip(plan.tasks, audios):
            samples_per_frame = self._samples_per_frame or max(
                int(audio.shape[-1]) // max(int(task.codes_TN.shape[0]), 1), 1
            )
            start = task.audio_offset_samples
            end = start + task.emit_frames * samples_per_frame
            delta = audio[start:end].detach().to(torch.float32).contiguous()
            task.state.next_emit_frame += task.emit_frames
            task.state.emitted_raw_frames = task.state.next_emit_frame
            if task.phase == "full_startup":
                task.state.startup_full_chunks_emitted += 1
            if delta.numel():
                out[task.request_id] = delta
        return out

    def _prepare_context_task(
        self, request_id: str, state: _HiggsStreamState
    ) -> _ContextDecodeTask | None:
        if not state.delayed_rows:
            return None
        num_codebooks, codebook_size = self._require_stream_contract(state, request_id)
        next_frame = state.next_emit_frame
        cache_len = len(state.delayed_rows)
        if self._startup_masked and next_frame < self._startup_masked_until_frames:
            emit_frames = min(
                self._startup_masked_emit_frames,
                self._startup_masked_until_frames - next_frame,
            )
            emit_frames = self._cap_emit_before_final_frame(
                state,
                next_frame=next_frame,
                emit_frames=emit_frames,
                codebook_size=codebook_size,
            )
            if emit_frames <= 0:
                return None
            frame_start = max(0, next_frame - self._context_frames)
            frame_end = next_frame + self._startup_masked_delay_rows
            if cache_len < frame_end:
                return None
            phase = "partial_masked_delay"
        else:
            if (
                self._startup_masked
                and next_frame < self._startup_reduced_context_until_frames
            ):
                emit_frames = min(
                    self._startup_full_chunk_frames,
                    self._startup_reduced_context_until_frames - next_frame,
                )
                left_context = self._startup_reduced_left_context_frames
                right_context = self._startup_reduced_context_frames
                phase = "full_reduced_context"
            elif (
                self._startup_masked
                and state.startup_full_chunks_emitted < self._startup_full_chunk_count
            ):
                emit_frames = self._startup_full_chunk_frames
                left_context = self._context_frames
                right_context = self._context_frames
                phase = "full_startup"
            else:
                emit_frames = max(1, self._stream_stride - num_codebooks + 1)
                left_context = self._context_frames
                right_context = self._context_frames
                phase = "full_context"
            emit_frames = self._cap_emit_before_final_frame(
                state,
                next_frame=next_frame,
                emit_frames=emit_frames,
                codebook_size=codebook_size,
            )
            if emit_frames <= 0:
                return None
            frame_start = max(0, next_frame - left_context)
            frame_end = next_frame + emit_frames + right_context
            if cache_len < frame_end + num_codebooks - 1:
                return None

        codes, counts = self._gather_rvq_window(
            state.delayed_rows,
            frame_start=frame_start,
            frame_end=frame_end,
            num_codebooks=num_codebooks,
        )
        codec_vocab = codebook_size - 2
        codes = torch.where(codes >= codec_vocab, torch.zeros_like(codes), codes)
        codes = torch.clamp(codes, 0, codec_vocab - 1)
        return _ContextDecodeTask(
            request_id=request_id,
            state=state,
            codes_TN=codes,
            counts_T=counts,
            audio_offset_samples=(next_frame - frame_start)
            * (self._samples_per_frame or 1),
            emit_frames=emit_frames,
            phase=phase,
        )

    @staticmethod
    def _cap_emit_before_final_frame(
        state: _HiggsStreamState,
        *,
        next_frame: int,
        emit_frames: int,
        codebook_size: int,
    ) -> int:
        eoc_id = int(codebook_size) - 1
        for row_index, row in enumerate(state.delayed_rows):
            if int(row[0].item()) == eoc_id:
                final_exclusive = max(0, row_index - 1)
                return min(int(emit_frames), max(0, final_exclusive - next_frame))
        return int(emit_frames)

    @staticmethod
    def _context_task_key(task: _ContextDecodeTask) -> tuple[Any, ...]:
        return (
            task.phase,
            tuple(task.codes_TN.shape),
            task.audio_offset_samples,
            task.emit_frames,
        )

    @staticmethod
    def _gather_rvq_window(
        delayed_rows: list[torch.Tensor],
        *,
        frame_start: int,
        frame_end: int,
        num_codebooks: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        codes = torch.zeros((frame_end - frame_start, num_codebooks), dtype=torch.long)
        counts = torch.empty((frame_end - frame_start,), dtype=torch.long)
        for output_frame, frame in enumerate(range(frame_start, frame_end)):
            count = min(num_codebooks, max(len(delayed_rows) - frame, 0))
            if count <= 0:
                raise ValueError(f"Higgs frame {frame} has no available codebooks")
            counts[output_frame] = count
            for codebook in range(count):
                codes[output_frame, codebook] = delayed_rows[frame + codebook][codebook]
        return codes, counts

    def stream_payload(self, request_id: str, waveform: torch.Tensor) -> dict[str, Any]:
        payload = audio_waveform_payload(
            waveform,
            sample_rate=self._sample_rate,
            modality="audio",
            source_hint="Higgs TTS streaming",
        )
        state = self._stream_states.get(request_id)
        if state is not None and state.words_pending:
            # a word rides with the first chunk whose audio reaches its onset
            end_ms = state.emitted_raw_frames * 1000 // 25
            ready = [w for w in state.words_pending if w["start_ms"] < end_ms]
            if ready:
                payload["words"] = ready
                state.words_pending = [
                    w for w in state.words_pending if w["start_ms"] >= end_ms
                ]
        return payload

    def fallback_full_decode(
        self, request_id: str, payload: StagePayload, state: _HiggsStreamState
    ) -> torch.Tensor | None:
        del request_id, state
        return self._decode_state_to_audio(HiggsTtsState.from_dict(payload.data))

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: _HiggsStreamState
    ) -> dict[str, Any]:
        del request_id, state
        final_data: dict[str, Any] = {
            "modality": "audio",
            "sample_rate": self._sample_rate,
        }
        final_state = HiggsTtsState.from_dict(payload.data)
        usage = build_usage(final_state)
        if usage is not None:
            final_data["usage"] = usage
        if final_state.omni_rollout is not None:
            final_data["omni_rollout"] = final_state.omni_rollout
        if final_state.words is not None:
            final_data["words"] = final_state.words
        return final_data

    @staticmethod
    def _latch_contract_values(
        request_id: str,
        state: _HiggsStreamState,
        *,
        num_codebooks: Any,
        codebook_size: Any,
        source: str,
    ) -> None:
        try:
            num_codebooks_i = int(num_codebooks)
            codebook_size_i = int(codebook_size)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Higgs {source} for {request_id!r} must include integer "
                "num_codebooks and codebook_size"
            ) from exc
        if num_codebooks_i <= 0 or codebook_size_i <= 2:
            raise ValueError(
                f"Higgs {source} for {request_id!r} has invalid "
                f"num_codebooks={num_codebooks_i}, codebook_size={codebook_size_i}"
            )
        if state.num_codebooks is not None and state.num_codebooks != num_codebooks_i:
            raise ValueError(
                f"Higgs stream num_codebooks changed for {request_id!r}: "
                f"{state.num_codebooks} -> {num_codebooks_i}"
            )
        if state.codebook_size is not None and state.codebook_size != codebook_size_i:
            raise ValueError(
                f"Higgs stream codebook_size changed for {request_id!r}: "
                f"{state.codebook_size} -> {codebook_size_i}"
            )
        state.num_codebooks = num_codebooks_i
        state.codebook_size = codebook_size_i

    @staticmethod
    def _require_stream_contract(
        state: _HiggsStreamState,
        request_id: str,
    ) -> tuple[int, int]:
        if state.num_codebooks is None or state.codebook_size is None:
            raise RuntimeError(
                f"Higgs stream contract for {request_id!r} is missing "
                "num_codebooks or codebook_size"
            )
        return state.num_codebooks, state.codebook_size

    def _vocode_payload(self, payload: StagePayload) -> StagePayload:
        return self._vocode_payloads([payload])[0]

    def _vocode_payloads(self, payloads: list[StagePayload]) -> list[StagePayload]:
        items = [self._prepare_vocoder_item(payload) for payload in payloads]
        valid = [(i, codes) for i, (_, codes) in enumerate(items) if codes is not None]
        waveforms: list[torch.Tensor | None] = [None] * len(items)
        if valid:
            indices, codes_list = zip(*valid)
            wavs = self._codec.decode_batch(list(codes_list))
            if len(wavs) != len(valid):
                raise RuntimeError(
                    f"Higgs vocoder decode_batch returned {len(wavs)} audios "
                    f"for {len(valid)} requests"
                )
            for idx, wav, codes in zip(indices, wavs, codes_list):
                waveforms[idx] = self._trim_codec_tail(
                    wav,
                    frames=int(codes.shape[0]),
                )
        return [
            self._store_vocoder_result(payload, state, wav)
            for payload, (state, _), wav in zip(payloads, items, waveforms)
        ]

    def _prepare_vocoder_item(
        self,
        payload: StagePayload,
    ) -> tuple[HiggsTtsState, torch.Tensor | None]:
        state = HiggsTtsState.from_dict(payload.data)
        if not state.output_audio:
            return state, None
        delayed_rows = state.output_codes_delayed
        if not delayed_rows:
            return state, None
        delayed_LN = torch.tensor(delayed_rows, dtype=torch.long)
        if delayed_LN.shape[0] < state.num_codebooks:
            return state, None
        codes_TN = reverse_delay_pattern(delayed_LN)
        if int(codes_TN.shape[0]) <= 1:
            return state, None
        codes_TN = codes_TN[:-1]
        codec_vocab = int(state.codebook_size) - 2
        return state, torch.where(
            codes_TN >= codec_vocab, torch.zeros_like(codes_TN), codes_TN
        )

    def _store_vocoder_result(
        self,
        payload: StagePayload,
        state: HiggsTtsState,
        waveform: torch.Tensor | None,
    ) -> StagePayload:
        data = audio_waveform_payload(
            waveform if waveform is not None else [],
            sample_rate=self._sample_rate,
            modality="audio",
            source_hint="Higgs TTS vocoder",
        )
        usage = build_usage(state)
        if usage is not None:
            data["usage"] = usage
        if state.omni_rollout is not None:
            data["omni_rollout"] = state.omni_rollout
        if state.words is not None:
            data["words"] = state.words
        payload.data = data
        return payload

    def _decode_state_to_audio(self, state: HiggsTtsState) -> torch.Tensor | None:
        delayed_rows = state.output_codes_delayed
        if not delayed_rows:
            return None
        rows = [torch.tensor(row, dtype=torch.long) for row in delayed_rows]
        if len(rows) < int(state.num_codebooks):
            return None
        return self._decode_delayed_rows(
            rows,
            num_codebooks=int(state.num_codebooks),
            codebook_size=int(state.codebook_size),
            drop_final_frame=True,
            trim_codec_tail=True,
        )

    def _decode_delayed_rows(
        self,
        rows: list[torch.Tensor],
        *,
        num_codebooks: int,
        codebook_size: int,
        drop_final_frame: bool = False,
        trim_codec_tail: bool = False,
    ) -> torch.Tensor:
        if len(rows) < int(num_codebooks):
            raise ValueError(
                f"Higgs delayed rows must include at least {num_codebooks} rows, "
                f"got {len(rows)}"
            )
        delayed_LN = torch.stack(rows, dim=0).to(torch.long)
        codes_TN = reverse_delay_pattern(delayed_LN)
        if drop_final_frame:
            codes_TN = codes_TN[:-1]
        if int(codes_TN.shape[0]) == 0:
            return torch.empty(0, dtype=torch.float32)
        codec_vocab = int(codebook_size) - 2
        codes_TN = torch.where(
            codes_TN >= codec_vocab, torch.zeros_like(codes_TN), codes_TN
        )
        audio = self._codec.decode(codes_TN).detach().to(torch.float32)
        if trim_codec_tail:
            audio = self._trim_codec_tail(audio, frames=int(codes_TN.shape[0]))
        return audio

    def _trim_codec_tail(self, audio: torch.Tensor, *, frames: int) -> torch.Tensor:
        if self._samples_per_frame is None:
            return audio
        samples = max(0, int(frames)) * self._samples_per_frame
        return audio[..., :samples].contiguous()

    @staticmethod
    def _resolve_samples_per_frame(codec: HiggsAudioCodec) -> int | None:
        hop_length = getattr(getattr(codec, "model", None), "config", None)
        hop_length = getattr(hop_length, "hop_length", None)
        if hop_length is None:
            return None
        hop_length_i = int(hop_length)
        return hop_length_i if hop_length_i > 0 else None


__all__ = ["HiggsStreamingVocoderScheduler"]
