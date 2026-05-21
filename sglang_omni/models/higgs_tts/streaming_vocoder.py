# SPDX-License-Identifier: Apache-2.0
"""vLLM-style streaming vocoder for Higgs TTS.

The AR stage emits one delayed-code row per decode step. This scheduler keeps
those rows in a per-request cache and mirrors vLLM 0.10's chunking policy:

* first chunk waits for ``chunk_size + num_codebooks - 1`` delayed rows;
* first emission decodes the whole cache but only releases
  ``chunk_size - num_codebooks + 1`` data frames;
* later chunks wait for ``chunk_size + overlap_size`` rows, release
  ``chunk_size`` frames, and retain the overlap rows in the cache;
* the decoded-but-not-yet-released tail is blended into the next chunk with a
  Hamming-window crossfade.
"""

from __future__ import annotations

import collections
import logging
import queue as _queue_mod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

_ABORTED_REQUEST_ID_LIMIT = 10000
_ABORTED_REQUEST_ID_RETAINED = 5000


@dataclass
class _HiggsStreamState:
    delayed_tokens_cache: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 0), dtype=torch.long)
    )
    is_first_chunk: bool = True
    fade_out_audio: torch.Tensor | None = None


def _codec_sample_rate(codec: Any) -> int:
    return int(getattr(codec, "SAMPLE_RATE", 24000))


def _codec_frame_length(codec: Any) -> int:
    return int(codec.model.config.hop_length)


def _codec_vocab_size(codec: Any) -> int:
    return int(codec.model.config.codebook_size)


def _reverse_delay_pattern(delayed_LN: torch.Tensor) -> torch.Tensor:
    if delayed_LN.ndim != 2:
        raise ValueError(
            f"delayed codes must be 2-D [L, N], got {tuple(delayed_LN.shape)}"
        )
    length, num_codebooks = delayed_LN.shape
    data_frames = length - (num_codebooks - 1)
    if data_frames <= 0:
        raise ValueError(f"need at least {num_codebooks} delayed rows, got {length}")
    out = torch.empty(
        (data_frames, num_codebooks),
        dtype=delayed_LN.dtype,
        device=delayed_LN.device,
    )
    for codebook in range(num_codebooks):
        out[:, codebook] = delayed_LN[codebook : codebook + data_frames, codebook]
    return out


def _hamming_crossfade(
    audio: torch.Tensor,
    fade_out_audio: torch.Tensor | None,
    *,
    max_window_len: int,
) -> torch.Tensor:
    if fade_out_audio is None or fade_out_audio.numel() == 0 or audio.numel() == 0:
        return audio

    window_len = min(
        2 * int(fade_out_audio.shape[-1]),
        int(max_window_len),
        2 * int(audio.shape[-1]),
    )
    window_len = (window_len // 2) * 2
    if window_len <= 0:
        return audio

    overlap = window_len // 2
    window = torch.hamming_window(
        window_len,
        periodic=False,
        dtype=audio.dtype,
        device=audio.device,
    )
    audio = audio.clone()
    audio[:overlap] = (
        audio[:overlap] * window[:overlap]
        + fade_out_audio[:overlap].to(device=audio.device, dtype=audio.dtype)
        * window[overlap:]
    )
    return audio


def _decode_delayed_tokens(
    delayed_tokens: torch.Tensor,
    *,
    codec: Any,
) -> torch.Tensor | None:
    if delayed_tokens.numel() == 0:
        return None
    if delayed_tokens.shape[0] < delayed_tokens.shape[1]:
        return None

    codec_vocab = _codec_vocab_size(codec)
    codes_TN = _reverse_delay_pattern(delayed_tokens.to(torch.long))
    codes_TN = torch.where(
        codes_TN >= codec_vocab, torch.zeros_like(codes_TN), codes_TN
    )
    codes_TN = torch.clamp(codes_TN, 0, codec_vocab - 1)
    with torch.no_grad():
        return codec.decode(codes_TN).detach().to(torch.float32).cpu()


def create_higgs_audio_chunk(
    delayed_tokens: torch.Tensor,
    audio_chunk_size: int,
    fade_out_audio: torch.Tensor | None,
    *,
    codec: Any,
    finalize: bool = False,
) -> tuple[dict[str, Any] | None, torch.Tensor | None]:
    audio = _decode_delayed_tokens(delayed_tokens, codec=codec)
    if audio is None or audio.numel() == 0:
        return None, fade_out_audio

    frame_length = _codec_frame_length(codec)
    audio = _hamming_crossfade(
        audio,
        fade_out_audio,
        max_window_len=frame_length,
    )

    emit_samples = max(int(audio_chunk_size), 0) * frame_length
    if finalize:
        next_fade_out = None
        ready_audio = audio
    else:
        next_fade_out = audio[emit_samples:].clone()
        ready_audio = audio[: min(emit_samples, int(audio.shape[-1]))]

    if ready_audio.numel() == 0:
        return None, next_fade_out
    return (
        _build_audio_chunk_payload(
            ready_audio,
            sample_rate=_codec_sample_rate(codec),
        ),
        next_fade_out,
    )


def _append_delayed_rows(
    state: _HiggsStreamState,
    codes: torch.Tensor,
    *,
    num_codebooks: int,
) -> None:
    if codes.ndim == 1:
        if codes.numel() % num_codebooks != 0:
            raise ValueError(
                f"stream code row has {codes.numel()} values, expected "
                f"multiples of {num_codebooks}"
            )
        codes = codes.reshape(-1, num_codebooks)
    if codes.ndim != 2 or codes.shape[1] != num_codebooks:
        raise ValueError(
            f"stream code chunk must be [L, {num_codebooks}], got "
            f"{tuple(codes.shape)}"
        )

    codes = codes.detach().to(dtype=torch.long, device="cpu")
    if state.delayed_tokens_cache.numel() == 0:
        state.delayed_tokens_cache = torch.empty((0, num_codebooks), dtype=torch.long)
    state.delayed_tokens_cache = torch.cat(
        [state.delayed_tokens_cache, codes],
        dim=0,
    )


def build_higgs_stream_chunk(
    state: _HiggsStreamState,
    codes: torch.Tensor,
    *,
    codec: Any,
    num_codebooks: int,
    audio_chunk_size: int,
    audio_chunk_overlap_size: int,
) -> dict[str, Any] | None:
    _validate_stream_sizes(
        num_codebooks=num_codebooks,
        audio_chunk_size=audio_chunk_size,
        audio_chunk_overlap_size=audio_chunk_overlap_size,
    )
    _append_delayed_rows(state, codes, num_codebooks=num_codebooks)

    cache_len = int(state.delayed_tokens_cache.shape[0])
    if state.is_first_chunk:
        first_threshold = int(audio_chunk_size) + int(num_codebooks) - 1
        if cache_len < first_threshold:
            return None
        emit_frames = int(audio_chunk_size) - int(num_codebooks) + 1
        chunk, state.fade_out_audio = create_higgs_audio_chunk(
            state.delayed_tokens_cache,
            emit_frames,
            state.fade_out_audio,
            codec=codec,
            finalize=False,
        )
        state.delayed_tokens_cache = state.delayed_tokens_cache[emit_frames:].clone()
        state.is_first_chunk = False
        return chunk

    threshold = int(audio_chunk_size) + int(audio_chunk_overlap_size)
    if cache_len < threshold:
        return None
    chunk, state.fade_out_audio = create_higgs_audio_chunk(
        state.delayed_tokens_cache,
        int(audio_chunk_size),
        state.fade_out_audio,
        codec=codec,
        finalize=False,
    )
    state.delayed_tokens_cache = state.delayed_tokens_cache[
        int(audio_chunk_size) :
    ].clone()
    return chunk


def flush_higgs_stream_chunk(
    state: _HiggsStreamState,
    *,
    codec: Any,
    num_codebooks: int,
    audio_chunk_size: int,
) -> dict[str, Any] | None:
    if state.delayed_tokens_cache.numel() == 0:
        tail = state.fade_out_audio
        state.fade_out_audio = None
        state.is_first_chunk = True
        if tail is None or tail.numel() == 0:
            return None
        return _build_audio_chunk_payload(tail, sample_rate=_codec_sample_rate(codec))

    chunk, _ = create_higgs_audio_chunk(
        state.delayed_tokens_cache,
        int(audio_chunk_size),
        state.fade_out_audio,
        codec=codec,
        finalize=True,
    )
    state.delayed_tokens_cache = torch.empty((0, num_codebooks), dtype=torch.long)
    state.fade_out_audio = None
    state.is_first_chunk = True
    return chunk


def _validate_stream_sizes(
    *,
    num_codebooks: int,
    audio_chunk_size: int,
    audio_chunk_overlap_size: int,
) -> None:
    if num_codebooks <= 0:
        raise ValueError("num_codebooks must be > 0")
    if audio_chunk_size < num_codebooks:
        raise ValueError("audio_chunk_size must be >= num_codebooks")
    if audio_chunk_overlap_size < 0:
        raise ValueError("audio_chunk_overlap_size must be >= 0")


def _build_audio_chunk_payload(
    audio_data: torch.Tensor,
    *,
    sample_rate: int,
) -> dict[str, Any]:
    return {
        "audio_data": audio_data.cpu().tolist(),
        "sample_rate": sample_rate,
        "modality": "audio",
    }


def _build_usage(state: HiggsTtsState) -> dict[str, Any] | None:
    if not (state.prompt_tokens or state.completion_tokens or state.engine_time_s):
        return None
    usage: dict[str, Any] = {
        "prompt_tokens": state.prompt_tokens,
        "completion_tokens": state.completion_tokens,
        "total_tokens": state.prompt_tokens + state.completion_tokens,
    }
    if state.engine_time_s:
        usage["engine_time_s"] = round(float(state.engine_time_s), 6)
    return usage


class HiggsVocoderScheduler:
    """Vocoder scheduler that supports full decode and vLLM-style streaming."""

    def __init__(
        self,
        codec: Any,
        *,
        device: str = "cpu",
        num_codebooks: int = 8,
        audio_chunk_size: int | None = None,
        audio_chunk_overlap_size: int | None = None,
        max_batch_size: int = 4,
        max_batch_wait_ms: int = 2,
    ) -> None:
        del device
        frame_length = _codec_frame_length(codec)
        tps = max(_codec_sample_rate(codec) // frame_length, 1)
        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()
        self._codec = codec
        self._num_codebooks = int(num_codebooks)
        self._audio_chunk_size = int(audio_chunk_size or tps)
        self._audio_chunk_overlap_size = int(audio_chunk_overlap_size or tps)
        _validate_stream_sizes(
            num_codebooks=self._num_codebooks,
            audio_chunk_size=self._audio_chunk_size,
            audio_chunk_overlap_size=self._audio_chunk_overlap_size,
        )
        self._max_batch_size = max(int(max_batch_size), 1)
        self._max_batch_wait_s = max(float(max_batch_wait_ms), 0.0) / 1000.0
        self._running = False
        self._pending_messages: collections.deque[IncomingMessage] = collections.deque()
        self._payloads: dict[str, StagePayload] = {}
        self._request_params: dict[str, dict[str, Any]] = {}
        self._stream_states: dict[str, _HiggsStreamState] = {}
        self._pending_done: set[str] = set()
        self._aborted_request_ids: set[str] = set()

    def start(self) -> None:
        self._running = True
        while self._running:
            msg = self._next_message()
            if msg is None:
                continue
            if msg.request_id in self._aborted_request_ids:
                continue
            try:
                if msg.type == "new_request":
                    self._handle_new_request(msg.request_id, msg.data)
                elif msg.type == "stream_chunk":
                    self._on_chunk(msg.request_id, msg.data)
                elif msg.type == "stream_done":
                    self._on_done(msg.request_id)
                else:
                    raise ValueError(f"Unsupported Higgs vocoder message: {msg.type}")
            except Exception as exc:
                logger.exception("HiggsVocoderScheduler failed for %s", msg.request_id)
                self.outbox.put(
                    OutgoingMessage(
                        request_id=msg.request_id,
                        type="error",
                        data=exc,
                    )
                )
                self.abort(msg.request_id)

    def stop(self) -> None:
        self._running = False

    def abort(self, request_id: str) -> None:
        self._aborted_request_ids.add(request_id)
        if len(self._aborted_request_ids) > _ABORTED_REQUEST_ID_LIMIT:
            excess = len(self._aborted_request_ids) - _ABORTED_REQUEST_ID_RETAINED
            for rid in list(self._aborted_request_ids)[:excess]:
                self._aborted_request_ids.discard(rid)
        self._clear_request_state(request_id, keep_aborted=True)

    def _next_message(self) -> IncomingMessage | None:
        if self._pending_messages:
            return self._pending_messages.popleft()
        try:
            return self.inbox.get(timeout=0.1)
        except _queue_mod.Empty:
            return None

    def _handle_new_request(self, request_id: str, payload: StagePayload) -> None:
        self._aborted_request_ids.discard(request_id)
        if not self._is_streaming_payload(payload):
            self._clear_request_state(request_id)
            result = self._vocode_full(payload)
            self.outbox.put(
                OutgoingMessage(request_id=request_id, type="result", data=result)
            )
            return

        self._payloads[request_id] = payload
        self._request_params[request_id] = dict(payload.request.params or {})
        self._stream_states.setdefault(request_id, _HiggsStreamState())
        if request_id in self._pending_done:
            self._pending_done.discard(request_id)
            self._on_done(request_id)

    def _on_chunk(self, request_id: str, chunk: Any) -> None:
        if request_id in self._aborted_request_ids:
            return
        self._remember_stream_request_params(request_id, chunk)
        codes = getattr(chunk, "data", chunk)
        if not isinstance(codes, torch.Tensor):
            codes = torch.as_tensor(codes, dtype=torch.long)
        state = self._stream_states.setdefault(request_id, _HiggsStreamState())
        audio_chunk_size, audio_chunk_overlap_size = self._stream_sizes_for_request(
            request_id
        )
        output = build_higgs_stream_chunk(
            state,
            codes,
            codec=self._codec,
            num_codebooks=self._num_codebooks,
            audio_chunk_size=audio_chunk_size,
            audio_chunk_overlap_size=audio_chunk_overlap_size,
        )
        if output is not None and request_id not in self._aborted_request_ids:
            self.outbox.put(
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data=output,
                    metadata={"modality": "audio"},
                )
            )

    def _on_done(self, request_id: str) -> None:
        if request_id in self._aborted_request_ids:
            return
        if request_id not in self._payloads:
            self._pending_done.add(request_id)
            return

        state = self._stream_states.setdefault(request_id, _HiggsStreamState())
        audio_chunk_size, _ = self._stream_sizes_for_request(request_id)
        output = flush_higgs_stream_chunk(
            state,
            codec=self._codec,
            num_codebooks=self._num_codebooks,
            audio_chunk_size=audio_chunk_size,
        )
        if output is not None and request_id not in self._aborted_request_ids:
            self.outbox.put(
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data=output,
                    metadata={"modality": "audio"},
                )
            )

        payload = self._payloads.get(request_id)
        if payload is None or request_id in self._aborted_request_ids:
            return
        result = self._finalize_streaming_payload(payload)
        self.outbox.put(
            OutgoingMessage(request_id=request_id, type="result", data=result)
        )
        self._clear_request_state(request_id)

    def _vocode_full(self, payload: StagePayload) -> StagePayload:
        state = HiggsTtsState.from_dict(payload.data)
        delayed_rows = state.output_codes_delayed
        out_data = dict(payload.data)
        out_data["sample_rate"] = _codec_sample_rate(self._codec)
        out_data["modality"] = "audio"
        if not delayed_rows:
            out_data["audio_data"] = []
            return StagePayload(
                request_id=payload.request_id,
                request=payload.request,
                data=out_data,
            )

        delayed = torch.tensor(delayed_rows, dtype=torch.long)
        audio = _decode_delayed_tokens(delayed, codec=self._codec)
        out_data["audio_data"] = [] if audio is None else audio.tolist()
        usage = _build_usage(state)
        if usage is not None:
            out_data["usage"] = usage
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=out_data,
        )

    def _finalize_streaming_payload(self, payload: StagePayload) -> StagePayload:
        state = HiggsTtsState.from_dict(payload.data)
        out_data = dict(payload.data)
        out_data["audio_data"] = []
        out_data["sample_rate"] = _codec_sample_rate(self._codec)
        out_data["modality"] = "audio"
        usage = _build_usage(state)
        if usage is not None:
            out_data["usage"] = usage
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=out_data,
        )

    def _clear_request_state(
        self,
        request_id: str,
        *,
        keep_aborted: bool = False,
    ) -> None:
        self._payloads.pop(request_id, None)
        self._request_params.pop(request_id, None)
        self._stream_states.pop(request_id, None)
        self._pending_done.discard(request_id)
        if not keep_aborted:
            self._aborted_request_ids.discard(request_id)

    @staticmethod
    def _is_streaming_payload(payload: StagePayload) -> bool:
        return bool((payload.request.params or {}).get("stream"))

    def _stream_sizes_for_request(self, request_id: str) -> tuple[int, int]:
        payload = self._payloads.get(request_id)
        if payload is not None:
            request_params = payload.request.params or {}
        else:
            request_params = self._request_params.get(request_id, {})
        vocoder_params = self._stage_vocoder_params(request_params)
        requested_chunk_size = vocoder_params.get("audio_chunk_size")
        audio_chunk_size = int(requested_chunk_size or self._audio_chunk_size)
        if "audio_chunk_overlap_size" in vocoder_params:
            audio_chunk_overlap_size = int(vocoder_params["audio_chunk_overlap_size"])
        elif requested_chunk_size is not None:
            audio_chunk_overlap_size = audio_chunk_size
        else:
            audio_chunk_overlap_size = self._audio_chunk_overlap_size
        _validate_stream_sizes(
            num_codebooks=self._num_codebooks,
            audio_chunk_size=audio_chunk_size,
            audio_chunk_overlap_size=audio_chunk_overlap_size,
        )
        return audio_chunk_size, audio_chunk_overlap_size

    def _remember_stream_request_params(self, request_id: str, chunk: Any) -> None:
        metadata = getattr(chunk, "metadata", None)
        if not isinstance(metadata, Mapping):
            return
        request_params = metadata.get("request_params")
        if isinstance(request_params, Mapping):
            self._request_params.setdefault(request_id, dict(request_params))

    @staticmethod
    def _stage_vocoder_params(request_params: Mapping[str, Any]) -> Mapping[str, Any]:
        stage_params = request_params.get("stage_params")
        if not isinstance(stage_params, Mapping):
            return {}
        vocoder_params = stage_params.get("vocoder")
        if not isinstance(vocoder_params, Mapping):
            return {}
        return vocoder_params


__all__ = [
    "HiggsVocoderScheduler",
    "_HiggsStreamState",
    "build_higgs_stream_chunk",
    "create_higgs_audio_chunk",
    "flush_higgs_stream_chunk",
]
