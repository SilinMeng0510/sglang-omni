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
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.higgs_tts.audio.utils import TAIL_TRIM_FRAMES, fade_out_tail
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
    next_emit_frame: int = 0
    startup_full_chunks_emitted: int = 0


@dataclass
class _ContextDecodeTask:
    request_id: str
    state: _HiggsStreamState
    codes_TN: torch.Tensor
    counts_T: torch.Tensor
    audio_offset_samples: int
    emit_frames: int
    phase: str


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


def _gather_rvq_window(
    delayed_LN: torch.Tensor,
    *,
    frame_start: int,
    frame_end: int,
    num_codebooks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather every genuinely available leading RVQ codebook per raw frame."""
    if not 0 <= frame_start < frame_end:
        raise ValueError(f"invalid frame window [{frame_start}, {frame_end})")
    available_rows = int(delayed_LN.shape[0])
    codes = torch.zeros((frame_end - frame_start, num_codebooks), dtype=torch.long)
    counts = torch.empty((frame_end - frame_start,), dtype=torch.long)
    for output_frame, frame in enumerate(range(frame_start, frame_end)):
        count = min(num_codebooks, max(available_rows - frame, 0))
        if count <= 0:
            raise ValueError(f"frame {frame} has no available codebooks")
        counts[output_frame] = count
        for codebook in range(count):
            codes[output_frame, codebook] = delayed_LN[frame + codebook, codebook]
    return codes, counts


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


def _decode_delayed_tokens_batch(
    delayed_tokens: list[torch.Tensor], *, codec: Any
) -> list[torch.Tensor]:
    """Decode equal-shaped first chunks together, with a serial fallback."""
    if len(delayed_tokens) <= 1 or not hasattr(codec, "decode_batch"):
        return [
            audio
            for tokens in delayed_tokens
            if (audio := _decode_delayed_tokens(tokens, codec=codec)) is not None
        ]

    codec_vocab = _codec_vocab_size(codec)
    codes = torch.stack(
        [_reverse_delay_pattern(tokens.to(torch.long)) for tokens in delayed_tokens]
    )
    codes = torch.where(codes >= codec_vocab, torch.zeros_like(codes), codes)
    codes = torch.clamp(codes, 0, codec_vocab - 1)
    with torch.no_grad():
        audio = codec.decode_batch(codes).detach().to(torch.float32).cpu()
    return [row for row in audio]


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
        ready_audio = fade_out_tail(audio, _codec_sample_rate(codec))
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
        emit_frames = int(audio_chunk_size) - int(num_codebooks) + 1
        if cache_len < first_threshold:
            return None
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
    # Trim the click-prone wind-down frame(s) before the final decode: one
    # trailing delayed row == one final data frame (see audio.utils).
    cache = state.delayed_tokens_cache
    cache = cache[: max(int(cache.shape[0]) - TAIL_TRIM_FRAMES, 0)]
    fade_out_audio = state.fade_out_audio
    state.delayed_tokens_cache = torch.empty((0, num_codebooks), dtype=torch.long)
    state.fade_out_audio = None
    state.is_first_chunk = True

    if cache.shape[0] < num_codebooks:
        # Not enough rows left to decode; the utterance ends on the retained
        # decoded-but-unreleased tail.
        if fade_out_audio is None or fade_out_audio.numel() == 0:
            return None
        tail = fade_out_tail(fade_out_audio, _codec_sample_rate(codec))
        return _build_audio_chunk_payload(tail, sample_rate=_codec_sample_rate(codec))

    chunk, _ = create_higgs_audio_chunk(
        cache,
        int(audio_chunk_size),
        fade_out_audio,
        codec=codec,
        finalize=True,
    )
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
        "audio_data": audio_data.detach()
        .to(dtype=torch.float32, device="cpu")
        .tolist(),
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
        startup_full_chunk_frames: int = 8,
        startup_full_chunk_count: int = 8,
        full_context_streaming: bool = False,
        context_frames: int = 9,
        startup_reduced_context_frames: int | None = None,
        startup_reduced_left_context_frames: int | None = None,
        startup_reduced_context_until_frames: int = 0,
        startup_masked_delay_rows: int | None = None,
        startup_masked_emit_frames: int = 2,
        startup_masked_until_frames: int = 0,
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
        self._startup_full_chunk_frames = int(startup_full_chunk_frames)
        self._startup_full_chunk_count = int(startup_full_chunk_count)
        self._full_context_streaming = bool(full_context_streaming)
        self._context_frames = int(context_frames)
        if self._context_frames < 0:
            raise ValueError("context_frames must be >= 0")
        self._startup_reduced_context_frames = (
            int(startup_reduced_context_frames)
            if startup_reduced_context_frames is not None
            else None
        )
        self._startup_reduced_left_context_frames = (
            int(startup_reduced_left_context_frames)
            if startup_reduced_left_context_frames is not None
            else self._context_frames
        )
        self._startup_reduced_context_until_frames = int(
            startup_reduced_context_until_frames
        )
        self._startup_masked_delay_rows = (
            int(startup_masked_delay_rows)
            if startup_masked_delay_rows is not None
            else None
        )
        self._startup_masked_emit_frames = int(startup_masked_emit_frames)
        self._startup_masked_until_frames = int(startup_masked_until_frames)
        if (
            self._startup_reduced_context_frames is not None
            and not 0 <= self._startup_reduced_context_frames <= self._context_frames
        ):
            raise ValueError(
                "startup_reduced_context_frames must be within [0, context_frames]"
            )
        if not 0 <= self._startup_reduced_left_context_frames <= self._context_frames:
            raise ValueError(
                "startup_reduced_left_context_frames must be within "
                "[0, context_frames]"
            )
        if self._startup_reduced_context_until_frames < 0:
            raise ValueError("startup_reduced_context_until_frames must be >= 0")
        if (
            self._startup_reduced_context_until_frames
            and self._startup_reduced_context_frames is None
        ):
            raise ValueError(
                "startup_reduced_context_until_frames requires reduced context"
            )
        if self._startup_masked_delay_rows is not None:
            if not self._full_context_streaming:
                raise ValueError("masked delay startup requires full_context_streaming")
            if (
                not 1
                <= self._startup_masked_emit_frames
                <= self._startup_masked_delay_rows
            ):
                raise ValueError(
                    "startup_masked_emit_frames must be within masked delay rows"
                )
            if self._startup_masked_delay_rows > self._num_codebooks:
                raise ValueError("startup_masked_delay_rows exceeds num_codebooks")
            if self._startup_masked_until_frames < self._startup_masked_emit_frames:
                raise ValueError("startup_masked_until_frames is too small")
            # A short complete-RVQ transition with reduced real right context
            # may follow the masked region. The masked branch is selected
            # first; reduced-context decoding begins only after it ends.
        elif self._startup_masked_until_frames:
            raise ValueError("startup_masked_until_frames requires masked delay rows")
        if self._startup_full_chunk_frames < 1:
            raise ValueError("startup_full_chunk_frames must be >= 1")
        if self._startup_full_chunk_count < 0:
            raise ValueError("startup_full_chunk_count must be >= 0")
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
                    self._on_chunk_batch(self._drain_stream_chunk_batch(msg))
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

    def _drain_stream_chunk_batch(
        self, first: IncomingMessage
    ) -> list[IncomingMessage]:
        messages = [first]
        deadline = time.monotonic() + self._max_batch_wait_s
        while len(messages) < self._max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                msg = self.inbox.get(timeout=remaining)
            except _queue_mod.Empty:
                break
            if msg.type != "stream_chunk":
                self._pending_messages.append(msg)
                break
            messages.append(msg)
        return messages

    def _on_chunk_batch(self, messages: list[IncomingMessage]) -> None:
        if self._full_context_streaming:
            self._on_context_chunk_batch(messages)
            return
        first_ready: list[tuple[str, _HiggsStreamState, int]] = []
        followups: set[str] = set()
        for msg in messages:
            request_id = msg.request_id
            if request_id in self._aborted_request_ids:
                continue
            self._remember_stream_request_params(request_id, msg.data)
            codes = getattr(msg.data, "data", msg.data)
            if not isinstance(codes, torch.Tensor):
                codes = torch.as_tensor(codes, dtype=torch.long)
            state = self._stream_states.setdefault(request_id, _HiggsStreamState())
            audio_chunk_size, _ = self._stream_sizes_for_request(request_id)
            _append_delayed_rows(state, codes, num_codebooks=self._num_codebooks)
            emit_frames = audio_chunk_size - self._num_codebooks + 1
            threshold = audio_chunk_size + self._num_codebooks - 1
            if (
                state.is_first_chunk
                and state.delayed_tokens_cache.shape[0] >= threshold
            ):
                first_ready.append((request_id, state, emit_frames))
            elif not state.is_first_chunk:
                followups.add(request_id)

        # Ready first chunks have the same configured frame count and therefore
        # equal tensor shapes; decode them in one GPU batch across requests.
        if first_ready:
            audios = _decode_delayed_tokens_batch(
                [state.delayed_tokens_cache for _, state, _ in first_ready],
                codec=self._codec,
            )
            for (request_id, state, emit_frames), audio in zip(first_ready, audios):
                emit_samples = emit_frames * _codec_frame_length(self._codec)
                state.fade_out_audio = audio[emit_samples:].clone()
                state.delayed_tokens_cache = state.delayed_tokens_cache[
                    emit_frames:
                ].clone()
                state.is_first_chunk = False
                ready_audio = audio[:emit_samples]
                if ready_audio.numel() and request_id not in self._aborted_request_ids:
                    self.outbox.put(
                        OutgoingMessage(
                            request_id=request_id,
                            type="stream",
                            data=_build_audio_chunk_payload(
                                ready_audio,
                                sample_rate=_codec_sample_rate(self._codec),
                            ),
                            metadata={"modality": "audio"},
                        )
                    )

        # Group equal-shaped follow-up chunks for one codec call, then apply
        # each request's established crossfade and cache update independently.
        groups: dict[tuple[int, int], list[tuple[str, _HiggsStreamState]]] = {}
        for request_id in followups:
            state = self._stream_states[request_id]
            audio_chunk_size, audio_chunk_overlap_size = self._stream_sizes_for_request(
                request_id
            )
            cache_len = int(state.delayed_tokens_cache.shape[0])
            if cache_len < audio_chunk_size + audio_chunk_overlap_size:
                continue
            groups.setdefault((cache_len, audio_chunk_size), []).append(
                (request_id, state)
            )

        for (_, audio_chunk_size), group in groups.items():
            audios = _decode_delayed_tokens_batch(
                [state.delayed_tokens_cache for _, state in group],
                codec=self._codec,
            )
            emit_samples = audio_chunk_size * _codec_frame_length(self._codec)
            for (request_id, state), audio in zip(group, audios):
                audio = _hamming_crossfade(
                    audio,
                    state.fade_out_audio,
                    max_window_len=_codec_frame_length(self._codec),
                )
                state.fade_out_audio = audio[emit_samples:].clone()
                ready_audio = audio[:emit_samples]
                state.delayed_tokens_cache = state.delayed_tokens_cache[
                    audio_chunk_size:
                ].clone()
                if ready_audio.numel() and request_id not in self._aborted_request_ids:
                    self.outbox.put(
                        OutgoingMessage(
                            request_id=request_id,
                            type="stream",
                            data=_build_audio_chunk_payload(
                                ready_audio,
                                sample_rate=_codec_sample_rate(self._codec),
                            ),
                            metadata={"modality": "audio"},
                        )
                    )

    def _prepare_context_task(
        self, request_id: str, state: _HiggsStreamState
    ) -> _ContextDecodeTask | None:
        next_frame = int(state.next_emit_frame)
        cache_len = int(state.delayed_tokens_cache.shape[0])
        frame_length = _codec_frame_length(self._codec)
        if (
            self._startup_masked_delay_rows is not None
            and next_frame < self._startup_masked_until_frames
        ):
            emit_frames = min(
                self._startup_masked_emit_frames,
                self._startup_masked_until_frames - next_frame,
            )
            frame_start = max(0, next_frame - self._context_frames)
            frame_end = next_frame + self._startup_masked_delay_rows
            if cache_len < frame_end:
                return None
            phase = "partial_masked_delay"
        elif (
            self._startup_reduced_context_frames is not None
            and next_frame < self._startup_reduced_context_until_frames
        ):
            configured_emit = self._startup_full_chunk_frames
            emit_frames = min(
                configured_emit,
                self._startup_reduced_context_until_frames - next_frame,
            )
            active_left_context = self._startup_reduced_left_context_frames
            active_right_context = self._startup_reduced_context_frames
            phase = "full_reduced_context"
        elif state.startup_full_chunks_emitted < self._startup_full_chunk_count:
            emit_frames = self._startup_full_chunk_frames
            active_left_context = self._context_frames
            active_right_context = self._context_frames
            phase = "full_startup"
        else:
            emit_frames, _ = self._stream_sizes_for_request(request_id)
            active_left_context = self._context_frames
            active_right_context = self._context_frames
            phase = "full_context"

        if phase != "partial_masked_delay":
            frame_start = max(0, next_frame - active_left_context)
            frame_end = next_frame + emit_frames + active_right_context
            if cache_len < frame_end + self._num_codebooks - 1:
                return None

        codes, counts = _gather_rvq_window(
            state.delayed_tokens_cache,
            frame_start=frame_start,
            frame_end=frame_end,
            num_codebooks=self._num_codebooks,
        )
        codec_vocab = _codec_vocab_size(self._codec)
        codes = torch.where(codes >= codec_vocab, torch.zeros_like(codes), codes)
        codes = torch.clamp(codes, 0, codec_vocab - 1)
        return _ContextDecodeTask(
            request_id=request_id,
            state=state,
            codes_TN=codes,
            counts_T=counts,
            audio_offset_samples=(next_frame - frame_start) * frame_length,
            emit_frames=emit_frames,
            phase=phase,
        )

    def _on_context_chunk_batch(self, messages: list[IncomingMessage]) -> None:
        touched: set[str] = set()
        for msg in messages:
            request_id = msg.request_id
            if request_id in self._aborted_request_ids:
                continue
            self._remember_stream_request_params(request_id, msg.data)
            codes = getattr(msg.data, "data", msg.data)
            if not isinstance(codes, torch.Tensor):
                codes = torch.as_tensor(codes, dtype=torch.long)
            state = self._stream_states.setdefault(request_id, _HiggsStreamState())
            _append_delayed_rows(state, codes, num_codebooks=self._num_codebooks)
            touched.add(request_id)

        while True:
            tasks = [
                task
                for request_id in touched
                if (
                    task := self._prepare_context_task(
                        request_id, self._stream_states[request_id]
                    )
                )
                is not None
            ]
            if not tasks:
                break
            groups: dict[tuple[Any, ...], list[_ContextDecodeTask]] = {}
            for task in tasks:
                key = (
                    task.phase.startswith("partial"),
                    tuple(task.codes_TN.shape),
                    task.audio_offset_samples,
                    task.emit_frames,
                )
                groups.setdefault(key, []).append(task)

            # Under rolling concurrency, follow-up vocoder work from an older
            # request can otherwise run ahead of a newly ready first chunk.
            # Prioritize the smallest next-emission position. This changes only
            # cross-request GPU submission order; codes and per-request audio
            # windows remain identical.
            ordered_groups = sorted(
                groups.items(),
                key=lambda item: min(
                    int(task.state.next_emit_frame) for task in item[1]
                ),
            )
            for (masked, _, _, _), group in ordered_groups:
                code_batch = torch.stack([task.codes_TN for task in group])
                if masked:
                    count_batch = torch.stack([task.counts_T for task in group])
                    audios = self._codec_decode_masked_batch(code_batch, count_batch)
                else:
                    audios = self._codec_decode_codes_batch(code_batch)
                for task, audio in zip(group, audios):
                    state = task.state
                    emit_samples = task.emit_frames * _codec_frame_length(self._codec)
                    start = task.audio_offset_samples
                    ready_audio = audio[start : start + emit_samples]
                    state.next_emit_frame += task.emit_frames
                    state.is_first_chunk = False
                    if task.phase == "full_startup":
                        state.startup_full_chunks_emitted += 1
                    if (
                        ready_audio.numel()
                        and task.request_id not in self._aborted_request_ids
                    ):
                        self.outbox.put(
                            OutgoingMessage(
                                request_id=task.request_id,
                                type="stream",
                                data=_build_audio_chunk_payload(
                                    ready_audio,
                                    sample_rate=_codec_sample_rate(self._codec),
                                ),
                                metadata={"modality": "audio"},
                            )
                        )

    def _codec_decode_masked_batch(
        self, codes: torch.Tensor, counts: torch.Tensor
    ) -> list[torch.Tensor]:
        if not hasattr(self._codec, "decode_masked_batch"):
            raise TypeError(
                "masked startup requires a codec with decode_masked_batch support"
            )
        audio = self._codec.decode_masked_batch(codes, counts).to(torch.float32)
        return [row for row in audio]

    def _codec_decode_codes_batch(
        self, codes: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        if isinstance(codes, torch.Tensor):
            code_batch = codes
        else:
            code_batch = torch.stack(codes)
        if len(codes) > 1 and hasattr(self._codec, "decode_batch"):
            audio = self._codec.decode_batch(code_batch).to(torch.float32)
            return [row for row in audio]
        return [self._codec.decode(item).to(torch.float32) for item in code_batch]

    def _on_done(self, request_id: str) -> None:
        if request_id in self._aborted_request_ids:
            return
        if request_id not in self._payloads:
            self._pending_done.add(request_id)
            return

        state = self._stream_states.setdefault(request_id, _HiggsStreamState())
        audio_chunk_size, _ = self._stream_sizes_for_request(request_id)
        if self._full_context_streaming:
            output = self._flush_context_stream(state)
        else:
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

    def _flush_context_stream(self, state: _HiggsStreamState) -> dict[str, Any] | None:
        cache = state.delayed_tokens_cache
        total_frames = max(
            int(cache.shape[0]) - (self._num_codebooks - 1) - TAIL_TRIM_FRAMES,
            0,
        )
        next_frame = int(state.next_emit_frame)
        if total_frames <= next_frame:
            return None
        frame_start = max(0, next_frame - self._context_frames)
        codes, counts = _gather_rvq_window(
            cache,
            frame_start=frame_start,
            frame_end=total_frames,
            num_codebooks=self._num_codebooks,
        )
        if torch.any(counts != self._num_codebooks):
            raise ValueError("final context frames are not delay-complete")
        codec_vocab = _codec_vocab_size(self._codec)
        codes = torch.where(codes >= codec_vocab, torch.zeros_like(codes), codes)
        codes = torch.clamp(codes, 0, codec_vocab - 1)
        audio = self._codec.decode(codes).to(torch.float32)
        offset = (next_frame - frame_start) * _codec_frame_length(self._codec)
        audio = audio[offset:]
        audio = fade_out_tail(audio, _codec_sample_rate(self._codec))
        state.delayed_tokens_cache = torch.empty(
            (0, self._num_codebooks), dtype=torch.long
        )
        state.fade_out_audio = None
        return _build_audio_chunk_payload(
            audio, sample_rate=_codec_sample_rate(self._codec)
        )

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
        if audio is not None:
            audio = fade_out_tail(audio, _codec_sample_rate(self._codec))
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
