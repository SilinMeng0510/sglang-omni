# SPDX-License-Identifier: Apache-2.0
"""Per-request data + StagePayload <-> scheduler adapters for Higgs TTS (V1)."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.models.higgs_tts.session import session_extra_key
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData


@dataclass
class HiggsSGLangRequestData(SGLangARRequestData):
    """Per-request state for the Higgs TTS scheduler."""

    reference_codes_delayed: list[list[int]] | None = None
    num_ref_codes_consumed: int = 0
    num_codebooks: int = 8
    codebook_size: int = 1026
    output_codes: list[torch.Tensor] = field(default_factory=list)
    generation_done: bool = False
    # One freshly sampled delayed-code row for the streaming vocoder. The
    # OmniScheduler forwards it and clears the field every decode step.
    latest_stream_code_chunk: torch.Tensor | None = None
    engine_start_s: float = 0.0
    # Continuity: the result adapter commits this chunk's codes + text under
    # ``session_id`` so the next chunk conditions on it; ``session_final`` evicts.
    session_id: str | None = None
    session_final: bool = False
    session_text_token_ids: list[int] | None = None


class _ResettableHiggsModel(Protocol):
    def reset_request(self, req_id: str) -> None: ...


_HiggsRequestBuilder = Callable[[StagePayload], HiggsSGLangRequestData]
_HiggsResultAdapter = Callable[[HiggsSGLangRequestData], StagePayload]


def _ref_audio_fingerprint(codes: list[list[int]] | None) -> str | None:
    """Stable hash of the full N-codebook ref-audio sequence.

    Returned as a short hex string used as ``Req.extra_key``. ``None`` for
    zero-shot (no ref audio) so all zero-shot requests share the radix subtree.
    Each codec value packs into 2 bytes (range 0..1025) so the hash is
    sensitive to every codebook, not just cb0.
    """
    if not codes:
        return None
    buf = bytearray(2 * sum(len(row) for row in codes))
    i = 0
    for row in codes:
        for c in row:
            buf[i] = c & 0xFF
            buf[i + 1] = (c >> 8) & 0xFF
            i += 2
    return hashlib.blake2b(bytes(buf), digest_size=16).hexdigest()


def build_sglang_higgs_request(
    state: HiggsTtsState,
    *,
    request_id: str = "",
    extra_key_override: str | None = None,
) -> HiggsSGLangRequestData:
    input_ids_list = list(state.prompt_token_ids)
    input_ids = torch.tensor(input_ids_list, dtype=torch.long)

    sp_kwargs: dict[str, Any] = {
        "max_new_tokens": int(state.max_new_tokens),
        "temperature": float(state.temperature),
    }
    if state.top_p is not None:
        sp_kwargs["top_p"] = float(state.top_p)
    if state.top_k is not None:
        sp_kwargs["top_k"] = int(state.top_k)
    if state.seed is not None:
        sp_kwargs["seed"] = int(state.seed)
    sampling_params = SamplingParams(**sp_kwargs)
    # tokenizer_manager.normalize() is bypassed in our custom pipeline;
    # without it stop_strs / stop_regex_strs stay None and the upstream
    # scheduler's check_finished trips on ``len(None)``.
    sampling_params.normalize(tokenizer=None)

    # vocab_size = backbone text vocab so cb0 rides sglang's standard sampler path.
    # extra_key namespaces the radix cache (identical -100 placeholder prefixes
    # must not share KV): single-shot keys per ref-audio fingerprint, continuity
    # sessions key per session (prefix reuse within, isolation across).
    extra_key = (
        extra_key_override
        if extra_key_override is not None
        else _ref_audio_fingerprint(state.reference_codes_delayed)
    )
    req = Req(
        rid=request_id,
        origin_input_text="",
        origin_input_ids=input_ids_list,
        sampling_params=sampling_params,
        vocab_size=151_936,
        extra_key=extra_key,
    )
    # V1's prefill manager probes these attrs; absence triggers AttributeError.
    req._codec_suppress_tokens = None
    req._input_embeds_are_projected = False

    return HiggsSGLangRequestData(
        input_ids=input_ids,
        req=req,
        reference_codes_delayed=state.reference_codes_delayed,
        num_codebooks=int(state.num_codebooks),
        codebook_size=int(state.codebook_size),
        max_new_tokens=int(state.max_new_tokens),
        temperature=float(state.temperature),
        top_p=float(state.top_p) if state.top_p is not None else 1.0,
        top_k=int(state.top_k) if state.top_k is not None else -1,
    )


def apply_higgs_result(state: HiggsTtsState, data: HiggsSGLangRequestData) -> None:
    if data.output_codes:
        codes = torch.stack(data.output_codes, dim=0).to(torch.long)
        state.output_codes_delayed = codes.tolist()
        state.completion_tokens = int(codes.shape[0])
    else:
        state.output_codes_delayed = None
    state.prompt_tokens = len(data.input_ids)


def make_higgs_scheduler_adapters(
    model: _ResettableHiggsModel,
    *,
    max_new_tokens_cap: int | None = None,
    adapter: Any = None,
    session_store: Any = None,
) -> tuple[_HiggsRequestBuilder, _HiggsResultAdapter]:
    """Build (request_builder, result_adapter) closures bound to a
    :class:`HiggsTTSModel` instance.

    The result adapter drops the model's per-request slot (sampler state +
    accumulated codes) once a result is emitted so a long-running server
    doesn't accumulate dead slots.

    With ``adapter`` + ``session_store``, a request carrying
    ``state.session_id`` is reassembled into the interleaved continuity prompt:
    prior chunks' codes (from the store, never surfaced to serve) are woven in
    as ``<|text|> tok(t_i) <|audio|> [a_i]`` blocks and appended after the ref
    codes for the runner's order-based ``-100`` overlay. Both closures run in
    the engine process, so the store needs no cross-process sync.
    """

    def request_builder(payload: StagePayload) -> HiggsSGLangRequestData:
        state = HiggsTtsState.from_dict(payload.data)
        if max_new_tokens_cap is not None:
            state.max_new_tokens = min(
                int(state.max_new_tokens),
                int(max_new_tokens_cap),
            )

        extra_key_override: str | None = None
        session_text_token_ids: list[int] | None = None
        session_id = state.session_id
        if session_id and session_store is not None and adapter is not None:
            prompt_history, overlay_codes = session_store.history_for(session_id)
            num_ref_rows = len(state.reference_codes_delayed or [])
            session_text_token_ids = list(state.target_text_token_ids or [])
            state.prompt_token_ids = adapter.build_prompt_from_ids(
                session_text_token_ids,
                num_ref_tokens=num_ref_rows,
                reference_text_ids=state.reference_text_token_ids,
                history=prompt_history,
            )
            ref_codes = list(state.reference_codes_delayed or [])
            ref_codes.extend(overlay_codes)
            state.reference_codes_delayed = ref_codes or None
            extra_key_override = session_extra_key(session_id)

        data = build_sglang_higgs_request(
            state,
            request_id=payload.request_id,
            extra_key_override=extra_key_override,
        )
        data.session_id = session_id
        data.session_final = state.session_final
        data.session_text_token_ids = session_text_token_ids
        data.engine_start_s = time.perf_counter()
        data.stage_payload = payload
        return data

    def result_adapter(data: HiggsSGLangRequestData) -> StagePayload:
        payload = data.stage_payload
        state = HiggsTtsState.from_dict(payload.data)
        apply_higgs_result(state, data)
        if data.engine_start_s:
            state.engine_time_s = time.perf_counter() - data.engine_start_s
        if (
            session_store is not None
            and data.session_id
            and state.output_codes_delayed
        ):
            # Codes stay engine-side — never written onto the payload serve sees.
            session_store.commit(
                data.session_id,
                data.session_text_token_ids or [],
                state.output_codes_delayed,
            )
            if data.session_final:
                session_store.evict(data.session_id)
        model.reset_request(payload.request_id)
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
        )

    return request_builder, result_adapter


__all__ = [
    "HiggsSGLangRequestData",
    "apply_higgs_result",
    "build_sglang_higgs_request",
    "make_higgs_scheduler_adapters",
]
