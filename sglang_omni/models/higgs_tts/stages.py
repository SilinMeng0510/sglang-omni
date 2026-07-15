# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Higgs TTS pipeline.

Pipeline shape::

    preprocessing → audio_encoder → tts_engine → vocoder

- ``create_preprocessing_executor``: text tokenize + (if raw audio path)
  load waveform; fast path also delay-encodes client-supplied
  ``reference_codes`` and builds the prompt. Returns a
  :class:`ThreadedSimpleScheduler` for CPU-heavy work.
- ``create_audio_encoder_executor``: GPU codec encode for the raw-audio
  path → delayed ref codes + prompt assembly. No-op on the fast path.
- ``create_sglang_tts_engine_executor``: runs :class:`HiggsTTSModel` under
  sglang's worker; the model runner computes the fused multi-codebook
  embedding inline in prefill from ``reference_codes_delayed`` and overlays
  it at ``-100`` placeholder positions. Returns a :class:`OmniScheduler`.
- ``create_vocoder_executor``: reverses the delay pattern, decodes via
  :class:`HiggsAudioCodec` into a mono 24 kHz waveform. Returns a
  :class:`SimpleScheduler`.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import torch
import torchaudio.functional as F_audio
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from sglang_omni.models.higgs_tts.audio import HiggsAudioCodec
from sglang_omni.models.higgs_tts.audio.utils import (
    apply_delay_pattern,
    fade_out_tail,
    get_or_load_codec,
    load_audio_to_24k,
    reverse_delay_pattern,
    to_codes_TN,
)
from sglang_omni.models.higgs_tts.lora import validate_lora_adapter_model
from sglang_omni.models.higgs_tts.model_runner import HiggsTTSModelRunner
from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.models.higgs_tts.request_builders import make_higgs_scheduler_adapters
from sglang_omni.models.higgs_tts.session import SessionStore
from sglang_omni.models.higgs_tts.text.normalizer import normalize_punctuation
from sglang_omni.models.higgs_tts.text.tokenizer import HiggsTokenizerAdapter
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend import (
    SGLangOutputProcessor,
    build_sglang_server_args,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler

logger = logging.getLogger(__name__)


def truncate_rope_to_bf16(model: torch.nn.Module) -> None:
    """bf16-truncate sglang's fp32 ``cos_sin_cache`` in-place (stored as fp32)
    to match Higgs's bf16 training-time RoPE."""
    for module in model.modules():
        if hasattr(module, "cos_sin_cache"):
            module.cos_sin_cache.data = module.cos_sin_cache.data.to(torch.bfloat16).to(
                torch.float32
            )


def resolve_checkpoint(checkpoint: str) -> str:
    """Local dir or HF repo id → local snapshot path."""
    if Path(checkpoint).is_dir():
        return checkpoint
    return snapshot_download(checkpoint)


# Reject ref audio past this many seconds
_MAX_REF_AUDIO_SEC = 30
_HIGGS_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _resolve_voice(params: Any, metadata: Any) -> str:
    """Resolve the speech voice from public-API metadata or legacy params."""
    params = params if isinstance(params, dict) else {}
    metadata = metadata if isinstance(metadata, dict) else {}
    tts_params = metadata.get("tts_params")
    tts_params = tts_params if isinstance(tts_params, dict) else {}
    return str(tts_params.get("voice") or params.get("voice") or "default")


def _resolve_lora_adapter_path(metadata: Any) -> str | None:
    """Read the request-scoped LoRA adapter path from speech metadata."""
    metadata = metadata if isinstance(metadata, dict) else {}
    tts_params = metadata.get("tts_params")
    tts_params = tts_params if isinstance(tts_params, dict) else {}
    config = tts_params.get("lora_adapter")
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError("lora_adapter must be an object containing 'path'")
    path = config.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("lora_adapter.path must be a non-empty string")
    return path.strip()


class DynamicLoraCache:
    """Load each adapter path once and reuse SGLang's GPU-resident weights."""

    def __init__(
        self,
        load_adapter: Any,
        *,
        max_cached_adapters: int,
        serve_model_name: str,
        initial_refs: Any = (),
    ) -> None:
        self._load_adapter = load_adapter
        self._max_cached_adapters = int(max_cached_adapters)
        self._serve_model_name = serve_model_name
        self._refs_by_path: dict[str, Any] = {}
        for ref in initial_refs:
            path = getattr(ref, "lora_path", None)
            if path:
                resolved = str(Path(path).expanduser().resolve())
                validate_lora_adapter_model(resolved, self._serve_model_name)
                self._refs_by_path[resolved] = ref

    def get_or_load(self, path: str) -> str:
        try:
            resolved = Path(path).expanduser().resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"LoRA adapter path does not exist: {path}") from exc
        if not resolved.is_dir():
            raise ValueError(f"LoRA adapter path is not a directory: {resolved}")
        cache_key = str(resolved)
        cached = self._refs_by_path.get(cache_key)
        if cached is not None:
            logger.info("Dynamic LoRA cache hit: %s", cache_key)
            return cached.lora_id
        validate_lora_adapter_model(resolved, self._serve_model_name)
        if len(self._refs_by_path) >= self._max_cached_adapters:
            raise ValueError(
                "Dynamic LoRA cache is full "
                f"({self._max_cached_adapters} adapters); restart with a larger "
                "lora_max_cached_adapters value"
            )

        from sglang.srt.lora.lora_registry import LoRARef

        digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]
        name = f"dynamic-{digest}"
        ref = LoRARef(
            lora_id=LoRARef.deterministic_id(name, cache_key),
            lora_name=name,
            lora_path=cache_key,
            pinned=False,
        )
        logger.info("Dynamic LoRA cache miss; loading adapter: %s", cache_key)
        result = self._load_adapter(ref)
        if not getattr(result, "success", False):
            message = getattr(result, "error_message", "unknown loading error")
            raise ValueError(f"Failed to load LoRA adapter {cache_key}: {message}")
        self._refs_by_path[cache_key] = ref
        logger.info("Dynamic LoRA adapter cached: %s", cache_key)
        return ref.lora_id


def _with_dynamic_lora_cache(request_builder: Any, cache: DynamicLoraCache | None):
    def _build(payload: StagePayload):
        state = HiggsTtsState.from_dict(payload.data)
        if state.lora_adapter_path is not None:
            if cache is None:
                raise ValueError(
                    "Request specifies lora_adapter, but dynamic LoRA loading is "
                    "disabled; set enable_dynamic_lora=true"
                )
            state.lora_id = cache.get_or_load(state.lora_adapter_path)
            payload.data = state.to_dict()
        return request_builder(payload)

    return _build


def _build_higgs_audio_code_stream_outputs(
    request_id: str,
    data: Any,
    _req_output: Any,
):
    payload = getattr(data, "stage_payload", None)
    request_params = dict(payload.request.params or {}) if payload is not None else {}
    if not request_params.get("stream"):
        data.latest_stream_code_chunk = None
        return

    codes = getattr(data, "latest_stream_code_chunk", None)
    if codes is None:
        return

    yield OutgoingMessage(
        request_id=request_id,
        type="stream",
        data=codes,
        target="vocoder",
        metadata={
            "modality": "audio_codes",
            "request_params": request_params,
        },
    )
    data.latest_stream_code_chunk = None


def create_preprocessing_executor(
    model_path: str,
    *,
    num_codebooks: int = 8,
    codebook_size: int = 1026,
    max_concurrency: int = 8,
    lora_voices: dict[str, str] | None = None,
):
    """CPU stage: text tokenize + optional ref-audio file IO.

    Builds the full prompt + delays the codes when the client supplied
    pre-encoded ``reference_codes``. When raw audio is supplied, defers
    codec encoding (and prompt assembly) to the audio_encoder stage —
    only the loaded waveform is shipped forward.
    """
    checkpoint_dir = resolve_checkpoint(model_path)

    # Higgs ckpt tokenizer_config.json uses transformers v5 metadata and crashes
    # transformers<5's from_pretrained; load tokenizer.json directly to avoid it.
    raw = Tokenizer.from_file(os.path.join(checkpoint_dir, "tokenizer.json"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw)
    adapter = HiggsTokenizerAdapter(tokenizer)

    # ServerArgs assigns initial adapters stable UUID5 ids from name + path.
    # The tokenizer manager normally performs this lookup, but the Omni TTS
    # pipeline intentionally bypasses it, so preprocessing carries the same id
    # through StagePayload instead.
    lora_voice_ids: dict[str, str] = {}
    if lora_voices:
        from sglang.srt.lora.lora_registry import LoRARef

        for lora_name, lora_path in lora_voices.items():
            if not lora_name or not lora_path:
                raise ValueError("Higgs LoRA voice names and paths must be non-empty")
            if not Path(lora_path).is_dir():
                raise FileNotFoundError(
                    f"Higgs LoRA voice {lora_name!r} does not exist: {lora_path}"
                )
            lora_voice_ids[lora_name] = LoRARef.deterministic_id(lora_name, lora_path)

    def _preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs or {}
        params = payload.request.params or {}
        if isinstance(inputs, str):
            inputs = {"text": inputs}

        raw_refs = inputs.get("references")
        if raw_refs and isinstance(raw_refs, list):
            first = raw_refs[0]
            if isinstance(first, dict):
                inputs = dict(inputs)
                if first.get("text") and not inputs.get("reference_text"):
                    inputs["reference_text"] = first["text"]
                if inputs.get("reference_audio") is None:
                    if "bytes" in first or "base64" in first or "data" in first:
                        inputs["reference_audio"] = first
                    else:
                        inputs["reference_audio"] = first.get(
                            "audio_path"
                        ) or first.get("path")

        # The OpenAI speech API carries speech-only options in
        # metadata["tts_params"]. Keep params as a fallback for internal and
        # legacy callers that construct GenerateRequest directly.
        voice = _resolve_voice(params, payload.request.metadata)
        dynamic_lora_adapter_path = _resolve_lora_adapter_path(payload.request.metadata)

        # Continuity session tag (set by serve for chunked utterances); the
        # engine accumulates the audio and rebuilds the prompt. Unset = single-shot.
        session = payload.request.metadata.get("tts_session") or {}
        session_id = session.get("id")
        session_final = bool(session.get("final", False))
        session_index = int(session.get("index", -1))
        session_truncate_after = session.get("truncate_after")

        # Normalize CJK/full-width punctuation to ASCII just before tokenizing.
        text = normalize_punctuation(inputs.get("input") or inputs.get("text") or "")
        reference_text = inputs.get("reference_text") or None
        target_text_token_ids = list(tokenizer.encode(text, add_special_tokens=False))
        reference_text_token_ids = (
            list(tokenizer.encode(reference_text, add_special_tokens=False))
            if reference_text
            else None
        )
        ref_codes_TN = to_codes_TN(inputs.get("reference_codes"), num_codebooks)
        if (
            ref_codes_TN is not None
            and ref_codes_TN.shape[0] > _MAX_REF_AUDIO_SEC * HiggsAudioCodec.FRAME_RATE
        ):
            raise ValueError(
                f"reference_codes is too long ({ref_codes_TN.shape[0]} frames); "
                f"cap at {_MAX_REF_AUDIO_SEC}s of audio "
                f"(~{_MAX_REF_AUDIO_SEC * HiggsAudioCodec.FRAME_RATE} frames at "
                f"{HiggsAudioCodec.FRAME_RATE} Hz)."
            )

        waveform_tensor = None
        if ref_codes_TN is None and inputs.get("reference_audio") is not None:
            waveform_np, sample_rate = load_audio_to_24k(inputs["reference_audio"])
            wav = torch.from_numpy(waveform_np)
            if sample_rate != 24000:
                wav = F_audio.resample(wav, sample_rate, 24000)
            if wav.shape[-1] > _MAX_REF_AUDIO_SEC * 24000:
                raise ValueError(
                    f"reference_audio is too long "
                    f"({wav.shape[-1] / 24000:.1f}s); cap at {_MAX_REF_AUDIO_SEC}s."
                )
            waveform_tensor = wav.view(1, 1, -1).contiguous().float()

        # Built from the already-tokenized ids (no re-tokenize). ref_codes are
        # reference-only; the session path appends prior chunks' codes engine-side.
        if ref_codes_TN is not None:
            delayed = apply_delay_pattern(ref_codes_TN)
            prompt_ids = adapter.build_prompt_from_ids(
                target_text_token_ids,
                num_ref_tokens=delayed.shape[0],
                reference_text_ids=reference_text_token_ids,
            )
            ref_codes_delayed: list[list[int]] | None = delayed.tolist()
        elif waveform_tensor is None:
            prompt_ids = adapter.build_prompt_from_ids(
                target_text_token_ids,
                num_ref_tokens=0,
                reference_text_ids=reference_text_token_ids,
            )
            ref_codes_delayed = None
        else:
            # Raw-audio path: codec encode + prompt assembly happen in audio_encoder.
            prompt_ids = []
            ref_codes_delayed = None

        state = HiggsTtsState(
            prompt_token_ids=prompt_ids,
            reference_codes_delayed=ref_codes_delayed,
            reference_waveform=waveform_tensor,
            session_id=session_id,
            session_final=session_final,
            session_index=session_index,
            session_truncate_after=session_truncate_after,
            target_text_token_ids=target_text_token_ids,
            reference_text_token_ids=reference_text_token_ids,
            lora_id=None if dynamic_lora_adapter_path else lora_voice_ids.get(voice),
            lora_adapter_path=dynamic_lora_adapter_path,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            max_new_tokens=int(params.get("max_new_tokens", 1024)),
            temperature=float(params.get("temperature", 1.0)),
            top_p=params.get("top_p"),
            top_k=params.get("top_k"),
            seed=params.get("seed"),
        )
        payload.data = state.to_dict()
        return payload

    return ThreadedSimpleScheduler(_preprocess, max_concurrency=max_concurrency)


def create_audio_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    num_codebooks: int = 8,
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
):
    """GPU stage: codec-encode raw ref audio → delayed codes + prompt assembly.

    No-op when preprocessing already produced ``reference_codes_delayed`` (the
    client-supplied pre-encoded fast path). Codec weights are extracted from
    the TTS checkpoint itself (bundled at ``tied.embedding.modality_embeddings``).
    """
    checkpoint_dir = resolve_checkpoint(model_path)
    raw = Tokenizer.from_file(os.path.join(checkpoint_dir, "tokenizer.json"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw)
    adapter = HiggsTokenizerAdapter(tokenizer)

    codec = None

    def _encode(payload: StagePayload) -> StagePayload:
        nonlocal codec
        state = HiggsTtsState.from_dict(payload.data)
        waveform = state.reference_waveform
        if waveform is None:
            return payload

        if codec is None:
            codec = get_or_load_codec(checkpoint_dir, device, dtype)
        ref_codes_TN = codec.encode_reference(waveform, sample_rate=24000).to(
            torch.long
        )
        if ref_codes_TN.ndim != 2 or ref_codes_TN.shape[1] != num_codebooks:
            raise ValueError(
                f"codec output must be [T, {num_codebooks}], got "
                f"{tuple(ref_codes_TN.shape)}"
            )
        delayed = apply_delay_pattern(ref_codes_TN)
        state.prompt_token_ids = adapter.build_prompt_from_ids(
            state.target_text_token_ids or [],
            num_ref_tokens=delayed.shape[0],
            reference_text_ids=state.reference_text_token_ids,
        )
        # Reference-only; the session path appends prior chunks' codes engine-side.
        state.reference_codes_delayed = delayed.tolist()
        state.reference_waveform = None
        payload.data = state.to_dict()
        return payload

    return SimpleScheduler(
        _encode,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    max_new_tokens: int | None = 1024,
    max_history_chunks: int = 4,
    server_args_overrides: dict[str, Any] | None = None,
    lora_voices: dict[str, str] | None = None,
    lora_backend: str = "triton",
    lora_max_rank: int = 32,
    enable_dynamic_lora: bool = False,
    lora_max_cached_adapters: int = 8,
    serve_model_name: str | None = None,
):
    """sglang-backed AR engine for Higgs TTS.

    ``max_history_chunks`` is the continuity sliding-window cap (prior chunks
    conditioning the next; ``0`` disables it). Set it via the pipeline config's
    top-level ``max_history_chunks`` (yaml or CLI ``max_history_chunks=N``),
    which routes it here.
    """
    serve_model_name = serve_model_name or model_path
    for adapter_path in (lora_voices or {}).values():
        validate_lora_adapter_model(adapter_path, serve_model_name)
    checkpoint_dir = resolve_checkpoint(model_path)
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0

    overrides: dict[str, Any] = {
        "disable_cuda_graph": False,
        "cuda_graph_max_bs": 32,
        "mem_fraction_static": 0.85,
        "max_running_requests": 16,
        "chunked_prefill_size": 8192,
        "dtype": "bfloat16",
        # Radix cache is namespaced per ref-audio via Req.extra_key (set in
        # build_sglang_higgs_request); shared -100 placeholder prefixes from
        # different ref audios can't cross-contaminate the KV tree.
    }
    if server_args_overrides:
        overrides.update(server_args_overrides)
    if lora_voices or enable_dynamic_lora:
        initial_lora_count = len(lora_voices or {})
        max_loras_per_batch = (
            max(lora_max_cached_adapters, initial_lora_count) + 1
            if enable_dynamic_lora
            else initial_lora_count + 1
        )
        # SGLang loads only the small adapter tensors and applies them per
        # request, so base and multiple voices can safely share one batch.
        overrides.update(
            enable_lora=True,
            lora_paths=dict(lora_voices or {}),
            max_lora_rank=lora_max_rank,
            max_loaded_loras=max_loras_per_batch,
            max_loras_per_batch=max_loras_per_batch,
            lora_backend=lora_backend,
        )
        if enable_dynamic_lora:
            # With no initial adapter, SGLang cannot auto-detect which modules
            # need LoRA buffers. Higgs training targets these seven projections.
            overrides["lora_target_modules"] = list(_HIGGS_LORA_TARGET_MODULES)

    server_args = build_sglang_server_args(
        checkpoint_dir,
        context_length=4096,
        **overrides,
    )
    # ``build_sglang_server_args`` constructs the dataclass directly rather
    # than going through SGLang's CLI parser.  Normalize name/path mappings to
    # LoRARef objects here; LoRAManager intentionally accepts only LoRARef.
    if isinstance(getattr(server_args, "lora_paths", None), dict):
        server_args.check_lora_server_args()
    server_args.disable_overlap_schedule = True

    want_cuda_graph = not bool(getattr(server_args, "disable_cuda_graph", False))
    if want_cuda_graph:
        server_args.disable_cuda_graph = True

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = create_sglang_infrastructure(server_args, gpu_id)

    if want_cuda_graph:
        server_args.disable_cuda_graph = False

    truncate_rope_to_bf16(model_worker.model_runner.model)
    if want_cuda_graph:
        model_worker.model_runner.model.enable_cuda_graph_decoder(
            max_batch_size=server_args.max_running_requests,
        )
        model_worker.model_runner.init_device_graphs()

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model_worker.model_runner.model,
    )
    model_runner = HiggsTTSModelRunner(model_worker, output_proc)
    model = model_worker.model_runner.model

    # Continuity: a tokenizer adapter (to rebuild the prompt) + a per-session
    # code store, shared by the request builder and result adapter.
    raw_tok = Tokenizer.from_file(os.path.join(checkpoint_dir, "tokenizer.json"))
    engine_adapter = HiggsTokenizerAdapter(
        PreTrainedTokenizerFast(tokenizer_object=raw_tok)
    )
    session_store = SessionStore(max_history_chunks=max_history_chunks)

    request_builder, result_adapter = make_higgs_scheduler_adapters(
        model,
        max_new_tokens_cap=max_new_tokens,
        adapter=engine_adapter,
        session_store=session_store,
    )
    dynamic_lora_cache = None
    if enable_dynamic_lora:
        lora_manager = model_worker.model_runner.lora_manager
        dynamic_lora_cache = DynamicLoraCache(
            model_worker.model_runner.load_lora_adapter,
            max_cached_adapters=max(
                lora_max_cached_adapters,
                len(lora_manager.lora_refs),
            ),
            serve_model_name=serve_model_name,
            initial_refs=lora_manager.lora_refs.values(),
        )
    request_builder = _with_dynamic_lora_cache(
        request_builder,
        dynamic_lora_cache,
    )

    return OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        model_runner=model_runner,
        request_builder=request_builder,
        result_adapter=result_adapter,
        stream_output_builder=_build_higgs_audio_code_stream_outputs,
        abort_callback=model.reset_request,
    )


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    max_batch_size: int = 4,
    max_batch_wait_ms: int = 2,
    streaming: bool = False,
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
    num_codebooks: int = 8,
):
    """Decode Higgs delayed codes to a mono 24 kHz waveform.

    Codec weights are extracted from the TTS checkpoint itself.

    When ``streaming=True``, return a scheduler that accepts per-step delayed
    code rows from the AR stage and streams audio chunks before the terminal
    result. Non-streaming requests still use the full one-shot decode path.
    """
    checkpoint_dir = resolve_checkpoint(model_path)
    codec = get_or_load_codec(checkpoint_dir, device, dtype)
    sample_rate = HiggsAudioCodec.SAMPLE_RATE

    if streaming:
        from sglang_omni.models.higgs_tts.streaming_vocoder import HiggsVocoderScheduler

        return HiggsVocoderScheduler(
            codec,
            device=device,
            num_codebooks=num_codebooks,
            audio_chunk_size=audio_chunk_size,
            audio_chunk_overlap_size=audio_chunk_overlap_size,
            startup_full_chunk_frames=startup_full_chunk_frames,
            startup_full_chunk_count=startup_full_chunk_count,
            full_context_streaming=full_context_streaming,
            context_frames=context_frames,
            startup_reduced_context_frames=startup_reduced_context_frames,
            startup_reduced_left_context_frames=(startup_reduced_left_context_frames),
            startup_reduced_context_until_frames=(startup_reduced_context_until_frames),
            startup_masked_delay_rows=startup_masked_delay_rows,
            startup_masked_emit_frames=startup_masked_emit_frames,
            startup_masked_until_frames=startup_masked_until_frames,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )

    def _vocode(payload: StagePayload) -> StagePayload:
        state = HiggsTtsState.from_dict(payload.data)
        delayed_rows = state.output_codes_delayed

        if not delayed_rows:
            payload.data["audio_data"] = []
            payload.data["sample_rate"] = sample_rate
            payload.data["modality"] = "audio"
            return payload

        delayed_LN = torch.tensor(delayed_rows, dtype=torch.long)
        N = state.num_codebooks
        if delayed_LN.shape[0] < N:
            payload.data["audio_data"] = []
            payload.data["sample_rate"] = sample_rate
            payload.data["modality"] = "audio"
            return payload

        codes_TN = reverse_delay_pattern(delayed_LN)
        codec_vocab = state.codebook_size - 2  # 1026 - BOC - EOC
        codes_TN = torch.where(
            codes_TN >= codec_vocab, torch.zeros_like(codes_TN), codes_TN
        )
        waveform = codec.decode(codes_TN)
        waveform = fade_out_tail(waveform, sample_rate)
        audio_np = waveform.detach().to(torch.float32).cpu().numpy()

        payload.data["audio_data"] = audio_np.tolist()
        payload.data["sample_rate"] = sample_rate
        payload.data["modality"] = "audio"
        if state.prompt_tokens or state.completion_tokens or state.engine_time_s:
            usage = {
                "prompt_tokens": state.prompt_tokens,
                "completion_tokens": state.completion_tokens,
                "total_tokens": state.prompt_tokens + state.completion_tokens,
            }
            if state.engine_time_s:
                usage["engine_time_s"] = round(state.engine_time_s, 6)
            payload.data["usage"] = usage
        return payload

    return SimpleScheduler(
        _vocode,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )


__all__ = [
    "create_audio_encoder_executor",
    "create_preprocessing_executor",
    "create_sglang_tts_engine_executor",
    "create_vocoder_executor",
]
