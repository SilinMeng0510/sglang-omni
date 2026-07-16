# SPDX-License-Identifier: Apache-2.0
"""Higgs TTS SGLang engine builder."""

from __future__ import annotations

import importlib
from typing import Any

from sglang_omni.models.higgs_tts import request_builders
from sglang_omni.models.higgs_tts import stages as higgs_stages
from sglang_omni.models.higgs_tts import utils as higgs_utils
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder


class HiggsTtsEngineBuilder(TtsEngineBuilder):
    model_name = "Higgs TTS"
    context_length = 4096

    def __init__(
        self,
        *,
        max_new_tokens: int | None,
        max_running_requests: int,
        cuda_graph_max_bs: int,
        enable_async_decode: bool,
        async_decode_min_batch_size: int,
        enable_dynamic_lora: bool = False,
        lora_base_dir: str | None = None,
        lora_backend: str = "triton",
        lora_max_rank: int = 32,
        lora_max_cached_adapters: int = 8,
        serve_model_name: str = "",
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.max_running_requests = max_running_requests
        self.cuda_graph_max_bs = cuda_graph_max_bs
        self.enable_async_decode = enable_async_decode
        self.async_decode_min_batch_size = async_decode_min_batch_size
        self.enable_dynamic_lora = enable_dynamic_lora
        self.lora_base_dir = lora_base_dir
        self.lora_backend = lora_backend
        self.lora_max_rank = lora_max_rank
        self.lora_max_cached_adapters = lora_max_cached_adapters
        self.serve_model_name = serve_model_name
        self.model: Any | None = None
        self.model_worker: Any | None = None

    def resolve_checkpoint(self, model_path: str) -> str:
        return higgs_stages.resolve_checkpoint(model_path)

    def generation_defaults(
        self,
        *,
        dtype: str,
    ) -> dict[str, Any]:
        del dtype
        # note (luojiaxuan): Radix cache is namespaced per ref-audio via
        # Req.extra_key (set in build_sglang_higgs_request); shared -100
        # placeholder prefixes from different ref audios can't cross-contaminate
        # the KV tree.
        return {
            "max_running_requests": self.max_running_requests,
            "cuda_graph_max_bs": self.cuda_graph_max_bs,
            "disable_cuda_graph": False,
            "mem_fraction_static": 0.85,
            "chunked_prefill_size": 8192,
            "dtype": "bfloat16",
        }

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        if not self.enable_dynamic_lora:
            return
        from sglang_omni.models.higgs_tts.lora import HIGGS_LORA_TARGET_MODULES

        capacity = int(self.lora_max_cached_adapters) + 1
        overrides.update(
            enable_lora=True,
            lora_paths={},
            max_lora_rank=int(self.lora_max_rank),
            max_loaded_loras=capacity,
            max_loras_per_batch=capacity,
            lora_backend=self.lora_backend,
            lora_target_modules=list(HIGGS_LORA_TARGET_MODULES),
        )

    def customize_server_args(self, server_args: Any) -> None:
        if self.enable_dynamic_lora:
            server_args.check_lora_server_args()
        server_args.disable_overlap_schedule = True

    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        del checkpoint_dir, device, gpu_id, server_args
        self.model_worker = model_worker
        self.model = model_worker.model_runner.model
        higgs_utils.truncate_rope_to_bf16(self.model)

    def get_model_buffer_bs(self, model: Any) -> int | None:
        return model.sampler_pool_max_running_requests

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        model_runner_mod = importlib.import_module(
            "sglang_omni.models.higgs_tts.model_runner"
        )

        return model_runner_mod.HiggsTTSModelRunner(model_worker, output_proc)

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        request_builder, result_adapter = (
            request_builders.make_higgs_scheduler_adapters(
                model,
                max_new_tokens_cap=self.max_new_tokens,
            )
        )
        if not self.enable_dynamic_lora:
            return request_builder, result_adapter

        from sglang_omni.models.higgs_tts.lora import DynamicLoraCache
        from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState

        assert self.model_worker is not None
        model_runner = self.model_worker.model_runner
        cache = DynamicLoraCache(
            model_runner.load_lora_adapter,
            max_cached_adapters=self.lora_max_cached_adapters,
            serve_model_name=self.serve_model_name,
            base_dir=self.lora_base_dir,
        )

        def dynamic_lora_request_builder(payload: Any) -> Any:
            state = HiggsTtsState.from_dict(payload.data)
            if state.lora_adapter_path is not None:
                state.lora_id = cache.get_or_load(state.lora_adapter_path)
                payload.data = state.to_dict()
            return request_builder(payload)

        return dynamic_lora_request_builder, result_adapter

    def make_abort_callback(self) -> Any | None:
        assert self.model is not None
        return self.model.reset_request

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        return {
            "enable_async_decode": self.enable_async_decode,
            "async_decode_min_batch_size": self.async_decode_min_batch_size,
        }

    def post_scheduler_setup(self, scheduler: Any, model_runner: Any) -> None:
        model_runner.set_stream_outbox(scheduler.outbox)
