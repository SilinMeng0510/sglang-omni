# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Higgs TTS (V1)."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.higgs_tts"


class HiggsTtsPipelineConfig(PipelineConfig):
    """4-stage TTS pipeline: preprocessing → audio_encoder → tts_engine → vocoder.

    Mirrors the V0 layout: preprocessing tokenises text + delay-pattern-encodes
    the reference audio codes; audio_encoder runs the fused multi-codebook
    embedding once on the delayed ref codes (CPU- or GPU-side); tts_engine
    drives the AR loop on the sglang backbone with the precomputed embed
    pasted at ``-100`` placeholder positions; vocoder reverses the delay
    pattern and decodes to waveform via the higgs-audio-v2-tokenizer codec.
    """

    architecture: ClassVar[str] = "HiggsMultimodalQwen3ForConditionalGeneration"
    requires_model_capabilities: ClassVar[bool] = True

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": "tts_engine"}

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "tts_engine"}

    model_path: str
    enable_dynamic_lora: bool = False
    lora_base_dir: str | None = None
    lora_backend: str = "triton"
    lora_max_rank: int = Field(default=32, ge=1)
    lora_max_cached_adapters: int = Field(default=8, ge=1)
    separate_vocoder_process: bool = False
    stages: list[StageConfig] = Field(
        default_factory=lambda: [
            StageConfig(
                name="preprocessing",
                process="pipeline",
                factory=f"{_PKG}.stages.create_preprocessing_executor",
                next="audio_encoder",
            ),
            StageConfig(
                name="audio_encoder",
                process="pipeline",
                factory=f"{_PKG}.stages.create_audio_encoder_executor",
                factory_args={"device": "cuda"},
                gpu=0,
                next="tts_engine",
            ),
            StageConfig(
                name="tts_engine",
                process="pipeline",
                factory=f"{_PKG}.stages.create_sglang_tts_engine_executor",
                factory_args={
                    "device": "cuda",
                    "max_new_tokens": 2048,
                    "enable_async_decode": True,
                },
                gpu=0,
                next="vocoder",
                stream_to=["vocoder"],
            ),
            StageConfig(
                name="vocoder",
                process="pipeline",
                factory=f"{_PKG}.stages.create_vocoder_executor",
                factory_args={"device": "cuda"},
                gpu=0,
                terminal=True,
                can_accept_stream_before_payload=True,
            ),
        ]
    )

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        for stage in self.stages:
            if stage.name == "tts_engine":
                stage.factory_args.update(
                    enable_dynamic_lora=self.enable_dynamic_lora,
                    lora_base_dir=self.lora_base_dir,
                    lora_backend=self.lora_backend,
                    lora_max_rank=self.lora_max_rank,
                    lora_max_cached_adapters=self.lora_max_cached_adapters,
                    serve_model_name=self.name,
                )
            elif stage.name == "vocoder" and self.separate_vocoder_process:
                stage.process = "vocoder"

    def requires_uploaded_voice_for_named_voice(self) -> bool:
        return True

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = HiggsTtsPipelineConfig
