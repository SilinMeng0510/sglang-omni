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
    # steady emission = stride - num_codebooks + 1 = 8 frames (0.32 s), the
    # same granularity as the listening-tested K8 startup chunks: continuous
    # live playback with no multi-second lumps. The masked decode context is
    # a fixed 9 frames either side, so small chunks cost bounded extra
    # vocoder compute (~3x per frame) and zero quality — measured same-seed
    # spectral distance 0.032 vs whole-utterance decode, identical to
    # stride 75. Throughput deployments may still raise this per config.
    vocoder_stream_stride: int = Field(default=15, ge=1)
    vocoder_stream_followup_stride: int = Field(default=15, ge=1)
    # masked full-context decoding: every streamed chunk is decoded with the
    # complete left context, so chunk seams carry no timbre discontinuity
    # (windowed mode re-decodes only 8 rows of context and is audibly worse —
    # confirmed by same-seed listening A/B). The masked path also ships its own
    # listening-tested small-chunk startup schedule (K8/M3).
    vocoder_full_context_streaming: bool = True
    vocoder_startup_masked_delay_rows: int = Field(default=8, ge=1)
    vocoder_startup_masked_emit_frames: int = Field(default=3, ge=1)
    vocoder_startup_masked_until_frames: int = Field(default=8, ge=1)
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
        if self.enable_dynamic_lora and not self.lora_base_dir:
            raise ValueError("enable_dynamic_lora requires lora_base_dir")
        colocated_fractions = {
            "audio_encoder": 0.01,
            "tts_engine": 0.90,
            "vocoder": 0.09,
        }
        for stage in self.stages:
            if self.separate_vocoder_process and stage.name in colocated_fractions:
                resources = stage.runtime.resources
                if resources.total_gpu_memory_fraction is None:
                    resources.total_gpu_memory_fraction = colocated_fractions[
                        stage.name
                    ]
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
            if stage.name == "vocoder":
                stage.factory_args.update(
                    stream_stride=self.vocoder_stream_stride,
                    stream_followup_stride=self.vocoder_stream_followup_stride,
                    full_context_streaming=self.vocoder_full_context_streaming,
                    startup_masked_delay_rows=self.vocoder_startup_masked_delay_rows,
                    startup_masked_emit_frames=self.vocoder_startup_masked_emit_frames,
                    startup_masked_until_frames=self.vocoder_startup_masked_until_frames,
                )

    def requires_uploaded_voice_for_named_voice(self) -> bool:
        return True

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = HiggsTtsPipelineConfig
