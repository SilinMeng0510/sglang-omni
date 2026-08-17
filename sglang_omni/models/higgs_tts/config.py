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
    # steady emission = stride - num_codebooks + 1 = 25 frames (1 s), the
    # value the masked deployment config (higgs_tts_4b_masked.yaml) has run
    # in production. Emission size does not affect quality — every emitted
    # frame gets the same fixed 9-frame masked contexts either side
    # (measured same-seed spectral distance vs whole-utterance decode:
    # 0.032 at stride 15 and 75 alike) — it only trades steady batching
    # delay (~avg half a chunk) against vocoder calls.
    vocoder_stream_stride: int = Field(default=32, ge=1)
    vocoder_stream_followup_stride: int = Field(default=32, ge=1)
    # Streaming decode is always masked full-context (there is no windowed
    # streaming mode — it was audibly worse in same-seed listening A/B).
    # This flag only chooses the startup behavior: True = the
    # listening-tested K8/M3 schedule emits the first frames with reduced
    # lookahead for fast TTFA; False = uniform stride emission from frame 0.
    vocoder_low_latency_startup: bool = True
    # steady-phase decode context per side; 11 >= the codec decoder's
    # receptive field (+-10.4 frames), which makes steady chunk seams
    # bit-exact vs whole-utterance decode — stride becomes pure cadence.
    # Costs 2 extra frames (80 ms) of steady lookahead vs the old 9.
    vocoder_context_frames: int = Field(default=11, ge=0)
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
                    low_latency_startup=self.vocoder_low_latency_startup,
                    context_frames=self.vocoder_context_frames,
                    startup_masked_delay_rows=self.vocoder_startup_masked_delay_rows,
                    startup_masked_emit_frames=self.vocoder_startup_masked_emit_frames,
                    startup_masked_until_frames=self.vocoder_startup_masked_until_frames,
                )

    def requires_uploaded_voice_for_named_voice(self) -> bool:
        return True

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = HiggsTtsPipelineConfig
