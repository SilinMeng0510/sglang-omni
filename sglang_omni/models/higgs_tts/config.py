# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Higgs TTS (V1)."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from pydantic import Field

from sglang_omni.config import PipelineConfig, StageConfig

if TYPE_CHECKING:
    from sglang_omni.models.higgs_tts.text.chunker import ChunkerOptions

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

    model_path: str
    chunker_max_seconds: float = Field(default=8.0, gt=0)
    chunker_cps: float = Field(default=10.0, gt=0)
    max_history_chunks: int = Field(default=4, ge=0)
    preprocessing_max_concurrency: int = Field(default=8, ge=1)
    separate_vocoder_process: bool = False
    # Voice IDs whose acoustic model is a PEFT LoRA adapter.  The mapping is
    # passed to both preprocessing (voice -> deterministic adapter id) and the
    # AR engine (adapter name -> adapter path).
    lora_voices: dict[str, str] = Field(default_factory=dict)
    lora_backend: str = "triton"
    lora_max_rank: int = Field(default=32, ge=1)
    enable_dynamic_lora: bool = False
    lora_max_cached_adapters: int = Field(default=8, ge=1)
    startup_full_chunk_frames: int = Field(default=8, ge=1)
    startup_full_chunk_count: int = Field(default=8, ge=0)
    vocoder_full_context_streaming: bool = False
    vocoder_context_frames: int = Field(default=9, ge=0)
    vocoder_startup_reduced_context_frames: int | None = Field(default=None, ge=0)
    vocoder_startup_reduced_left_context_frames: int | None = Field(default=None, ge=0)
    vocoder_startup_reduced_context_until_frames: int = Field(default=0, ge=0)
    vocoder_startup_masked_delay_rows: int | None = Field(default=None, ge=1)
    vocoder_startup_masked_emit_frames: int = Field(default=2, ge=1)
    vocoder_startup_masked_until_frames: int = Field(default=0, ge=0)
    vocoder_max_batch_size: int = Field(default=4, ge=1)
    vocoder_max_batch_wait_ms: int = Field(default=2, ge=0)
    vocoder_audio_chunk_size: int | None = Field(default=None, ge=1)
    vocoder_audio_chunk_overlap_size: int | None = Field(default=None, ge=0)
    stages: list[StageConfig] = [
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
            factory_args={"device": "cuda", "max_new_tokens": 1024},
            gpu=0,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory=f"{_PKG}.stages.create_vocoder_executor",
            factory_args={
                "device": "cuda",
                "streaming": True,
                "audio_chunk_size": 16,
                "audio_chunk_overlap_size": 16,
            },
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]

    def model_post_init(self, __context: object = None) -> None:
        super().model_post_init(__context)
        colocated_fractions = {
            "audio_encoder": 0.01,
            "tts_engine": 0.90,
            "vocoder": 0.09,
        }
        for stage in self.stages:
            if (
                self.separate_vocoder_process
                and stage.name in colocated_fractions
                and stage.runtime.resources.total_gpu_memory_fraction is None
            ):
                stage.runtime.resources.total_gpu_memory_fraction = colocated_fractions[
                    stage.name
                ]
            if stage.name == "preprocessing":
                stage.factory_args = {
                    **stage.factory_args,
                    "max_concurrency": self.preprocessing_max_concurrency,
                }
                if self.lora_voices:
                    stage.factory_args["lora_voices"] = dict(self.lora_voices)
            if stage.name == "tts_engine":
                stage.factory_args = {
                    **stage.factory_args,
                    "max_history_chunks": self.max_history_chunks,
                    "serve_model_name": self.name,
                }
                if self.lora_voices or self.enable_dynamic_lora:
                    stage.factory_args["lora_voices"] = dict(self.lora_voices)
                    stage.factory_args["lora_backend"] = self.lora_backend
                    stage.factory_args["lora_max_rank"] = self.lora_max_rank
                    stage.factory_args["enable_dynamic_lora"] = self.enable_dynamic_lora
                    stage.factory_args["lora_max_cached_adapters"] = (
                        self.lora_max_cached_adapters
                    )
            if stage.name == "vocoder":
                if self.separate_vocoder_process:
                    stage.process = "vocoder"
                stage.factory_args = {
                    **stage.factory_args,
                    "max_batch_size": self.vocoder_max_batch_size,
                    "max_batch_wait_ms": self.vocoder_max_batch_wait_ms,
                }
                if self.vocoder_audio_chunk_size is not None:
                    stage.factory_args["audio_chunk_size"] = (
                        self.vocoder_audio_chunk_size
                    )
                if self.vocoder_audio_chunk_overlap_size is not None:
                    stage.factory_args["audio_chunk_overlap_size"] = (
                        self.vocoder_audio_chunk_overlap_size
                    )
                if self.vocoder_full_context_streaming:
                    stage.factory_args.update(
                        full_context_streaming=True,
                        context_frames=self.vocoder_context_frames,
                        startup_reduced_context_frames=(
                            self.vocoder_startup_reduced_context_frames
                        ),
                        startup_reduced_left_context_frames=(
                            self.vocoder_startup_reduced_left_context_frames
                        ),
                        startup_reduced_context_until_frames=(
                            self.vocoder_startup_reduced_context_until_frames
                        ),
                        startup_full_chunk_frames=self.startup_full_chunk_frames,
                        startup_full_chunk_count=self.startup_full_chunk_count,
                        startup_masked_delay_rows=(
                            self.vocoder_startup_masked_delay_rows
                        ),
                        startup_masked_emit_frames=(
                            self.vocoder_startup_masked_emit_frames
                        ),
                        startup_masked_until_frames=(
                            self.vocoder_startup_masked_until_frames
                        ),
                    )

    def create_generate_orchestrator(self):
        """The chunking middleware the launcher plugs into the shared Client."""
        from sglang_omni.models.higgs_tts.text.chunked_generate import (
            HiggsChunkedGenerate,
        )
        from sglang_omni.models.higgs_tts.text.chunker import HiggsTextChunker

        options = self._chunker_options()
        return HiggsChunkedGenerate(
            text_chunker=HiggsTextChunker(options),
            chunker_factory=HiggsTextChunker,
            chunker_options=options,
        )

    def _chunker_options(self) -> "ChunkerOptions":
        """Chunker knobs from the top-level config fields."""
        from sglang_omni.models.higgs_tts.audio import HiggsAudioCodec
        from sglang_omni.models.higgs_tts.text.chunker import ChunkerOptions

        return ChunkerOptions(
            max_seconds=self.chunker_max_seconds,
            cps=self.chunker_cps,
            codec_frame_rate=float(HiggsAudioCodec.FRAME_RATE),
        )


EntryClass = HiggsTtsPipelineConfig
