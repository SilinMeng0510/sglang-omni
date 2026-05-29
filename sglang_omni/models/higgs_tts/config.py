# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Higgs TTS (V1)."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from pydantic import Field

from sglang_omni.config import PipelineConfig, StageConfig

if TYPE_CHECKING:
    from sglang_omni.models.higgs_tts.text_chunker import ChunkerOptions

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
    chunker_max_seconds: float | None = Field(default=None, gt=0)
    max_history_chunks: int = Field(default=4, ge=0)
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
            factory_args={"device": "cuda", "max_new_tokens": 2048},
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
        for stage in self.stages:
            if stage.name == "tts_engine":
                stage.factory_args = {
                    **stage.factory_args,
                    "max_history_chunks": self.max_history_chunks,
                }
                break

    def create_generate_orchestrator(self):
        """The chunking middleware the launcher plugs into the shared Client."""
        from sglang_omni.models.higgs_tts.chunked_generate import HiggsChunkedGenerate
        from sglang_omni.models.higgs_tts.text_chunker import HiggsTextChunker

        options = self._chunker_options()
        return HiggsChunkedGenerate(
            text_chunker=HiggsTextChunker(options),
            chunker_factory=HiggsTextChunker,
            chunker_options=options,
        )

    def _chunker_options(self) -> "ChunkerOptions":
        """Codec frame rate + the optional ``chunker_max_seconds`` override."""
        from sglang_omni.models.higgs_tts.audio_codec import HiggsAudioCodec
        from sglang_omni.models.higgs_tts.text_chunker import ChunkerOptions

        kwargs: dict = {"codec_frame_rate": float(HiggsAudioCodec.FRAME_RATE)}
        if self.chunker_max_seconds is not None:
            kwargs["max_seconds"] = self.chunker_max_seconds
        return ChunkerOptions(**kwargs)


EntryClass = HiggsTtsPipelineConfig
