# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Higgs TTS (V1)."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

if TYPE_CHECKING:
    from sglang_omni.utils.streaming_text import (
        StreamingTextOptions,
        StreamingTextSplitter,
    )

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

    def create_streaming_text_splitter(
        self, options: "StreamingTextOptions"
    ) -> "StreamingTextSplitter":
        """Batch streaming text into TTS-ready sentences for Higgs.

        Uses a fixed conservative time budget (no per-session audio I/O): the
        chars-per-second rate only gates oversized-sentence sub-splitting, so
        precise per-speaker pacing is not worth loading the reference audio.
        """
        from sglang_omni.models.higgs_tts.text_chunker import StreamingTextChunker

        return StreamingTextChunker(
            options.split_granularity,
            max_buffer_chars=options.max_buffer_chars,
        )


EntryClass = HiggsTtsPipelineConfig
