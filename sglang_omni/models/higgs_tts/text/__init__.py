# SPDX-License-Identifier: Apache-2.0
"""Text side of Higgs TTS: tokenization + prompt assembly, sentence/clause
chunking, CJK punctuation normalization, and the chunked-generate orchestrator.

The chunker / normalizer / tokenizer are dependency-light leaves (safe to import
standalone in unit tests). :class:`HiggsChunkedGenerate` pulls in the full
``Client`` pipeline, so it is exposed lazily (PEP 562) — importing this package,
or any leaf submodule, does not drag in the client stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang_omni.models.higgs_tts.text.chunker import (
    ChunkerOptions,
    HiggsTextChunker,
    TextChunker,
    estimate_seconds,
)
from sglang_omni.models.higgs_tts.text.normalizer import normalize_punctuation
from sglang_omni.models.higgs_tts.text.tokenizer import (
    AUDIO_PLACEHOLDER_ID,
    HiggsTokenizerAdapter,
)

if TYPE_CHECKING:
    from sglang_omni.models.higgs_tts.text.chunked_generate import HiggsChunkedGenerate

__all__ = [
    "AUDIO_PLACEHOLDER_ID",
    "ChunkerOptions",
    "HiggsChunkedGenerate",
    "HiggsTextChunker",
    "HiggsTokenizerAdapter",
    "TextChunker",
    "estimate_seconds",
    "normalize_punctuation",
]


def __getattr__(name: str):
    """Lazily resolve :class:`HiggsChunkedGenerate` (heavy: pulls in ``Client``)."""
    if name == "HiggsChunkedGenerate":
        from sglang_omni.models.higgs_tts.text.chunked_generate import (
            HiggsChunkedGenerate,
        )

        return HiggsChunkedGenerate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
