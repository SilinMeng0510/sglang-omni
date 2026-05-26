# SPDX-License-Identifier: Apache-2.0
"""Streaming text-input strategy interface for ``/v1/audio/speech/stream``.

Different TTS models want different handling of incrementally-arriving text:
some batch it into TTS-ready sentences, others synthesize each fragment the
moment it arrives. The strategy is **owned by the model**: a model's
``PipelineConfig`` implements
:meth:`~sglang_omni.config.PipelineConfig.create_streaming_text_splitter` and
the launcher injects it into the serve app, so the WebSocket handler never
branches on the model name.

This module holds only the model-neutral pieces: the :class:`StreamingTextSplitter`
protocol, a :class:`PassthroughTextSplitter` default (token-in → straight to the
engine), and :class:`StreamingTextOptions` (the raw inputs a factory may use).
Model-specific strategies (e.g. Higgs TTS's
:mod:`sglang_omni.models.higgs_tts.text_chunker`) live with their model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class StreamingTextSplitter(Protocol):
    """Turns incrementally-fed text into dispatch-ready pieces.

    ``add_text`` returns the pieces ready to synthesize given everything seen so
    far (possibly empty); ``flush`` drains whatever remains at end-of-input.
    """

    def add_text(self, text: str) -> list[str]: ...

    def flush(self) -> list[str]: ...


class PassthroughTextSplitter:
    """Emit each fragment immediately, unmodified.

    The default strategy: for models that synthesize incrementally and need no
    sentence batching, a fragment goes straight to the engine as it arrives, and
    there is nothing to drain on ``flush``.
    """

    def add_text(self, text: str) -> list[str]:
        return [text] if text else []

    def flush(self) -> list[str]:
        return []


@dataclass(frozen=True)
class StreamingTextOptions:
    """Model-neutral inputs handed to a model's splitter factory.

    A factory uses whichever fields it needs. Extend this with defaults so
    existing factories keep working unchanged.
    """

    split_granularity: str = "sentence"
    max_buffer_chars: int | None = None


__all__ = [
    "PassthroughTextSplitter",
    "StreamingTextOptions",
    "StreamingTextSplitter",
]
