# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the model-neutral streaming text-input interface."""

from __future__ import annotations

from sglang_omni.utils.streaming_text import (
    PassthroughTextSplitter,
    StreamingTextOptions,
    StreamingTextSplitter,
)


def test_passthrough_emits_each_fragment_immediately():
    splitter = PassthroughTextSplitter()
    assert splitter.add_text("Hello world. How ") == ["Hello world. How "]
    assert splitter.add_text("are you?") == ["are you?"]
    assert splitter.flush() == []


def test_passthrough_ignores_empty_fragments():
    assert PassthroughTextSplitter().add_text("") == []


def test_passthrough_satisfies_protocol():
    assert isinstance(PassthroughTextSplitter(), StreamingTextSplitter)


def test_options_defaults():
    options = StreamingTextOptions()
    assert options.split_granularity == "sentence"
    assert options.max_buffer_chars is None
