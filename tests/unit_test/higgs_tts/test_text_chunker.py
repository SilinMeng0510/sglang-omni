# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the streaming/batch text chunker."""

from __future__ import annotations

import pytest

from sglang_omni.models.higgs_tts.text_chunker import (
    StreamingTextChunker,
    chunk_text,
    estimate_seconds,
)


def _run(fragments, granularity="sentence", **kwargs):
    chunker = StreamingTextChunker(granularity, **kwargs)
    out: list[str] = []
    for fragment in fragments:
        out.extend(chunker.add_text(fragment))
    out.extend(chunker.flush())
    return out


# ---------------------------------------------------------------------------
# Backward-compatible parity with the previous _SpeechTextSplitter behaviour
# ---------------------------------------------------------------------------
def test_english_sentence_split_on_whitespace() -> None:
    assert _run(["Hello world. How are you?"]) == ["Hello world.", "How are you?"]


def test_ascii_period_without_whitespace_is_not_a_boundary() -> None:
    # vLLM-Omni rule: "Hello.World" must stay a single chunk.
    assert _run(["Hello.World"]) == ["Hello.World"]


def test_clause_mode_splits_fullwidth_only() -> None:
    # ASCII comma does not split; fullwidth ；does.
    assert _run(["alpha, beta；gamma"], "clause") == ["alpha, beta；", "gamma"]


# ---------------------------------------------------------------------------
# Incremental streaming
# ---------------------------------------------------------------------------
def test_fragmented_input_reassembles_sentences() -> None:
    assert _run(["Hel", "lo wor", "ld. How ", "are you?"]) == [
        "Hello world.",
        "How are you?",
    ]


def test_period_held_until_following_whitespace_arrives() -> None:
    chunker = StreamingTextChunker()
    # A trailing period is ambiguous until we see what follows it.
    assert chunker.add_text("The value is 3.") == []
    assert chunker.add_text("14 exactly.") == []
    assert chunker.flush() == ["The value is 3.14 exactly."]


def test_emits_completed_sentence_before_input_done() -> None:
    chunker = StreamingTextChunker()
    assert chunker.add_text("First sentence. Second") == ["First sentence."]
    assert chunker.flush() == ["Second"]


# ---------------------------------------------------------------------------
# Language-agnostic sentence detection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("你好世界。这是测试！", ["你好世界。", "这是测试！"]),
        ("مرحبا بالعالم؟ كيف حالك", ["مرحبا بالعالم؟", "كيف حالك"]),
        ("नमस्ते दुनिया। यह परीक्षण है।", ["नमस्ते दुनिया।", "यह परीक्षण है।"]),
    ],
)
def test_multiscript_boundaries(text, expected) -> None:
    assert _run([text]) == expected


# ---------------------------------------------------------------------------
# Time-budget sub-splitting of oversized sentences
# ---------------------------------------------------------------------------
def test_oversized_sentence_is_subsplit_under_budget() -> None:
    # 60 words, no internal sentence punctuation, 2s budget at 10 cps (=20 chars).
    chunks = chunk_text("word " * 60, max_seconds=2.0, cps=10.0)
    assert len(chunks) > 1
    for chunk in chunks:
        assert estimate_seconds(chunk, 10.0) <= 2.0 + 1e-9


def test_short_sentence_is_not_subsplit() -> None:
    assert chunk_text("Hello world.", max_seconds=8.0, cps=10.0) == ["Hello world."]


# ---------------------------------------------------------------------------
# Tag handling
# ---------------------------------------------------------------------------
def test_state_tag_prefixes_following_chunks() -> None:
    assert _run(["<|emotion:joy|>Hello there. World now."]) == [
        "<|emotion:joy|>Hello there.",
        "<|emotion:joy|>World now.",
    ]


def test_clear_tag_resets_state() -> None:
    assert _run(["<|emotion:joy|>Hi. <|clear|>Bye now."]) == [
        "<|emotion:joy|>Hi.",
        "Bye now.",
    ]


def test_state_tag_split_across_fragments() -> None:
    assert _run(["<|emo", "tion:sad|>Sad text here."]) == [
        "<|emotion:sad|>Sad text here."
    ]


def test_transient_tag_stays_inline() -> None:
    assert _run(["Hi there <|prosody:pause|> friend. Bye."]) == [
        "Hi there <|prosody:pause|> friend.",
        "Bye.",
    ]


def test_state_carries_across_add_text_calls() -> None:
    chunker = StreamingTextChunker()
    assert chunker.add_text("<|emotion:joy|>First. ") == ["<|emotion:joy|>First."]
    assert chunker.add_text("Second. ") == ["<|emotion:joy|>Second."]


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def test_empty_and_whitespace_input() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []
    assert _run(["   "]) == []


def test_buffer_limit_enforced() -> None:
    chunker = StreamingTextChunker(max_buffer_chars=16)
    with pytest.raises(ValueError, match="maximum size"):
        chunker.add_text("no terminator here keeps growing")


# ---------------------------------------------------------------------------
# Model-owned strategy: HiggsTtsPipelineConfig declares the chunker
# ---------------------------------------------------------------------------
def test_higgs_config_creates_sentence_chunker() -> None:
    from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig
    from sglang_omni.utils.streaming_text import (
        StreamingTextOptions,
        StreamingTextSplitter,
    )

    config = HiggsTtsPipelineConfig(model_path="dummy/path")
    splitter = config.create_streaming_text_splitter(StreamingTextOptions())
    assert isinstance(splitter, StreamingTextSplitter)
    assert isinstance(splitter, StreamingTextChunker)
    assert splitter.add_text("Hello world. How are you?") + splitter.flush() == [
        "Hello world.",
        "How are you?",
    ]


def test_base_pipeline_config_defaults_to_passthrough() -> None:
    from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig
    from sglang_omni.config import PipelineConfig
    from sglang_omni.utils.streaming_text import (
        PassthroughTextSplitter,
        StreamingTextOptions,
    )

    # The base hook (inherited, not overridden) returns pass-through.
    splitter = PipelineConfig.create_streaming_text_splitter(
        HiggsTtsPipelineConfig(model_path="dummy/path"), StreamingTextOptions()
    )
    assert isinstance(splitter, PassthroughTextSplitter)
