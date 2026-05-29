# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Higgs TTS text chunker (batch + streaming)."""

from __future__ import annotations

import pytest

from sglang_omni.models.higgs_tts.text_chunker import (
    ChunkerOptions,
    HiggsTextChunker,
    estimate_seconds,
)

# Public batch entry point. ``chunk`` is stateless for batch input, so one
# shared instance backs all the module-level batch tests.
_OPTS = ChunkerOptions(max_seconds=8.0, cps=10.0)
chunk = HiggsTextChunker(_OPTS).chunk


# ---------------------------------------------------------------------------
# Batch chunking
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty_list() -> None:
    assert chunk("") == []
    assert chunk("   \n\t   ") == []


def test_single_short_sentence_one_chunk_fast_path() -> None:
    # When the whole input fits in one budget, we must return exactly 1 chunk
    # so callers (Client.generate orchestrator) fast-path to single-shot.
    assert chunk("Hello world.") == ["Hello world."]


def test_ascii_period_inside_decimal_does_not_split() -> None:
    # "3.14" must stay intact — naive STerm splitting would cut "3."/"14".
    assert chunk("Pi is 3.14 approximately.") == ["Pi is 3.14 approximately."]


def test_ascii_period_inside_abbreviation_does_not_split() -> None:
    # "U.S.A" should remain intact (no trailing whitespace after the periods).
    assert chunk("I live in U.S.A.") == ["I live in U.S.A."]


def test_multiple_ascii_sentences_split_at_whitespace() -> None:
    # ASCII boundary requires trailing whitespace OR end-of-input.
    out = chunk("Hello world. This is a test. Goodbye.")
    assert len(out) == 3
    assert out[0] == "Hello world."
    assert out[1] == " This is a test."  # leading ws preserved
    assert out[2] == " Goodbye."


def test_cjk_terminators_cut_immediately() -> None:
    # Non-ASCII STerm characters (。！？) cut without needing trailing ws.
    out = chunk("你好。再见！世界？")
    assert out == ["你好。", "再见！", "世界？"]


def test_arabic_question_mark_cuts() -> None:
    out = chunk("مرحبا. كيف حالك؟ سلام.")
    assert len(out) == 3


def test_rejoin_invariant_for_clean_input() -> None:
    src = "Hello world. This is a test. Goodbye."
    assert "".join(chunk(src, cps=20)) == src


def test_oversized_sentence_drops_through_tier_refine() -> None:
    # 600 'a' chars at 10 cps = 60s ≫ 8s budget. With no sentence terminator,
    # no clause, no whitespace, only tier-5 per-character refine can shrink it.
    big = "a" * 600
    out = chunk(big, cps=10)
    assert len(out) > 1
    assert all(estimate_seconds(c, 10) <= 8.0 + 1e-6 for c in out)
    # Tier-5 preserves every character; concatenation reconstructs the input.
    assert "".join(out) == big


def test_oversized_sentence_uses_whitespace_tier() -> None:
    # Single long sentence with whitespace — tier-4 word splits cap each piece.
    out = chunk(("word " * 300).strip() + ".", cps=10)
    assert len(out) > 1
    assert all(estimate_seconds(c, 10) <= 8.0 + 1e-6 for c in out)


def test_adjacent_sentences_never_merged_even_if_they_fit() -> None:
    # Two tiny sentences could fit together under budget, but we keep them
    # separate to preserve prosody boundaries.
    out = chunk("Hi. Bye.", cps=10)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# estimate_seconds
# ---------------------------------------------------------------------------


def test_estimate_seconds_excludes_whitespace() -> None:
    # "Hello world" is 10 non-ws chars; at 10 cps = 1.0 s.
    assert estimate_seconds("Hello world", cps=10) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# HiggsTextChunker streaming wrapper
# ---------------------------------------------------------------------------


def test_streaming_buffers_until_confirmed() -> None:
    c = HiggsTextChunker(_OPTS)
    # No terminator yet → buffered.
    assert c.add_text("Hello world.") == []
    # Trailing whitespace confirms the ASCII boundary.
    assert c.add_text(" ") == ["Hello world."]
    # Nothing left to drain.
    assert c.flush() == []


def test_streaming_cjk_cuts_immediately() -> None:
    c = HiggsTextChunker(_OPTS)
    assert c.add_text("你好世界") == []  # no terminator yet
    assert c.add_text("。再见") == ["你好世界。"]
    # "再见" has no terminator — flush drains it.
    assert c.flush() == ["再见"]


def test_streaming_holds_ascii_period_until_whitespace() -> None:
    c = HiggsTextChunker(_OPTS)
    got: list[str] = []
    # "3.14" arrives — must NOT cut at the embedded period.
    got += c.add_text("Pi is 3.14")
    got += c.add_text(" approximately. ")
    assert got == ["Pi is 3.14 approximately."]


def test_streaming_two_in_one_add() -> None:
    c = HiggsTextChunker(_OPTS)
    out = c.add_text("First sentence. Second sentence. ")
    # Leading whitespace on second chunk is preserved (matches batch chunker).
    assert out == ["First sentence.", " Second sentence."]


def test_streaming_empty_add_is_noop() -> None:
    c = HiggsTextChunker(_OPTS)
    assert c.add_text("") == []
    assert c.flush() == []


def test_streaming_flush_drains_remaining_buffer() -> None:
    c = HiggsTextChunker(_OPTS)
    c.add_text("Incomplete sentence with no terminator")
    out = c.flush()
    assert out == ["Incomplete sentence with no terminator"]
    # Subsequent flush returns nothing (buffer cleared).
    assert c.flush() == []


def test_fastout_releases_first_clause_then_sentences() -> None:
    # First chunk cuts at the earliest clause boundary (CJK comma); every later
    # chunk uses sentence boundaries, so the comma in "三，四。" does NOT split.
    c = HiggsTextChunker(
        ChunkerOptions(max_seconds=8.0, cps=10.0, fastout=True)
    )
    assert c.add_text("一，二。三，四。") == ["一，", "二。", "三，四。"]
    assert c.flush() == []


def test_fastout_only_affects_the_first_chunk() -> None:
    c = HiggsTextChunker(
        ChunkerOptions(max_seconds=8.0, cps=10.0, fastout=True)
    )
    # Earliest clause boundary releases chunk 1 right away.
    assert c.add_text("第一句，") == ["第一句，"]
    # Now in sentence mode: a clause comma alone is NOT a cut point.
    assert c.add_text("还有逗号，继续") == []
    # A sentence terminator releases the buffered remainder as one chunk.
    assert c.add_text("结束。") == ["还有逗号，继续结束。"]


def test_default_no_fastout_is_sentence_only() -> None:
    # Without the flag, a leading clause comma must NOT release early.
    c = HiggsTextChunker(ChunkerOptions(max_seconds=8.0, cps=10.0))
    assert c.add_text("一，二") == []
    assert c.add_text("。") == ["一，二。"]


def test_streaming_options_max_seconds_respected() -> None:
    # Custom small budget should force tier-refine on oversized sentences.
    c = HiggsTextChunker(ChunkerOptions(max_seconds=2.0, cps=10.0))
    out = c.chunk("a" * 100)
    # 100 chars / 10 cps = 10s, budget 2s → at least 5 chunks.
    assert len(out) >= 5


# ---------------------------------------------------------------------------
# Higgs config override wires the chunker through
# ---------------------------------------------------------------------------


def test_higgs_config_chunker_options() -> None:
    from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig

    opts = HiggsTtsPipelineConfig(model_path="m")._chunker_options()
    assert opts.codec_frame_rate == 25.0
    assert opts.max_seconds == 8.0  # config-default budget
    assert opts.cps == 10.0  # config-default fallback CPS
    assert isinstance(HiggsTextChunker(opts), HiggsTextChunker)
    # overrides flow through
    overridden = HiggsTtsPipelineConfig(
        model_path="m", chunker_max_seconds=15, chunker_cps=6
    )._chunker_options()
    assert (overridden.max_seconds, overridden.cps) == (15.0, 6.0)
