# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``HiggsTokenizerAdapter.build_prompt(history=...)``.

The interleaved prompt layout (cross-chunk continuity) is pure-Python and
doesn't need the real Higgs tokenizer — a deterministic fake suffices for
checking the token emission order and ``-100`` placeholder counts.
"""

from __future__ import annotations

import pytest

from sglang_omni.models.higgs_tts.text_tokenizer import (
    AUDIO_PLACEHOLDER_ID,
    HiggsTokenizerAdapter,
)


# Distinct, easy-to-spot specials for assertion readability.
_SPECIALS = {
    "<|tts|>": 100,
    "<|ref_audio|>": 101,
    "<|text|>": 102,
    "<|audio|>": 103,
    "<|ref_text|>": 104,
}


class _FakeTokenizer:
    """Minimal tokenizer matching the adapter contract.

    ``encode`` maps printable chars to ``1000 + ord(c)``; whitespace is
    skipped so test inputs stay readable but assertions are deterministic.
    """

    def get_added_vocab(self) -> dict[str, int]:
        return dict(_SPECIALS)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [1000 + ord(c) for c in text if not c.isspace()]


@pytest.fixture
def adapter() -> HiggsTokenizerAdapter:
    return HiggsTokenizerAdapter(_FakeTokenizer())


# ---------------------------------------------------------------------------
# Backward-compat: history=None / empty == today's behaviour
# ---------------------------------------------------------------------------


def test_zero_shot_no_history(adapter: HiggsTokenizerAdapter) -> None:
    ids = adapter.build_prompt("AB", num_ref_tokens=0)
    assert ids == [
        _SPECIALS["<|tts|>"],
        _SPECIALS["<|text|>"], 1065, 1066,
        _SPECIALS["<|audio|>"],
    ]


def test_ref_only_no_history(adapter: HiggsTokenizerAdapter) -> None:
    ids = adapter.build_prompt(
        "AB", num_ref_tokens=3, reference_text="X"
    )
    P = AUDIO_PLACEHOLDER_ID
    assert ids == [
        _SPECIALS["<|tts|>"],
        _SPECIALS["<|ref_text|>"], 1088,
        _SPECIALS["<|ref_audio|>"], P, P, P,
        _SPECIALS["<|text|>"], 1065, 1066,
        _SPECIALS["<|audio|>"],
    ]


def test_empty_history_list_matches_no_history(adapter: HiggsTokenizerAdapter) -> None:
    no_hist = adapter.build_prompt("Z", num_ref_tokens=2, history=None)
    empty_hist = adapter.build_prompt("Z", num_ref_tokens=2, history=[])
    assert no_hist == empty_hist


# ---------------------------------------------------------------------------
# History layout
# ---------------------------------------------------------------------------


def test_one_history_segment(adapter: HiggsTokenizerAdapter) -> None:
    hist = [([1077, 1078], 2)]  # "MN"-ish tokens + 2 audio placeholder rows
    ids = adapter.build_prompt(
        "AB", num_ref_tokens=3, reference_text="X", history=hist
    )
    P = AUDIO_PLACEHOLDER_ID
    assert ids == [
        _SPECIALS["<|tts|>"],
        _SPECIALS["<|ref_text|>"], 1088,                # ref_text + tok('X')
        _SPECIALS["<|ref_audio|>"], P, P, P,            # ref_audio + 3 -100s
        _SPECIALS["<|text|>"], 1077, 1078,              # history block
        _SPECIALS["<|audio|>"], P, P,
        _SPECIALS["<|text|>"], 1065, 1066,              # final text(AB)
        _SPECIALS["<|audio|>"],
    ]


def test_many_history_segments(adapter: HiggsTokenizerAdapter) -> None:
    hist = [([1001], 1), ([1002, 1003], 2), ([1004], 3)]
    ids = adapter.build_prompt("Z", num_ref_tokens=2, history=hist)
    P = AUDIO_PLACEHOLDER_ID
    assert ids == [
        _SPECIALS["<|tts|>"],
        _SPECIALS["<|ref_audio|>"], P, P,
        _SPECIALS["<|text|>"], 1001, _SPECIALS["<|audio|>"], P,
        _SPECIALS["<|text|>"], 1002, 1003, _SPECIALS["<|audio|>"], P, P,
        _SPECIALS["<|text|>"], 1004, _SPECIALS["<|audio|>"], P, P, P,
        _SPECIALS["<|text|>"], 1090,
        _SPECIALS["<|audio|>"],
    ]


def test_zero_shot_with_history(adapter: HiggsTokenizerAdapter) -> None:
    # No reference audio but history still threads through — valid layout
    # for the orchestrator's first sub-chunk when there's no voice ref.
    hist = [([1001], 1)]
    ids = adapter.build_prompt("Z", num_ref_tokens=0, history=hist)
    P = AUDIO_PLACEHOLDER_ID
    assert ids == [
        _SPECIALS["<|tts|>"],
        _SPECIALS["<|text|>"], 1001, _SPECIALS["<|audio|>"], P,
        _SPECIALS["<|text|>"], 1090,
        _SPECIALS["<|audio|>"],
    ]


def test_placeholder_count_matches_history_audio_rows(
    adapter: HiggsTokenizerAdapter,
) -> None:
    """Order-based overlay invariant: total ``-100``s must equal the sum of
    ref rows + each history segment's audio row count, in order."""
    hist = [([1001, 1002], 3), ([1003], 2)]
    ids = adapter.build_prompt("Z", num_ref_tokens=4, history=hist)
    assert ids.count(AUDIO_PLACEHOLDER_ID) == 4 + 3 + 2


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_negative_history_audio_rows_rejected(
    adapter: HiggsTokenizerAdapter,
) -> None:
    with pytest.raises(ValueError, match="history audio-row counts"):
        adapter.build_prompt("Z", num_ref_tokens=2, history=[([1], -1)])


def test_negative_num_ref_tokens_rejected(
    adapter: HiggsTokenizerAdapter,
) -> None:
    with pytest.raises(ValueError, match="num_ref_tokens"):
        adapter.build_prompt("Z", num_ref_tokens=-1)
