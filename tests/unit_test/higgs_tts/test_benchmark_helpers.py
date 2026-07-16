from pathlib import Path

import pytest

from benchmarks.higgs_tts.common import (
    BYTES_PER_SECOND,
    continuity,
    percentile,
    speech_payload,
    stats,
)
from benchmarks.higgs_tts.compare_audio import normalized_words, word_errors
from benchmarks.higgs_tts.gallery_server import RangeRequestHandler
from benchmarks.higgs_tts.performance import load_prompts

HERE = Path(__file__).resolve().parents[3] / "benchmarks" / "higgs_tts"


def test_percentile_and_stats() -> None:
    assert percentile([1, 2, 3], 50) == 2
    assert stats([1, 2, 3])["p85"] == pytest.approx(2.7)
    assert stats([])["mean"] is None


def test_continuity_detects_underrun() -> None:
    chunks = [
        (10.0, int(BYTES_PER_SECOND * 0.1)),
        (10.25, int(BYTES_PER_SECOND * 0.2)),
    ]

    jitter, speed = continuity(chunks)

    assert jitter == pytest.approx(0.15)
    assert speed == pytest.approx(0.8)


def test_audio_comparison_word_errors_normalize_case_and_punctuation() -> None:
    assert normalized_words("HTTPS, ParseJSON!") == ["https", "parsejson"]
    assert word_errors("That sounds good.", "that sounds good") == (0, 3)
    assert word_errors("one two three", "one four three extra") == (2, 3)


def test_bundled_prompts_are_valid() -> None:
    prompts = load_prompts(HERE / "sharegpt_10k.jsonl", 1)

    assert len(prompts) == 10_000
    assert all(isinstance(value, str) and value for value in prompts)


def test_ab_sample_texts_are_valid() -> None:
    samples = [
        line.strip()
        for line in (HERE / "sample_text.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(samples) >= 10


def test_performance_payload_supports_dynamic_lora() -> None:
    payload = speech_payload(
        "hello",
        model="higgs-tts",
        voice="default",
        seed=1,
        lora_adapter_path="/models/ap2/adapter",
    )

    assert payload["lora_adapter"] == {"path": "/models/ap2/adapter"}


@pytest.mark.parametrize(
    ("header", "size", "expected"),
    [
        (None, 100, None),
        ("bytes=0-9", 100, (0, 9)),
        ("bytes=90-", 100, (90, 99)),
        ("bytes=-10", 100, (90, 99)),
        ("bytes=0-999", 100, (0, 99)),
        ("bytes=100-", 100, False),
        ("bytes=20-10", 100, False),
        ("items=0-9", 100, False),
    ],
)
def test_gallery_server_parses_byte_ranges(
    header: str | None,
    size: int,
    expected: tuple[int, int] | None | bool,
) -> None:
    assert RangeRequestHandler._parse_range(header, size) == expected
