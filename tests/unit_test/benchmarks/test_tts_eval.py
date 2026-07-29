# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from benchmarks.tts_eval.data import BLOG_LANGUAGES, load_benchmark_samples
from benchmarks.tts_eval.metrics import (
    cv3_components,
    score_benchmark,
    seedtts_components,
)
from benchmarks.tts_eval.transcribe import _load_completed


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_blog_language_sets_are_fixed() -> None:
    assert len(BLOG_LANGUAGES["seedtts"]) == 2
    assert len(BLOG_LANGUAGES["cv3"]) == 9
    assert len(BLOG_LANGUAGES["minimax"]) == 23
    assert "yue" not in BLOG_LANGUAGES["minimax"]


def test_seedtts_loader_and_deterministic_sharding(tmp_path: Path) -> None:
    root = tmp_path / "seed_tts" / "en"
    for index in range(3):
        _touch(root / f"{index}.wav")
    (root / "meta.lst").write_text(
        "\n".join(
            f"utt-{index}|reference {index}|{index}.wav|target {index}"
            for index in range(3)
        ),
        encoding="utf-8",
    )

    first = load_benchmark_samples(
        "seedtts",
        data_root=tmp_path,
        languages=["en"],
        shard_index=0,
        num_shards=2,
    )
    second = load_benchmark_samples(
        "seedtts",
        data_root=tmp_path,
        languages=["en"],
        shard_index=1,
        num_shards=2,
    )

    assert [sample.sample_id for sample in first] == ["utt-0", "utt-2"]
    assert [sample.sample_id for sample in second] == ["utt-1"]
    assert {sample.key for sample in first}.isdisjoint(sample.key for sample in second)


def test_jsonl_loader_resolves_audio_under_language_directory(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "multi_lingual_tts"
    _touch(dataset / "fr" / "speaker.wav")
    (dataset / "fr.jsonl").write_text(
        json.dumps(
            {
                "prompt_text": "bonjour",
                "prompt_audio_path": "speaker.wav",
                "text_to_synthesize": "au revoir",
                "misc": {"uttid": "fr-1"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    [sample] = load_benchmark_samples("cv3", data_root=tmp_path, languages=["fr"])

    assert sample.sample_id == "fr-1"
    assert sample.ref_audio == str((dataset / "fr" / "speaker.wav").resolve())


@pytest.mark.parametrize(
    ("scorer", "lang", "reference", "hypothesis", "expected"),
    [
        (seedtts_components, "en", "It's GOOD!", "it's good", (0, 2)),
        (seedtts_components, "zh", "你，好。", "你好", (0, 2)),
        (cv3_components, "ja", "今日は。", "今日は", (0, 3)),
        (cv3_components, "fr", "Bonjour, ami!", "bonjour ami", (0, 2)),
    ],
)
def test_seedtts_and_cv3_normalization(
    scorer, lang: str, reference: str, hypothesis: str, expected: tuple[int, int]
) -> None:
    assert scorer(reference, hypothesis, lang) == expected


def test_score_uses_language_macro_of_per_sample_rates() -> None:
    rows = [
        {
            "key": "seedtts/en/1",
            "lang": "en",
            "target_text": "one two",
            "transcript": "one",
            "is_success": True,
        },
        {
            "key": "seedtts/en/2",
            "lang": "en",
            "target_text": "one",
            "transcript": "one",
            "is_success": True,
        },
        {
            "key": "seedtts/zh/1",
            "lang": "zh",
            "target_text": "你好",
            "transcript": "",
            "is_success": True,
        },
    ]

    result = score_benchmark("seedtts", rows)

    # en mean=(0.5+0)/2=.25, zh mean=1, language macro=.625.
    assert result["wer_cer_macro"] == pytest.approx(0.625)
    assert result["wer_cer_x100"] == pytest.approx(62.5)
    assert result["wer_cer_micro"] == pytest.approx(3 / 5)


def test_transcription_resume_only_skips_successes(tmp_path: Path) -> None:
    path = tmp_path / "transcripts.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "key": "new-success",
                    "is_success": True,
                    "transcription_success": True,
                    "transcript": "ok",
                },
                {
                    "key": "new-failure",
                    "is_success": True,
                    "transcription_success": False,
                    "transcript": "",
                },
                {
                    "key": "legacy-success",
                    "is_success": True,
                    "transcript": "ok",
                },
                {
                    "key": "legacy-failure",
                    "is_success": False,
                    "transcript": None,
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert _load_completed(path) == {"new-success", "legacy-success"}
