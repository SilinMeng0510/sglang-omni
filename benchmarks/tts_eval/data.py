# SPDX-License-Identifier: Apache-2.0
"""Dataset adapters for the multilingual TTS quality benchmarks."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_ROOT = Path("/ceph/data/higgs_audio_eval/zero_shot_tts")

# Exact language sets used by the joint Boson AI / SGLang-Omni reproduction.
# See https://www.lmsys.org/blog/2026-06-04-higgs-audio-v3-tts/
BLOG_LANGUAGES: dict[str, tuple[str, ...]] = {
    "seedtts": ("en", "zh"),
    "cv3": ("en", "ko", "ja", "fr", "de", "es", "it", "ru", "zh"),
    "minimax": (
        "ar",
        "zh",
        "cs",
        "nl",
        "en",
        "fi",
        "fr",
        "de",
        "el",
        "hi",
        "id",
        "it",
        "ja",
        "ko",
        "pl",
        "pt",
        "ro",
        "ru",
        "es",
        "th",
        "tr",
        "uk",
        "vi",
    ),
}

_DATASET_DIRS = {
    "seedtts": "seed_tts",
    "cv3": "multi_lingual_tts",
    "minimax": "minimax_multi_lingual_tts",
}


@dataclass(frozen=True)
class TTSEvalSample:
    benchmark: str
    lang: str
    sample_id: str
    ref_text: str
    ref_audio: str
    target_text: str

    @property
    def key(self) -> str:
        return f"{self.benchmark}/{self.lang}/{self.sample_id}"


def _parse_seedtts(meta_path: Path, lang: str) -> Iterable[TTSEvalSample]:
    with meta_path.open(encoding="utf-8") as handle:
        for line_index, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("|") if "|" in line else line.split("\t")
            if len(parts) < 4:
                raise ValueError(
                    f"{meta_path}:{line_index + 1}: expected at least 4 fields"
                )
            sample_id, ref_text, ref_audio, target_text = (
                value.strip() for value in parts[:4]
            )
            yield TTSEvalSample(
                benchmark="seedtts",
                lang=lang,
                sample_id=sample_id,
                ref_text=ref_text,
                ref_audio=str((meta_path.parent / ref_audio).resolve()),
                target_text=target_text,
            )


def _parse_jsonl(
    metadata_path: Path,
    audio_root: Path,
    benchmark: str,
    lang: str,
) -> Iterable[TTSEvalSample]:
    with metadata_path.open(encoding="utf-8") as handle:
        for line_index, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = str(
                row.get("misc", {}).get("uttid") or f"{lang}_{line_index:05d}"
            )
            ref_audio = Path(str(row["prompt_audio_path"]))
            if not ref_audio.is_absolute():
                ref_audio = audio_root / ref_audio
            yield TTSEvalSample(
                benchmark=benchmark,
                lang=lang,
                sample_id=sample_id,
                ref_text=str(row["prompt_text"]),
                ref_audio=str(ref_audio.resolve()),
                target_text=str(row["text_to_synthesize"]),
            )


def _language_samples(
    data_root: Path,
    benchmark: str,
    lang: str,
) -> list[TTSEvalSample]:
    dataset_root = data_root / _DATASET_DIRS[benchmark]
    if benchmark == "seedtts":
        metadata_path = dataset_root / lang / "meta.lst"
        audio_root = metadata_path.parent
        parser = _parse_seedtts(metadata_path, lang)
    else:
        metadata_path = dataset_root / f"{lang}.jsonl"
        audio_root = dataset_root / lang
        parser = _parse_jsonl(metadata_path, audio_root, benchmark, lang)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing {benchmark}/{lang} metadata: {metadata_path}")
    samples = list(parser)
    if not samples:
        raise ValueError(f"No samples found in {metadata_path}")
    missing = [
        sample.ref_audio for sample in samples if not Path(sample.ref_audio).is_file()
    ]
    if missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(
            f"{benchmark}/{lang}: {len(missing)} reference audio files are missing: "
            f"{preview}"
        )
    return samples


def load_benchmark_samples(
    benchmark: str,
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    languages: Iterable[str] | None = None,
    max_samples_per_language: int | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
) -> list[TTSEvalSample]:
    """Load one benchmark with deterministic global sharding.

    ``languages=None`` selects the exact 2/9/23-language blog reproduction.
    Passing languages explicitly permits newer or extended dataset revisions.
    """
    benchmark = benchmark.lower()
    if benchmark not in BLOG_LANGUAGES:
        raise ValueError(
            f"Unknown benchmark {benchmark!r}; choose from {sorted(BLOG_LANGUAGES)}"
        )
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError(
            f"Expected 0 <= shard_index < num_shards, got {shard_index}/{num_shards}"
        )
    selected_languages = tuple(languages or BLOG_LANGUAGES[benchmark])
    if not selected_languages:
        raise ValueError("At least one language is required")

    samples: list[TTSEvalSample] = []
    for lang in selected_languages:
        language_samples = _language_samples(Path(data_root), benchmark, lang)
        if max_samples_per_language is not None:
            if max_samples_per_language < 0:
                raise ValueError("max_samples_per_language must be non-negative")
            language_samples = language_samples[:max_samples_per_language]
        samples.extend(language_samples)
    return [
        sample
        for global_index, sample in enumerate(samples)
        if global_index % num_shards == shard_index
    ]
