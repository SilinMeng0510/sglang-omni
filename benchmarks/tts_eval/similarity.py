# SPDX-License-Identifier: Apache-2.0
"""Offline WavLM speaker similarity for generated TTS benchmark manifests."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from benchmarks.metrics.speaker_similarity import WavLMSpeakerSimilarity


def _load_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = str(row["key"])
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
    return rows


def _resolve(path: str, replacements: list[tuple[str, str]]) -> str:
    candidate = Path(path)
    if candidate.is_file():
        return str(candidate)
    for old, new in replacements:
        if path.startswith(old):
            replaced = Path(new + path[len(old) :])
            if replaced.is_file():
                return str(replaced)
    return str(candidate)


def _chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def score(
    *,
    manifests: list[Path],
    output: Path,
    finetune_checkpoint: Path,
    wavlm_base: Path,
    device: str,
    batch_size: int,
    replacements: list[tuple[str, str]],
) -> dict[str, Any]:
    rows = _load_rows(manifests)
    scoreable: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in rows:
        ref_audio = _resolve(str(row["ref_audio"]), replacements)
        wav_path = _resolve(str(row["wav_path"]), replacements)
        if (
            row.get("is_success")
            and Path(ref_audio).is_file()
            and Path(wav_path).is_file()
        ):
            scoreable.append({**row, "ref_audio": ref_audio, "wav_path": wav_path})
        else:
            skipped.append(
                {
                    "key": row["key"],
                    "lang": row["lang"],
                    "speaker_similarity": None,
                    "error": "generation failed or audio path is missing",
                }
            )
    if not scoreable:
        raise RuntimeError("no scoreable generated audio found")

    scorer = WavLMSpeakerSimilarity(
        finetune_checkpoint=finetune_checkpoint,
        wavlm_base=wavlm_base,
        device=device,
    )

    # Reference prompts repeat heavily in multilingual benchmarks. Embed every
    # distinct prompt only once, then batch only generated audio.
    unique_refs = list(dict.fromkeys(str(row["ref_audio"]) for row in scoreable))
    ref_embeddings: dict[str, torch.Tensor] = {}
    for batch in _chunks(unique_refs, batch_size):
        embedded = scorer.embed(batch).detach().cpu()
        ref_embeddings.update(zip(batch, embedded))

    scored: list[dict[str, Any]] = []
    for batch in _chunks(scoreable, batch_size):
        generated = scorer.embed([str(row["wav_path"]) for row in batch])
        references = torch.stack(
            [ref_embeddings[str(row["ref_audio"])] for row in batch]
        ).to(generated.device)
        values = (F.cosine_similarity(references, generated, dim=-1) * 100.0).cpu()
        for row, value in zip(batch, values.tolist()):
            scored.append(
                {
                    "key": row["key"],
                    "benchmark": row["benchmark"],
                    "lang": row["lang"],
                    "sample_id": row["sample_id"],
                    "ref_audio": row["ref_audio"],
                    "wav_path": row["wav_path"],
                    "speaker_similarity": float(value),
                    "error": None,
                }
            )

    by_language: dict[str, list[float]] = defaultdict(list)
    for row in scored:
        by_language[str(row["lang"])].append(float(row["speaker_similarity"]))
    language_means = {
        lang: statistics.mean(values) for lang, values in sorted(by_language.items())
    }
    summary = {
        "benchmark": scoreable[0]["benchmark"],
        "total": len(rows),
        "evaluated": len(scored),
        "skipped": len(skipped),
        "speaker_similarity_mean": statistics.mean(
            float(row["speaker_similarity"]) for row in scored
        ),
        "speaker_similarity_language_macro_mean": statistics.mean(
            language_means.values()
        ),
        "speaker_similarity_by_language": language_means,
        "finetune_checkpoint": str(finetune_checkpoint),
        "wavlm_base": str(wavlm_base),
        "manifests": [str(path) for path in manifests],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"summary": summary, "per_sample": scored + skipped}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finetune-checkpoint", type=Path, required=True)
    parser.add_argument("--wavlm-base", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--replace-prefix",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Rewrite container-side paths from generation manifests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replacements = []
    for value in args.replace_prefix:
        old, separator, new = value.partition("=")
        if not separator:
            raise ValueError(f"expected OLD=NEW, got {value!r}")
        replacements.append((old, new))
    summary = score(
        manifests=args.manifest,
        output=args.output,
        finetune_checkpoint=args.finetune_checkpoint,
        wavlm_base=args.wavlm_base,
        device=args.device,
        batch_size=args.batch_size,
        replacements=replacements,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
