#!/usr/bin/env python3
"""Compare matched BF16/FP8 Higgs TTS streaming performance runs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

try:
    from .common import percentile, write_json
except ImportError:
    from common import percentile, write_json


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _paired_p85(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    field = "continuous_playback_start_s"
    baseline = {
        str(row["text"]): float(row[field])
        for row in baseline_rows
        if row.get("success") and row.get(field) is not None
    }
    candidate = {
        str(row["text"]): float(row[field])
        for row in candidate_rows
        if row.get("success") and row.get(field) is not None
    }
    keys = sorted(baseline.keys() & candidate.keys())
    if not keys:
        raise ValueError("runs have no matched successful prompts")
    baseline_values = [baseline[key] for key in keys]
    candidate_values = [candidate[key] for key in keys]
    baseline_p85 = float(percentile(baseline_values, 85))
    candidate_p85 = float(percentile(candidate_values, 85))
    improvement = (baseline_p85 - candidate_p85) / baseline_p85 * 100.0

    rng = random.Random(seed)
    bootstrapped: list[float] = []
    for _ in range(bootstrap_samples):
        indices = [rng.randrange(len(keys)) for _ in keys]
        base = float(percentile([baseline_values[i] for i in indices], 85))
        cand = float(percentile([candidate_values[i] for i in indices], 85))
        if base > 0:
            bootstrapped.append((base - cand) / base * 100.0)
    return {
        "matched_prompts": len(keys),
        "bf16_p85_s": baseline_p85,
        "fp8_p85_s": candidate_p85,
        "improvement_percent": improvement,
        "improvement_bootstrap_95ci_percent": [
            percentile(bootstrapped, 2.5),
            percentile(bootstrapped, 97.5),
        ],
    }


def compare(
    *,
    bf16_dir: Path,
    fp8_dir: Path,
    output: Path,
    concurrencies: list[int],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    levels = []
    for concurrency in concurrencies:
        bf16 = _load(bf16_dir / f"c{concurrency}.json")
        fp8 = _load(fp8_dir / f"c{concurrency}.json")
        paired = _paired_p85(
            bf16["rows"],
            fp8["rows"],
            bootstrap_samples=bootstrap_samples,
            seed=seed + concurrency,
        )
        levels.append(
            {
                "concurrency": concurrency,
                **paired,
                "bf16_qps": bf16["qps"],
                "fp8_qps": fp8["qps"],
                "qps_improvement_percent": (
                    (fp8["qps"] - bf16["qps"]) / bf16["qps"] * 100.0
                ),
                "bf16_audio_seconds_per_wall_second": bf16[
                    "audio_seconds_per_wall_second"
                ],
                "fp8_audio_seconds_per_wall_second": fp8[
                    "audio_seconds_per_wall_second"
                ],
            }
        )
    result = {
        "metric": (
            "continuous_playback_start_s = time_to_first_audio_byte + minimum "
            "jitter buffer required for stall-free playback"
        ),
        "lower_is_better": True,
        "protocol": {
            "model": "Higgs TTS 4B",
            "vocoder": "K8/M3",
            "same_gpu": "NVIDIA H100 80GB HBM3, physical GPU 6",
            "duration_s_per_level": 60,
            "warmup_batches_per_level": 1,
            "max_running_requests": 32,
            "cuda_graph_max_bs": 32,
            "kv_cache_dtype": "bfloat16",
            "only_model_difference": "runtime quantization fp8",
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
        },
        "levels": levels,
    }
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16-dir", type=Path, required=True)
    parser.add_argument("--fp8-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrencies", default="10,11,12,13,14,15,16,17,18,19,20")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    result = compare(
        bf16_dir=args.bf16_dir,
        fp8_dir=args.fp8_dir,
        output=args.output,
        concurrencies=[int(value) for value in args.concurrencies.split(",")],
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    for row in result["levels"]:
        print(
            f"c{row['concurrency']}: "
            f"P85 {row['bf16_p85_s']:.3f}s -> {row['fp8_p85_s']:.3f}s "
            f"({row['improvement_percent']:+.1f}%), "
            f"QPS {row['bf16_qps']:.3f} -> {row['fp8_qps']:.3f} "
            f"({row['qps_improvement_percent']:+.1f}%)"
        )


if __name__ == "__main__":
    main()
