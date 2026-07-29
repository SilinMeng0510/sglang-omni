# SPDX-License-Identifier: Apache-2.0
"""CLI for SeedTTS, CV3, and MiniMax multilingual TTS evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from benchmarks.tts_eval.data import (
    BLOG_LANGUAGES,
    DEFAULT_DATA_ROOT,
    load_benchmark_samples,
)
from benchmarks.tts_eval.metrics import load_jsonl, score_benchmark

DEFAULT_MODEL = "bosonai/higgs-tts-3-4b"
DEFAULT_WHISPER = "/ceph/models/whisper-large-v3"
DEFAULT_PARAFORMER = (
    "/ceph/models/modelscope_cache/hub/iic/"
    "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
)


def _csv(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("benchmark", choices=sorted(BLOG_LANGUAGES))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--languages", help="Comma-separated override")
    parser.add_argument("--max-samples-per-language", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multilingual TTS quality evaluation")
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser("inspect")
    _add_dataset_args(inspect_parser)

    generate_parser = commands.add_parser("generate")
    _add_dataset_args(generate_parser)
    generate_parser.add_argument("--base-url", required=True)
    generate_parser.add_argument("--model", default=DEFAULT_MODEL)
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.add_argument("--concurrency", type=int, default=16)
    generate_parser.add_argument("--temperature", type=float, default=0.8)
    generate_parser.add_argument("--top-k", type=int, default=50)
    generate_parser.add_argument("--max-new-tokens", type=int, default=750)
    generate_parser.add_argument("--seed", type=int)
    generate_parser.add_argument("--timeout", type=float, default=300)
    generate_parser.add_argument("--retries", type=int, default=2)
    generate_parser.add_argument(
        "--reference-mode",
        choices=("base64", "path"),
        default="base64",
        help="Use base64 unless the server can see the dataset paths",
    )

    transcribe_parser = commands.add_parser("transcribe")
    transcribe_parser.add_argument("--output-dir", type=Path, required=True)
    transcribe_parser.add_argument("--transcript-name", default="transcripts.jsonl")
    transcribe_parser.add_argument("--languages")
    transcribe_parser.add_argument("--whisper-model", default=DEFAULT_WHISPER)
    transcribe_parser.add_argument("--paraformer-model", default=DEFAULT_PARAFORMER)
    transcribe_parser.add_argument("--device", default="cuda:0")
    transcribe_parser.add_argument("--batch-size", type=int, default=1)
    transcribe_parser.add_argument("--whisper-for-zh", action="store_true")

    score_parser = commands.add_parser("score")
    score_parser.add_argument("benchmark", choices=sorted(BLOG_LANGUAGES))
    score_parser.add_argument("--transcripts", type=Path, nargs="+", required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    return parser


def _load_samples(args: argparse.Namespace):
    return load_benchmark_samples(
        args.benchmark,
        data_root=args.data_root,
        languages=_csv(args.languages),
        max_samples_per_language=args.max_samples_per_language,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "inspect":
        samples = _load_samples(args)
        counts = {}
        for sample in samples:
            counts[sample.lang] = counts.get(sample.lang, 0) + 1
        print(json.dumps({"total": len(samples), "per_language": counts}, indent=2))
        return
    if args.command == "generate":
        from benchmarks.tts_eval.generate import generate_samples

        samples = _load_samples(args)
        result = asyncio.run(
            generate_samples(
                samples,
                base_url=args.base_url,
                model=args.model,
                output_dir=args.output_dir,
                concurrency=args.concurrency,
                shard_index=args.shard_index,
                temperature=args.temperature,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
                timeout_s=args.timeout,
                retries=args.retries,
                reference_mode=args.reference_mode,
            )
        )
        print(json.dumps(result, indent=2))
        return
    if args.command == "transcribe":
        from benchmarks.tts_eval.transcribe import transcribe_generated

        result = transcribe_generated(
            output_dir=args.output_dir,
            transcript_name=args.transcript_name,
            whisper_model=args.whisper_model,
            paraformer_model=args.paraformer_model,
            device=args.device,
            batch_size=args.batch_size,
            languages=set(_csv(args.languages) or ()) or None,
            use_paraformer_for_zh=not args.whisper_for_zh,
        )
        print(json.dumps(result, indent=2))
        return
    rows_by_key = {}
    for path in args.transcripts:
        for row in load_jsonl(path):
            rows_by_key[row["key"]] = row
    result = score_benchmark(args.benchmark, rows_by_key.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = {key: value for key, value in result.items() if key != "rows"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
