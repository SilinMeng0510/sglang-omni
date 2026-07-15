#!/usr/bin/env python3
"""Concurrent raw-PCM performance sweep for Higgs TTS."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any

import aiohttp
from common import request_speech, speech_payload, stats, write_json


def load_prompts(path: Path, seed: int) -> list[str]:
    prompts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            text = row.get("text") or row.get("prompt")
            if text:
                prompts.append(str(text))
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    random.Random(seed).shuffle(prompts)
    return prompts


async def run_level(
    args: argparse.Namespace, concurrency: int, prompts: list[str]
) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=0)
    rows: list[dict[str, Any]] = []
    lock = asyncio.Lock()
    cursor = 0
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

        async def one(text: str, seed: int) -> None:
            result, _ = await request_speech(
                session,
                base_url=args.base_url,
                payload=speech_payload(
                    text,
                    model=args.model,
                    voice=args.voice,
                    seed=seed,
                    lora_adapter_path=args.lora_adapter_path,
                ),
            )
            async with lock:
                rows.append({"text": text, **result.json()})

        # Warm the exact concurrent shape before measurement.
        for batch in range(args.warmup_batches):
            await asyncio.gather(
                *[
                    one(
                        prompts[(batch * concurrency + i) % len(prompts)],
                        700_000 + batch * concurrency + i,
                    )
                    for i in range(concurrency)
                ]
            )
        rows.clear()
        started = time.perf_counter()
        deadline = started + args.duration

        async def worker(worker_id: int) -> None:
            nonlocal cursor
            del worker_id
            while time.perf_counter() < deadline:
                async with lock:
                    if cursor >= args.max_requests:
                        return
                    index = cursor
                    cursor += 1
                await one(prompts[index % len(prompts)], args.seed + index)

        await asyncio.gather(*[worker(i) for i in range(concurrency)])
    wall = time.perf_counter() - started
    successful = [row for row in rows if row["success"]]
    continuous = [row["continuous_playback_start_s"] for row in successful]
    summary = {
        "concurrency": concurrency,
        "requests": len(rows),
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "wall_clock_s": wall,
        "qps": len(successful) / wall if wall else None,
        "audio_seconds_per_wall_second": (
            sum(r["audio_duration_s"] for r in successful) / wall if wall else None
        ),
        "ttfab_s": stats([r["ttfab_s"] for r in successful]),
        "continuous_playback_start_s": stats(continuous),
        "required_jitter_buffer_s": stats(
            [r["required_jitter_buffer_s"] for r in successful]
        ),
        "latency_s": stats([r["latency_s"] for r in successful]),
        "stall_free_at_100ms_fraction": (
            sum(r["required_jitter_buffer_s"] <= 0.1 for r in successful)
            / len(successful)
            if successful
            else None
        ),
        "rows": rows,
    }
    return summary


async def main_async(args: argparse.Namespace) -> None:
    prompts = load_prompts(args.prompts, args.seed)
    levels = [int(v) for v in args.concurrencies.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for level in levels:
        print(f"performance: concurrency={level}", flush=True)
        summary = await run_level(args, level, prompts)
        write_json(args.output_dir / f"c{level}.json", summary)
        summaries.append({k: v for k, v in summary.items() if k != "rows"})
    write_json(
        args.output_dir / "summary.json",
        {
            "config": {
                "base_url": args.base_url,
                "model": args.model,
                "voice": args.voice,
                "lora_adapter_path": args.lora_adapter_path,
                "prompt_file": str(args.prompts),
                "concurrencies": levels,
                "duration_s_per_level": args.duration,
                "warmup_batches_per_level": args.warmup_batches,
                "timeout_s": args.timeout,
                "seed": args.seed,
                "max_requests_per_level": args.max_requests,
            },
            "levels": summaries,
        },
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--voice", default="default")
    parser.add_argument("--lora-adapter-path")
    parser.add_argument("--prompts", type=Path, default=root / "sharegpt_10k.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concurrencies", default=",".join(map(str, range(1, 17))))
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--max-requests", type=int, default=10_000)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
