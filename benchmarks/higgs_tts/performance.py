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

try:
    from .common import request_speech, speech_payload, stats, write_json
except ImportError:  # Direct execution: python benchmarks/higgs_tts/performance.py
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
    pace_lock = asyncio.Lock()
    cursor = 0
    warmup_cursor = 0
    next_request_start = 0.0
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

        async def one(text: str, seed: int, *, record: bool = True) -> None:
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
            if record:
                async with lock:
                    rows.append({"text": text, **result.json()})

        stagger_s = args.worker_start_stagger_ms / 1000.0
        request_interval_s = 1.0 / args.request_rate if args.request_rate else None
        steady_state_mode = (
            stagger_s > 0
            or args.steady_state_warmup_s > 0
            or request_interval_s is not None
        )
        if steady_state_mode:
            # A synchronized warmup batch can keep closed-loop workers phase-aligned
            # for the full run. Ramp workers independently, keep them active during
            # an unmeasured settling period, and record only requests that start
            # after the steady-state boundary.
            ramp_started = time.perf_counter()
            started = (
                ramp_started
                + max(0, concurrency - 1) * stagger_s
                + args.steady_state_warmup_s
            )
            deadline = started + args.duration

            async def pace_request_start() -> None:
                nonlocal next_request_start
                if request_interval_s is None:
                    return
                async with pace_lock:
                    now = time.perf_counter()
                    scheduled = max(now, next_request_start)
                    next_request_start = scheduled + request_interval_s
                delay = scheduled - now
                if delay > 0:
                    await asyncio.sleep(delay)

            async def worker(worker_id: int) -> None:
                nonlocal cursor, warmup_cursor
                if worker_id:
                    await asyncio.sleep(worker_id * stagger_s)
                while time.perf_counter() < deadline:
                    await pace_request_start()
                    request_started = time.perf_counter()
                    if request_started >= deadline:
                        return
                    record = request_started >= started
                    async with lock:
                        if record:
                            if cursor >= args.max_requests:
                                return
                            index = cursor
                            cursor += 1
                            seed = args.seed + index
                        else:
                            index = warmup_cursor
                            warmup_cursor += 1
                            seed = 800_000 + index
                    await one(
                        prompts[index % len(prompts)],
                        seed,
                        record=record,
                    )

            await asyncio.gather(*[worker(i) for i in range(concurrency)])
        else:
            # Legacy mode: preserve the PR benchmark protocol exactly.
            for batch in range(args.warmup_batches):
                await asyncio.gather(
                    *[
                        one(
                            prompts[(batch * concurrency + i) % len(prompts)],
                            700_000 + batch * concurrency + i,
                            record=False,
                        )
                        for i in range(concurrency)
                    ]
                )
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
                "worker_start_stagger_ms": args.worker_start_stagger_ms,
                "steady_state_warmup_s": args.steady_state_warmup_s,
                "request_rate": args.request_rate,
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
    parser.add_argument(
        "--worker-start-stagger-ms",
        type=float,
        default=0,
        help=(
            "Delay each successive closed-loop worker by this many milliseconds. "
            "A positive value enables steady-state worker mode."
        ),
    )
    parser.add_argument(
        "--steady-state-warmup-s",
        type=float,
        default=0,
        help=(
            "Keep staggered workers running for this many seconds after the last "
            "worker starts before recording measured requests."
        ),
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        help=(
            "Maximum aggregate request starts per second across all workers. "
            "Starts are evenly paced without catch-up bursts."
        ),
    )
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    if args.worker_start_stagger_ms < 0:
        parser.error("--worker-start-stagger-ms must be non-negative")
    if args.steady_state_warmup_s < 0:
        parser.error("--steady-state-warmup-s must be non-negative")
    if args.request_rate is not None and args.request_rate <= 0:
        parser.error("--request-rate must be positive")
    return args


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
