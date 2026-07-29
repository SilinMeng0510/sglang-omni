# SPDX-License-Identifier: Apache-2.0
"""Resumable concurrent audio generation through /v1/audio/speech."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import mimetypes
import time
from dataclasses import asdict
from pathlib import Path

import aiohttp
import soundfile as sf

from benchmarks.tts_eval.data import TTSEvalSample

logger = logging.getLogger(__name__)


def _audio_path(output_dir: Path, sample: TTSEvalSample) -> Path:
    digest = hashlib.sha1(sample.key.encode()).hexdigest()[:16]
    return output_dir / "audio" / sample.benchmark / sample.lang / f"{digest}.wav"


def _load_completed(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("is_success"):
                completed.add(str(row["key"]))
    return completed


def _reference_payload(sample: TTSEvalSample, mode: str) -> dict:
    if mode == "path":
        return {"audio_path": sample.ref_audio, "text": sample.ref_text}
    if mode != "base64":
        raise ValueError(f"Unknown reference mode: {mode!r}")
    media_type = mimetypes.guess_type(sample.ref_audio)[0] or "audio/wav"
    encoded = base64.b64encode(Path(sample.ref_audio).read_bytes()).decode("ascii")
    return {"data": encoded, "media_type": media_type, "text": sample.ref_text}


async def generate_samples(
    samples: list[TTSEvalSample],
    *,
    base_url: str,
    model: str,
    output_dir: str | Path,
    concurrency: int,
    shard_index: int,
    temperature: float = 0.8,
    top_k: int = 50,
    max_new_tokens: int = 750,
    seed: int | None = None,
    timeout_s: float = 300,
    retries: int = 2,
    reference_mode: str = "base64",
) -> dict:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"generation_shard_{shard_index:03d}.jsonl"
    completed = _load_completed(result_path)
    pending = [sample for sample in samples if sample.key not in completed]
    logger.info(
        "Generation shard %d: %d pending, %d already complete",
        shard_index,
        len(pending),
        len(samples) - len(pending),
    )

    endpoint = base_url.rstrip("/") + "/v1/audio/speech"
    queue: asyncio.Queue[TTSEvalSample] = asyncio.Queue()
    for sample in pending:
        queue.put_nowait(sample)
    write_lock = asyncio.Lock()
    counts = {"successful": 0, "failed": 0}
    started = time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    connector = aiohttp.TCPConnector(limit=0)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

        async def worker() -> None:
            while True:
                try:
                    sample = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                wav_path = _audio_path(output_dir, sample)
                wav_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "model": model,
                    "voice": "default",
                    "input": sample.target_text,
                    "references": [_reference_payload(sample, reference_mode)],
                    "response_format": "wav",
                    "temperature": temperature,
                    "top_k": top_k,
                    "max_new_tokens": max_new_tokens,
                }
                if seed is not None:
                    payload["seed"] = seed
                row = {
                    **asdict(sample),
                    "key": sample.key,
                    "wav_path": str(wav_path),
                    "is_success": False,
                }
                request_started = time.perf_counter()
                for attempt in range(retries + 1):
                    try:
                        async with session.post(endpoint, json=payload) as response:
                            if response.status != 200:
                                message = await response.text()
                                raise RuntimeError(f"HTTP {response.status}: {message}")
                            wav_bytes = await response.read()
                        wav_path.write_bytes(wav_bytes)
                        row["audio_duration_s"] = sf.info(str(wav_path)).duration
                        row["is_success"] = True
                        row["error"] = None
                        break
                    except (
                        Exception
                    ) as exc:  # noqa: BLE001 - retry any request/write failure
                        row["error"] = str(exc)
                        if attempt < retries:
                            await asyncio.sleep(2**attempt)
                row["latency_s"] = time.perf_counter() - request_started
                counts["successful" if row["is_success"] else "failed"] += 1
                async with write_lock:
                    with result_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        handle.flush()
                queue.task_done()

        await asyncio.gather(*(worker() for _ in range(concurrency)))
    counts.update(
        {
            "requested": len(pending),
            "skipped_complete": len(samples) - len(pending),
            "wall_clock_s": time.perf_counter() - started,
            "result_path": str(result_path),
        }
    )
    return counts
