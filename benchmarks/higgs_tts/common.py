#!/usr/bin/env python3
"""Shared primitives for the Higgs TTS end-to-end suite."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import statistics
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp

SAMPLE_RATE = 24_000
SAMPLE_WIDTH = 2
CHANNELS = 1
BYTES_PER_SECOND = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS


@dataclass
class SpeechResult:
    success: bool
    latency_s: float
    ttfab_s: float | None
    audio_duration_s: float
    required_jitter_buffer_s: float | None
    continuous_playback_start_s: float | None
    post_first_audio_speed: float | None
    audio_chunk_count: int
    audio_bytes: int
    sha256: str | None
    status_code: int | None = None
    error: str | None = None

    def json(self) -> dict[str, Any]:
        return asdict(self)


def percentile(values: list[float], pct: float) -> float | None:
    finite = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not finite:
        return None
    position = (len(finite) - 1) * pct / 100.0
    lower = int(position)
    upper = min(lower + 1, len(finite) - 1)
    fraction = position - lower
    return finite[lower] * (1 - fraction) + finite[upper] * fraction


def stats(values: list[float]) -> dict[str, float | None]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return {
        "mean": statistics.mean(finite) if finite else None,
        "p50": percentile(finite, 50),
        "p85": percentile(finite, 85),
        "p95": percentile(finite, 95),
        "p99": percentile(finite, 99),
    }


def continuity(chunks: list[tuple[float, int]]) -> tuple[float, float | None]:
    """Return minimum jitter buffer and post-first audio production speed."""
    if not chunks:
        return 0.0, None
    first = chunks[0][0]
    available = chunks[0][1] / BYTES_PER_SECOND
    required = 0.0
    for arrival, size in chunks[1:]:
        required = max(required, arrival - first - available)
        available += size / BYTES_PER_SECOND
    remaining = sum(size for _, size in chunks[1:]) / BYTES_PER_SECOND
    elapsed = chunks[-1][0] - first
    speed = remaining / elapsed if remaining > 0 and elapsed > 0 else None
    return max(0.0, required), speed


def write_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


async def request_speech(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    payload: dict[str, Any],
) -> tuple[SpeechResult, bytes]:
    """Call the OpenAI-compatible endpoint and measure the full chunk timeline."""
    started = time.perf_counter()
    chunks: list[tuple[float, int]] = []
    pcm = bytearray()
    status: int | None = None
    try:
        async with session.post(
            f"{base_url.rstrip('/')}/v1/audio/speech", json=payload
        ) as response:
            status = response.status
            if status != 200:
                raise RuntimeError(f"HTTP {status}: {(await response.text())[:1000]}")
            content_type = (response.headers.get("content-type") or "").lower()
            if content_type.startswith("audio/"):
                async for data in response.content.iter_chunked(64 * 1024):
                    if data:
                        chunks.append((time.perf_counter(), len(data)))
                        pcm.extend(data)
            else:
                while raw := await response.content.readline():
                    line = raw.decode("utf-8", errors="replace").strip()
                    if line == "data: [DONE]":
                        break
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    encoded = (event.get("audio") or {}).get("data")
                    if encoded:
                        data = base64.b64decode(encoded)
                        chunks.append((time.perf_counter(), len(data)))
                        pcm.extend(data)
        ended = time.perf_counter()
        if not chunks or not pcm:
            raise RuntimeError("response contained no audio")
        jitter, speed = continuity(chunks)
        ttfab = chunks[0][0] - started
        duration = len(pcm) / BYTES_PER_SECOND
        return SpeechResult(
            success=True,
            latency_s=ended - started,
            ttfab_s=ttfab,
            audio_duration_s=duration,
            required_jitter_buffer_s=jitter,
            continuous_playback_start_s=ttfab + jitter,
            post_first_audio_speed=speed,
            audio_chunk_count=len(chunks),
            audio_bytes=len(pcm),
            sha256=hashlib.sha256(pcm).hexdigest(),
            status_code=status,
        ), bytes(pcm)
    except Exception as exc:
        return (
            SpeechResult(
                success=False,
                latency_s=time.perf_counter() - started,
                ttfab_s=None,
                audio_duration_s=0.0,
                required_jitter_buffer_s=None,
                continuous_playback_start_s=None,
                post_first_audio_speed=None,
                audio_chunk_count=0,
                audio_bytes=0,
                sha256=None,
                status_code=status,
                error=str(exc),
            ),
            b"",
        )


def speech_payload(
    text: str,
    *,
    model: str,
    voice: str,
    seed: int,
    references: list[dict[str, str]] | None = None,
    lora_adapter_path: str | None = None,
    max_new_tokens: int = 1024,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model": model,
        "input": text,
        "voice": voice,
        "stream": True,
        "response_format": "pcm",
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 50,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
    }
    if references:
        result["references"] = references
    if lora_adapter_path:
        result["lora_adapter"] = {"path": lora_adapter_path}
    return result
