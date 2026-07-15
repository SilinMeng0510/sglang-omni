#!/usr/bin/env python3
"""Tiny deterministic raw-PCM server for CPU-only benchmark smoke tests."""

from __future__ import annotations

import argparse
import asyncio
import math
import struct

from aiohttp import web

SAMPLE_RATE = 24_000


async def health(_: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def speech(request: web.Request) -> web.StreamResponse:
    payload = await request.json()
    text = payload.get("input", "")
    duration = max(0.12, min(0.5, len(text) / 400))
    frames = int(duration * SAMPLE_RATE)
    pcm = b"".join(
        struct.pack("<h", int(5000 * math.sin(2 * math.pi * 220 * i / SAMPLE_RATE)))
        for i in range(frames)
    )
    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "audio/pcm",
            "X-Audio-Sample-Rate": str(SAMPLE_RATE),
            "X-Audio-Sample-Format": "s16le",
            "X-Audio-Channels": "1",
        },
    )
    await response.prepare(request)
    midpoint = len(pcm) // 2
    for chunk in (pcm[:midpoint], pcm[midpoint:]):
        await response.write(chunk)
        await asyncio.sleep(0.002)
    await response.write_eof()
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18999)
    args = parser.parse_args()
    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_post("/v1/audio/speech", speech)
    web.run_app(app, host="127.0.0.1", port=args.port)
