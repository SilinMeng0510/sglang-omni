# SPDX-License-Identifier: Apache-2.0
"""Multilingual TTS quality evaluation for SeedTTS, CV3, and MiniMax."""

from benchmarks.tts_eval.data import (
    BLOG_LANGUAGES,
    TTSEvalSample,
    load_benchmark_samples,
)
from benchmarks.tts_eval.metrics import score_benchmark

__all__ = [
    "BLOG_LANGUAGES",
    "TTSEvalSample",
    "load_benchmark_samples",
    "score_benchmark",
]
