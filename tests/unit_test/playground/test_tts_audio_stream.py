# SPDX-License-Identifier: Apache-2.0

import io
import wave

import numpy as np
import pytest

from playground.tts.audio_stream import BufferedPcmChunkEmitter, PcmChunkAccumulator


def _read_wav(audio_bytes: bytes) -> tuple[int, int, int, bytes]:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    return channels, sample_width, sample_rate, frames


def test_pcm_accumulator_writes_single_wav_artifact() -> None:
    first_chunk = np.array([0, 1000, -1000], dtype="<i2").tobytes()
    second_chunk = np.array([2000, -2000], dtype="<i2").tobytes()

    accumulator = PcmChunkAccumulator()
    accumulator.add_pcm_chunk(first_chunk, 24000)
    accumulator.add_pcm_chunk(second_chunk, 24000)

    wav_bytes = accumulator.to_wav_bytes()

    assert wav_bytes is not None
    channels, sample_width, sample_rate, frames = _read_wav(wav_bytes)
    assert channels == 1
    assert sample_width == 2
    assert sample_rate == 24000
    assert frames == first_chunk + second_chunk


def test_pcm_emitter_wraps_buffered_pcm_as_wav() -> None:
    pcm_chunk = np.arange(16, dtype="<i2").tobytes()
    emitter = BufferedPcmChunkEmitter(
        min_emit_duration_s=100.0,
        max_buffered_chunks=1,
    )

    wav_bytes = emitter.add_pcm_chunk(pcm_chunk, 24000)

    assert wav_bytes is not None
    assert _read_wav(wav_bytes) == (1, 2, 24000, pcm_chunk)
    assert emitter.flush() is None


def test_pcm_accumulator_rejects_sample_rate_changes() -> None:
    accumulator = PcmChunkAccumulator()
    accumulator.add_pcm_chunk(np.array([0], dtype="<i2").tobytes(), 24000)

    with pytest.raises(ValueError, match="Inconsistent PCM chunk sample rate"):
        accumulator.add_pcm_chunk(np.array([1], dtype="<i2").tobytes(), 16000)


def test_pcm_accumulator_rejects_incomplete_int16_sample() -> None:
    accumulator = PcmChunkAccumulator()

    with pytest.raises(ValueError, match="complete int16"):
        accumulator.add_pcm_chunk(b"\x00", 24000)
