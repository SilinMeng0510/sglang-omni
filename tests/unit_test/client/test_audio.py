"""``encode_audio`` must produce real compressed audio without a system ffmpeg.

mp3/aac/opus are encoded via PyAV (which links the FFmpeg libraries directly), so
these formats work even when no ``ffmpeg`` binary is installed. Previously they
went through pydub -> ffmpeg and raised FileNotFoundError when ffmpeg was absent.
"""

import io

import numpy as np
import pytest

from sglang_omni.client.audio import encode_audio


def _sine(sample_rate: int = 24000, seconds: float = 0.3, freq: float = 440.0):
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _detect(data: bytes) -> str:
    head = bytes(data[:4])
    if head[:4] == b"RIFF":
        return "wav"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] == b"OggS":
        return "opus"
    if head[:3] == b"ID3" or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "mp3"
    if head[0] == 0xFF and (head[1] & 0xF6) == 0xF0:
        return "aac"
    return "raw"


@pytest.mark.parametrize(
    "fmt, mime, detected",
    [
        ("wav", "audio/wav", "wav"),
        ("flac", "audio/flac", "flac"),
        ("mp3", "audio/mpeg", "mp3"),
        ("opus", "audio/opus", "opus"),
        ("aac", "audio/aac", "aac"),
    ],
)
def test_encode_audio_produces_real_format(fmt, mime, detected):
    data, got_mime = encode_audio(_sine(), response_format=fmt, sample_rate=24000)
    assert got_mime == mime
    # The bytes are the real format, not a silent fallback to WAV.
    assert _detect(data) == detected


def test_encode_pcm_is_raw_int16():
    sample_rate = 24000
    data, mime = encode_audio(
        _sine(sample_rate), response_format="pcm", sample_rate=sample_rate
    )
    assert mime == "audio/pcm"
    assert len(data) == int(sample_rate * 0.3) * 2  # int16 mono, no container


def test_compressed_output_is_decodable():
    soundfile = pytest.importorskip("soundfile")
    for fmt in ("opus", "flac"):
        data, _ = encode_audio(_sine(), response_format=fmt, sample_rate=24000)
        audio, sample_rate = soundfile.read(io.BytesIO(data))
        assert len(audio) > 0 and sample_rate > 0


def test_opus_resamples_unsupported_sample_rate():
    # 44100 is not a valid Opus input rate; the encoder must resample, not crash.
    data, mime = encode_audio(_sine(44100), response_format="opus", sample_rate=44100)
    assert mime == "audio/opus"
    assert _detect(data) == "opus"


def test_unknown_format_falls_back_to_wav():
    data, mime = encode_audio(
        _sine(), response_format="does-not-exist", sample_rate=24000
    )
    assert mime == "audio/wav"
    assert _detect(data) == "wav"
