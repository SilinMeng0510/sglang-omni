# SPDX-License-Identifier: Apache-2.0
"""Audio encoding utilities.

Converts raw audio data (numpy arrays, torch tensors, or raw bytes) into
various output formats (WAV, MP3, FLAC, etc.) for API responses.
"""

from __future__ import annotations

import base64
import io
import logging
import struct
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Supported output formats and their MIME types
FORMAT_MIME_TYPES: dict[str, str] = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "pcm": "audio/pcm",
}

# Default sample rate for generated audio
DEFAULT_SAMPLE_RATE = 24000


def to_numpy(audio: Any) -> np.ndarray:
    """Convert audio data to a numpy float32 array.

    Accepts:
    - numpy ndarray
    - torch Tensor
    - list / tuple of numbers
    - bytes (assumed 16-bit PCM)
    """
    if isinstance(audio, np.ndarray):
        return audio.astype(np.float32, copy=False)

    # torch Tensor
    if hasattr(audio, "cpu") and hasattr(audio, "numpy"):
        arr = audio.detach().cpu().float().numpy()
        return arr.astype(np.float32, copy=False)

    if isinstance(audio, (list, tuple)):
        return np.array(audio, dtype=np.float32)

    if isinstance(audio, bytes):
        # Assume 16-bit signed PCM
        arr = np.frombuffer(audio, dtype="<i2")
        return (arr.astype(np.float32) / 32768.0).astype(np.float32)

    raise TypeError(f"Unsupported audio type: {type(audio)}")


def apply_speed(
    audio: np.ndarray, speed: float, sample_rate: int
) -> tuple[np.ndarray, int]:
    """Apply speed adjustment by resampling.

    Returns (adjusted_audio, adjusted_sample_rate).

    Raises:
        ValueError: If *speed* is zero or negative.
    """
    if speed <= 0.0:
        raise ValueError(f"speed must be positive, got {speed}")
    if speed == 1.0:
        return audio, sample_rate

    # Speed up/slow down by changing the effective sample rate
    # Then resample to the original rate
    new_length = max(int(round(len(audio) / speed)), 1)
    old_idx = np.arange(len(audio), dtype=np.float64)
    new_idx = np.linspace(0.0, len(audio) - 1, num=new_length, dtype=np.float64)
    resampled = np.interp(new_idx, old_idx, audio).astype(np.float32)
    return resampled, sample_rate


def encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode audio as a WAV file (16-bit PCM)."""
    # Clamp to [-1, 1]
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    pcm_bytes = pcm.tobytes()

    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm_bytes)

    buf = io.BytesIO()
    # RIFF header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    # fmt chunk
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # chunk size
    buf.write(
        struct.pack(
            "<HHIIHH",
            1,
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
        )
    )
    # data chunk
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm_bytes)

    return buf.getvalue()


def encode_pcm(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode audio as raw 16-bit PCM bytes."""
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16).tobytes()


# response_format -> (container format, codec name) for the PyAV encoder.
_AV_FORMAT_SPEC: dict[str, tuple[str, str]] = {
    "mp3": ("mp3", "mp3"),
    "aac": ("adts", "aac"),
    "opus": ("ogg", "libopus"),
}
# Opus only accepts these input sample rates; others are resampled to 48 kHz.
_OPUS_SAMPLE_RATES = (8000, 12000, 16000, 24000, 48000)


def _encode_with_av(audio: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    """Encode mono audio to a compressed format (mp3/aac/opus) using PyAV.

    PyAV links the FFmpeg libraries directly, so this works without a system
    ``ffmpeg`` binary (the previous pydub path shelled out to ffmpeg and failed
    with FileNotFoundError when it was absent).
    """
    import av

    container_fmt, codec_name = _AV_FORMAT_SPEC[fmt]

    samples = np.clip(np.asarray(audio), -1.0, 1.0)
    if samples.dtype.kind == "f":
        samples = (samples * 32767.0).astype("<i2")
    else:
        samples = samples.astype("<i2")
    samples = samples.reshape(1, -1)  # (channels=1, n_samples) for mono s16

    out_rate = sample_rate
    if fmt == "opus" and sample_rate not in _OPUS_SAMPLE_RATES:
        out_rate = 48000

    buf = io.BytesIO()
    with av.open(buf, mode="w", format=container_fmt) as container:
        stream = container.add_stream(codec_name, rate=out_rate)
        stream.layout = "mono"

        frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.rate = sample_rate

        if out_rate != sample_rate:
            resampler = av.AudioResampler(format="s16", layout="mono", rate=out_rate)
            frames = list(resampler.resample(frame)) + list(resampler.resample(None))
        else:
            frames = [frame]

        for fr in frames:
            for packet in stream.encode(fr):
                container.mux(packet)
        for packet in stream.encode(None):  # flush encoder
            container.mux(packet)

    return buf.getvalue()


def encode_audio(
    audio: Any,
    *,
    response_format: str = "wav",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    speed: float = 1.0,
) -> tuple[bytes, str]:
    """Encode audio data to the requested format.

    Args:
        audio: Raw audio data (numpy, torch tensor, list, bytes)
        response_format: Target format (wav, mp3, flac, opus, aac, pcm)
        sample_rate: Audio sample rate in Hz
        speed: Speed adjustment factor (1.0 = normal)

    Returns:
        (encoded_bytes, mime_type)
    """
    arr = to_numpy(audio)

    # Flatten to 1D if needed
    if arr.ndim > 1:
        arr = arr.squeeze()
    if arr.ndim > 1:
        # Multi-channel: take first channel.
        # Handle both (channels, samples) and (samples, channels).
        if arr.shape[0] < arr.shape[-1]:
            arr = arr[0]
        else:
            arr = arr[:, 0]

    # Apply speed
    if speed != 1.0:
        arr, sample_rate = apply_speed(arr, speed, sample_rate)

    fmt = response_format.lower().strip()
    mime = FORMAT_MIME_TYPES.get(fmt, "application/octet-stream")

    if fmt == "wav":
        return encode_wav(arr, sample_rate), mime

    if fmt == "pcm":
        return encode_pcm(arr, sample_rate), mime

    if fmt == "flac":
        try:
            import soundfile as sf

            buf = io.BytesIO()
            sf.write(buf, arr, sample_rate, format="FLAC")
            return buf.getvalue(), mime
        except Exception:  # pragma: no cover - depends on optional codec support
            logger.warning("Failed to encode FLAC; falling back to WAV", exc_info=True)
            return encode_wav(arr, sample_rate), FORMAT_MIME_TYPES["wav"]

    if fmt in ("mp3", "aac", "opus"):
        try:
            return _encode_with_av(arr, sample_rate, fmt), mime
        except Exception:  # pragma: no cover - depends on optional codec support
            logger.warning(
                "Failed to encode %s; falling back to WAV", fmt, exc_info=True
            )
            return encode_wav(arr, sample_rate), FORMAT_MIME_TYPES["wav"]

    # Unknown format -> fall back to WAV
    logger.warning("Unknown audio format '%s'; falling back to WAV", fmt)
    return encode_wav(arr, sample_rate), FORMAT_MIME_TYPES["wav"]


def audio_to_base64(
    audio: Any,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    output_format: str = "wav",
) -> str:
    """Encode audio data to a base64 string.

    Useful for embedding audio in JSON responses (e.g. chat completions
    with audio modality).
    """
    audio_bytes, _ = encode_audio(
        audio, response_format=output_format, sample_rate=sample_rate
    )
    return base64.b64encode(audio_bytes).decode("ascii")
