# SPDX-License-Identifier: Apache-2.0
"""Audio-codec helpers for Higgs TTS.

- Delay pattern: :func:`apply_delay_pattern` / :func:`reverse_delay_pattern`
  shift codebook ``c`` by ``c`` steps, BOC/EOC padding inside the codebook
  vocab (ids 1024 / 1025 for the default 1026 vocab).
- Tail cleanup: :data:`TAIL_TRIM_FRAMES` / :func:`fade_out_tail` kill the
  end-of-utterance click (see the constants' comment).
- Codec cache + ref-codes coercion + ref-audio loading from path / URL /
  bytes / base64.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from sglang_omni.models.higgs_tts.audio.codec import HiggsAudioCodec
from sglang_omni.preprocessing.audio import AudioMediaIO
from sglang_omni.preprocessing.base import _is_url
from sglang_omni.preprocessing.resource_connector import global_http_connection

# Codec-vocab specials (inside the [N*V] codebook space, NOT the text vocab).
BOC_ID = 1024
EOC_ID = 1025

# End-of-utterance click suppression. Once cb0 emits EOC, the final data
# frame's remaining codebooks (cb1..cbN-1) come from the delay-pattern
# wind-down steps — the model's least-certain predictions — and the decoded
# last codec frame (~40 ms) often carries an audible spike. Drop it, then
# fade the last few ms of waveform to remove any residual discontinuity.
TAIL_TRIM_FRAMES = 1
TAIL_FADE_MS = 8.0


def fade_out_tail(
    audio: torch.Tensor, sample_rate: int, fade_ms: float = TAIL_FADE_MS
) -> torch.Tensor:
    """Linear fade over the last ``fade_ms`` of ``audio`` (last dim = samples)."""
    n = min(int(sample_rate * fade_ms / 1000.0), int(audio.shape[-1]))
    if n <= 0:
        return audio
    audio = audio.clone()
    ramp = torch.linspace(1.0, 0.0, n, dtype=audio.dtype, device=audio.device)
    audio[..., -n:] *= ramp
    return audio


# Shared between audio_encoder + vocoder; one codec load saves ~1 GB VRAM.
_CODEC_CACHE: dict[tuple[str, str, str], HiggsAudioCodec] = {}


def apply_delay_pattern(codes_TN: torch.Tensor) -> torch.Tensor:
    """``[T, N]`` raw codes → ``[T + N - 1, N]`` delayed, BOC/EOC padded."""
    if codes_TN.ndim != 2:
        raise ValueError(
            f"codes_TN must be 2-D [T, N], got shape {tuple(codes_TN.shape)}"
        )
    T, N = codes_TN.shape
    out = torch.full(
        (T + N - 1, N), EOC_ID, device=codes_TN.device, dtype=codes_TN.dtype
    )
    t_idx = torch.arange(T + N - 1, device=codes_TN.device)
    for c in range(N):
        out[t_idx < c, c] = BOC_ID
        out[c : c + T, c] = codes_TN[:, c]
    return out


def reverse_delay_pattern(delayed_LN: torch.Tensor) -> torch.Tensor:
    """``[L, N]`` delayed (L >= N) → ``[L - (N - 1), N]`` raw codes."""
    if delayed_LN.ndim != 2:
        raise ValueError(
            f"delayed_LN must be 2-D [L, N], got shape {tuple(delayed_LN.shape)}"
        )
    L, N = delayed_LN.shape
    T = L - (N - 1)
    if T <= 0:
        raise ValueError(
            f"delayed_LN has L={L}, N={N}; need L >= N so at least one "
            f"data row can be recovered."
        )
    out = torch.empty((T, N), device=delayed_LN.device, dtype=delayed_LN.dtype)
    for c in range(N):
        out[:, c] = delayed_LN[c : c + T, c]
    return out


def get_or_load_codec(path: str, device: str, dtype: str) -> HiggsAudioCodec:
    """Process-wide cached :class:`HiggsAudioCodec` per (path, device, dtype)."""
    key = (str(path), str(device), str(dtype))
    cached = _CODEC_CACHE.get(key)
    if cached is not None:
        return cached
    codec = HiggsAudioCodec.from_pretrained(
        path, device=device, dtype=getattr(torch, dtype)
    )
    _CODEC_CACHE[key] = codec
    return codec


def to_codes_TN(raw: Any, num_codebooks: int) -> torch.Tensor | None:
    """Coerce client-supplied ``reference_codes`` to a ``[T, N]`` int64 tensor."""
    if raw is None:
        return None
    t = raw if isinstance(raw, torch.Tensor) else torch.tensor(raw)
    if t.numel() == 0:
        return None
    if t.ndim != 2 or t.shape[1] != num_codebooks:
        raise ValueError(
            f"reference_codes must have shape [T, {num_codebooks}], got {tuple(t.shape)}"
        )
    return t.to(torch.long)


def load_audio_to_24k(reference_audio: Any) -> tuple[np.ndarray, int]:
    """Load ``inputs["reference_audio"]`` as 24 kHz mono float32.

    Accepts local path, HTTP/HTTPS URL, or ``{audio_path|path|bytes|base64|data}`` dict.
    """
    io = AudioMediaIO(target_sr=HiggsAudioCodec.SAMPLE_RATE)

    def _load_path_or_url(src: str | Path) -> tuple[np.ndarray, int]:
        if isinstance(src, str) and _is_url(src):
            response = global_http_connection.get_sync_client().get(src)
            response.raise_for_status()
            audio, sr = io.load_bytes(response.content)
        else:
            audio, sr = io.load_file(Path(src))
        return np.asarray(audio, dtype=np.float32), int(sr)

    if isinstance(reference_audio, (str, Path)):
        return _load_path_or_url(reference_audio)

    if "audio_path" in reference_audio or "path" in reference_audio:
        return _load_path_or_url(
            reference_audio.get("audio_path") or reference_audio["path"]
        )
    if "bytes" in reference_audio:
        audio, sr = io.load_bytes(reference_audio["bytes"])
        return np.asarray(audio, dtype=np.float32), int(sr)
    media_type = reference_audio.get("media_type", "audio/wav")
    data = reference_audio.get("base64") or reference_audio["data"]
    audio, sr = io.load_base64(media_type, data)
    return np.asarray(audio, dtype=np.float32), int(sr)


__all__ = [
    "BOC_ID",
    "EOC_ID",
    "TAIL_FADE_MS",
    "TAIL_TRIM_FRAMES",
    "apply_delay_pattern",
    "fade_out_tail",
    "get_or_load_codec",
    "load_audio_to_24k",
    "reverse_delay_pattern",
    "to_codes_TN",
]
