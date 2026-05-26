# SPDX-License-Identifier: Apache-2.0
"""load_audio_to_24k accepts data: URIs / local paths and rejects remote URLs."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
import soundfile as sf

from sglang_omni.models.higgs_tts.utils import load_audio_to_24k


def _wav_bytes(seconds: float = 0.1, sr: int = 24000) -> bytes:
    samples = np.zeros(int(seconds * sr), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV")
    return buf.getvalue()


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/a.wav",
        "https://hf.co/x.wav",
        "ftp://host/a.wav",
    ],
)
def test_rejects_remote_url(url) -> None:
    with pytest.raises(ValueError, match="SSRF"):
        load_audio_to_24k(url)


def test_accepts_data_uri() -> None:
    b64 = base64.b64encode(_wav_bytes()).decode()
    audio, sr = load_audio_to_24k(f"data:audio/wav;base64,{b64}")
    assert sr == 24000
    assert audio.dtype == np.float32
    assert audio.shape[-1] > 0


def test_accepts_local_path(tmp_path) -> None:
    path = tmp_path / "ref.wav"
    path.write_bytes(_wav_bytes())
    audio, sr = load_audio_to_24k(str(path))
    assert sr == 24000
    assert audio.shape[-1] > 0
