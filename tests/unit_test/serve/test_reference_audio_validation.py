# SPDX-License-Identifier: Apache-2.0
"""Reference-audio must be inline base64 / local path — never a remote URL (SSRF)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sglang_omni.serve.protocol import (
    CreateSpeechRequest,
    SpeechReference,
    StreamingSpeechSessionConfig,
)

_REJECTED_URLS = [
    "http://example.com/a.wav",
    "https://hf.co/datasets/x/prompt.wav",
    "HTTPS://HF.CO/x.wav",
    "ftp://host/a.wav",
    "file:///etc/passwd",
]
_ACCEPTED = [
    "/srv/refs/voice.wav",
    "refs/voice.wav",
    "data:audio/wav;base64,UklGRiQAAABXQVZF",
    None,
]


@pytest.mark.parametrize("url", _REJECTED_URLS)
def test_speech_reference_rejects_remote_url(url) -> None:
    with pytest.raises(ValidationError, match="SSRF"):
        SpeechReference(audio_path=url)


@pytest.mark.parametrize("url", _REJECTED_URLS)
def test_create_speech_request_rejects_remote_url(url) -> None:
    with pytest.raises(ValidationError, match="SSRF"):
        CreateSpeechRequest(input="hi", ref_audio=url)


@pytest.mark.parametrize("url", _REJECTED_URLS)
def test_create_speech_request_rejects_url_in_references(url) -> None:
    with pytest.raises(ValidationError, match="SSRF"):
        CreateSpeechRequest(input="hi", references=[{"audio_path": url}])


@pytest.mark.parametrize("url", _REJECTED_URLS)
def test_streaming_config_rejects_remote_url(url) -> None:
    with pytest.raises(ValidationError, match="SSRF"):
        StreamingSpeechSessionConfig(ref_audio=url)


@pytest.mark.parametrize("value", _ACCEPTED)
def test_local_path_and_data_uri_accepted(value) -> None:
    assert SpeechReference(audio_path=value).audio_path == value
    assert CreateSpeechRequest(input="hi", ref_audio=value).ref_audio == value
    assert StreamingSpeechSessionConfig(ref_audio=value).ref_audio == value
