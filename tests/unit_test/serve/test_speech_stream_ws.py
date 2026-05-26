# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from sglang_omni.client import GenerateChunk, SpeechResult
from sglang_omni.client.types import GenerateRequest
from sglang_omni.serve import create_app
from sglang_omni.serve.openai_api import (
    _build_streaming_speech_request,
    build_speech_generate_request,
)
from sglang_omni.serve.protocol import StreamingSpeechSessionConfig


class StreamingSpeechWsClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.stream_prompts: list[str] = []
        self.aborted: list[str] = []

    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def speech(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        response_format: str = "wav",
        speed: float = 1.0,
    ) -> SpeechResult:
        del request_id, response_format, speed
        assert isinstance(request.prompt, str)
        self.prompts.append(request.prompt)
        return SpeechResult(
            audio_bytes=f"audio:{request.prompt}".encode(),
            mime_type="audio/wav",
            format="wav",
            usage=None,
        )

    async def generate(self, request: GenerateRequest, request_id: str | None = None):
        del request_id
        assert isinstance(request.prompt, str)
        self.stream_prompts.append(request.prompt)
        yield GenerateChunk(
            request_id="speech-stream-1",
            modality="audio",
            audio_data=[0.0, 0.1, -0.1, 0.0],
            sample_rate=24000,
        )
        yield GenerateChunk(
            request_id="speech-stream-1",
            modality="audio",
            finish_reason="stop",
        )

    async def abort(self, request_id: str):
        self.aborted.append(request_id)


def test_streaming_speech_ws_splits_text_and_returns_audio_frames() -> None:
    speech_client = StreamingSpeechWsClient()
    client = TestClient(create_app(speech_client, model_name="higgs"))

    with client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(
            {
                "type": "session.config",
                "response_format": "wav",
                "voice": "default",
            }
        )
        ws.send_json({"type": "input.text", "text": "Hello world. How are you?"})
        ws.send_json({"type": "input.done"})

        assert ws.receive_json() == {
            "type": "audio.start",
            "sentence_index": 0,
            "sentence_text": "Hello world.",
            "format": "wav",
        }
        assert ws.receive_bytes() == b"audio:Hello world."
        assert ws.receive_json() == {
            "type": "audio.done",
            "sentence_index": 0,
            "total_bytes": len(b"audio:Hello world."),
            "error": False,
        }

        assert ws.receive_json() == {
            "type": "audio.start",
            "sentence_index": 1,
            "sentence_text": "How are you?",
            "format": "wav",
        }
        assert ws.receive_bytes() == b"audio:How are you?"
        assert ws.receive_json() == {
            "type": "audio.done",
            "sentence_index": 1,
            "total_bytes": len(b"audio:How are you?"),
            "error": False,
        }

        assert ws.receive_json() == {
            "type": "session.done",
            "total_sentences": 2,
        }

    assert speech_client.prompts == ["Hello world.", "How are you?"]


def test_streaming_speech_ws_matches_vllm_english_sentence_boundary() -> None:
    speech_client = StreamingSpeechWsClient()
    client = TestClient(create_app(speech_client, model_name="higgs"))

    with client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json({"type": "session.config"})
        ws.send_json({"type": "input.text", "text": "Hello.World"})
        ws.send_json({"type": "input.done"})

        assert ws.receive_json()["sentence_text"] == "Hello.World"
        assert ws.receive_bytes() == b"audio:Hello.World"
        assert ws.receive_json()["type"] == "audio.done"
        assert ws.receive_json() == {
            "type": "session.done",
            "total_sentences": 1,
        }

    assert speech_client.prompts == ["Hello.World"]


def test_streaming_speech_ws_clause_mode_matches_vllm_boundaries() -> None:
    speech_client = StreamingSpeechWsClient()
    client = TestClient(create_app(speech_client, model_name="higgs"))

    with client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json({"type": "session.config", "split_granularity": "clause"})
        ws.send_json({"type": "input.text", "text": "alpha, beta；gamma"})
        ws.send_json({"type": "input.done"})

        assert ws.receive_json()["sentence_text"] == "alpha, beta；"
        assert ws.receive_bytes() == "audio:alpha, beta；".encode()
        assert ws.receive_json()["type"] == "audio.done"
        assert ws.receive_json()["sentence_text"] == "gamma"
        assert ws.receive_bytes() == b"audio:gamma"
        assert ws.receive_json()["type"] == "audio.done"
        assert ws.receive_json() == {
            "type": "session.done",
            "total_sentences": 2,
        }

    assert speech_client.prompts == ["alpha, beta；", "gamma"]


def test_streaming_speech_ws_requires_config_first() -> None:
    client = TestClient(create_app(StreamingSpeechWsClient(), model_name="higgs"))

    with client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json({"type": "input.text", "text": "hello"})
        msg = ws.receive_json()

    assert msg["type"] == "error"
    assert "Expected session.config" in msg["message"]


def test_streaming_speech_ws_validates_stream_audio_pcm() -> None:
    client = TestClient(create_app(StreamingSpeechWsClient(), model_name="higgs"))

    with client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(
            {
                "type": "session.config",
                "stream_audio": True,
                "response_format": "wav",
            }
        )
        msg = ws.receive_json()

    assert msg["type"] == "error"
    assert "requires response_format='pcm'" in msg["message"]


def test_streaming_speech_ws_stream_audio_sends_pcm_chunks() -> None:
    speech_client = StreamingSpeechWsClient()
    client = TestClient(create_app(speech_client, model_name="higgs"))

    with client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json(
            {
                "type": "session.config",
                "stream_audio": True,
                "response_format": "pcm",
            }
        )
        ws.send_json({"type": "input.text", "text": "Hello."})
        ws.send_json({"type": "input.done"})

        assert ws.receive_json() == {
            "type": "audio.start",
            "sentence_index": 0,
            "sentence_text": "Hello.",
            "format": "pcm",
            "sample_rate": 24000,
        }
        assert len(ws.receive_bytes()) > 0
        assert ws.receive_json()["type"] == "audio.done"
        assert ws.receive_json() == {
            "type": "session.done",
            "total_sentences": 1,
        }

    assert speech_client.stream_prompts == ["Hello."]


def test_streaming_speech_config_accepts_speaker_alias_and_passthrough() -> None:
    config = StreamingSpeechSessionConfig(
        speaker="alice",
        x_vector_only_mode=True,
        initial_codec_chunk_frames=8,
        speaker_embedding=[0.1, 0.2],
    )

    speech_req = _build_streaming_speech_request(config, "Hello.")
    gen_req = build_speech_generate_request(speech_req, "qwen3-tts")

    assert speech_req.voice == "alice"
    assert gen_req.metadata["tts_params"]["x_vector_only_mode"] is True
    assert gen_req.metadata["tts_params"]["initial_codec_chunk_frames"] == 8
    assert gen_req.metadata["tts_params"]["speaker_embedding"] == [0.1, 0.2]
