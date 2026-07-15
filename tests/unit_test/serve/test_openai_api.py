# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sglang_omni.client import Client, ClientError, GenerateChunk
from sglang_omni.client.types import GenerateRequest
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import CompleteMessage, OmniRequest, StreamMessage
from sglang_omni.serve import create_app
from sglang_omni.serve.openai_api import (
    _build_speech_generate_request,
    _chat_stream,
    _speech_stream,
    _streaming_speech_pcm_chunks,
    build_speech_generate_request,
)
from sglang_omni.serve.protocol import (
    MAX_REF_AUDIO_CHARS,
    MAX_REQUEST_BODY_BYTES,
    MAX_SPEECH_INPUT_CHARS,
    ChatCompletionRequest,
    CreateSpeechRequest,
    LoRAAdapterConfig,
)
from tests.unit_test.fixtures.pipeline_fakes import RecordingCoordinatorControlPlane

MODEL_FAMILIES = {
    "qwen3-omni": "code2wav",
    "ming-omni": "talker",
    "s2-pro": "vocoder",
    "voxtral": "vocoder",
}


class FaultInjectingCoordinator(Coordinator):
    """Inject a model-stage failure through the real Coordinator/Client path."""

    def __init__(self, terminal_stage: str):
        super().__init__(
            completion_endpoint="inproc://complete",
            abort_endpoint="inproc://abort",
            entry_stage="preprocess",
            terminal_stages=[terminal_stage],
        )
        self.control_plane = RecordingCoordinatorControlPlane()
        self.terminal_stage = terminal_stage
        self.register_stage("preprocess", "inproc://preprocess")

    async def _submit_request(
        self, request_id: str, request: OmniRequest | Any
    ) -> None:
        await super()._submit_request(request_id, request)
        if not isinstance(request, OmniRequest):
            request = OmniRequest(inputs=request)
        if bool(request.params.get("stream", False)):
            await self._handle_stream(self._partial_stream_message(request_id, request))
        await self._handle_completion(
            CompleteMessage(
                request_id=request_id,
                from_stage=self.terminal_stage,
                success=False,
                error="cuda out of memory",
            )
        )

    def _partial_stream_message(
        self, request_id: str, request: OmniRequest
    ) -> StreamMessage:
        if "tts_params" in request.metadata:
            chunk = {
                "audio_data": [0.0, 0.1],
                "sample_rate": 24000,
                "modality": "audio",
            }
            modality = "audio"
        else:
            chunk = {"text": "partial", "modality": "text"}
            modality = "text"
        return StreamMessage(
            request_id=request_id,
            from_stage=self.terminal_stage,
            chunk=chunk,
            stage_name=self.terminal_stage,
            modality=modality,
        )


def _fault_client(model_name: str) -> Client:
    return Client(FaultInjectingCoordinator(MODEL_FAMILIES[model_name]))


class SuccessfulSpeechClient:
    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def generate(self, request: Any, request_id: str | None = None):
        del request
        yield GenerateChunk(
            request_id=request_id or "speech-1",
            modality="audio",
            audio_data=[0.0, 0.1, -0.1, 0.0],
            sample_rate=24000,
            finish_reason="stop",
        )


class FailingSpeechClient:
    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def generate(self, request: Any, request_id: str | None = None):
        del request, request_id
        yield GenerateChunk(
            request_id="speech-1",
            modality="audio",
            audio_data=[0.0, 0.1, -0.1, 0.0],
            sample_rate=24000,
        )
        raise ClientError("stream failed")


class LoRALoadFailureSpeechClient:
    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def generate(self, request: Any, request_id: str | None = None):
        del request, request_id
        if False:
            yield
        raise ClientError(
            "Failed to load LoRA adapter /models/ap2: "
            "adapter rank 64 exceeds max rank 32"
        )


class RefAudioTooLongSpeechClient:
    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def speech(self, request: Any, **kwargs: Any):
        del request, kwargs
        raise ClientError("reference_audio is too long (2400.0s); cap at 30s.")


@pytest.mark.parametrize("model_name", MODEL_FAMILIES)
def test_non_streaming_http_faults_return_500(model_name: str) -> None:
    client = TestClient(create_app(_fault_client(model_name), model_name=model_name))

    chat_resp = client.post(
        "/v1/chat/completions",
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )
    assert chat_resp.status_code == 500
    assert "cuda out of memory" in chat_resp.json()["detail"]

    speech_resp = client.post(
        "/v1/audio/speech",
        json={
            "model": model_name,
            "input": "hello",
            "stream": False,
            "response_format": "wav",
        },
    )
    assert speech_resp.status_code == 500
    assert "cuda out of memory" in speech_resp.json()["detail"]


def test_non_streaming_speech_ref_audio_too_long_returns_400() -> None:
    client = TestClient(
        create_app(RefAudioTooLongSpeechClient(), model_name="higgs-tts")
    )

    resp = client.post(
        "/v1/audio/speech",
        json={
            "model": "higgs-tts",
            "input": "hello",
            "response_format": "wav",
            "ref_audio": "https://example.com/very-long.wav",
        },
    )

    assert resp.status_code == 400
    assert "reference_audio is too long" in resp.json()["detail"]


def test_speech_request_body_size_limit_returns_413() -> None:
    client = TestClient(create_app(SuccessfulSpeechClient(), model_name="higgs-tts"))
    big = "u" * (MAX_REQUEST_BODY_BYTES + 10)

    resp = client.post("/v1/audio/speech", json={"input": "hi", "ref_audio": big})

    assert resp.status_code == 413
    assert "Request body exceeds" in resp.json()["error"]["message"]


def test_speech_ref_audio_field_limit_returns_compact_422() -> None:
    client = TestClient(create_app(SuccessfulSpeechClient(), model_name="higgs-tts"))
    big = "u" * (MAX_REF_AUDIO_CHARS + 10)

    resp = client.post("/v1/audio/speech", json={"input": "hi", "ref_audio": big})

    assert resp.status_code == 422
    assert "ref_audio exceeds" in resp.text
    assert "uuu" not in resp.text
    assert len(resp.content) < 10_000


def test_speech_input_field_limit_returns_compact_422() -> None:
    client = TestClient(create_app(SuccessfulSpeechClient(), model_name="higgs-tts"))
    big = "x" * (MAX_SPEECH_INPUT_CHARS + 1)

    resp = client.post("/v1/audio/speech", json={"input": big})

    assert resp.status_code == 422
    assert "input exceeds" in resp.text
    assert "xxx" not in resp.text
    assert len(resp.content) < 10_000


def test_chat_stream_failure_closes_without_done_sentinel() -> None:
    chunks: list[str] = []
    client = _fault_client("qwen3-omni")
    req = ChatCompletionRequest(
        model="qwen3-omni",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )

    async def _drive() -> None:
        async for chunk in _chat_stream(
            client=client,
            gen_req=GenerateRequest(model="qwen3-omni", prompt="hello", stream=True),
            request_id="req-1",
            response_id="chatcmpl-req-1",
            created=0,
            model="qwen3-omni",
            req=req,
            audio_format="wav",
        ):
            chunks.append(chunk)

    with pytest.raises(RuntimeError, match="cuda out of memory"):
        asyncio.run(_drive())

    assert chunks
    assert all(chunk != "data: [DONE]\n\n" for chunk in chunks)


async def _collect_speech_stream(client: Any) -> list[str]:
    chunks: list[str] = []
    async for chunk in _speech_stream(
        client=client,
        gen_req=GenerateRequest(model="s2-pro", prompt="hello", stream=True),
        request_id="req-1",
        response_format="wav",
        speed=1.0,
    ):
        chunks.append(chunk)
    return chunks


def test_speech_stream_success_emits_done_sentinel() -> None:
    chunks = asyncio.run(_collect_speech_stream(SuccessfulSpeechClient()))

    assert chunks[-1] == "data: [DONE]\n\n"
    payload = json.loads(chunks[-2][len("data: ") :])
    assert payload["audio"] is None
    assert payload["finish_reason"] == "stop"


def test_pcm_speech_stream_returns_raw_audio_bytes() -> None:
    client = TestClient(create_app(SuccessfulSpeechClient(), model_name="higgs-tts"))

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "higgs-tts",
            "input": "hello",
            "stream": True,
            "response_format": "pcm",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/pcm")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert response.headers["x-audio-sample-rate"] == "24000"
    assert response.headers["x-audio-sample-format"] == "s16le"
    assert response.headers["x-audio-channels"] == "1"
    assert response.content == b"\x00\x00\xcc\x0c\x34\xf3\x00\x00"
    assert not response.content.startswith(b"data:")


def test_pcm_stream_returns_http_error_when_lora_load_fails() -> None:
    client = TestClient(
        create_app(LoRALoadFailureSpeechClient(), model_name="higgs-tts")
    )

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "higgs-tts",
            "input": "hello",
            "stream": True,
            "response_format": "pcm",
            "lora_adapter": {"path": "/models/ap2"},
        },
    )

    assert response.status_code == 400
    assert "rank 64 exceeds max rank 32" in response.json()["detail"]


def test_pcm_stream_logs_failure_after_response_start(caplog) -> None:
    async def consume() -> None:
        stream = _streaming_speech_pcm_chunks(
            client=FailingSpeechClient(),
            gen_req=object(),
            request_id="speech-midstream-failure",
        )
        with pytest.raises(ClientError, match="stream failed"):
            async for _ in stream:
                pass

    with caplog.at_level("ERROR"):
        asyncio.run(consume())

    assert "PCM speech stream failed after response start" in caplog.text


def test_speech_stream_returns_error_event_after_chunk_failure() -> None:
    """Preserves deterministic SSE termination after a mid-stream client error."""
    client = TestClient(create_app(FailingSpeechClient(), model_name="s2-pro"))

    with client.stream(
        "POST",
        "/v1/audio/speech",
        json={
            "model": "s2-pro",
            "input": "hello",
            "stream": True,
            "response_format": "wav",
        },
        timeout=5.0,
    ) as resp:
        assert resp.status_code == 200
        events = []
        done = False
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                done = True
                break
            events.append(json.loads(payload))

    assert done
    assert len(events) == 2
    assert events[0]["audio"] is not None
    assert events[0]["finish_reason"] is None
    assert events[1]["audio"] is None
    assert events[1]["finish_reason"] == "error"
    assert events[1]["error"] == {
        "type": "ClientError",
        "message": "stream failed",
    }


def test_speech_request_records_explicit_generation_params() -> None:
    req = CreateSpeechRequest(
        input="hello",
        temperature=0.8,
        top_k=30,
        seed=123,
    )

    gen_req = build_speech_generate_request(req, "qwen3-tts")

    assert _build_speech_generate_request is build_speech_generate_request
    assert gen_req.sampling.temperature == 0.8
    assert gen_req.sampling.top_k == 30
    assert gen_req.sampling.seed == 123
    assert gen_req.metadata["tts_params"]["explicit_generation_params"] == [
        "seed",
        "temperature",
        "top_k",
    ]


def test_speech_request_passes_request_scoped_lora_adapter() -> None:
    request = build_speech_generate_request(
        CreateSpeechRequest(
            input="hello",
            lora_adapter=LoRAAdapterConfig(path="/models/ap2/adapter"),
        ),
        "higgs-tts",
    )

    assert request.metadata["tts_params"]["lora_adapter"] == {
        "path": "/models/ap2/adapter"
    }


def test_speech_request_uses_higgs_tts_sampling_defaults() -> None:
    request = _build_speech_generate_request(
        CreateSpeechRequest(input="hello", voice="default"),
        "boson-sglang/higgs-audio-v3-tts-4b-base",
    )

    assert request.sampling.temperature == 0.8
    assert request.sampling.top_p == 0.95
    assert request.sampling.top_k == 50
    assert request.sampling.repetition_penalty == 1.0


def test_speech_request_maps_ref_audio_raw_base64() -> None:
    ref_audio = base64.b64encode(b"RIFF....WAVE").decode("ascii")
    req = CreateSpeechRequest(
        input="hello",
        ref_audio=ref_audio,
        ref_text="reference transcript",
    )

    gen_req = build_speech_generate_request(req, "higgs-tts")

    assert gen_req.prompt == {
        "text": "hello",
        "references": [
            {
                "base64": ref_audio,
                "media_type": "audio/wav",
                "text": "reference transcript",
            }
        ],
    }


def test_speech_request_keeps_ref_audio_url_as_audio_path() -> None:
    req = CreateSpeechRequest(
        input="hello",
        ref_audio="https://example.com/ref.wav",
        ref_text="reference transcript",
    )

    gen_req = build_speech_generate_request(req, "higgs-tts")

    assert gen_req.prompt == {
        "text": "hello",
        "references": [
            {
                "audio_path": "https://example.com/ref.wav",
                "text": "reference transcript",
            }
        ],
    }


def test_speech_request_maps_ref_audio_data_uri_base64() -> None:
    ref_audio = base64.b64encode(b"OggS....").decode("ascii")
    req = CreateSpeechRequest(
        input="hello",
        ref_audio=f"data:audio/ogg;base64,{ref_audio}",
        ref_text="reference transcript",
    )

    gen_req = build_speech_generate_request(req, "higgs-tts")

    assert gen_req.prompt == {
        "text": "hello",
        "references": [
            {
                "base64": ref_audio,
                "media_type": "audio/ogg",
                "text": "reference transcript",
            }
        ],
    }


def test_speech_request_preserves_stage_params() -> None:
    request = _build_speech_generate_request(
        CreateSpeechRequest(
            input="hello",
            voice="default",
            stage_params={
                "vocoder": {
                    "audio_chunk_size": 12,
                    "audio_chunk_overlap_size": 12,
                }
            },
        ),
        "boson-sglang/higgs-audio-v3-tts-4b-base",
    )

    assert request.stage_params == {
        "vocoder": {
            "audio_chunk_size": 12,
            "audio_chunk_overlap_size": 12,
        }
    }


def test_speech_request_keeps_s2_pro_sampling_defaults() -> None:
    request = _build_speech_generate_request(
        CreateSpeechRequest(input="hello", voice="default"),
        "fishaudio-s2-pro",
    )

    assert request.sampling.temperature == 0.8
    assert request.sampling.top_p == 0.8
    assert request.sampling.top_k == 30
    assert request.sampling.repetition_penalty == 1.1
