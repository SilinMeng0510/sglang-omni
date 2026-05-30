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


class _FakeMiddleware:
    """Stand-in for the generate-middleware: the WS handler only needs its
    ``new_streaming_chunker`` to obtain a per-connection sentence splitter."""

    def new_streaming_chunker(self, *, fastout: bool = False):
        from sglang_omni.models.higgs_tts.text_chunker import (
            ChunkerOptions,
            HiggsTextChunker,
        )

        return HiggsTextChunker(
            ChunkerOptions(
                max_seconds=8.0,
                cps=10.0,
                fastout=fastout,
            )
        )


class StreamingSpeechWsClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.stream_prompts: list[str] = []
        self.aborted: list[str] = []
        # tts_session metadata seen per generated chunk (continuity + barge-in).
        self.sessions: list[dict[str, Any]] = []
        # The WS handler reads ``client.generate_middleware.new_streaming_chunker``.
        self.generate_middleware = _FakeMiddleware()

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
        self.sessions.append(dict(request.metadata.get("tts_session") or {}))
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


def test_streaming_speech_ws_fastout_clause_then_sentence() -> None:
    # fastout: the FIRST chunk is released at the earliest clause
    # boundary (CJK comma) for low first-audio latency; every later chunk uses
    # sentence boundaries — so the comma inside the final piece does NOT split.
    speech_client = StreamingSpeechWsClient()
    client = TestClient(create_app(speech_client, model_name="higgs"))

    with client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json({"type": "session.config", "fastout": True})
        ws.send_json({"type": "input.text", "text": "一，二。三，四"})
        ws.send_json({"type": "input.done"})

        # chunk 0 — earliest clause boundary (fast first audio)
        assert ws.receive_json()["sentence_text"] == "一，"
        assert ws.receive_bytes() == "audio:一，".encode()
        assert ws.receive_json()["type"] == "audio.done"
        # chunk 1 — sentence boundary
        assert ws.receive_json()["sentence_text"] == "二。"
        assert ws.receive_bytes() == "audio:二。".encode()
        assert ws.receive_json()["type"] == "audio.done"
        # chunk 2 (flush) — the comma did NOT split: sentence mode after first
        assert ws.receive_json()["sentence_text"] == "三，四"
        assert ws.receive_bytes() == "audio:三，四".encode()
        assert ws.receive_json()["type"] == "audio.done"
        assert ws.receive_json() == {
            "type": "session.done",
            "total_sentences": 3,
        }

    assert speech_client.prompts == ["一，", "二。", "三，四"]


def test_streaming_speech_ws_input_wait_rearms_fastout() -> None:
    # input.wait re-arms fastout: turn 2's first chunk is a clause again. Without
    # the wait, "三，四。" would stay one sentence chunk (sentence mode after first).
    speech_client = StreamingSpeechWsClient()
    client = TestClient(create_app(speech_client, model_name="higgs"))

    with client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json({"type": "session.config", "fastout": True})
        # turn 1
        ws.send_json({"type": "input.text", "text": "一，二。"})
        assert ws.receive_json()["sentence_text"] == "一，"
        assert ws.receive_bytes() == "audio:一，".encode()
        assert ws.receive_json()["type"] == "audio.done"
        assert ws.receive_json()["sentence_text"] == "二。"
        assert ws.receive_bytes() == "audio:二。".encode()
        assert ws.receive_json()["type"] == "audio.done"
        # agent paused (user speaking) → keepalive that re-arms fastout
        ws.send_json({"type": "input.wait"})
        # turn 2 — first chunk is a clause again thanks to the re-arm
        ws.send_json({"type": "input.text", "text": "三，四。"})
        assert ws.receive_json()["sentence_text"] == "三，"
        assert ws.receive_bytes() == "audio:三，".encode()
        assert ws.receive_json()["type"] == "audio.done"
        assert ws.receive_json()["sentence_text"] == "四。"
        assert ws.receive_bytes() == "audio:四。".encode()
        assert ws.receive_json()["type"] == "audio.done"
        ws.send_json({"type": "input.done"})
        assert ws.receive_json() == {
            "type": "session.done",
            "total_sentences": 4,
        }

    assert speech_client.prompts == ["一，", "二。", "三，", "四。"]


def test_streaming_speech_ws_input_wait_flushes_unterminated_tail() -> None:
    # A tail with no sentence terminator ("night!" — '!' not followed by space)
    # stays buffered on input.text. input.wait drains it like input.done, but
    # keeps the session open so a later turn still works.
    speech_client = StreamingSpeechWsClient()
    client = TestClient(create_app(speech_client, model_name="higgs"))

    with client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json({"type": "session.config", "fastout": True})
        ws.send_json({"type": "input.text", "text": "See you guys tomorrow night!"})
        # Nothing emitted yet — the whole sentence is held in the chunker buffer.
        ws.send_json({"type": "input.wait"})
        assert (
            ws.receive_json()["sentence_text"] == "See you guys tomorrow night!"
        )
        assert ws.receive_bytes() == "audio:See you guys tomorrow night!".encode()
        assert ws.receive_json()["type"] == "audio.done"
        # Session still open: a second turn (also unterminated) flushes on done.
        ws.send_json({"type": "input.text", "text": "Bye."})
        ws.send_json({"type": "input.done"})
        assert ws.receive_json()["sentence_text"] == "Bye."
        assert ws.receive_bytes() == "audio:Bye.".encode()
        assert ws.receive_json()["type"] == "audio.done"
        assert ws.receive_json() == {
            "type": "session.done",
            "total_sentences": 2,
        }

    assert speech_client.prompts == ["See you guys tomorrow night!", "Bye."]


def test_streaming_speech_ws_input_stop_rolls_back_and_rewinds_index() -> None:
    # Barge-in: the server generated chunks 0..3 (ahead of playback), but the user
    # only heard up to chunk 1. input.stop{chunk:1} aborts the in-flight request,
    # resets the chunker (dropping the unspoken tail), rolls history back to ≤1,
    # and REWINDS the index so the resume starts at chunk 2 (not 4) — old chunks
    # 2,3 are discarded on both sides, indices stay a contiguous restartable run.
    speech_client = StreamingSpeechWsClient()
    client = TestClient(create_app(speech_client, model_name="higgs"))

    with client.websocket_connect("/v1/audio/speech/stream") as ws:
        ws.send_json({"type": "session.config"})
        # turn 1 → chunks 0,1,2,3 + an unterminated tail held in the buffer.
        ws.send_json(
            {"type": "input.text", "text": "One. Two. Three. Four. dangling tail"}
        )
        for _ in range(4):
            assert ws.receive_json()["type"] == "audio.start"
            ws.receive_bytes()
            assert ws.receive_json()["type"] == "audio.done"
        # User barges in having only heard up to chunk 1.
        ws.send_json({"type": "input.stop", "chunk": 1})
        assert ws.receive_json() == {"type": "generation.stopped", "chunk": 1}
        # Resume turn — its chunk must be index 2 (rewound), carrying the rollback.
        ws.send_json({"type": "input.text", "text": "Five."})
        ws.send_json({"type": "input.done"})
        assert ws.receive_json()["sentence_text"].strip() == "Five."
        ws.receive_bytes()
        assert ws.receive_json()["type"] == "audio.done"
        assert ws.receive_json()["type"] == "session.done"

    # The in-flight/stale request was aborted on stop.
    assert speech_client.aborted
    # The held tail was dropped by reset() — it never reached generation.
    assert not any("dangling" in p for p in speech_client.prompts)
    # Index rewound: the resume reuses index 2 (not 4), proving the rewind.
    assert [s["index"] for s in speech_client.sessions] == [0, 1, 2, 3, 2]
    # Only the resume chunk carries the rollback, to K=1.
    rollback = [s for s in speech_client.sessions if "truncate_after" in s]
    assert len(rollback) == 1
    assert rollback[0]["truncate_after"] == 1
    assert rollback[0]["index"] == 2
    assert speech_client.sessions[0].get("truncate_after") is None


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
