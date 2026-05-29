# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Higgs TTS chunking orchestrator (generate middleware).

Engine-side continuity model: the orchestrator (a ``generate_middleware`` on
the shared single-shot ``Client``) splits long text and tags every chunk of one
utterance with a shared ``session_id``; the Higgs engine accumulates the prior
chunks' audio itself. The orchestrator no longer threads codes — these tests
assert the session tagging, the sequential submission, and the ``finish_reason``
gating. Run with the full dependency stack (msgpack/zmq); no GPU.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateRequest
from sglang_omni.models.higgs_tts.chunked_generate import HiggsChunkedGenerate


class _FakeCoord:
    """Minimal coordinator stand-in: records each submission's id, inputs, and
    metadata so the session tagging can be inspected."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any, dict]] = []

    async def submit(self, req_id: str, omni_req: Any) -> dict[str, Any]:
        self.calls.append((req_id, omni_req.inputs, dict(omni_req.metadata)))
        return {
            "audio_data": [0.1],
            "sample_rate": 24000,
            "modality": "audio",
        }

    async def stream(self, req_id: str, omni_req: Any):
        from sglang_omni.proto import CompleteMessage

        self.calls.append((req_id, omni_req.inputs, dict(omni_req.metadata)))
        yield CompleteMessage(
            request_id=req_id,
            success=True,
            error=None,
            result={
                "audio_data": [0.5],
                "sample_rate": 24000,
                "modality": "audio",
            },
            from_stage="vocoder",
        )


class _FixedChunker:
    """Deterministic chunker — always emits ``n`` chunks."""

    def __init__(self, n: int) -> None:
        self._chunks = [f"chunk{i}" for i in range(n)]

    def chunk(self, text: str, *, cps: float | None = None) -> list[str]:
        return list(self._chunks)

    def add_text(self, text: str) -> list[str]:
        return []

    def flush(self) -> list[str]:
        return []


def _client(n_chunks: int | None) -> tuple[Client, _FakeCoord]:
    """Client with (or without) a chunking middleware."""
    coord = _FakeCoord()
    middleware = (
        HiggsChunkedGenerate(text_chunker=_FixedChunker(n_chunks))
        if n_chunks is not None
        else None
    )
    return Client(coord, generate_middleware=middleware), coord


def _session(metadata: dict) -> dict | None:
    return metadata.get("tts_session")


# ---------------------------------------------------------------------------
# Fast paths (no orchestration, no session)
# ---------------------------------------------------------------------------


def test_no_middleware_uses_single_shot_path() -> None:
    client, coord = _client(None)
    req = GenerateRequest(prompt="x", metadata={"task": "tts"}, stream=False)
    out = asyncio.run(_drain(client.generate(req, request_id="r")))
    assert len(out) == 1
    assert len(coord.calls) == 1
    assert coord.calls[0][0] == "r"
    assert _session(coord.calls[0][2]) is None


def test_one_chunk_uses_single_shot_path() -> None:
    client, coord = _client(1)
    req = GenerateRequest(prompt="short", metadata={"task": "tts"}, stream=False)
    asyncio.run(_drain(client.generate(req, request_id="r")))
    assert len(coord.calls) == 1
    inputs = coord.calls[0][1]
    assert inputs == "short" or (
        isinstance(inputs, dict) and inputs.get("text") == "short"
    )
    assert _session(coord.calls[0][2]) is None


def test_non_tts_task_skips_orchestrator() -> None:
    client, coord = _client(3)
    req = GenerateRequest(prompt="x", metadata={"task": "chat"}, stream=False)
    out = asyncio.run(_drain(client.generate(req, request_id="r")))
    assert len(out) == 1
    assert len(coord.calls) == 1


def test_token_ids_input_skips_orchestrator() -> None:
    client, coord = _client(3)
    req = GenerateRequest(
        prompt=None,
        prompt_token_ids=[1, 2, 3],
        metadata={"task": "tts"},
        stream=False,
    )
    out = asyncio.run(_drain(client.generate(req, request_id="r")))
    assert len(out) == 1
    assert len(coord.calls) == 1


# ---------------------------------------------------------------------------
# Multi-chunk orchestration (ephemeral session from the outer request id)
# ---------------------------------------------------------------------------


def test_three_chunks_session_tagging_and_finish_reason_gating() -> None:
    client, coord = _client(3)
    req = GenerateRequest(
        prompt={"text": "long", "references": [{"audio_path": "/x.wav"}]},
        metadata={"task": "tts"},
        stream=False,
    )
    out = asyncio.run(_drain(client.generate(req, request_id="r")))
    assert len(out) == 3
    assert [cid for cid, _, _ in coord.calls] == ["r:c0", "r:c1", "r:c2"]
    assert all(ch.request_id == "r" for ch in out)

    for i, (_, inputs, metadata) in enumerate(coord.calls):
        assert _session(metadata) == {"id": "r", "final": i == 2}
        assert inputs["text"] == f"chunk{i}"
        assert inputs.get("references") == [{"audio_path": "/x.wav"}]
        assert "_history" not in inputs

    assert out[0].finish_reason is None
    assert out[1].finish_reason is None
    assert out[2].finish_reason == "stop"


def test_streaming_path_orchestrator() -> None:
    client, coord = _client(2)
    req = GenerateRequest(prompt="long", metadata={"task": "tts"}, stream=True)
    out = asyncio.run(_drain(client.generate(req, request_id="r")))
    assert len(out) == 2
    assert out[0].finish_reason is None
    assert out[1].finish_reason == "stop"
    assert _session(coord.calls[0][2]) == {"id": "r", "final": False}
    assert _session(coord.calls[1][2]) == {"id": "r", "final": True}


# ---------------------------------------------------------------------------
# Caller-managed session (the WebSocket handler tags one session per socket)
# ---------------------------------------------------------------------------


def test_single_shot_with_caller_session_is_tagged() -> None:
    client, coord = _client(1)
    req = GenerateRequest(
        prompt="x",
        metadata={"task": "tts", "tts_session": {"id": "conn", "final": False}},
        stream=False,
    )
    asyncio.run(_drain(client.generate(req, request_id="r")))
    assert len(coord.calls) == 1
    assert _session(coord.calls[0][2]) == {"id": "conn", "final": False}


def test_caller_session_propagates_across_chunks_not_final() -> None:
    client, coord = _client(3)
    req = GenerateRequest(
        prompt="long",
        metadata={"task": "tts", "tts_session": {"id": "conn", "final": False}},
        stream=False,
    )
    asyncio.run(_drain(client.generate(req, request_id="r")))
    assert [_session(m) for _, _, m in coord.calls] == [
        {"id": "conn", "final": False},
        {"id": "conn", "final": False},
        {"id": "conn", "final": False},
    ]


def test_caller_session_final_only_on_last_chunk() -> None:
    client, coord = _client(3)
    req = GenerateRequest(
        prompt="long",
        metadata={"task": "tts", "tts_session": {"id": "conn", "final": True}},
        stream=False,
    )
    asyncio.run(_drain(client.generate(req, request_id="r")))
    assert [_session(m) for _, _, m in coord.calls] == [
        {"id": "conn", "final": False},
        {"id": "conn", "final": False},
        {"id": "conn", "final": True},
    ]


# ---------------------------------------------------------------------------
# CPS resolution (no audio I/O — code count / tps only)
# ---------------------------------------------------------------------------


def _orch(n: int = 1) -> HiggsChunkedGenerate:
    return HiggsChunkedGenerate(text_chunker=_FixedChunker(n))


def test_cps_from_pre_encoded_vq_codes() -> None:
    req = GenerateRequest(
        prompt={"text": "x", "references": [{"text": "AB", "vq_codes": [[1, 2]] * 50}]},
        metadata={"task": "tts"},
    )
    orch = _orch()
    orch._text_chunker.codec_frame_rate = 25.0  # 50 rows / 25 Hz = 2s; "AB"/2 = 1.0
    assert orch._compute_cps_from_request(req) == pytest.approx(1.0)


def test_cps_none_without_codes() -> None:
    # No calibration source → None (the chunker applies its own default CPS).
    req = GenerateRequest(prompt="x", metadata={"task": "tts"})
    assert _orch()._compute_cps_from_request(req) is None


def test_cps_audio_path_never_loads_audio() -> None:
    # A raw audio_path (no codes) must NOT trigger any audio I/O — duration is
    # unknown so CPS is left to the chunker. (vq_codes is the only source.)
    req = GenerateRequest(
        prompt={"text": "x", "references": [{"text": "AB", "audio_path": "/local.wav"}]},
        metadata={"task": "tts"},
    )
    orch = _orch()
    orch._text_chunker.codec_frame_rate = 25.0
    assert orch._compute_cps_from_request(req) is None


def test_cps_vq_codes_skipped_when_chunker_lacks_frame_rate() -> None:
    req = GenerateRequest(
        prompt={"text": "x", "references": [{"text": "AB", "vq_codes": [[1, 2]] * 50}]},
        metadata={"task": "tts"},
    )
    orch = _orch()
    assert not hasattr(orch._text_chunker, "codec_frame_rate")
    assert orch._compute_cps_from_request(req) is None


# ---------------------------------------------------------------------------
# Prompt-template helper
# ---------------------------------------------------------------------------


def test_chunk_base_prompt_preserves_references() -> None:
    req = GenerateRequest(
        prompt={"text": "x", "references": [{"audio_path": "/r.wav"}]}
    )
    assert HiggsChunkedGenerate._chunk_base_prompt(req)["references"] == [
        {"audio_path": "/r.wav"}
    ]


def test_chunk_base_prompt_converts_string_prompt() -> None:
    req = GenerateRequest(prompt="hello")
    assert HiggsChunkedGenerate._chunk_base_prompt(req) == {"text": "hello"}


async def _drain(stream) -> list[Any]:
    return [ch async for ch in stream]
