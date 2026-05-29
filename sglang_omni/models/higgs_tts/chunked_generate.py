# SPDX-License-Identifier: Apache-2.0
"""Higgs TTS request-level chunking orchestrator (a ``Client`` middleware).

Splits long TTS text and issues N sequential submissions, each tagged with a
shared ``session_id``; the engine ties them together (continuity is engine-side
— see :mod:`sglang_omni.models.higgs_tts.session`). Plugged into the shared
:class:`~sglang_omni.client.client.Client` as ``generate_middleware``, so the
base client stays free of TTS logic and ``speech``/``completion[_stream]``
inherit chunking via ``client.generate``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

from sglang_omni.client.types import GenerateChunk, GenerateRequest
from sglang_omni.models.higgs_tts.text_chunker import ChunkerOptions, TextChunker

if TYPE_CHECKING:
    from sglang_omni.client.client import Client


class HiggsChunkedGenerate:
    """``generate_middleware`` that chunks long TTS text into a continuity
    session and runs the chunks as sequential engine submissions."""

    def __init__(
        self,
        *,
        text_chunker: TextChunker,
        chunker_factory: Callable[[ChunkerOptions], "TextChunker | None"] | None = None,
        chunker_options: ChunkerOptions | None = None,
    ) -> None:
        self._text_chunker = text_chunker
        self._chunker_factory = chunker_factory
        self._chunker_options = chunker_options

    async def __call__(
        self,
        client: "Client",
        request: GenerateRequest,
        request_id: str,
    ) -> AsyncIterator[GenerateChunk]:
        caller_session = request.metadata.get("tts_session") or {}
        caller_session_id = caller_session.get("id")

        chunks = self._chunk_text(request)
        if not chunks or len(chunks) <= 1:
            # Single submission, but it may belong to a caller-managed session
            # (the WS handler threads one session across sentences) — forward
            # the tag so the engine conditions it on the session's prior audio.
            if caller_session_id:
                request = self._tag_session(
                    request, caller_session_id, final=bool(caller_session.get("final"))
                )
            async for chunk in client._generate_single(request, request_id):
                yield chunk
            return

        # Reuse the caller's session id when present (WS) so cross-sentence and
        # cross-chunk share one engine session; else mint one from request_id.
        session_id = caller_session_id or request_id
        session_final = (
            bool(caller_session.get("final")) if caller_session_id else True
        )
        async for chunk in self._generate_chunked(
            client, request, chunks, request_id, session_id, session_final
        ):
            yield chunk

    def new_streaming_chunker(
        self,
        *,
        fastout: bool = False,
    ) -> "TextChunker | None":
        """Fresh per-WS-connection streaming chunker (each needs its own
        ``add_text`` buffer); ``None`` if no factory was supplied."""
        if self._chunker_factory is None or self._chunker_options is None:
            return None
        opts = self._chunker_options
        if fastout != opts.fastout:
            opts = replace(opts, fastout=fastout)
        return self._chunker_factory(opts)

    def _chunk_text(self, request: GenerateRequest) -> list[str] | None:
        """Split the request's TTS text; ``None`` if nothing to split (non-TTS
        task or no text input)."""
        text = self._extract_chunkable_text(request)
        if not text:
            return None
        return self._text_chunker.chunk(text, cps=self._compute_cps_from_request(request))

    @staticmethod
    def _extract_chunkable_text(request: GenerateRequest) -> str | None:
        if request.metadata.get("task") != "tts":
            return None
        prompt = request.prompt
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, dict):
            return prompt.get("text") or prompt.get("input")
        return None

    def _compute_cps_from_request(self, request: GenerateRequest) -> float | None:
        """Chars/sec from the reference, computed from pre-encoded ``vq_codes``
        row count (``len(vq) / codec_frame_rate``) — no audio I/O. ``None`` when
        there's nothing to calibrate from; the chunker then uses its default.
        Only sizes oversized-sentence sub-splits, so a default is fine."""
        codec_frame_rate = getattr(self._text_chunker, "codec_frame_rate", None)
        ref_text = (request.metadata.get("tts_params") or {}).get("ref_text")
        ref_dur_s = 0.0

        if isinstance(request.prompt, dict):
            refs = request.prompt.get("references")
            if refs and isinstance(refs, list) and isinstance(refs[0], dict):
                first = refs[0]
                ref_text = ref_text or first.get("text")
                vq = first.get("vq_codes")
                if codec_frame_rate and isinstance(vq, list) and vq and isinstance(
                    vq[0], list
                ):
                    ref_dur_s = len(vq) / codec_frame_rate

        if ref_dur_s <= 0 or not ref_text:
            return None
        nws = sum(1 for c in ref_text if not c.isspace())
        return nws / ref_dur_s if nws else None

    async def _generate_chunked(
        self,
        client: "Client",
        request: GenerateRequest,
        chunks: list[str],
        request_id: str,
        session_id: str,
        session_final: bool,
    ) -> AsyncIterator[GenerateChunk]:
        """Submit chunks in order (each completes — engine commits its codes —
        before the next), all tagged with ``session_id``. The caller sees one
        stream under ``request_id``; non-final chunks' finish reasons are
        suppressed so end-of-stream fires only after the last chunk."""
        base_prompt = self._chunk_base_prompt(request)
        for i, chunk_str in enumerate(chunks):
            is_last = i == len(chunks) - 1
            chunk_prompt: dict[str, Any] = dict(base_prompt)
            chunk_prompt["text"] = chunk_str
            sub_request = self._tag_session(
                replace(request, prompt=chunk_prompt),
                session_id,
                final=session_final and is_last,
            )
            async for sub_chunk in client._generate_single(
                sub_request, f"{request_id}:c{i}"
            ):
                outer = replace(sub_chunk, request_id=request_id)
                if not is_last and outer.finish_reason is not None:
                    outer.finish_reason = None
                yield outer

    @staticmethod
    def _chunk_base_prompt(request: GenerateRequest) -> dict[str, Any]:
        if isinstance(request.prompt, dict):
            return dict(request.prompt)
        return {"text": request.prompt}

    @staticmethod
    def _tag_session(
        request: GenerateRequest, session_id: str, *, final: bool
    ) -> GenerateRequest:
        metadata = dict(request.metadata)
        metadata["tts_session"] = {"id": session_id, "final": final}
        return replace(request, metadata=metadata)


__all__ = ["HiggsChunkedGenerate"]
