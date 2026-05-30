# SPDX-License-Identifier: Apache-2.0
"""Per-request pipeline state for Higgs TTS.

Carried between stages via :class:`sglang_omni.proto.StagePayload.data`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HiggsTtsState:
    """Per-request state threaded through preprocessing → audio_encoder →
    tts_engine → vocoder. Fields populate lazily so a deserialised state is
    valid at any stage boundary."""

    # preprocessing / audio_encoder
    prompt_token_ids: list[int] = field(default_factory=list)
    reference_codes_delayed: list[list[int]] | None = None
    reference_waveform: Any | None = None
    target_text_token_ids: list[int] | None = None
    reference_text_token_ids: list[int] | None = None

    # Cross-chunk continuity
    session_id: str | None = None
    session_final: bool = False
    session_index: int = -1  # chunk's serve sentence index (barge-in rollback)
    session_truncate_after: int | None = None  # input.stop: drop history past this

    num_codebooks: int = 8
    codebook_size: int = 1026  # 1024 data + <|boc|> + <|eoc|>

    # generation params
    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None

    # tts_engine
    output_codes_delayed: list[list[int]] | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    engine_time_s: float = 0.0

    # vocoder
    audio_samples: Any | None = None
    sample_rate: int = 24000

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "prompt_token_ids": list(self.prompt_token_ids),
            "num_codebooks": self.num_codebooks,
            "codebook_size": self.codebook_size,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
        if self.reference_codes_delayed is not None:
            data["reference_codes_delayed"] = self.reference_codes_delayed
        if self.reference_waveform is not None:
            data["reference_waveform"] = self.reference_waveform
        if self.session_id is not None:
            data["session_id"] = self.session_id
            data["session_final"] = self.session_final
            data["session_index"] = self.session_index
            if self.session_truncate_after is not None:
                data["session_truncate_after"] = self.session_truncate_after
        if self.target_text_token_ids is not None:
            data["target_text_token_ids"] = self.target_text_token_ids
        if self.reference_text_token_ids is not None:
            data["reference_text_token_ids"] = self.reference_text_token_ids
        for key in ("top_p", "top_k", "seed"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        if self.output_codes_delayed is not None:
            data["output_codes_delayed"] = self.output_codes_delayed
        for key in ("prompt_tokens", "completion_tokens", "engine_time_s"):
            value = getattr(self, key)
            if value:
                data[key] = value
        if self.audio_samples is not None:
            data["audio_samples"] = self.audio_samples
            data["sample_rate"] = self.sample_rate
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HiggsTtsState:
        return cls(
            prompt_token_ids=list(data.get("prompt_token_ids", [])),
            reference_codes_delayed=data.get("reference_codes_delayed"),
            reference_waveform=data.get("reference_waveform"),
            session_id=data.get("session_id"),
            session_final=data.get("session_final", False),
            session_index=data.get("session_index", -1),
            session_truncate_after=data.get("session_truncate_after"),
            target_text_token_ids=data.get("target_text_token_ids"),
            reference_text_token_ids=data.get("reference_text_token_ids"),
            num_codebooks=data.get("num_codebooks", 8),
            codebook_size=data.get("codebook_size", 1026),
            max_new_tokens=data.get("max_new_tokens", 2048),
            temperature=data.get("temperature", 1.0),
            top_p=data.get("top_p"),
            top_k=data.get("top_k"),
            seed=data.get("seed"),
            output_codes_delayed=data.get("output_codes_delayed"),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            engine_time_s=data.get("engine_time_s", 0.0),
            audio_samples=data.get("audio_samples"),
            sample_rate=data.get("sample_rate", 24000),
        )


__all__ = ["HiggsTtsState"]
