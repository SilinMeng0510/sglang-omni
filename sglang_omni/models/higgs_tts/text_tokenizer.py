# SPDX-License-Identifier: Apache-2.0
"""Text-prompt builder for HiggsMultimodalQwen3 TTS.

Assembles, depending on ref-audio + ref-text presence:

- voice-clone + transcript: ``<|tts|> <|ref_text|> tok(ref) <|ref_audio|> [-100]×N <|text|> tok(text) <|audio|>``
- voice-clone, no transcript: ``<|tts|> <|ref_audio|> [-100]×N <|text|> tok(text) <|audio|>``
- zero-shot: ``<|tts|> <|text|> tok(text) <|audio|>``

``<|tts|>`` selects task mode (vs ASR); missing it yields fluent-but-wrong
output. ``-100`` placeholders are spliced by :class:`HiggsFusedMultiTextEmbedding`
at runtime; ``num_ref_tokens`` must match the *delayed* ref-code row count
(``T + num_codebooks - 1``).
"""

from __future__ import annotations

from typing import Any

# Matches Higgs ``audio_token_id`` and transformers' ``IGNORE_INDEX`` convention.
AUDIO_PLACEHOLDER_ID = -100

_REQUIRED_SPECIALS: tuple[str, ...] = (
    "<|tts|>",
    "<|ref_audio|>",
    "<|text|>",
    "<|audio|>",
)


class HiggsTokenizerAdapter:
    def __init__(self, tokenizer: Any) -> None:
        self._tok = tokenizer
        vocab = dict(tokenizer.get_added_vocab())
        missing = [t for t in _REQUIRED_SPECIALS if t not in vocab]
        if missing:
            raise ValueError(f"Tokenizer is missing Higgs TTS specials: {missing}")
        self.tts_id: int = vocab["<|tts|>"]
        self.ref_audio_id: int = vocab["<|ref_audio|>"]
        self.text_id: int = vocab["<|text|>"]
        self.audio_id: int = vocab["<|audio|>"]
        # Newer ckpts only; older ckpts fall back to audio-only voice-cloning.
        self.ref_text_id: int | None = vocab.get("<|ref_text|>")
        # Streaming-protocol specials (streaming-trained ckpts only).
        self.streaming_tts_id: int | None = vocab.get("<|streaming_tts|>")
        self.text_end_id: int | None = vocab.get("<|text_end|>")

    @property
    def tokenizer(self) -> Any:
        return self._tok

    def build_prompt(
        self,
        prompt_text: str,
        *,
        num_ref_tokens: int = 0,
        reference_text: str | None = None,
    ) -> list[int]:
        """``num_ref_tokens=0`` → zero-shot; non-zero must match delayed row count."""
        if num_ref_tokens < 0:
            raise ValueError(f"num_ref_tokens must be >= 0, got {num_ref_tokens}")
        ids: list[int] = [self.tts_id]
        if reference_text and num_ref_tokens > 0 and self.ref_text_id is not None:
            ids.append(self.ref_text_id)
            ids.extend(self._tok.encode(reference_text, add_special_tokens=False))
        if num_ref_tokens > 0:
            ids.append(self.ref_audio_id)
            ids.extend([AUDIO_PLACEHOLDER_ID] * num_ref_tokens)
        ids.append(self.text_id)
        ids.extend(self._tok.encode(prompt_text, add_special_tokens=False))
        ids.append(self.audio_id)
        return ids

    def build_streaming_prompt(
        self,
        prompt_text: str,
        *,
        num_ref_tokens: int = 0,
        reference_text: str | None = None,
    ) -> tuple[list[int], list[int]]:
        """Streaming-protocol preamble, mirroring higgs-mm
        ``build_streaming_preamble``:

        ``<|streaming_tts|> [<|ref_text|> tok(ref)] [<|ref_audio|> [-100]*N]
        <|text|> T0``

        Returns ``(prompt_ids, inject_ids)`` where ``prompt_ids`` end with the
        FIRST target-text token and ``inject_ids`` are the remaining target
        tokens injected one per protocol cycle during decode.
        """
        if self.streaming_tts_id is None or self.text_end_id is None:
            raise ValueError(
                "Tokenizer lacks <|streaming_tts|>/<|text_end|>; this "
                "checkpoint does not support the streaming TTS protocol."
            )
        if num_ref_tokens < 0:
            raise ValueError(f"num_ref_tokens must be >= 0, got {num_ref_tokens}")
        text_ids = self._tok.encode(prompt_text, add_special_tokens=False)
        if not text_ids:
            raise ValueError("streaming TTS requires non-empty input text")
        ids: list[int] = [self.streaming_tts_id]
        if reference_text and num_ref_tokens > 0 and self.ref_text_id is not None:
            ids.append(self.ref_text_id)
            ids.extend(self._tok.encode(reference_text, add_special_tokens=False))
        if num_ref_tokens > 0:
            ids.append(self.ref_audio_id)
            ids.extend([AUDIO_PLACEHOLDER_ID] * num_ref_tokens)
        ids.append(self.text_id)
        ids.append(text_ids[0])
        return ids, list(text_ids[1:])


__all__ = ["AUDIO_PLACEHOLDER_ID", "HiggsTokenizerAdapter"]
