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

    @property
    def tokenizer(self) -> Any:
        return self._tok

    def build_prompt(
        self,
        prompt_text: str,
        *,
        num_ref_tokens: int = 0,
        reference_text: str | None = None,
        history: list[tuple[list[int], int]] | None = None,
    ) -> list[int]:
        """String-taking wrapper over :meth:`build_prompt_from_ids` (tokenizes
        ``prompt_text`` / ``reference_text`` first)."""
        ref_text_ids = (
            self._tok.encode(reference_text, add_special_tokens=False)
            if reference_text
            else None
        )
        return self.build_prompt_from_ids(
            self._tok.encode(prompt_text, add_special_tokens=False),
            num_ref_tokens=num_ref_tokens,
            reference_text_ids=ref_text_ids,
            history=history,
        )

    def build_prompt_from_ids(
        self,
        prompt_token_ids: list[int],
        *,
        num_ref_tokens: int = 0,
        reference_text_ids: list[int] | None = None,
        history: list[tuple[list[int], int]] | None = None,
    ) -> list[int]:
        """Assemble the prompt from already-tokenized pieces.
        ``num_ref_tokens=0`` → zero-shot; non-zero must match the delayed ref
        row count.

        Each ``history`` pair ``(text_ids_i, num_audio_rows_i)`` (a prior chunk,
        chronological) becomes a ``<|text|> tok(t_i) <|audio|> [-100]×N_i`` block
        between the reference and the new prompt; ``N_i`` must equal chunk
        ``i``'s delayed row count (``T_i + num_codebooks - 1``). Empty/``None``
        history → byte-identical to the single-shot prompt.
        """
        if num_ref_tokens < 0:
            raise ValueError(f"num_ref_tokens must be >= 0, got {num_ref_tokens}")
        ids: list[int] = [self.tts_id]
        if reference_text_ids and num_ref_tokens > 0 and self.ref_text_id is not None:
            ids.append(self.ref_text_id)
            ids.extend(reference_text_ids)
        if num_ref_tokens > 0:
            ids.append(self.ref_audio_id)
            ids.extend([AUDIO_PLACEHOLDER_ID] * num_ref_tokens)
        for hist_text_ids, hist_num_audio_rows in history or []:
            if hist_num_audio_rows < 0:
                raise ValueError(
                    "history audio-row counts must be >= 0, got "
                    f"{hist_num_audio_rows}"
                )
            ids.append(self.text_id)
            ids.extend(hist_text_ids)
            ids.append(self.audio_id)
            ids.extend([AUDIO_PLACEHOLDER_ID] * hist_num_audio_rows)
        ids.append(self.text_id)
        ids.extend(prompt_token_ids)
        ids.append(self.audio_id)
        return ids


__all__ = ["AUDIO_PLACEHOLDER_ID", "HiggsTokenizerAdapter"]
