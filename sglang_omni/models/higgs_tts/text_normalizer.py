# SPDX-License-Identifier: Apache-2.0
"""Punctuation normalization for Higgs TTS prompts.

Maps CJK / full-width punctuation to its ASCII form (。→ ., ，→ ,, …) so the
model sees one punctuation style regardless of input locale. Applied per chunk
right before tokenization; spoken content is unchanged.
"""

from __future__ import annotations

# Single char → ASCII.
_PUNCT_MAP = {
    "。": ".",
    "，": ", ",
    "！": "!",
    "？": "?",
    "；": ";",
    "：": ": ",
    "～": "~",
    "（": "(",
    "）": ")",
    "【": "[",
    "】": "]",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
}

_TRANSLATION = {ord(k): v for k, v in _PUNCT_MAP.items()}


def normalize_punctuation(text: str) -> str:
    """Replace CJK / full-width punctuation with ASCII equivalents."""
    return text.translate(_TRANSLATION)


__all__ = ["normalize_punctuation"]
