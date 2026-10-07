# SPDX-License-Identifier: Apache-2.0
"""Word timestamps for Higgs TTS from a frozen-backbone pointer probe.

A trained ``PointerHead`` (copied from higgs-mm ``higgs_mm/nn/word_align.py``;
export layout from ``experiments/tts/word_align.py:write_export``) reads the
backbone's layer-``k`` hidden state at every audio step and scores it against
the prompt's text-token hiddens of the same layer::

    x = BatchNorm(h)            # running stats, folded into the linears below
    s[t, j] = audio(x_t) . text(x_j) / sqrt(rank)   over the chunk's text tokens
    s[t, null] = null(x_t)                          gap / silence

The causal decoder (copied from ``higgs_mm/eval/word_align.py:online_onsets``)
forward-filters the chain ``G0 W0 G1 .. W_{n-1} G_n`` (stay ``1 - p_adv``,
every word held >= ``dwell`` frames) and commits word ``j``'s onset at the
first frame where ``P(state >= W_j) > 0.5``.

Conventions (export ``config.json``): the hidden that PREDICTS frame ``t`` --
the decode position fed frame ``t-1``'s codebook row; frame 0 <- the
``<|audio|>`` prompt token -- scores frame ``t``; the softmax runs over the
text tokens from the first word's first token up to ``<|audio|>`` (head
control tags excluded, inline tags / punctuation left in as distractors);
onset frame ``t`` -> ``t * 40 ms`` from the start of the generated audio.
"""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from typing import Any

import numpy as np
import torch

FRAME_MS = 40
_TAG = re.compile(r"<\|[^|<>]*\|>")
_CJK_RANGES = (
    (0x3040, 0x30FF),  # hiragana, katakana
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0xAC00, 0xD7AF),  # hangul
    (0x20000, 0x2FA1F),
)


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in _CJK_RANGES)


def _content(ch: str) -> bool:
    return not ch.isspace() and unicodedata.category(ch)[0] not in "PSZ"


def text_words(text: str) -> list[tuple[int, int]]:
    """Char spans of the words of a prompt text: whitespace-delimited pieces
    outside ``<|...|>`` tags, every CJK character its own word (the server has
    no segmenter); punctuation-only pieces join the previous word."""
    plain = _TAG.sub(lambda m: " " * len(m.group()), text)
    pieces: list[tuple[int, int]] = []
    for m in re.finditer(r"\S+", plain):
        start = None
        for i in range(m.start(), m.end()):
            if _is_cjk(plain[i]):
                if start is not None:
                    pieces.append((start, i))
                    start = None
                pieces.append((i, i + 1))
            elif start is None:
                start = i
        if start is not None:
            pieces.append((start, m.end()))
    out: list[list[int]] = []
    for s, e in pieces:
        if out and not any(_content(c) for c in text[s:e]):
            out[-1][1] = e
        else:
            out.append([s, e])
    return [(s, e) for s, e in out if any(_content(c) for c in text[s:e])]


def word_token_ranges(
    offsets: list[tuple[int, int]], words: list[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    """[lo, hi) token range of every word: the tokens overlapping its chars;
    None if one has none."""
    s = np.array([o[0] for o in offsets])
    e = np.array([o[1] for o in offsets])
    out = []
    for c0, c1 in words:
        hit = np.flatnonzero((s < c1) & (e > c0) & (e > s))
        if hit.size == 0:
            return None
        out.append((int(hit[0]), int(hit[-1]) + 1))
    return out


def plan_word_align(tokenizer: Any, text: str, text_cap: int) -> dict[str, Any]:
    """Preprocessing-stage plan: the words of ``text`` with their token ranges
    in ``tokenizer.encode(text, add_special_tokens=False)``. Consecutive words
    that add no token of their own (CJK characters sharing one token) are
    merged, since the probe cannot tell them apart."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    n_text = len(enc["input_ids"])
    if n_text > text_cap:
        raise ValueError(
            f"word_timestamps supports at most {text_cap} text tokens, got {n_text}"
        )
    spans = text_words(text)
    ranges = word_token_ranges(enc["offset_mapping"], spans) if spans else None
    if not ranges:
        raise ValueError("word_timestamps: no alignable word in the input text")
    words: list[str] = []
    merged: list[list[int]] = []
    for (s, e), (lo, hi) in zip(spans, ranges):
        if merged and hi <= merged[-1][1]:
            words[-1] += text[s:e]
            continue
        words.append(text[s:e])
        merged.append([lo, hi])
    return {"words": words, "ranges": merged, "n_text": n_text}


class WordAlignHead:
    """The exported PointerHead on the GPU, BatchNorm folded into the linears,
    plus a per-sampler-row bank of projected text-token hiddens."""

    def __init__(
        self, export_dir: str, *, pool_size: int, device: Any, text_cap: int = 1024
    ) -> None:
        with open(os.path.join(export_dir, "config.json")) as f:
            cfg = json.load(f)
        ck = torch.load(os.path.join(export_dir, "head.pt"), map_location="cpu")
        if len(ck["layers"]) != 1:
            raise ValueError(f"serving takes a single-layer export, got {ck['layers']}")
        sd = ck["state_dicts"][0]
        self.layer = int(ck["layers"][0])
        self.rank = int(ck["rank"])
        self.hidden_size = int(ck["dim"])
        self.dwell = int(cfg.get("dwell", 2))
        self.p_adv = float(cfg.get("p_adv", 0.1))
        self.text_cap = int(text_cap)
        self.scale = 1.0 / math.sqrt(self.rank)
        inv_sd = torch.rsqrt(sd["norm.running_var"].float() + 1e-5)
        shift = sd["norm.running_mean"].float() * inv_sd

        def fold(name: str) -> tuple[torch.Tensor, torch.Tensor]:
            w, b = sd[f"{name}.weight"].float(), sd[f"{name}.bias"].float()
            return (w * inv_sd).to(device), (b - w @ shift).to(device)

        self.w_audio, self.b_audio = fold("audio")
        self.w_text, self.b_text = fold("text")
        self.w_null, self.b_null = fold("null")
        self.bank = torch.zeros(pool_size, text_cap, self.rank, device=device)
        self._col = torch.arange(text_cap, device=device)[None]

    @torch.no_grad()
    def set_text(self, row: int, h_text: torch.Tensor) -> None:
        """Project the prompt's ``[J, D]`` text-token hiddens into row ``row``."""
        self.bank[row, : h_text.shape[0]] = h_text.float() @ self.w_text.T + self.b_text

    @torch.no_grad()
    def score(
        self, h: torch.Tensor, rows: torch.Tensor, lo: torch.Tensor, n_text: torch.Tensor
    ) -> torch.Tensor:
        """``[P, text_cap + 1]`` log-probs (last column = null) of ``P`` audio-step
        hiddens against their rows' text banks, columns outside ``[lo, n_text)``
        masked."""
        x = h.float()
        a = x @ self.w_audio.T + self.b_audio
        z = x @ self.w_null.T + self.b_null
        s = torch.bmm(self.bank[rows], a.unsqueeze(-1)).squeeze(-1) * self.scale
        mask = (self._col < lo[:, None]) | (self._col >= n_text[:, None])
        s = s.masked_fill(mask, float("-inf"))
        return torch.log_softmax(torch.cat([s, z], -1), -1)


class WordAlignRequest:
    """Per-request decoder state: word log-probs from the token log-probs, the
    causal forward filter, and the committed onsets."""

    def __init__(self, plan: dict[str, Any], *, dwell: int, p_adv: float) -> None:
        self.words: list[str] = list(plan["words"])
        ranges = [tuple(r) for r in plan["ranges"]]
        self.n_text = int(plan["n_text"])
        self.lo = int(ranges[0][0])
        n, j_span = len(self.words), self.n_text - self.lo
        self.cover = np.zeros((j_span + 1, n + 1), dtype=np.float32)
        for j, (a, b) in enumerate(ranges):
            self.cover[a - self.lo : b - self.lo, j] = 1.0
        self.cover[j_span, n] = 1.0
        self.col = np.array([n] + [c for j in range(n) for c in [j] * dwell + [n]])
        self.first = np.array([1 + j * (dwell + 1) for j in range(n)])
        self.la = np.full(len(self.col), -np.inf)
        self.onset = np.full(n, -1)
        self.log_stay, self.log_adv = math.log1p(-p_adv), math.log(p_adv)
        self.t = 0
        self.finished = False

    def step(self, logp: np.ndarray) -> list[dict[str, Any]]:
        """Feed one frame's ``[text_cap + 1]`` token log-probs; returns the words
        whose onset this frame commits."""
        n = len(self.words)
        tok = np.concatenate([logp[self.lo : self.n_text], logp[-1:]]).astype(np.float64)
        wlp = np.log(np.exp(tok) @ self.cover + 1e-30)
        if self.t == 0:
            self.la[0], self.la[1] = wlp[n], wlp[0]
        else:
            a1 = np.concatenate([[-np.inf], self.la[:-1]])
            a2 = np.full(len(self.col), -np.inf)
            a2[self.first[1:]] = self.la[self.first[1:] - 2]
            self.la = np.logaddexp.reduce(
                np.stack([self.la + self.log_stay, a1 + self.log_adv, a2 + self.log_adv]), 0
            ) + wlp[self.col]
        p = np.exp(self.la - np.logaddexp.reduce(self.la))
        tail = np.cumsum(p[::-1])[::-1][self.first]
        new = np.flatnonzero((self.onset < 0) & (tail > 0.5))
        self.onset[new] = self.t
        self.t += 1
        return [self._word(j) for j in new]

    def finish(self) -> list[dict[str, Any]]:
        """End of the audio: the words still open take the last frame."""
        if self.finished:
            return []
        self.finished = True
        left = np.flatnonzero(self.onset < 0)
        self.onset[left] = max(self.t - 1, 0)
        return [self._word(j) for j in left]

    def all_words(self) -> list[dict[str, Any]]:
        return [self._word(j) for j in range(len(self.words))]

    def _word(self, j: int) -> dict[str, Any]:
        return {"index": int(j), "text": self.words[j], "start_ms": int(self.onset[j]) * FRAME_MS}


__all__ = [
    "FRAME_MS",
    "WordAlignHead",
    "WordAlignRequest",
    "plan_word_align",
    "text_words",
    "word_token_ranges",
]
