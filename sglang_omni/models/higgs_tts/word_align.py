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
first frame where ``P(state >= W_j) > 0.5`` and -- a serving extension, the
reference returns onsets only -- its end at the first frame where
``P(state >= G_{j+1}) > 0.5`` (the mass has left the word's dwell states, into
the gap or straight into word ``j + 1``; no gap means ``end == next onset``).

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


def plan_word_align(
    tokenizer: Any, text: str, text_cap: int, unit: str = "word", dwell: int | None = None
) -> dict[str, Any]:
    """Preprocessing-stage plan: the units of ``text`` with their token ranges
    in ``tokenizer.encode(text, add_special_tokens=False)`` and char spans.
    ``unit="word"``: words per ``text_words``, consecutive words that add no
    token of their own (CJK characters sharing one token) merged, since the
    probe cannot tell them apart. ``unit="token"``: every text token outside
    ``<|...|>`` tags is a unit (scripts without word boundaries, th/km/lo/my;
    clients regroup by the char spans); tokens with no content char
    (whitespace, punctuation) join the previous unit."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    n_text = len(enc["input_ids"])
    if n_text > text_cap:
        raise ValueError(
            f"word_timestamps supports at most {text_cap} text tokens, got {n_text}"
        )
    if unit == "token":
        tagged = [False] * len(text)
        for m in _TAG.finditer(text):
            tagged[m.start() : m.end()] = [True] * (m.end() - m.start())
        spans = [
            (s, e) for s, e in enc["offset_mapping"]
            if e > s and not any(tagged[s:e])
        ]
        ranges = word_token_ranges(enc["offset_mapping"], spans) if spans else None
    elif unit == "word":
        spans = text_words(text)
        ranges = word_token_ranges(enc["offset_mapping"], spans) if spans else None
    else:
        raise ValueError(f"word_timestamps_unit must be 'word' or 'token', got {unit!r}")
    if not ranges:
        raise ValueError("word_timestamps: no alignable word in the input text")
    words: list[str] = []
    merged: list[list[int]] = []
    chars: list[list[int]] = []
    for (s, e), (lo, hi) in zip(spans, ranges):
        content = any(_content(c) for c in text[s:e])
        if merged and (hi <= merged[-1][1] or not content):
            words[-1] += text[s:e]
            merged[-1][1] = max(merged[-1][1], hi)
            chars[-1][1] = e
            continue
        if not content:
            continue
        words.append(text[s:e])
        merged.append([lo, hi])
        chars.append([s, e])
    if not words:
        raise ValueError("word_timestamps: no alignable word in the input text")
    plan = {"words": words, "ranges": merged, "chars": chars, "n_text": n_text, "unit": unit}
    if dwell is not None:
        plan["dwell"] = int(dwell)
    return plan


class WordAlignHead:
    """The exported PointerHead on the GPU, BatchNorm folded into the linears,
    plus a per-sampler-row bank of projected text-token hiddens.

    The null logit is folded into the bank so one ``bmm`` yields every column:
    the audio projection is ``[audio(x); null(x)]`` (rank + 1), text column
    ``j`` is ``[text(x_j) / sqrt(rank); 0]`` and the null column (index
    ``text_cap``) is ``[0; 1]``; ``bias`` is 0 on the live columns and -inf
    elsewhere. ``score`` is then 4 kernels that run inside the decode CUDA
    graph (``HiggsTTSModel.forward``) indexed by the static row buffer."""

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
        inv_sd = torch.rsqrt(sd["norm.running_var"].float() + 1e-5)
        shift = sd["norm.running_mean"].float() * inv_sd

        def fold(name: str) -> tuple[torch.Tensor, torch.Tensor]:
            w, b = sd[f"{name}.weight"].float(), sd[f"{name}.bias"].float()
            return w * inv_sd, b - w @ shift

        w_audio, b_audio = fold("audio")
        w_null, b_null = fold("null")
        self.w_text, self.b_text = (t.to(device) for t in fold("text"))
        self.w_proj = torch.cat([w_audio, w_null]).to(device)
        self.b_proj = torch.cat([b_audio, b_null]).to(device)
        self.bank = torch.zeros(pool_size, text_cap + 1, self.rank + 1, device=device)
        self.bank[:, text_cap, self.rank] = 1.0
        self.bias = torch.full((pool_size, text_cap + 1), float("-inf"), device=device)

    @torch.no_grad()
    def set_text(self, row: int, h_text: torch.Tensor, lo: int) -> None:
        """Project the prompt's ``[J, D]`` text-token hiddens into row ``row``;
        columns ``[lo, J)`` and the null column score, the rest is masked."""
        n = h_text.shape[0]
        t = (h_text.float() @ self.w_text.T + self.b_text) / math.sqrt(self.rank)
        self.bank[row, : self.text_cap, : self.rank] = 0.0
        self.bank[row, :n, : self.rank] = t
        self.bias[row] = float("-inf")
        self.bias[row, lo:n] = 0.0
        self.bias[row, self.text_cap] = 0.0

    @torch.no_grad()
    def score(self, h: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """``[P, text_cap + 1]`` log-probs (last column = null) of ``P`` audio-step
        hiddens against their sampler rows' text banks."""
        a = h.float() @ self.w_proj.T + self.b_proj
        s = torch.bmm(self.bank[rows], a.unsqueeze(-1)).squeeze(-1) + self.bias[rows]
        return torch.log_softmax(s, -1)


class WordAlignRequest:
    """Per-request decoder state: word log-probs from the token log-probs, the
    causal forward filter, and the committed onsets / ends. ``step`` returns
    an entry when a word's onset commits (``start_ms``) and again, possibly
    frames later, when its end commits (``start_ms`` + ``end_ms``)."""

    def __init__(self, plan: dict[str, Any], *, dwell: int, p_adv: float) -> None:
        self.words: list[str] = list(plan["words"])
        self.chars = [tuple(c) for c in plan.get("chars", [])]
        dwell = max(1, int(plan.get("dwell", dwell)))
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
        self.end = np.full(n, -1)
        self.dwell = dwell
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
        tail = np.cumsum(p[::-1])[::-1]
        new = np.flatnonzero((self.onset < 0) & (tail[self.first] > 0.5))
        self.onset[new] = self.t
        done = np.flatnonzero((self.end < 0) & (tail[self.first + self.dwell] > 0.5))
        self.end[done] = self.t
        self.t += 1
        return [self._word(j) for j in sorted(set(new) | set(done))]

    def finish(self) -> list[dict[str, Any]]:
        """End of the audio: open onsets take the last frame, open ends the
        audio end."""
        if self.finished:
            return []
        self.finished = True
        left = np.flatnonzero(self.onset < 0)
        self.onset[left] = max(self.t - 1, 0)
        open_end = np.flatnonzero(self.end < 0)
        self.end[open_end] = self.t
        return [self._word(j) for j in sorted(set(left) | set(open_end))]

    def all_words(self) -> list[dict[str, Any]]:
        return [self._word(j) for j in range(len(self.words))]

    def _word(self, j: int) -> dict[str, Any]:
        w = {"index": int(j), "text": self.words[j], "start_ms": int(self.onset[j]) * FRAME_MS}
        if self.chars:
            w["start_char"], w["end_char"] = self.chars[j]
        if self.end[j] >= 0:
            w["end_ms"] = int(self.end[j]) * FRAME_MS
        return w


__all__ = [
    "FRAME_MS",
    "WordAlignHead",
    "WordAlignRequest",
    "plan_word_align",
    "text_words",
    "word_token_ranges",
]
