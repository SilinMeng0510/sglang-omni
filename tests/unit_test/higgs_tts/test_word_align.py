# SPDX-License-Identifier: Apache-2.0
"""Word-align probe helpers: token units, the online decoder against
the higgs-mm batch reference, and the folded head against the plain math."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from sglang_omni.models.higgs_tts.word_align import (
    FRAME_MS,
    WordAlignHead,
    WordAlignRequest,
    plan_word_align,
    release_words,
)


def _reference_online_onsets(wlp: np.ndarray, dwell: int = 2, p_adv: float = 0.1):
    """higgs_mm.eval.word_align.online_onsets, verbatim."""
    T, n = wlp.shape[0], wlp.shape[1] - 1
    col = [n] + [c for j in range(n) for c in [j] * dwell + [n]]
    first = np.array([1 + j * (dwell + 1) for j in range(n)])
    C = len(col)
    la = np.full(C, -np.inf)
    la[0], la[1] = wlp[0, n], wlp[0, 0]
    onset = np.full(n, -1)
    for t in range(T):
        if t:
            a1 = np.concatenate([[-np.inf], la[:-1]])
            a2 = np.full(C, -np.inf)
            a2[first[1:]] = la[first[1:] - 2]
            la = np.logaddexp.reduce(
                np.stack(
                    [la + math.log1p(-p_adv), a1 + math.log(p_adv), a2 + math.log(p_adv)]
                ),
                0,
            ) + wlp[t, col]
        p = np.exp(la - np.logaddexp.reduce(la))
        tail = np.cumsum(p[::-1])[::-1][first]
        onset[(onset < 0) & (tail > 0.5)] = t
    return np.where(onset < 0, T - 1, onset)


class _FakeTok:
    """Char-level tokenizer with one two-char token (``"ab"``)."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids, offs, i = [], [], 0
        while i < len(text):
            n = 2 if text[i : i + 2] == "ab" else 1
            ids.append(ord(text[i]))
            offs.append((i, i + n))
            i += n
        return {"input_ids": ids, "offset_mapping": offs}


def test_plan_word_align_token_units():
    # one unit per token; whitespace / punctuation tokens join the previous
    # one; tag tokens are masked; char spans come from the offsets
    plan = plan_word_align(_FakeTok(), "<|t|>ab c,d", text_cap=64, dwell=1)
    assert plan["words"] == ["ab ", "c,", "d"]
    assert plan["ranges"] == [[5, 7], [7, 9], [9, 10]]
    assert plan["chars"] == [[5, 8], [8, 10], [10, 11]]
    assert plan["n_text"] == 10 and plan["dwell"] == 1
    req = WordAlignRequest(plan, dwell=2, p_adv=0.1)
    assert req.dwell == 1 and req.lo == 5
    with pytest.raises(ValueError):
        plan_word_align(_FakeTok(), "x y", text_cap=2)
    with pytest.raises(ValueError):
        plan_word_align(_FakeTok(), "<|sfx:laughter|>", text_cap=64)


def test_online_decoder_matches_batch_reference():
    rng = np.random.default_rng(0)
    T, n_text, lo = 60, 12, 2
    ranges = [[2, 4], [4, 5], [5, 9], [9, 12]]
    plan = {"words": ["a", "b", "c", "d"], "ranges": ranges, "n_text": n_text}
    logits = rng.normal(size=(T, n_text + 1))
    logits[:, :lo] = -np.inf
    logp = logits - np.logaddexp.reduce(logits, -1, keepdims=True)
    req = WordAlignRequest(plan, dwell=2, p_adv=0.1)
    seen, ends = {}, {}
    for t in range(T):
        for w in req.step(logp[t]):
            seen.setdefault(w["index"], w["start_ms"])
            if "end_ms" in w:
                ends[w["index"]] = w["end_ms"]
                assert w["end_ms"] >= seen[w["index"]]
    for w in req.finish():
        seen.setdefault(w["index"], w["start_ms"])
        ends.setdefault(w["index"], w["end_ms"])
    assert sorted(ends) == [0, 1, 2, 3]
    assert all("end_ms" in w for w in req.all_words())
    # an end never precedes the next onset (the chain is monotone)
    assert all(ends[j] <= seen[j + 1] for j in range(3))
    # reference word log-probs over the span tokens
    cover = np.zeros((n_text - lo + 1, 5))
    for j, (a, b) in enumerate(ranges):
        cover[a - lo : b - lo, j] = 1
    cover[-1, -1] = 1
    span = np.concatenate([logp[:, lo:n_text], logp[:, -1:]], -1)
    wlp = np.log(np.exp(span) @ cover + 1e-30)
    ref = _reference_online_onsets(wlp)
    assert [seen[j] for j in range(4)] == [int(f) * FRAME_MS for f in ref]
    assert req.finish() == []


def test_head_folds_batchnorm_exactly(tmp_path):
    torch.manual_seed(0)
    dim, rank, cap = 16, 4, 8
    sd = {
        "norm.running_mean": torch.randn(dim) * 3,
        "norm.running_var": torch.rand(dim) * 5 + 0.1,
        "norm.num_batches_tracked": torch.tensor(1),
        "audio.weight": torch.randn(rank, dim),
        "audio.bias": torch.randn(rank),
        "text.weight": torch.randn(rank, dim),
        "text.bias": torch.randn(rank),
        "null.weight": torch.randn(1, dim),
        "null.bias": torch.randn(1),
    }
    torch.save({"layers": [3], "rank": rank, "dim": dim, "state_dicts": [sd]}, tmp_path / "head.pt")
    (tmp_path / "config.json").write_text(json.dumps({"dwell": 2, "p_adv": 0.1}))
    head = WordAlignHead(str(tmp_path), pool_size=3, device="cpu", text_cap=cap)
    assert head.layer == 3
    J, lo = 6, 1
    h_text, h_audio = torch.randn(J, dim), torch.randn(1, dim)
    head.set_text(2, h_text, lo)
    logp = head.score(h_audio, torch.tensor([2]))[0]
    # plain PointerHead math (higgs_mm.nn.word_align) in eval mode
    bn = lambda x: (x - sd["norm.running_mean"]) / torch.sqrt(sd["norm.running_var"] + 1e-5)
    a = bn(h_audio) @ sd["audio.weight"].T + sd["audio.bias"]
    t = bn(h_text) @ sd["text.weight"].T + sd["text.bias"]
    z = bn(h_audio) @ sd["null.weight"].T + sd["null.bias"]
    s = torch.cat([(a @ t.T / math.sqrt(rank))[0, lo:], z[0]])
    ref = torch.log_softmax(s, -1)
    assert torch.allclose(torch.cat([logp[lo:J], logp[-1:]]), ref, atol=1e-5)
    assert torch.isinf(logp[:lo]).all() and torch.isinf(logp[J:cap]).all()


def test_release_words_one_entry_per_index_per_event():
    pending = [
        {"index": 0, "text": "a", "start_ms": 120},
        {"index": 0, "text": "a", "start_ms": 120, "end_ms": 320},
        {"index": 1, "text": "b", "start_ms": 320},
        {"index": 1, "text": "b", "start_ms": 320, "end_ms": 640},
        {"index": 2, "text": "c", "start_ms": 700},
    ]
    ready, left = release_words(pending, 400)
    # onset + end of index 0 merge into one entry; index 1's end (640) waits
    assert ready == [{"index": 0, "text": "a", "start_ms": 120, "end_ms": 320},
                     {"index": 1, "text": "b", "start_ms": 320}]
    assert left == pending[3:]
    ready, left = release_words(left, 800)
    assert ready == [{"index": 1, "text": "b", "start_ms": 320, "end_ms": 640},
                     {"index": 2, "text": "c", "start_ms": 700}] and left == []
