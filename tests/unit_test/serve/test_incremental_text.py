# SPDX-License-Identifier: Apache-2.0
"""Tests for stable-prefix incremental tokenization."""

from __future__ import annotations

import os
import random

import pytest

from sglang_omni.serve.incremental_text import (
    StablePrefixTokenizer,
    derive_hold_params,
)

_CKPT_TOKENIZER = os.environ.get(
    "HIGGS_STREAMING_TOKENIZER_JSON",
    "/hot-data/checkpoints/TTS/927fdf6881844516b1682f3134c06c9a/step_08000/tokenizer.json",
)


def _simple_encode(text: str) -> list[int]:
    """Deterministic fake encoder: one id per character."""
    return [ord(c) for c in text]


def test_holds_back_until_boundary():
    tok = StablePrefixTokenizer(_simple_encode)
    assert tok.push("Hel") == []
    assert tok.push("lo") == []
    # the space is a boundary: "Hello" is released, " wor" stays buffered
    released = tok.push(" wor")
    assert released == _simple_encode("Hello")
    assert tok.pending_text == " wor"


def test_space_attaches_forward():
    tok = StablePrefixTokenizer(_simple_encode)
    tok.push("Hello wor")
    # a second space closes " wor" -> " world" pattern
    released = tok.push("ld again")
    assert released == _simple_encode(" world")
    assert tok.pending_text == " again"


def test_cjk_punct_boundary_needs_following_char():
    tok = StablePrefixTokenizer(_simple_encode)
    assert tok.push("你好。") == []  # trailing punct could extend (e.g. 。。)
    released = tok.push("再")
    assert released == _simple_encode("你好。")
    assert tok.pending_text == "再"


def test_flush_releases_everything():
    tok = StablePrefixTokenizer(_simple_encode)
    tok.push("no boundary here")
    tail = tok.flush()
    assert tail == _simple_encode(" here")  # after "no boundary" released at space
    assert tok.pending_text == ""
    assert tok.flush() == []


def test_fallback_cut_on_overlong_unpunctuated_run():
    tok = StablePrefixTokenizer(_simple_encode, max_hold_chars=8, fallback_hold_chars=2)
    text = "计算机科学与技术专业"  # 10 chars, no punct
    released = tok.push(text)
    assert released == _simple_encode(text[:-2])
    assert tok.pending_text == text[-2:]


def test_released_plus_flush_reconstructs_text():
    tok = StablePrefixTokenizer(_simple_encode)
    text = "The quick brown fox. 你好，世界！Jumps over 13 lazy dogs?"
    out: list[int] = []
    for ch in text:
        out.extend(tok.push(ch))
    out.extend(tok.flush())
    assert out == _simple_encode(text) or "".join(chr(t) for t in out) == text


@pytest.mark.skipif(
    not os.path.exists(_CKPT_TOKENIZER), reason="streaming ckpt tokenizer not available"
)
class TestWithRealTokenizer:
    @pytest.fixture(scope="class")
    def encode(self):
        from tokenizers import Tokenizer

        raw = Tokenizer.from_file(_CKPT_TOKENIZER)

        def _enc(s: str) -> list[int]:
            return raw.encode(s, add_special_tokens=False).ids

        return _enc

    @pytest.fixture(scope="class")
    def decode(self):
        from tokenizers import Tokenizer

        raw = Tokenizer.from_file(_CKPT_TOKENIZER)

        def _dec(ids: list[int]) -> str:
            return raw.decode(ids)

        return _dec

    TEXTS = [
        "The quick brown fox jumps over the lazy dog. It keeps latency low "
        "even for long passages, doesn't it?",
        "你好，这是一段流式语音合成的测试。导航开始，全程二十公里，预计需要十二分钟。",
        "Mixed 中英 mixed text with numbers 3.14 and abbreviations e.g. NASA! "
        "以及很长的一段没有标点的中文内容测试回退切分逻辑是否可靠",
    ]

    def test_char_by_char_matches_canonical_at_boundaries(self, encode, decode):
        for text in self.TEXTS:
            tok = StablePrefixTokenizer(encode)
            out: list[int] = []
            for ch in text:
                out.extend(tok.push(ch))
            out.extend(tok.flush())
            # released stream must decode back to the exact original text
            assert decode(out) == text

    def test_random_chunking_is_deterministic_and_lossless(self, encode, decode):
        rng = random.Random(0)
        for text in self.TEXTS:
            for _ in range(5):
                tok = StablePrefixTokenizer(encode)
                out: list[int] = []
                i = 0
                while i < len(text):
                    step = rng.randint(1, 7)
                    out.extend(tok.push(text[i : i + step]))
                    i += step
                out.extend(tok.flush())
                assert decode(out) == text

    def test_derived_hold_covers_vocab_tokens(self):
        from tokenizers import Tokenizer

        raw = Tokenizer.from_file(_CKPT_TOKENIZER)
        hold = derive_hold_params(raw)
        # the forced-cut holdback must cover the longest letter-bearing
        # vocab token (CJK tokens reach 7 chars in Qwen3; code-identifier
        # tokens reach the 40s), and stay within the configured cap
        assert 7 <= hold.fallback_hold_chars <= 64
        assert hold.max_hold_chars == hold.fallback_hold_chars * 3
        # the CJK window is much tighter than the global one
        assert 2 <= hold.cjk_hold_chars <= hold.fallback_hold_chars
        # cached: second call returns the same params
        assert derive_hold_params(raw) == hold

    def test_cjk_continuous_release_before_punctuation(self, encode, decode):
        """Unpunctuated Chinese must release well before any punctuation
        arrives, with the vocab-derived CJK holdback — and losslessly."""
        from tokenizers import Tokenizer

        raw = Tokenizer.from_file(_CKPT_TOKENIZER)
        hold = derive_hold_params(raw)

        def encode_full(s):
            enc = raw.encode(s, add_special_tokens=False)
            return enc.ids, enc.offsets

        text = "人工智能语音合成技术在实时对话场景中的应用越来越广泛而且效果显著"
        tok = StablePrefixTokenizer(
            encode,
            max_hold_chars=hold.max_hold_chars,
            fallback_hold_chars=hold.fallback_hold_chars,
            cjk_hold_chars=hold.cjk_hold_chars,
            encode_full=encode_full,
        )
        out: list[int] = []
        first_release_at = None
        for i, ch in enumerate(text):
            released = tok.push(ch)
            if released and first_release_at is None:
                first_release_at = i
            out.extend(released)
        out.extend(tok.flush())
        # released long before the end despite zero punctuation
        assert first_release_at is not None
        assert first_release_at <= hold.cjk_hold_chars + 3
        # and the stream is lossless
        assert decode(out) == text
        # holdback honored: pending tail stayed short after each release
        assert tok.pending_text == ""

    def test_space_boundary_cuts_are_canonical(self, encode):
        # pure-Latin text with spaces: incremental must equal full encode
        text = "Streaming text to speech keeps the time to first audio low"
        tok = StablePrefixTokenizer(encode)
        out: list[int] = []
        for ch in text:
            out.extend(tok.push(ch))
        out.extend(tok.flush())
        assert out == encode(text)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
