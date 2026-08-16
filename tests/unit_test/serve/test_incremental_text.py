# SPDX-License-Identifier: Apache-2.0
"""Tests for canonical safe-prefix incremental tokenization."""

from __future__ import annotations

import os
import random
import re

import pytest

from sglang_omni.serve.incremental_text import StablePrefixTokenizer

_CKPT_TOKENIZER = os.environ.get(
    "HIGGS_STREAMING_TOKENIZER_JSON",
    "/hot-data/checkpoints/TTS/927fdf6881844516b1682f3134c06c9a/step_08000/tokenizer.json",
)

# fake pre-tokenizer: a piece is [spaces]word, trailing spaces their own piece
_FAKE_PIECE_RE = re.compile(r"\s*\S+|\s+")


def _fake_pre_tokenize(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in _FAKE_PIECE_RE.finditer(text)]


def _fake_encode_full(text: str):
    """One id per character."""
    return [ord(c) for c in text], [(i, i + 1) for i in range(len(text))]


def _make(**kwargs) -> StablePrefixTokenizer:
    return StablePrefixTokenizer(_fake_encode_full, _fake_pre_tokenize, **kwargs)


def test_complete_pieces_release_immediately():
    # large guard suppresses soft commits: only piece completion releases
    tok = _make(guard_tokens=100)
    assert tok.push("Hel") == []
    assert tok.push("lo") == []
    # the space starts a new piece -> "Hello" is complete and releases
    assert tok.push(" wor") == [ord(c) for c in "Hello"]
    assert tok.pending_text == " wor"
    assert tok.released_text == "Hello"


def test_soft_commit_inside_trailing_piece():
    tok = _make(guard_tokens=4)
    # first sight of the piece: no previous tokenization to agree with
    assert tok.push("abcdefgh") == []
    # second push: LCP with previous = 8, guard keeps 4 behind the tip of 9
    assert tok.push("i") == [ord(c) for c in "abcde"]
    assert tok.pending_text == "fghi"
    assert tok.released_text == "abcde"
    # committed chars stay part of the piece and are not re-released
    assert tok.push("j") == [ord("f")]
    assert tok.flush() == [ord(c) for c in "ghij"]
    assert tok.released_text == "abcdefghij"
    assert tok.pending_text == ""


def test_flush_releases_everything():
    tok = _make()
    tok.push("no boundary here")
    tail = tok.flush()
    assert tail == [ord(c) for c in " here"]  # "no boundary" released at spaces
    assert tok.pending_text == ""
    assert tok.flush() == []


def test_reconstruction_invariant_under_random_chunking():
    rng = random.Random(7)
    text = "The quick brown fox. 你好，世界！Jumps over 13 lazy dogs?  end"
    tok = _make()
    out: list[int] = []
    i = 0
    while i < len(text):
        step = rng.randint(1, 5)
        out.extend(tok.push(text[i : i + step]))
        i += step
        assert tok.released_text + tok.pending_text == text[:i]
    out.extend(tok.flush())
    assert "".join(chr(t) for t in out) == text


def test_guard_breach_seals_and_recovers():
    """An encoder whose early ids flip once the piece grows long must trip
    the committed-prefix verification, seal, and keep the text lossless."""

    def breach_encode_full(text: str):
        ids = [ord(c) for c in text]
        if len(text) >= 10:
            ids[0] = 9999  # canonical tokenization rewrites the first token
        return ids, [(i, i + 1) for i in range(len(text))]

    tok = StablePrefixTokenizer(breach_encode_full, _fake_pre_tokenize, guard_tokens=4)
    tok.push("abcdefgh")
    released = tok.push("i")  # commits "abcde"
    assert released == [ord(c) for c in "abcde"]
    out = tok.push("jk")  # len 11 -> id flip inside the committed prefix
    assert tok.guard_breaches == 1
    tail = tok.flush()
    # committed text was sealed; remainder re-tokenized fresh (len < 10)
    assert tok.released_text == "abcdefghijk"
    assert "".join(chr(t) for t in out + tail) == "fghijk"


def test_forced_reseal_bounds_buffer_growth():
    tok = _make(guard_tokens=2, max_buffer_chars=16)
    for ch in "abcdefghijklmnopqrstuvwxyz":
        tok.push(ch)
    assert tok.forced_reseals >= 1
    assert len(tok.pending_text) <= 16
    tok.flush()
    assert tok.released_text == "abcdefghijklmnopqrstuvwxyz"


@pytest.mark.skipif(
    not os.path.exists(_CKPT_TOKENIZER), reason="streaming ckpt tokenizer not available"
)
class TestWithRealTokenizer:
    @pytest.fixture(scope="class")
    def raw(self):
        from tokenizers import Tokenizer

        return Tokenizer.from_file(_CKPT_TOKENIZER)

    @pytest.fixture(scope="class")
    def make(self, raw):
        def _factory() -> StablePrefixTokenizer:
            def encode_full(s: str):
                enc = raw.encode(s, add_special_tokens=False)
                return enc.ids, enc.offsets

            def pre_tokenize(s: str):
                return [span for _, span in raw.pre_tokenizer.pre_tokenize_str(s)]

            return StablePrefixTokenizer(encode_full, pre_tokenize)

        return _factory

    @pytest.fixture(scope="class")
    def canonical(self, raw):
        def _enc(s: str) -> list[int]:
            return raw.encode(s, add_special_tokens=False).ids

        return _enc

    TEXTS = [
        "The quick brown fox jumps over the lazy dog. It keeps latency low "
        "even for long passages, doesn't it?",
        "你好，这是一段流式语音合成的测试。导航开始，全程二十公里，预计需要十二分钟。",
        "Mixed 中英 mixed text with numbers 3.14 and abbreviations e.g. NASA! "
        "以及很长的一段没有标点的中文内容测试回退切分逻辑是否可靠",
        "こんにちは、音声合成のストリーミングテストです。サーバーの遅延を確認します",
        "internationalization antidisestablishmentarianism "
        "Donaudampfschifffahrtsgesellschaftskapitän 12345678901234567890",
        "spaced   out\t\ttabs\n\nand    newlines end",
    ]

    def test_char_by_char_stream_is_exactly_canonical(self, make, canonical):
        for text in self.TEXTS:
            tok = make()
            out: list[int] = []
            for ch in text:
                out.extend(tok.push(ch))
            out.extend(tok.flush())
            # not merely lossless: token-for-token identical to the offline
            # full-text tokenization (zero distribution shift), and the
            # empirical guard was never breached
            assert out == canonical(text), text
            assert tok.guard_breaches == 0
            assert tok.forced_reseals == 0

    def test_random_chunking_is_exactly_canonical(self, make, canonical):
        rng = random.Random(0)
        for text in self.TEXTS:
            for _ in range(5):
                tok = make()
                out: list[int] = []
                i = 0
                while i < len(text):
                    step = rng.randint(1, 7)
                    out.extend(tok.push(text[i : i + step]))
                    i += step
                out.extend(tok.flush())
                assert out == canonical(text), text
                assert tok.guard_breaches == 0

    def test_unpunctuated_cjk_releases_continuously(self, make, canonical):
        """A boundary-free CJK run is a single pre-token: only the guard
        commit path can release it, and it must do so long before the end."""
        text = "人工智能语音合成技术在实时对话场景中的应用越来越广泛而且效果显著"
        tok = make()
        out: list[int] = []
        first_release_at = None
        for i, ch in enumerate(text):
            released = tok.push(ch)
            if released and first_release_at is None:
                first_release_at = i
            out.extend(released)
        out.extend(tok.flush())
        assert first_release_at is not None and first_release_at <= 12
        assert out == canonical(text)
        assert tok.guard_breaches == 0

    def test_unpunctuated_kana_releases_continuously(self, make, canonical):
        # kana had no release window under the old vocab-derived scheme;
        # the pre-tokenizer + guard path must handle it like any script
        text = "きょうはとてもいいてんきですねさんぽにいきましょうかそれともうちでやすみましょうか"
        tok = make()
        out: list[int] = []
        first_release_at = None
        for i, ch in enumerate(text):
            released = tok.push(ch)
            if released and first_release_at is None:
                first_release_at = i
            out.extend(released)
        out.extend(tok.flush())
        assert first_release_at is not None and first_release_at <= 14
        assert out == canonical(text)
        assert tok.guard_breaches == 0

    def test_emoji_and_multibyte_are_lossless(self, make, raw):
        # ZWJ emoji split across byte-level tokens: the shared-character
        # cut check must keep every release on a character boundary
        text = "family: 👨‍👩‍👧‍👦 flags 🇯🇵🇺🇸 done. 好👍的"
        tok = make()
        out: list[int] = []
        for ch in text:
            out.extend(tok.push(ch))
        out.extend(tok.flush())
        assert raw.decode(out) == text

    def test_word_stream_is_exactly_canonical(self, make, canonical):
        # LLM-style word-sized chunks (the production arrival pattern)
        text = self.TEXTS[0]
        tok = make()
        out: list[int] = []
        for m in re.finditer(r"\S+\s*", text):
            out.extend(tok.push(m.group(0)))
        out.extend(tok.flush())
        assert out == canonical(text)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
