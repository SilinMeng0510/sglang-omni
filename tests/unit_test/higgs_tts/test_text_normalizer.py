# SPDX-License-Identifier: Apache-2.0
from sglang_omni.models.higgs_tts.text_normalizer import normalize_punctuation


def test_cjk_sentence_punctuation_to_ascii():
    assert normalize_punctuation("你好，世界。") == "你好,世界."
    assert normalize_punctuation("真的吗？太好了！") == "真的吗?太好了!"


def test_quotes_and_brackets():
    assert normalize_punctuation("他说“你好”") == '他说"你好"'
    assert normalize_punctuation("（注）【重点】") == '(注)[重点]'


def test_ascii_text_unchanged():
    s = "Hello, world. Really? Yes!"
    assert normalize_punctuation(s) == s


def test_empty_string():
    assert normalize_punctuation("") == ""
