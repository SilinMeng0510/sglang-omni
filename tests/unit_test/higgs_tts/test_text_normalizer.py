# SPDX-License-Identifier: Apache-2.0
from sglang_omni.models.higgs_tts.text.normalizer import normalize_punctuation


def test_cjk_sentence_punctuation_to_ascii():
    # Terminators map to their ASCII form plus a trailing space; the CJK comma
    # 、，map to a bare space (no ASCII comma) so they read as a soft pause.
    assert normalize_punctuation("你好，世界。") == "你好 世界. "
    assert normalize_punctuation("真的吗？太好了！") == "真的吗? 太好了! "


def test_comma_maps_to_space_colon_keeps_ascii():
    # ，、collapse to a plain space; ：；keep their ASCII form + trailing space.
    assert normalize_punctuation("一，二，三") == "一 二 三"
    assert normalize_punctuation("顿、号") == "顿 号"
    assert normalize_punctuation("项目：内容") == "项目: 内容"
    assert normalize_punctuation("分号；连接") == "分号; 连接"


def test_quotes_and_brackets():
    # Quotes map tight; each bracket gets a space on its outer side
    # (（→ " (", ）→ ") ", 【→ " [", 】→ "] "), so ）【 yields two spaces.
    assert normalize_punctuation("他说“你好”") == '他说"你好"'
    assert normalize_punctuation("（注）【重点】") == ' (注)  [重点] '


def test_ascii_text_unchanged():
    s = "Hello, world. Really? Yes!"
    assert normalize_punctuation(s) == s


def test_empty_string():
    assert normalize_punctuation("") == ""
