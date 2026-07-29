# SPDX-License-Identifier: Apache-2.0
"""Benchmark-specific WER/CER normalization and aggregation."""

from __future__ import annotations

import importlib.util
import json
import re
import string
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

_ZH_PUNCTUATION = (
    "＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～"
    "｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿"
    "–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。"
)
_BENCHMARK_PUNCTUATION = _ZH_PUNCTUATION + string.punctuation

_NORM_CONFIG_PATH = Path(__file__).with_name("_omnilingual_norm_config.py")
_NORM_SPEC = importlib.util.spec_from_file_location(
    "_tts_eval_omnilingual_norm_config", _NORM_CONFIG_PATH
)
if _NORM_SPEC is None or _NORM_SPEC.loader is None:
    raise RuntimeError(f"Cannot load normalization config from {_NORM_CONFIG_PATH}")
_NORM_MODULE = importlib.util.module_from_spec(_NORM_SPEC)
_NORM_SPEC.loader.exec_module(_NORM_MODULE)
_NORM_CONFIG = _NORM_MODULE.norm_config

_MINIMAX_ISO1_TO_ISO3 = {
    "ar": "arb",
    "cs": "ces",
    "de": "deu",
    "el": "ell",
    "en": "eng",
    "es": "spa",
    "fi": "fin",
    "fr": "fra",
    "hi": "hin",
    "id": "ind",
    "it": "ita",
    "ja": "jpn",
    "ko": "kor",
    "nl": "nld",
    "pl": "pol",
    "pt": "por",
    "ro": "ron",
    "ru": "rus",
    "th": "tha",
    "tr": "tur",
    "uk": "ukr",
    "vi": "vie",
    "zh": "cmn",
    "yue": "yue",
}


def _edit_components(reference: str, hypothesis: str) -> tuple[int, int]:
    ref = reference.split()
    hyp = hypothesis.split()
    previous = list(range(len(hyp) + 1))
    for ref_index, ref_token in enumerate(ref, start=1):
        current = [ref_index]
        for hyp_index, hyp_token in enumerate(hyp, start=1):
            substitution = previous[hyp_index - 1] + (ref_token != hyp_token)
            current.append(min(previous[hyp_index] + 1, current[-1] + 1, substitution))
        previous = current
    return previous[-1], len(ref)


def _strip_benchmark_punctuation(text: str) -> str:
    for value in _BENCHMARK_PUNCTUATION:
        if value != "'":
            text = text.replace(value, "")
    return text.replace("  ", " ")


def seedtts_components(ref: str, hyp: str, lang: str) -> tuple[int, int]:
    ref = _strip_benchmark_punctuation(ref)
    hyp = _strip_benchmark_punctuation(hyp)
    if lang == "zh":
        ref, hyp = " ".join(ref), " ".join(hyp)
    elif lang == "en":
        ref, hyp = ref.lower(), hyp.lower()
    else:
        raise ValueError(f"SeedTTS only supports en/zh, got {lang!r}")
    return _edit_components(ref, hyp)


def cv3_components(ref: str, hyp: str, lang: str) -> tuple[int, int]:
    ref = _strip_benchmark_punctuation(ref)
    hyp = _strip_benchmark_punctuation(hyp)
    if lang in {"zh", "ja", "ko"}:
        ref, hyp = " ".join(ref), " ".join(hyp)
    else:
        ref, hyp = ref.lower(), hyp.lower()
    return _edit_components(ref, hyp)


def _text_normalize(
    text: str,
    iso_code: str,
    *,
    lower_case: bool = True,
    remove_numbers: bool = False,
    remove_brackets: bool = False,
) -> str:
    config = {**_NORM_CONFIG["*"], **_NORM_CONFIG.get(iso_code, {})}
    text = unicodedata.normalize(config["unicode_norm"], text)
    if config["lower_case"] and lower_case:
        text = text.lower()
    text = re.sub(r"\([^)]*\d[^)]*\)", " ", text)
    if remove_brackets:
        text = re.sub(r"\([^)]*\)", " ", text)
    for old, new in config["mapping"].items():
        text = re.sub(old, new, text)
    text = re.sub(r"[" + config["punc_set"] + "]", " ", text)
    text = re.sub(r"[" + config["del_set"] + "]", "", text)
    if remove_numbers:
        digits = "[" + config["digit_set"] + "]+"
        text = re.sub(
            rf"^{digits}(?=\s)|(?<=\s){digits}(?=\s)|(?<=\s){digits}$", " ", text
        )
    if config["rm_diacritics"]:
        try:
            from unidecode import unidecode
        except ImportError as exc:
            raise RuntimeError(
                "MiniMax normalization needs `pip install Unidecode`"
            ) from exc
        text = unidecode(text)
    return re.sub(r"\s+", " ", text).strip()


def _minimax_post_process(text: str, lang: str) -> str:
    iso_code = _MINIMAX_ISO1_TO_ISO3.get(lang)
    if iso_code is not None:
        text = _text_normalize(text, iso_code)
    if lang in {"zh", "yue"}:
        try:
            import zhconv
        except ImportError as exc:
            raise RuntimeError(
                "Chinese MiniMax normalization needs `pip install zhconv`"
            ) from exc
        text = zhconv.convert(text, "zh-cn")
    if lang in {"zh", "yue", "ja"}:
        text = " ".join(text.replace(" ", ""))
    elif lang in {"ko", "th", "ar", "vi", "hi", "el"}:
        text = " ".join(text.replace(" ", "|"))
    return text.lower().strip()


def minimax_components(ref: str, hyp: str, lang: str) -> tuple[int, int]:
    return _edit_components(
        _minimax_post_process(ref, lang),
        _minimax_post_process(hyp, lang),
    )


_SCORERS: dict[str, Callable[[str, str, str], tuple[int, int]]] = {
    "seedtts": seedtts_components,
    "cv3": cv3_components,
    "minimax": minimax_components,
}


def score_benchmark(benchmark: str, rows: Iterable[dict]) -> dict:
    """Score transcript rows and return blog-compatible macro WER/CER."""
    benchmark = benchmark.lower()
    scorer = _SCORERS[benchmark]
    per_language: dict[str, list[dict]] = defaultdict(list)
    scored_rows: list[dict] = []
    for row in rows:
        if not row.get("is_success", True) or row.get("transcript") is None:
            continue
        lang = str(row["lang"])
        edits, ref_len = scorer(str(row["target_text"]), str(row["transcript"]), lang)
        scored = {
            **row,
            "wer_edits": edits,
            "wer_ref_len": ref_len,
            "wer": edits / max(1, ref_len),
        }
        per_language[lang].append(scored)
        scored_rows.append(scored)
    if not scored_rows:
        raise ValueError("No successful transcript rows to score")

    language_metrics = {}
    for lang, values in sorted(per_language.items()):
        total_edits = sum(row["wer_edits"] for row in values)
        total_ref = sum(row["wer_ref_len"] for row in values)
        language_metrics[lang] = {
            "samples": len(values),
            "wer_cer_mean": sum(row["wer"] for row in values) / len(values),
            "wer_cer_micro": total_edits / max(1, total_ref),
        }
    macro = sum(v["wer_cer_mean"] for v in language_metrics.values()) / len(
        language_metrics
    )
    micro = sum(row["wer_edits"] for row in scored_rows) / max(
        1, sum(row["wer_ref_len"] for row in scored_rows)
    )
    return {
        "benchmark": benchmark,
        "languages": len(language_metrics),
        "evaluated": len(scored_rows),
        "wer_cer_macro": macro,
        "wer_cer_x100": macro * 100,
        "wer_cer_micro": micro,
        "per_language": language_metrics,
        "rows": scored_rows,
    }


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
