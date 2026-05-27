# SPDX-License-Identifier: Apache-2.0
"""Streaming chunker under an LLM token stream (the voice-agent scenario).

Upstream is a language model emitting one token at a time; the server feeds each
token to the chunker and synthesizes a sentence the moment its boundary is
confirmed.
"""

from __future__ import annotations

import regex

from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig
from sglang_omni.models.higgs_tts.text_chunker import chunk_text
from sglang_omni.utils.streaming_text import StreamingTextOptions

# Crude BPE-like tokenizer: a control tag is one token; ASCII words carry their
# leading space (like Llama/GPT BPE); punctuation and CJK characters are one
# token each. Good enough to mimic how an LLM dribbles text out.
_FAKE_LLM_TOKENIZER = regex.compile(
    r"<\|[^|]+\|>"  # whole control tag
    r"|\s*[A-Za-z]+"  # word + leading space
    r"|\s*[0-9]+"  # number + leading space
    r"|\s*[^\sA-Za-z0-9]"  # one punctuation / CJK char + leading space
)


def _fake_llm_tokens(text: str) -> list[str]:
    return [m.group(0) for m in _FAKE_LLM_TOKENIZER.finditer(text)]


def _new_splitter():
    # The real, model-owned Higgs strategy (sentence chunker, default budget).
    return HiggsTtsPipelineConfig(
        model_path="dummy/path"
    ).create_streaming_text_splitter(StreamingTextOptions(split_granularity="sentence"))


# An LLM voice-agent reply: control tag, EN sentences, a long clause-heavy
# sentence (triggers sub-splitting), CJK, and a tail with no final punctuation.
SAMPLE = (
    "<|emotion:cheerful|>Sure! Here is the plan. "
    "First, we book the flight, then we reserve the hotel, "
    "and finally we rent a car for the entire two-week trip. "
    "听起来不错吧？我们出发吧。"
    "Talk soon"
)

EXPECTED = [
    "<|emotion:cheerful|>Sure!",
    "<|emotion:cheerful|>Here is the plan.",
    "<|emotion:cheerful|>First, we book the flight, then we reserve the hotel,",
    "<|emotion:cheerful|>and finally we rent a car for the entire two-week trip.",
    "<|emotion:cheerful|>听起来不错吧？",
    "<|emotion:cheerful|>我们出发吧。",
    "<|emotion:cheerful|>Talk soon",
]


def test_llm_token_stream_emits_sentences_incrementally() -> None:
    splitter = _new_splitter()
    tokens = _fake_llm_tokens(SAMPLE)

    emitted: list[str] = []
    emitted_before_done = 0
    for tok in tokens:  # one LLM token at a time
        chunks = splitter.add_text(tok)
        emitted.extend(chunks)
        emitted_before_done += len(chunks)
    emitted.extend(splitter.flush())  # input.done (LLM hit EOS)

    assert emitted == EXPECTED
    # Sentences are emitted mid-stream, not all at the end (only the
    # punctuation-less tail "Talk soon" waits for flush).
    assert emitted_before_done == len(EXPECTED) - 1
    # The active emotion state prefixes every chunk (never cleared in SAMPLE).
    assert all(c.startswith("<|emotion:cheerful|>") for c in emitted)


def test_framing_is_irrelevant() -> None:
    """Token-by-token, char-by-char, and one-shot all yield identical chunks."""

    def run(pieces):
        sp = _new_splitter()
        out = []
        for p in pieces:
            out += sp.add_text(p)
        return out + sp.flush()

    by_token = run(_fake_llm_tokens(SAMPLE))
    by_char = run(list(SAMPLE))  # the extreme: one character per message
    one_shot = chunk_text(SAMPLE, split_granularity="sentence")

    assert by_token == by_char == one_shot == EXPECTED
