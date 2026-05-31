# SPDX-License-Identifier: Apache-2.0
"""Higgs audio tokenizer: the discrete audio codec.

``codec.py`` is the :class:`HiggsAudioCodec` facade (encode/decode + checkpoint
loading); ``tokenizer.py`` is the underlying model, vendored byte-for-byte from
https://huggingface.co/bosonai/higgs-audio-v2-tokenizer (droppable once
transformers >= 5.3.0 ships it), with its architecture config in ``config.json``.
"""

from sglang_omni.models.higgs_tts.audio.codec import HiggsAudioCodec

__all__ = ["HiggsAudioCodec"]
