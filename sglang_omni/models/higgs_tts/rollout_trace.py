# SPDX-License-Identifier: Apache-2.0
"""Serialize a Higgs rollout (delayed codes + logprobs) into the
``meta_info.omni_rollout`` schema (issue #780 §1.1.1): one ``higgs_codes``
discrete action stream with actions, logprobs, and the trainable-action mask.
"""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.models.higgs_tts.utils import delay_pattern_action_mask

OMNI_ROLLOUT_VERSION = 1


def build_omni_rollout_trace(
    delayed_codes: torch.Tensor,
    *,
    num_codebooks: int,
    codebook_vocab_size: int,
    delayed_logprobs: torch.Tensor | None = None,
    model_family: str = "higgs_tts",
    stage: str = "tts_engine",
    stream_name: str = "higgs_codes",
) -> dict[str, Any]:
    """Build the ``meta_info.omni_rollout`` dict from a delayed ``[L, N]`` code
    matrix and aligned ``[L, N]`` selected-action logprobs (``None`` if not
    requested).

    The logprob contract is the sampler's canonical Higgs RL signal: fp32
    selected-action values from full-vocab ``log_softmax(logits / T)`` at the
    sampled code. Greedy rows (``temperature ~= 0`` or ``top_k == 1``) use raw
    logits. Raises ``ValueError`` on shape disagreement or a non-finite logprob
    at a trainable action position.
    """
    if delayed_codes.ndim != 2:
        raise ValueError(
            f"delayed_codes must be 2-D [L, N], got shape {tuple(delayed_codes.shape)}"
        )
    L, N = delayed_codes.shape
    if N != num_codebooks:
        raise ValueError(
            f"delayed_codes has {N} codebooks but num_codebooks={num_codebooks}"
        )

    action_mask = delay_pattern_action_mask(delayed_codes)  # [L, N] bool

    if delayed_logprobs is not None:
        if tuple(delayed_logprobs.shape) != (L, N):
            raise ValueError(
                f"delayed_logprobs shape {tuple(delayed_logprobs.shape)} != "
                f"codes shape {(L, N)}"
            )
        # Every trainable action must carry a finite logprob.
        action_logprobs = delayed_logprobs[action_mask]
        if action_logprobs.numel() and not bool(torch.isfinite(action_logprobs).all()):
            raise ValueError(
                "non-finite logprob at a trainable action position; "
                "rollout logprob capture is inconsistent with the action mask"
            )

    stream: dict[str, Any] = {
        "name": stream_name,
        "stage": stage,
        "modality": "audio",
        "action_type": "discrete",
        "layout": "codebook_2d",
        "flatten_order": "time_major",
        "shape": [int(L), int(N)],
        "vocab_size": int(codebook_vocab_size),
        "actions": delayed_codes.to(torch.long).tolist(),
        "logprobs": (
            delayed_logprobs.to(torch.float32).tolist()
            if delayed_logprobs is not None
            else None
        ),
        "action_mask": action_mask.to(torch.int64).tolist(),
        # Omitted: action_mask==0 already marks every non-trainable position.
        "deterministic_mask": None,
        "channel_ids": list(range(N)),
        "channel_roles": [f"codebook_{c}" for c in range(N)],
    }

    return {
        "version": OMNI_ROLLOUT_VERSION,
        "model_family": model_family,
        "stages": [stage],
        "total_action_count": int(action_mask.sum().item()),
        "action_streams": [stream],
        "non_action_outputs": [],
    }


def append_streaming_rollout(
    trace: dict[str, Any],
    *,
    decision_actions: list[int],
    decision_logprobs: list[float] | None,
    protocol_trace: list,
    text_token_id: int,
    audio_token_id: int,
    frames_per_block: int,
    stage: str = "tts_engine",
) -> dict[str, Any]:
    """Extend an offline trace with streaming-protocol data: the trainable
    opening decisions (waits after the force-inserted first ``<|text|>``,
    then the block-opening ``<|audio|>``; 0=wait, 1=audio; greedy under the
    constrained two-way softmax) and the full step layout needed to replay
    the interleaved trajectory. Everything else (forced boundary ``<|audio|>``
    markers, injected text, discarded feed outputs) stays masked by omission.
    """
    k = len(decision_actions)
    if decision_logprobs is not None and len(decision_logprobs) != k:
        raise ValueError(
            f"decision logprobs length {len(decision_logprobs)} != actions {k}"
        )
    if k:
        trace["action_streams"].append(
            {
                "name": "opening_decisions",
                "stage": stage,
                "modality": "text",
                "action_type": "discrete",
                "layout": "sequence",
                "flatten_order": "time_major",
                "shape": [k],
                "vocab_size": 2,
                "actions": [int(a) for a in decision_actions],
                "logprobs": (
                    [float(v) for v in decision_logprobs]
                    if decision_logprobs is not None
                    else None
                ),
                "action_mask": [1] * k,
                "deterministic_mask": [1] * k,
                "channel_ids": [0],
                "channel_roles": ["opening_decision"],
                "action_token_ids": {
                    "0": int(text_token_id),
                    "1": int(audio_token_id),
                },
            }
        )
        trace["total_action_count"] = int(trace["total_action_count"]) + k
    trace["non_action_outputs"].append(
        {
            "name": "streaming_protocol_layout",
            "trace": [list(event) for event in protocol_trace],
            "text_token_id": int(text_token_id),
            "audio_token_id": int(audio_token_id),
            "frames_per_block": int(frames_per_block),
        }
    )
    return trace


__all__ = [
    "OMNI_ROLLOUT_VERSION",
    "append_streaming_rollout",
    "build_omni_rollout_trace",
]
