# SPDX-License-Identifier: Apache-2.0
"""Utilities for exporting native Higgs training LoRA checkpoints to PEFT.

The deep-clone trainer writes a distributed checkpoint containing the complete
trainer state.  Serving only needs ``body.layers.*.lora_[AB]``.  Loading those
tensors selectively avoids materialising the 10+ GB base model or optimizer.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import torch

_DCP_MODEL_PREFIX = "trainer.model."
_BACKBONE_PREFIX = "body.layers."
_LORA_SUFFIXES = (".lora_A", ".lora_B")
_EXPORT_METADATA_FILE = "export_metadata.json"
_SUPPORTED_EXPORT_DTYPES = {torch.bfloat16, torch.float32}
HIGGS_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def validate_lora_adapter_model(
    adapter_dir: str | Path, expected_model_name: str
) -> dict[str, Any]:
    """Validate that an exported adapter targets the model being served."""
    adapter_dir = Path(adapter_dir).expanduser().resolve()
    metadata_path = adapter_dir / _EXPORT_METADATA_FILE
    if not metadata_path.is_file():
        raise ValueError(
            f"LoRA adapter metadata is missing: {metadata_path}; "
            "re-export the adapter with model_name metadata"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Invalid LoRA adapter metadata: {metadata_path}: {exc}"
        ) from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid LoRA adapter metadata object: {metadata_path}")
    model_name = metadata.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError(
            f"LoRA adapter metadata has no valid model_name: {metadata_path}"
        )
    if model_name != expected_model_name:
        raise ValueError(
            f"LoRA adapter model_name {model_name!r} does not match current "
            f"serve model name {expected_model_name!r}"
        )
    return metadata


def dcp_name_to_peft(name: str) -> str | None:
    """Map one Higgs trainer tensor name to a standard PEFT adapter name."""
    if name.startswith(_DCP_MODEL_PREFIX):
        name = name[len(_DCP_MODEL_PREFIX) :]
    if not name.startswith(_BACKBONE_PREFIX) or not name.endswith(_LORA_SUFFIXES):
        return None
    name = "model.layers." + name[len(_BACKBONE_PREFIX) :]
    return f"base_model.model.{name}.weight"


def inspect_dcp_lora(checkpoint_dir: str | Path) -> tuple[dict[str, Any], list[str]]:
    """Return backbone tensor metadata and non-backbone LoRA tensor names."""
    checkpoint_dir = Path(checkpoint_dir)
    metadata_path = checkpoint_dir / ".metadata"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"DCP metadata not found: {metadata_path}")
    # DCP metadata is a pickle produced by torch.distributed.checkpoint. Only
    # inspect checkpoints from a trusted training pipeline; pickle can execute
    # arbitrary code while deserializing an untrusted file.
    with metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)

    backbone: dict[str, Any] = {}
    ignored: list[str] = []
    for full_name, tensor_meta in metadata.state_dict_metadata.items():
        if not full_name.startswith(_DCP_MODEL_PREFIX) or not full_name.endswith(
            _LORA_SUFFIXES
        ):
            continue
        peft_name = dcp_name_to_peft(full_name)
        if peft_name is None:
            ignored.append(full_name)
        else:
            backbone[full_name[len(_DCP_MODEL_PREFIX) :]] = tensor_meta
    if not backbone:
        raise ValueError(f"No backbone LoRA tensors found in {metadata_path}")
    return backbone, ignored


def export_dcp_lora_adapter(
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    *,
    rank: int,
    alpha: float,
    model_name: str,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    """Selectively load a DCP LoRA and write a PEFT/SGLang adapter directory."""
    import torch.distributed.checkpoint as dcp
    from safetensors.torch import save_file

    if not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    if dtype not in _SUPPORTED_EXPORT_DTYPES:
        supported = ", ".join(sorted(str(value) for value in _SUPPORTED_EXPORT_DTYPES))
        raise ValueError(f"Unsupported LoRA export dtype {dtype}; choose {supported}")
    checkpoint_dir = Path(checkpoint_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    tensor_meta, ignored = inspect_dcp_lora(checkpoint_dir)
    native_state = {
        name: torch.empty(meta.size, dtype=meta.properties.dtype)
        for name, meta in tensor_meta.items()
    }
    state = {"trainer": {"model": native_state}}
    # Partial DCP restore reads only the ~252 MiB FP32 adapter, not the base
    # model or optimizer state.
    dcp.load(state, checkpoint_id=str(checkpoint_dir), no_dist=True)

    peft_state = {
        dcp_name_to_peft(name): tensor.detach()
        .to(dtype=dtype, device="cpu")
        .contiguous()
        for name, tensor in native_state.items()
    }
    if None in peft_state:
        raise AssertionError("Unexpected non-backbone LoRA tensor after filtering")
    if any(
        tensor.ndim != 2 or rank not in tensor.shape for tensor in peft_state.values()
    ):
        raise ValueError(f"At least one LoRA tensor does not contain rank {rank}")

    save_file(peft_state, output_dir / "adapter_model.safetensors")
    adapter_config = {
        "base_model_name_or_path": model_name,
        "bias": "none",
        "inference_mode": True,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": rank,
        "target_modules": list(HIGGS_LORA_TARGET_MODULES),
        "task_type": "CAUSAL_LM",
    }
    (output_dir / "adapter_config.json").write_text(
        json.dumps(adapter_config, indent=2) + "\n", encoding="utf-8"
    )
    export_metadata = {
        "model_name": model_name,
        "source_checkpoint": str(checkpoint_dir),
        "format": "peft-lora",
        "dtype": str(dtype).removeprefix("torch."),
        "tensor_count": len(peft_state),
        "rank": rank,
        "alpha": alpha,
        "ignored_non_backbone_lora_tensors": len(ignored),
        "note": (
            "Non-backbone tensors belong to the reference-audio semantic encoder; "
            "they are not used by reference-free backbone inference."
        ),
    }
    (output_dir / _EXPORT_METADATA_FILE).write_text(
        json.dumps(export_metadata, indent=2) + "\n", encoding="utf-8"
    )
    return export_metadata


__all__ = [
    "HIGGS_LORA_TARGET_MODULES",
    "dcp_name_to_peft",
    "export_dcp_lora_adapter",
    "inspect_dcp_lora",
    "validate_lora_adapter_model",
]
