#!/usr/bin/env python3
"""Export a higgs-mm DCP LoRA checkpoint for SGLang per-request serving."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sglang_omni.models.higgs_tts.lora import export_dcp_lora_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--output-dir",
        help="Output adapter directory (default: <checkpoint-dir>/peft)",
    )
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=64.0)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="Saved LoRA weight dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--model-name",
        default="bosonai/higgs-tts-3-4b",
        help=(
            "Base/serving model name written to both PEFT config and export metadata"
        ),
    )
    args = parser.parse_args()
    output_dir = args.output_dir or str(Path(args.checkpoint_dir) / "peft")
    result = export_dcp_lora_adapter(
        args.checkpoint_dir,
        output_dir,
        rank=args.rank,
        alpha=args.alpha,
        model_name=args.model_name,
        dtype={"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype],
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
