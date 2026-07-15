from __future__ import annotations

import json

import pytest

from sglang_omni.models.higgs_tts.lora import (
    dcp_name_to_peft,
    validate_lora_adapter_model,
)


def test_dcp_name_to_peft_maps_backbone_lora() -> None:
    assert (
        dcp_name_to_peft("trainer.model.body.layers.3.self_attn.q_proj.lora_A")
        == "base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight"
    )
    assert (
        dcp_name_to_peft("body.layers.35.mlp.down_proj.lora_B")
        == "base_model.model.model.layers.35.mlp.down_proj.lora_B.weight"
    )


def test_dcp_name_to_peft_rejects_codec_and_base_weights() -> None:
    assert (
        dcp_name_to_peft(
            "trainer.model.tied.embedding.modality_embeddings.0.model."
            "semantic_model.encoder.layers.0.attention.q_proj.lora_A"
        )
        is None
    )
    assert (
        dcp_name_to_peft("trainer.model.body.layers.0.self_attn.q_proj.weight") is None
    )


def test_validate_lora_adapter_model_matches_metadata(tmp_path) -> None:
    (tmp_path / "export_metadata.json").write_text(
        json.dumps({"model_name": "served-model"}), encoding="utf-8"
    )

    metadata = validate_lora_adapter_model(tmp_path, "served-model")

    assert metadata["model_name"] == "served-model"


def test_validate_lora_adapter_model_rejects_mismatch(tmp_path) -> None:
    (tmp_path / "export_metadata.json").write_text(
        json.dumps({"model_name": "other-model"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="does not match current serve model name"):
        validate_lora_adapter_model(tmp_path, "served-model")


def test_validate_lora_adapter_model_requires_metadata(tmp_path) -> None:
    with pytest.raises(ValueError, match="metadata is missing"):
        validate_lora_adapter_model(tmp_path, "served-model")
