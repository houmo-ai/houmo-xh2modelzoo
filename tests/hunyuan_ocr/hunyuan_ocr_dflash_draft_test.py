# Copyright 2025 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from xhmodel_merak.xh_llm.models.hunyuan_ocr import (
    HunyuanOCRDFlashModel,
    HunyuanOCRDraftCacheController,
    XHHunYuanOCRDFlashConfig,
    dflash_graph_io_contract,
    load_hunyuan_ocr_dflash_checkpoint,
)


EXPORT_SCRIPT = Path("examples_merak/llm/hunyuan_ocr/export_dflash_draft_hmonnx.py")


def _load_export_script():
    spec = importlib.util.spec_from_file_location("hunyuan_ocr_dflash_draft_export_testmod", EXPORT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_checkpoints(tmp_path: Path, *, fc_width: int = 8) -> tuple[Path, Path]:
    draft = tmp_path / "draft"
    target = tmp_path / "target"
    draft.mkdir(parents=True)
    target.mkdir(parents=True)
    draft_config = {
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "num_hidden_layers": 5,
        "num_target_layers": 4,
        "vocab_size": 7,
        "block_size": 16,
        "eos_token_id": 6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "dtype": "float32",
        "dflash_config": {"target_layer_ids": [0, 3], "mask_token_id": 5},
    }
    (draft / "config.json").write_text(json.dumps(draft_config), encoding="utf-8")
    draft_state = {
        "fc.weight": torch.zeros(4, fc_width),
        "hidden_norm.weight": torch.ones(4),
        "norm.weight": torch.ones(4),
    }
    for layer_index in range(5):
        prefix = f"layers.{layer_index}."
        draft_state.update(
            {
                f"{prefix}input_layernorm.weight": torch.ones(4),
                f"{prefix}post_attention_layernorm.weight": torch.ones(4),
                f"{prefix}self_attn.q_proj.weight": torch.zeros(4, 4),
                f"{prefix}self_attn.k_proj.weight": torch.zeros(2, 4),
                f"{prefix}self_attn.v_proj.weight": torch.zeros(2, 4),
                f"{prefix}self_attn.o_proj.weight": torch.zeros(4, 4),
                f"{prefix}self_attn.q_norm.weight": torch.ones(2),
                f"{prefix}self_attn.k_norm.weight": torch.ones(2),
                f"{prefix}mlp.gate_proj.weight": torch.zeros(8, 4),
                f"{prefix}mlp.up_proj.weight": torch.zeros(8, 4),
                f"{prefix}mlp.down_proj.weight": torch.zeros(4, 8),
            }
        )
    save_file(draft_state, draft / "model.safetensors")

    target_config = {
        "text_config": {
            "hidden_size": 4,
            "vocab_size": 7,
            "tie_word_embeddings": True,
        }
    }
    (target / "config.json").write_text(json.dumps(target_config), encoding="utf-8")
    embedding = torch.arange(28, dtype=torch.float32).reshape(7, 4)
    save_file({"model.language_model.embed_tokens.weight": embedding}, target / "model.safetensors")
    return draft, target


def test_checkpoint_preflight_validates_shape_and_tied_output_head(tmp_path: Path) -> None:
    draft, target = _write_checkpoints(tmp_path)

    checkpoint = load_hunyuan_ocr_dflash_checkpoint(draft, target)

    assert checkpoint.config.target_layer_ids == (0, 3)
    assert checkpoint.output_head.tie_word_embeddings is True
    assert checkpoint.output_head.source_key == "model.language_model.embed_tokens.weight"
    assert checkpoint.output_head.weight.dtype == torch.float16
    assert checkpoint.output_head.deployment_dtype == "float16"

    bad_draft, bad_target = _write_checkpoints(tmp_path / "bad", fc_width=4)
    with pytest.raises(ValueError, match="fc.weight shape mismatch"):
        load_hunyuan_ocr_dflash_checkpoint(bad_draft, bad_target)


def test_cache_controller_commits_discards_rejects_stale_and_recovers_from_poison() -> None:
    controller = HunyuanOCRDraftCacheController(capacity=64)

    controller.commit_context(current_input_length=8)
    decode_id = controller.begin_decode()
    controller.discard_decode(transaction_id=decode_id)
    with pytest.raises(RuntimeError, match="stale transaction"):
        controller.discard_decode(transaction_id=decode_id)

    decode_id = controller.begin_decode()
    controller.begin_context_decode(accepted_draft_count=3, transaction_id=decode_id)
    assert controller.commit_context_decode(transaction_id=decode_id) == 12

    failed_id = controller.begin_decode()
    controller.mark_execution_failed(transaction_id=failed_id)
    assert controller.poisoned is True
    with pytest.raises(RuntimeError, match="poisoned"):
        controller.begin_decode()
    controller.reset()
    assert controller.committed_length == 0
    assert controller.poisoned is False


def test_three_graphs_have_fixed_io_contract() -> None:
    context = dflash_graph_io_contract(mode="context", num_hidden_layers=5)
    context_decode = dflash_graph_io_contract(mode="context_decode", num_hidden_layers=5)
    decode = dflash_graph_io_contract(mode="decode", num_hidden_layers=5)

    assert context == context_decode
    assert context["input_names"][:3] == ["target_hidden", "past_seq_length", "current_input_length"]
    assert context["output_names"] == [
        *[f"present_key_cache_{index}" for index in range(5)],
        *[f"present_value_cache_{index}" for index in range(5)],
    ]
    assert decode["input_names"][:4] == [
        "noise_embedding",
        "past_seq_length",
        "current_input_length",
        "attn_mask",
    ]
    assert decode["output_names"] == ["draft_logits"]


def test_dflash_core_and_export_config_keep_public_mode_contract() -> None:
    raw = {
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "num_hidden_layers": 5,
        "vocab_size": 7,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 64,
        "block_size": 16,
        "dflash_config": {"target_layer_ids": [0, 3]},
    }
    model = HunyuanOCRDFlashModel.from_config(raw, max_sequence_length=64)
    assert model.target_layer_ids == (0, 3)
    assert {
            mode: XHHunYuanOCRDFlashConfig(
            model_name=f"draft_{mode}",
            model_type="HunYuanOCR_DFlash_Draft",
            hf_model="/tmp/draft",
            target_model_dir="/tmp/target",
            mode=mode,
            context_max_length=64,
        ).input_sequence_length
        for mode in ("context", "context_decode", "decode")
    } == {"context": 256, "context_decode": 16, "decode": 16}


def test_exporter_preflight_and_metadata_upgrade_are_atomic(tmp_path: Path, monkeypatch) -> None:
    module = _load_export_script()
    from xhmodel_merak.xh_llm.models.hunyuan_ocr import HunyuanOCRTextExportMeta

    draft, target = _write_checkpoints(tmp_path / "models")
    export_dir = tmp_path / "export"
    for mode in ("context", "context_decode", "decode"):
        for suffix in ("model.onnx", "model_external_data"):
            artifact = export_dir / f"dflash_draft_{mode}" / suffix
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(b"fake")
    metadata = HunyuanOCRTextExportMeta(
        schema_version=2,
        model_config={"model_type": "HunYuanVLForConditionalGeneration"},
        max_sequence_length=64,
        generation_eos_token_id=[1, 2],
        output_names=["logits", "target_hidden"],
        spec_decode={
            "status": "target_verify_ready",
            "mode": "dflash",
            "target_layer_ids": [0, 3],
            "target_hidden_size": 4,
            "capabilities": {"target_verify": True},
        },
    )
    metadata_path = Path(metadata.save(export_dir / "text_meta_info.json"))
    preflight = module.preflight_draft_export(metadata_path, target, draft)
    artifacts = {
        mode: (
            export_dir / f"dflash_draft_{mode}" / "model.onnx",
            export_dir / f"dflash_draft_{mode}" / "model_external_data",
        )
        for mode in ("context", "context_decode", "decode")
    }

    module.upgrade_metadata_with_draft_graphs(preflight, artifacts=artifacts)
    upgraded = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert upgraded["spec_decode"]["status"] == "speculative_runtime_ready"
    loaded = HunyuanOCRTextExportMeta.from_file(metadata_path)
    assert loaded.dflash_context_hmonnx == str(export_dir / "dflash_draft_context/model.onnx")
    assert loaded.dflash_context_decode_hmonnx == str(export_dir / "dflash_draft_context_decode/model.onnx")
    assert loaded.dflash_decode_hmonnx == str(export_dir / "dflash_draft_decode/model.onnx")
    assert upgraded["spec_decode"]["draft"]["cache"]["capacity"] == 64
    assert upgraded["dflash_context_hmonnx"] == "dflash_draft_context/model.onnx"
    assert "--debug" not in EXPORT_SCRIPT.read_text(encoding="utf-8")