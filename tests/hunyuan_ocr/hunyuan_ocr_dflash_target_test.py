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
    HunyuanOCRTargetVerifyController,
    HunyuanOCRTextExportMeta,
    XHHunYuanOCRModel,
    XHHunYuanOCRModelConfig,
)
from xhmodel_merak.xh_llm.types import ModelSwitcher
from xhmodel_merak.xh_llm.vision_llm_model import VisionLLMModel


EXPORT_TARGET_SCRIPT = Path("examples_merak/llm/hunyuan_ocr/export_dflash_target_hmonnx.py")
def _load_script(path: Path):
    spec = importlib.util.spec_from_file_location("hunyuan_ocr_dflash_target_export_testmod", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_target_checkpoint(root: Path, *, hidden_size: int = 4, num_hidden_layers: int = 4) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["HunYuanVLForConditionalGeneration"],
                "text_config": {
                    "hidden_size": hidden_size,
                    "num_hidden_layers": num_hidden_layers,
                    "max_position_embeddings": 64,
                },
                "vision_config": {"cat_extra_token": True},
            }
        ),
        encoding="utf-8",
    )
    return root


def _write_dflash_checkpoint(
    root: Path,
    *,
    target_layer_ids: list[int],
    hidden_size: int = 4,
    num_target_layers: int = 4,
    fc_in_features: int | None = None,
) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": hidden_size,
                "num_target_layers": num_target_layers,
                "dflash_config": {"target_layer_ids": target_layer_ids},
            }
        ),
        encoding="utf-8",
    )
    if fc_in_features is None:
        fc_in_features = len(target_layer_ids) * hidden_size
    save_file({"fc.weight": torch.zeros(hidden_size, fc_in_features)}, root / "model.safetensors")
    return root


def _build_config(target: Path, draft: Path) -> XHHunYuanOCRModelConfig:
    return XHHunYuanOCRModelConfig(
        model_name="hunyuan_ocr_dflash_test",
        hf_model=str(target),
        spec_decode_mode="dflash",
        dflash_config={"hf_model": str(draft)},
    )


def test_dflash_target_contract_is_loaded_from_checkpoint_config(tmp_path: Path) -> None:
    target = _write_target_checkpoint(tmp_path / "target")
    draft = _write_dflash_checkpoint(tmp_path / "draft", target_layer_ids=[0, 3])

    config = _build_config(target, draft)

    assert config.spec_decode_mode == "dflash"
    assert config.dflash_target_contract == {
        "config_path": str(draft / "config.json"),
        "config_sha256": config.dflash_target_contract["config_sha256"],
        "target_layer_ids": [0, 3],
        "target_hidden_size": 4,
        "target_hidden_concat_size": 8,
        "hidden_layout": "concat_last_dim",
        "reference_dtype": "bfloat16",
        "deployment_dtype": "float16",
    }
    assert len(config.dflash_target_contract["config_sha256"]) == 64


def test_dflash_target_contract_drives_wrap_and_export_outputs(tmp_path: Path) -> None:
    target = _write_target_checkpoint(tmp_path / "target")
    draft = _write_dflash_checkpoint(tmp_path / "draft", target_layer_ids=[0, 3])
    model = XHHunYuanOCRModel(_build_config(target, draft))
    model.kvcache_config.num_layers = 4

    assert model.wrap_cfg["output_hidden_state_indices"] == [0, 3]
    assert model.get_export_cfg()["output_names"] == ["logits", "target_hidden"]
    assert model.config.num_draft_tokens == 15
    assert model.config.verify_input_length == 16


def test_w16_quantization_processes_prefill_and_decode_graphs_separately(
    monkeypatch,
    tmp_path: Path,
) -> None:
    target = _write_target_checkpoint(tmp_path / "target")
    draft = _write_dflash_checkpoint(tmp_path / "draft", target_layer_ids=[0, 3])
    model = XHHunYuanOCRModel(_build_config(target, draft))
    model._verify_frontend_model = object()
    frontend_model = ModelSwitcher({"prefill": object(), "decode": object()})
    frontend_model.set_activate_model("prefill")
    calls = []

    def fake_base_to_quanted(self, graph, state, **kwargs):
        calls.append((graph, state, kwargs))
        return ("quanted", graph)

    monkeypatch.setattr(VisionLLMModel, "_to_quanted", fake_base_to_quanted)
    monkeypatch.setattr(model, "set_prefill", lambda: calls.append("prefill"))
    monkeypatch.setattr(model, "set_decode", lambda: calls.append("decode"))

    quanted = model._to_quanted(frontend_model, "aligned")

    assert calls == [
        "prefill",
        (frontend_model.prefill, "aligned", {}),
        "decode",
        (frontend_model.decode, "aligned", {}),
        (model._verify_frontend_model, "aligned", {}),
        "prefill",
    ]
    assert isinstance(quanted, ModelSwitcher)
    assert quanted.prefill == ("quanted", frontend_model.prefill)
    assert quanted.decode == ("quanted", frontend_model.decode)
    assert quanted.activate_model is quanted.prefill
    assert model._verify_quanted_model == ("quanted", model._verify_frontend_model)


def test_dflash_text_metadata_v2_round_trips_target_contract(tmp_path: Path) -> None:
    target = _write_target_checkpoint(tmp_path / "target")
    draft = _write_dflash_checkpoint(tmp_path / "draft", target_layer_ids=[0, 3])
    contract = _build_config(target, draft).dflash_target_contract
    export_dir = tmp_path / "export"
    for relative_path in (
        "hf_config/config.json",
        "quant_embedding.pt",
        "prefill/model.onnx",
        "prefill/model_external_data/weights.bin",
        "decode/model.onnx",
        "decode/model_external_data/weights.bin",
    ):
        path = export_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake")

    metadata = HunyuanOCRTextExportMeta(
        schema_version=2,
        model_config={"model_type": "HunYuanVLForConditionalGeneration"},
        hf_config="hf_config",
        quant_embedding="quant_embedding.pt",
        prefill_hmonnx="prefill/model.onnx",
        prefill_external_data="prefill/model_external_data",
        decode_hmonnx="decode/model.onnx",
        decode_external_data="decode/model_external_data",
        max_sequence_length=64,
        prefill_chunk_length=4,
        kv_cache={
            "num_layers": 4,
            "kv_cache_shape": [1, 8, 64, 128],
            "cache_axis": 2,
            "cache_dtype": "float16",
        },
        output_names=["logits", "target_hidden"],
        spec_decode={
            "status": "target_only",
            "mode": "dflash",
            "hidden_output_name": "target_hidden",
            **{key: value for key, value in contract.items() if key != "config_path"},
        },
    )

    meta_path = Path(metadata.save(export_dir / "text_meta_info.json"))
    loaded = HunyuanOCRTextExportMeta.from_file(meta_path)

    assert loaded.schema_version == 2
    assert loaded.output_names == ["logits", "target_hidden"]
    assert loaded.spec_decode["target_layer_ids"] == [0, 3]
    assert loaded.spec_decode["hidden_layout"] == "concat_last_dim"


def test_plain_ar_metadata_v1_still_rejects_target_hidden_output(tmp_path: Path) -> None:
    data = {
        "schema_version": 1,
        "meta": {"class_name": "HunyuanOCRTextExportMeta"},
        "model_config": {"model_type": "HunYuanVLForConditionalGeneration"},
        "hf_config": "hf_config",
        "quant_embedding": "quant_embedding.pt",
        "prefill_hmonnx": "prefill/model.onnx",
        "prefill_external_data": "prefill/model_external_data",
        "decode_hmonnx": "decode/model.onnx",
        "decode_external_data": "decode/model_external_data",
        "max_sequence_length": 64,
        "prefill_chunk_length": 4,
        "position_id_names": list(HunyuanOCRTextExportMeta.POSITION_ID_NAMES),
        "cache_update_mode": "in_place",
        "output_names": ["logits", "target_hidden"],
        "kv_cache": {"num_layers": 4, "cache_axis": 2, "cache_dtype": "float16"},
        "logits_dtype": "float16",
    }

    with pytest.raises(ValueError, match="schema v1.*only the logits output"):
        HunyuanOCRTextExportMeta._validate_serialized_contract(data)


def test_target_only_export_cli_enables_dflash_without_creating_workflow_yaml(tmp_path: Path) -> None:
    module = _load_script(EXPORT_TARGET_SCRIPT)

    values = module.build_model_config(tmp_path / "checkpoint")

    assert values["spec_decode_mode"] == "dflash"
    assert values["dflash_config"]["hf_model"] == str(tmp_path / "checkpoint" / "dflash")
    assert values["num_logits_to_keep"] == 1
    assert values["quant_scheme"]["quant_type"] == "w16a16h0_sefp"


@pytest.mark.parametrize(
    ("target_layer_ids", "fc_in_features", "message"),
    [
        ([0, 0], None, "must not contain duplicates"),
        ([1, 0], None, "must be strictly increasing"),
        ([0, 4], None, "outside the target layer range"),
        ([0, 3], 7, "fc.weight shape mismatch"),
    ],
)
def test_dflash_target_contract_fails_before_model_loading(
    tmp_path: Path,
    target_layer_ids: list[int],
    fc_in_features: int | None,
    message: str,
) -> None:
    target = _write_target_checkpoint(tmp_path / "target")
    draft = _write_dflash_checkpoint(
        tmp_path / "draft",
        target_layer_ids=target_layer_ids,
        fc_in_features=fc_in_features,
    )

    with pytest.raises(ValueError, match=message):
        _build_config(target, draft)


def test_target_verify_controller_commits_prefix_and_truncates_at_eos() -> None:
    controller = HunyuanOCRTargetVerifyController(
        max_sequence_length=64,
        verify_input_length=4,
        generation_eos_token_ids=[9],
    )
    controller.restore_request_state(past_seq_length=20, rope_delta=-2)

    result = controller.run_target_verify(
        current_token_id=7,
        draft_token_ids=[8, 9, 10],
        executor=lambda **_: (torch.zeros(1, 4, 8), torch.zeros(1, 4, 4)),
    )

    assert result.valid_length == 4
    assert controller.commit_verify_prefix(
        transaction_id=result.transaction_id,
        accepted_draft_count=3,
    ) == 23
    assert controller.logical_past_length == 23


def test_target_verify_controller_poison_requires_reset() -> None:
    controller = HunyuanOCRTargetVerifyController(max_sequence_length=64, verify_input_length=4)
    controller.restore_request_state(past_seq_length=10, rope_delta=0)

    def failing_executor(**_):
        raise RuntimeError("device failed")

    with pytest.raises(RuntimeError, match="verify_execution_failed"):
        controller.run_target_verify(current_token_id=1, draft_token_ids=[2], executor=failing_executor)
    with pytest.raises(RuntimeError, match="verify_request_poisoned"):
        controller.run_target_verify(
            current_token_id=1,
            draft_token_ids=[2],
            executor=lambda **_: (torch.zeros(1, 4, 8), torch.zeros(1, 4, 4)),
        )

    controller.reset_generation_state()
    controller.restore_request_state(past_seq_length=10, rope_delta=0)
    result = controller.run_target_verify(
        current_token_id=1,
        draft_token_ids=[2],
        executor=lambda **_: (torch.zeros(1, 4, 8), torch.zeros(1, 4, 4)),
    )
    assert controller.commit_verify_prefix(transaction_id=result.transaction_id, accepted_draft_count=0) == 11