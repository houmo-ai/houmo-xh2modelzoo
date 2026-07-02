from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from xhmodel_merak.xh_llm.models.qwen3_5.quant_adapter import (
    build_qwen35_autoround_kwargs,
    build_qwen35_gptqmodel_kwargs,
    quantize_with_autoround_api,
    quantize_with_gptqmodel_api,
    validate_rotation_mtp_compatibility,
)


DENSE_EXPORT = {"model_type": "Qwen3_5ForConditionalGeneration", "use_mtp": False}
MOE_EXPORT = {"model_type": "Qwen3_5MoeForConditionalGeneration", "use_mtp": False}


def _local_pile10k(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    dataset = tmp_path / "data" / "calib_data" / "NeelNanda-pile-10k.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_text('{"text":"offline pile sample"}\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return dataset


def test_build_autoround_dense_kwargs_matches_script_defaults(tmp_path, monkeypatch):
    dataset = _local_pile10k(tmp_path, monkeypatch)
    kwargs = build_qwen35_autoround_kwargs(
        hf_model_dir="weights/Qwen3.5-9B",
        output_dir="work_dirs/qwen35_dense_ar",
        device="cuda:0",
        quant_cfg={
            "algorithm": "gptqmodel",
            "method": "autoround",
            "bits": 4,
            "group_size": 64,
            "calibration": {"dataset": "NeelNanda/pile-10k", "nsamples": 128, "seqlen": 2048},
            "runtime": {"batch_size": 8},
            "sym": True,
            "iters": 200,
            "format": "auto_gptq",
        },
        export_model_cfg=DENSE_EXPORT,
        workflow_seed=1024,
    )
    assert kwargs["topology"] == "dense"
    assert kwargs["bits"] == 4
    assert kwargs["group_size"] == 64
    assert kwargs["dataset"] == str(dataset.resolve())
    assert kwargs["nsamples"] == 128
    assert kwargs["seqlen"] == 2048
    assert kwargs["batch_size"] == 8
    assert kwargs["format"] == "auto_gptq"


def test_build_autoround_moe_kwargs_matches_script_defaults(tmp_path, monkeypatch):
    dataset = _local_pile10k(tmp_path, monkeypatch)
    kwargs = build_qwen35_autoround_kwargs(
        hf_model_dir="weights/Qwen3.6-35B-A3B",
        output_dir="work_dirs/qwen35_moe_ar",
        device="cuda:0",
        quant_cfg={
            "algorithm": "gptqmodel",
            "method": "autoround",
            "bits": 4,
            "group_size": 64,
            "calibration": {"dataset": "NeelNanda/pile-10k", "nsamples": 128, "seqlen": 2048},
            "runtime": {
                "batch_size": 8,
                "gradient_accumulate_steps": 1,
                "device_map": "0",
                "low_gpu_mem_usage": True,
            },
            "moe": {"attn_bits": 8, "shared_expert_bits": 8},
            "sym": True,
            "iters": 200,
            "format": "auto_round:gptqmodel",
        },
        export_model_cfg=MOE_EXPORT,
        workflow_seed=1024,
    )
    assert kwargs["topology"] == "moe"
    assert kwargs["batch_size"] == 8
    assert kwargs["gradient_accumulate_steps"] == 1
    assert kwargs["device_map"] == "0"
    assert kwargs["low_gpu_mem_usage"] is True
    assert kwargs["attn_bits"] == 8
    assert kwargs["shared_expert_bits"] == 8
    assert kwargs["dataset"] == str(dataset.resolve())
    assert kwargs["format"] == "auto_round:gptqmodel"


def test_autoround_default_dataset_requires_local_pile10k(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="data/calib_data/NeelNanda-pile-10k.jsonl"):
        build_qwen35_autoround_kwargs(
            hf_model_dir="weights/Qwen3.5-9B",
            output_dir="work_dirs/qwen35_dense_ar",
            device="cuda:0",
            quant_cfg={"algorithm": "gptqmodel", "method": "autoround", "bits": 4, "group_size": 64},
            export_model_cfg=DENSE_EXPORT,
            workflow_seed=1024,
        )


def test_autoround_default_dataset_finds_data_calib_data(tmp_path, monkeypatch):
    dataset = tmp_path / "data" / "calib_data" / "NeelNanda-pile-10k.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_text('{"text":"offline pile sample"}\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    kwargs = build_qwen35_autoround_kwargs(
        hf_model_dir="weights/Qwen3.5-9B",
        output_dir="work_dirs/qwen35_dense_ar",
        device="cuda:0",
        quant_cfg={"algorithm": "gptqmodel", "method": "autoround", "bits": 4, "group_size": 64},
        export_model_cfg=DENSE_EXPORT,
        workflow_seed=1024,
    )

    assert kwargs["dataset"] == str(dataset.resolve())


def test_build_gptqmodel_dense_kwargs_uses_dense_jsonl():
    kwargs = build_qwen35_gptqmodel_kwargs(
        hf_model_dir="weights/Qwen3.5-9B",
        output_dir="work_dirs/qwen35_dense_gptq",
        device="cuda:0",
        quant_cfg={"algorithm": "gptqmodel", "bits": 4, "group_size": 64, "rotation": False},
        export_model_cfg=DENSE_EXPORT,
        workflow_seed=1024,
    )
    assert kwargs["method"] == "gptq"
    assert kwargs["topology"] == "dense"
    assert kwargs["rotation"] is False
    assert kwargs["calibration_jsonl"].endswith("Qwen3.5-27B.jsonl")
    assert kwargs["nsamples"] == 256
    assert kwargs["seqlen"] == 1024
    assert kwargs["wait_for_submodule_finalizers"] is True


def test_build_gptqmodel_moe_kwargs_uses_moe_jsonl_and_bits():
    kwargs = build_qwen35_gptqmodel_kwargs(
        hf_model_dir="weights/Qwen3.6-35B-A3B",
        output_dir="work_dirs/qwen35_moe_gptq",
        device="cuda:0",
        quant_cfg={"algorithm": "gptqmodel", "bits": 4, "group_size": 64, "rotation": False},
        export_model_cfg=MOE_EXPORT,
        workflow_seed=1024,
    )
    assert kwargs["topology"] == "moe"
    assert kwargs["calibration_jsonl"].endswith("Qwen3-Next-80B-A3B-Instruct.jsonl")
    assert kwargs["self_attn_bits"] == 8
    assert kwargs["shared_expert_bits"] == 8
    assert kwargs["expert_bits"] == 4
    assert kwargs["expert_down_bits"] == 5
    assert kwargs["nsamples"] == 512
    assert kwargs["wait_for_submodule_finalizers"] is True


def test_rotation_with_mtp_is_rejected():
    with pytest.raises(ValueError, match="rotation.*MTP"):
        validate_rotation_mtp_compatibility({"rotation": "hadamard"}, {"use_mtp": True})


def test_rotation_with_spec_decode_mode_is_rejected():
    with pytest.raises(ValueError, match="rotation.*MTP"):
        validate_rotation_mtp_compatibility({"rotation": "hadamard"}, {"spec_decode_mode": "dflash"})


def test_rotation_allows_disabled_spec_decode_strings():
    validate_rotation_mtp_compatibility({"rotation": "hadamard"}, {"spec_decode_mode": "none", "mtp_config": "false"})


def test_adapter_calls_autoround_api(tmp_path, monkeypatch):
    dataset = _local_pile10k(tmp_path, monkeypatch)
    calls = {}
    module = types.ModuleType("gptqmodel.recipes.qwen35_autoround")

    def fake_quantize_qwen35_autoround(**kwargs):
        calls.update(kwargs)
        return {"output_dir": kwargs["output_dir"]}

    module.quantize_qwen35_autoround = fake_quantize_qwen35_autoround
    monkeypatch.setitem(sys.modules, "gptqmodel.recipes.qwen35_autoround", module)

    result = quantize_with_autoround_api(
        model_dir="weights/Qwen3.5-9B",
        output_dir="work_dirs/out",
        device="cuda:0",
        quant_cfg={"algorithm": "gptqmodel",
            "method": "autoround", "bits": 4, "group_size": 64},
        export_model_cfg=DENSE_EXPORT,
        workflow_seed=42,
    )
    assert calls["model_dir"] == "weights/Qwen3.5-9B"
    assert calls["dataset"] == str(dataset.resolve())
    assert result.quanted_model_dir.endswith("work_dirs/out")


def test_adapter_calls_gptqmodel_api(monkeypatch):
    calls = {}
    module = types.ModuleType("gptqmodel.recipes.qwen35")

    def fake_quantize_qwen35(**kwargs):
        calls.update(kwargs)
        return {"output_dir": kwargs["output_dir"]}

    module.quantize_qwen35 = fake_quantize_qwen35
    monkeypatch.setitem(sys.modules, "gptqmodel.recipes.qwen35", module)

    result = quantize_with_gptqmodel_api(
        model_dir="weights/Qwen3.5-9B",
        output_dir="work_dirs/out",
        device="cuda:0",
        quant_cfg={"algorithm": "gptqmodel", "bits": 4, "group_size": 64, "rotation": False},
        export_model_cfg=DENSE_EXPORT,
        workflow_seed=42,
    )
    assert calls["method"] == "gptq"
    assert result.quanted_model_dir.endswith("work_dirs/out")


def test_gptqmodel_branch_ignores_inherited_autoround_runtime_defaults():
    kwargs = build_qwen35_gptqmodel_kwargs(
        hf_model_dir="weights/Qwen3.5-9B",
        output_dir="work_dirs/qwen35_dense_gptq",
        device="cuda:0",
        quant_cfg={
            "algorithm": "gptqmodel",
            "bits": 4,
            "group_size": 64,
            "method": "gptq",
            "rotation": False,
            "calibration": {"dataset": "NeelNanda/pile-10k", "nsamples": 128, "seqlen": 2048},
            "runtime": {"batch_size": 8},
        },
        export_model_cfg=DENSE_EXPORT,
        workflow_seed=1024,
    )

    assert kwargs["method"] == "gptq"
    assert kwargs["batch_size"] == 1
    assert kwargs["nsamples"] == 256
    assert kwargs["seqlen"] == 1024
    assert "calibration_dataset" not in kwargs


def test_gptqmodel_branch_honors_explicit_calibration_overrides():
    kwargs = build_qwen35_gptqmodel_kwargs(
        hf_model_dir="weights/Qwen3.5-9B",
        output_dir="work_dirs/qwen35_dense_gptq",
        device="cuda:0",
        quant_cfg={
            "algorithm": "gptqmodel",
            "bits": 4,
            "group_size": 64,
            "method": "gptq",
            "rotation": False,
            "calibration": {"dataset": "NeelNanda/pile-10k", "nsamples": 16, "seqlen": 512},
            "runtime": {"batch_size": 8},
        },
        export_model_cfg=DENSE_EXPORT,
        workflow_seed=1024,
    )

    assert kwargs["method"] == "gptq"
    assert kwargs["batch_size"] == 1
    assert kwargs["nsamples"] == 16
    assert kwargs["seqlen"] == 512
