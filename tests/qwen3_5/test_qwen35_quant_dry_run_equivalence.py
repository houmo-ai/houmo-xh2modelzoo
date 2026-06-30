from __future__ import annotations

from pathlib import Path

import pytest

from xhmodel_merak.xh_llm.models.qwen3_5.quant_adapter import (
    build_qwen35_autoround_kwargs,
    build_qwen35_gptqmodel_kwargs,
)


@pytest.fixture()
def local_pile10k(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    dataset = tmp_path / "data" / "calib_data" / "NeelNanda-pile-10k.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_text('{"text":"offline pile sample"}\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return dataset


def test_xh2_dense_autoround_kwargs_match_gptqmodel_api(tmp_path: Path, local_pile10k: Path):
    kwargs = build_qwen35_autoround_kwargs(
        hf_model_dir="weights/Qwen3.5-9B",
        output_dir=str(tmp_path / "dense-ar"),
        device="cuda:0",
        quant_cfg={"algorithm": "gptqmodel",
            "method": "autoround", "bits": 4, "group_size": 64, "rotation": False},
        export_model_cfg={"model_type": "Qwen3_5ForConditionalGeneration"},
        workflow_seed=42,
    )
    assert kwargs["topology"] == "dense"
    assert kwargs["dataset"] == str(local_pile10k.resolve())
    assert kwargs["format"] == "auto_gptq"


def test_xh2_moe_autoround_kwargs_match_gptqmodel_api(tmp_path: Path, local_pile10k: Path):
    kwargs = build_qwen35_autoround_kwargs(
        hf_model_dir="weights/Qwen3.6-35B-A3B",
        output_dir=str(tmp_path / "moe-ar"),
        device="cuda:0",
        quant_cfg={"algorithm": "gptqmodel",
            "method": "autoround", "bits": 4, "group_size": 64, "rotation": False},
        export_model_cfg={"model_type": "Qwen3_5MoeForConditionalGeneration"},
        workflow_seed=42,
    )
    assert kwargs["topology"] == "moe"
    assert kwargs["batch_size"] == 8
    assert kwargs["gradient_accumulate_steps"] == 1
    assert kwargs["device_map"] == "0"
    assert kwargs["low_gpu_mem_usage"] is True
    assert kwargs["attn_bits"] == 8
    assert kwargs["shared_expert_bits"] == 8
    assert kwargs["dataset"] == str(local_pile10k.resolve())
    assert kwargs["format"] == "auto_round:gptqmodel"


def test_xh2_dense_gptqmodel_kwargs_have_readme_jsonl(tmp_path: Path):
    kwargs = build_qwen35_gptqmodel_kwargs(
        hf_model_dir="weights/Qwen3.5-9B",
        output_dir=str(tmp_path / "dense-gptq"),
        device="cuda:0",
        quant_cfg={"algorithm": "gptqmodel", "bits": 4, "group_size": 64, "rotation": False},
        export_model_cfg={"model_type": "Qwen3_5ForConditionalGeneration"},
        workflow_seed=42,
    )
    assert kwargs["method"] == "gptq"
    assert kwargs["calibration_jsonl"].endswith("Qwen3.5-27B.jsonl")
    assert kwargs["batch_size"] == 1
    assert kwargs["nsamples"] == 256
    assert kwargs["seqlen"] == 1024
    assert kwargs["rotation"] is False


def test_xh2_moe_gptqmodel_kwargs_have_readme_jsonl(tmp_path: Path):
    kwargs = build_qwen35_gptqmodel_kwargs(
        hf_model_dir="weights/Qwen3.6-35B-A3B",
        output_dir=str(tmp_path / "moe-gptq"),
        device="cuda:0",
        quant_cfg={"algorithm": "gptqmodel", "bits": 4, "group_size": 64, "rotation": False},
        export_model_cfg={"model_type": "Qwen3_5MoeForConditionalGeneration"},
        workflow_seed=42,
    )
    assert kwargs["method"] == "gptq"
    assert kwargs["calibration_jsonl"].endswith("Qwen3-Next-80B-A3B-Instruct.jsonl")
    assert kwargs["batch_size"] == 1
    assert kwargs["expert_down_bits"] == 5
    assert kwargs["nsamples"] == 512
