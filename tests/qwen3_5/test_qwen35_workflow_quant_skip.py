from __future__ import annotations

from pathlib import Path

import yaml


def _write_workflow_config(path: Path, *, quant):
    data = {
        "quant": quant,
        "export": {
            "model": {
                "chip_arch": "XH2a",
                "model_type": "Qwen3_5ForConditionalGeneration",
                "hf_model": None,
                "model_name": "xh2_qwen3_5_test",
                "context_max_length": 2048,
                "prefill_chunk_length": 256,
                "quant_scheme": {"quant_type": "w8a8h1_sefp", "ops": {}},
            }
        },
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_qwen35_quant_null_in_workflow_config_skips_quantization(tmp_path: Path):
    from xhmodel_merak.xh_llm.models.qwen3_5.workflow import Qwen35Workflow

    hf_model_dir = tmp_path / "qwen35-hf"
    hf_model_dir.mkdir()
    config_path = _write_workflow_config(tmp_path / "qwen35_base.yaml", quant=None)
    workflow = Qwen35Workflow.from_config(str(hf_model_dir), str(config_path))

    quant_result = workflow.quant(output_dir=str(tmp_path / "quant"), device="cpu")

    assert quant_result.skipped is True
    assert quant_result.raw_model_dir == str(hf_model_dir.resolve())
    assert quant_result.quanted_model_dir is None


def test_qwen35_quant_null_override_skips_quantization(tmp_path: Path):
    from xhmodel_merak.xh_llm.models.qwen3_5.workflow import Qwen35Workflow

    hf_model_dir = tmp_path / "qwen35-hf"
    hf_model_dir.mkdir()
    config_path = _write_workflow_config(
        tmp_path / "qwen35_quant.yaml",
        quant={"algorithm": "gptqmodel", "method": "autoround", "group_size": 64},
    )
    workflow = Qwen35Workflow.from_config(str(hf_model_dir), str(config_path))

    quant_result = workflow.quant(
        output_dir=str(tmp_path / "quant"),
        device="cpu",
        config_overrides={"quant": None},
    )

    assert quant_result.skipped is True
    assert quant_result.raw_model_dir == str(hf_model_dir.resolve())
    assert quant_result.quanted_model_dir is None


def test_qwen35_workflow_context_length_override_updates_target_and_draft_configs():
    from examples_merak.llm.qwen3_5.qwen3_5_workflow import _add_context_length_overrides

    workflow_data = {
        "export": {
            "model": {
                "context_max_length": 2048,
                "mtp_config": {"context_max_length": 2048},
                "dflash_config": {"max_sequence_length": 2048},
            }
        }
    }
    overrides = {}

    _add_context_length_overrides(overrides, workflow_data, 8192)

    assert overrides == {
        "export.model.context_max_length": 8192,
        "export.model.mtp_config.context_max_length": 8192,
        "export.model.dflash_config.max_sequence_length": 8192,
    }


def test_qwen35_workflow_context_length_override_skips_missing_optional_draft_configs():
    from examples_merak.llm.qwen3_5.qwen3_5_workflow import _add_context_length_overrides

    workflow_data = {"export": {"model": {"context_max_length": 2048}}}
    overrides = {"export.model.visual_config.max_size_h": 896}

    _add_context_length_overrides(overrides, workflow_data, 4096)

    assert overrides == {
        "export.model.visual_config.max_size_h": 896,
        "export.model.context_max_length": 4096,
    }


def test_qwen35_workflow_parse_args_accepts_context_length_alias(monkeypatch):
    from examples_merak.llm.qwen3_5.qwen3_5_workflow import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "qwen3_5_workflow.py",
            "--model-dir",
            "hf",
            "--config-path",
            "config.yaml",
            "--context-length",
            "8192",
        ],
    )

    args = parse_args()

    assert args.context_max_length == 8192
