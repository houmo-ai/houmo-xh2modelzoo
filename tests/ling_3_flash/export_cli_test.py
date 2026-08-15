import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


_EXPORT_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples_merak"
    / "llm"
    / "ling_3_flash"
    / "export.py"
)


def _load_export_module():
    spec = importlib.util.spec_from_file_location("ling_3_flash_export", _EXPORT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_low_memory_flag_enables_framework_huge_model_export(monkeypatch):
    module = _load_export_module()
    monkeypatch.delenv("HUGE_MODEL_EXPORT_ENABLED", raising=False)
    monkeypatch.delenv("XH_HUGE_MODEL_EXPORT", raising=False)

    module._configure_low_memory_export(True)

    assert os.environ["HUGE_MODEL_EXPORT_ENABLED"] == "1"
    assert "XH_HUGE_MODEL_EXPORT" not in os.environ


def test_no_low_memory_flag_disables_existing_framework_setting(monkeypatch):
    module = _load_export_module()
    monkeypatch.setenv("HUGE_MODEL_EXPORT_ENABLED", "1")

    module._configure_low_memory_export(False)

    assert os.environ["HUGE_MODEL_EXPORT_ENABLED"] == "0"


def test_unspecified_low_memory_mode_preserves_existing_framework_setting(monkeypatch):
    module = _load_export_module()
    monkeypatch.setenv("HUGE_MODEL_EXPORT_ENABLED", "1")

    module._configure_low_memory_export(None)

    assert os.environ["HUGE_MODEL_EXPORT_ENABLED"] == "1"


def test_export_from_quanted_model_accepts_gptqmodel_checkpoint(tmp_path):
    module = _load_export_module()
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": {"format": "gptq"}}),
        encoding="utf-8",
    )
    (tmp_path / "model.safetensors").write_bytes(b"packed")

    module._validate_quanted_checkpoint(tmp_path)


def test_export_from_quanted_model_rejects_float_checkpoint(tmp_path):
    module = _load_export_module()
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"float")

    with pytest.raises(ValueError, match="quantization_config"):
        module._validate_quanted_checkpoint(tmp_path)


def test_export_cli_dtype_defaults_to_workflow_config(monkeypatch):
    module = _load_export_module()
    monkeypatch.setattr(
        sys,
        "argv",
        ["export.py", "--model", "/tmp/model", "--output", "/tmp/output"],
    )

    args = module.parse_args()

    assert args.dtype is None
    assert args.config.endswith(
        "configs_merak/workflows/xh2a/llm_models/ling_3_flash/ling_3_flash_gptq.yaml"
    )


def test_export_cli_accepts_explicit_fp16_dtype(monkeypatch):
    module = _load_export_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export.py",
            "--model",
            "/tmp/model",
            "--output",
            "/tmp/output",
            "--dtype",
            "fp16",
        ],
    )

    assert module.parse_args().dtype == "fp16"
