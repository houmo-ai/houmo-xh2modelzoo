from __future__ import annotations

from pathlib import Path

import pytest

from xhmodel_merak.xh_llm.models.deepseek_v4.workflow import DeepSeekV4Workflow
from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow
from xhmodel_merak.xh_llm.workflows.base import BaseLLMWorkflow
from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig
from xhmodel_merak.xh_llm.workflows.result import QuantResult


_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_ROOT = _ROOT / "configs_merak/workflows/xh2a/llm_models/deepseek_v4/flash_0731"
_CONFIGS = {
    "deepseek_v4_flash_0731_w4a8.yaml": "w8a8h1_sefp",
    "deepseek_v4_flash_0731_w4a16.yaml": "w8a16h1_sefp",
}


@pytest.mark.parametrize(("filename", "quant_type"), _CONFIGS.items())
def test_workflow_config_selects_expected_activation_precision(filename: str, quant_type: str) -> None:
    config_path = _CONFIG_ROOT / filename
    config = WorkflowConfig.from_file(str(config_path))

    assert config.quant is None
    assert config.data["runtime"]["low_memory"] is True
    assert config.export["model"]["packed_weight_only"] is True
    assert config.export["model"]["quant_scheme"]["quant_type"] == quant_type
    assert config.export["model"]["quant_scheme"]["ops"] == {}

    workflow = AutoLLMWorkflow.from_config("/tmp/deepseek-v4", str(config_path))
    assert isinstance(workflow, DeepSeekV4Workflow)


def test_workflow_applies_low_memory_override(monkeypatch) -> None:
    config_path = _CONFIG_ROOT / "deepseek_v4_flash_0731_w4a8.yaml"
    workflow = DeepSeekV4Workflow("/tmp/deepseek-v4", str(config_path))
    configured = []
    expected = object()

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.deepseek_v4.workflow.configure_huge_model_export",
        configured.append,
    )
    monkeypatch.setattr(BaseLLMWorkflow, "export", lambda self, **kwargs: expected)

    result = workflow.export(
        QuantResult(raw_model_dir="/tmp/deepseek-v4", skipped=True),
        "/tmp/export",
        "cuda",
        {"runtime.low_memory": False},
    )

    assert result is expected
    assert configured == [False]
