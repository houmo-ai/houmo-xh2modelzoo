"""Public Qwen3.5/Qwen3.6 Merak workflow convenience API."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...workflows.result import ExportResult, QuantResult
from .workflow import Qwen35Workflow
from .workflow_defaults import (
    dump_yaml_template,
    export_config_template,
    get_default_export_config,
    get_default_quant_config,
    get_default_workflow_config,
    get_recommended_config_path,
    list_recommended_configs,
    quant_config_template,
)
from .workflow_docs import get_export_config_help, get_model_docs, get_quant_config_help


def dump_quant_config_template(output_path: str | Path) -> str:
    """Write the supported quant template YAML and return the written path."""
    return dump_yaml_template(quant_config_template(), output_path)


def dump_export_config_template(output_path: str | Path) -> str:
    """Write the supported export template YAML and return the written path."""
    return dump_yaml_template(export_config_template(), output_path)


def quant(
    *,
    hf_model_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    device: str,
    config_overrides: Mapping[str, Any] | None = None,
    seed: int = 1024,
    debug: bool = False,
) -> QuantResult:
    """Quantize a Qwen3.5/Qwen3.6 HF model using a workflow YAML."""
    workflow = Qwen35Workflow(
        hf_model_dir=str(hf_model_dir),
        config_path=str(config_path),
        seed=seed,
        debug=debug,
    )
    return workflow.quant(output_dir=str(output_dir), device=device, config_overrides=config_overrides)


def export(
    *,
    hf_model_dir: str | Path,
    config_path: str | Path,
    quant_result: QuantResult,
    output_dir: str | Path,
    device: str,
    config_overrides: Mapping[str, Any] | None = None,
    seed: int = 1024,
    debug: bool = False,
) -> ExportResult:
    """Export a Qwen3.5/Qwen3.6 workflow to HMONNX using a QuantResult."""
    workflow = Qwen35Workflow(
        hf_model_dir=str(hf_model_dir),
        config_path=str(config_path),
        seed=seed,
        debug=debug,
    )
    return workflow.export(
        quant_result=quant_result,
        output_dir=str(output_dir),
        device=device,
        config_overrides=config_overrides,
    )


__all__ = [
    "dump_export_config_template",
    "dump_quant_config_template",
    "export",
    "get_default_export_config",
    "get_default_quant_config",
    "get_default_workflow_config",
    "get_export_config_help",
    "get_model_docs",
    "get_quant_config_help",
    "get_recommended_config_path",
    "list_recommended_configs",
    "quant",
]
