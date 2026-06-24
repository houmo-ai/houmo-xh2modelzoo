"""Shared Merak workflow model-name generation helpers.

The generated ``model_name`` is an artifact-facing contract.  It should encode
what downstream users need to distinguish exported HMONNX directories, while
leaving internal quantization details such as ``h1_sefp`` inside
``quant_scheme.quant_type``.
"""
from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import WorkflowConfig


AUTO_MODEL_NAME = "auto"
DEFAULT_MAX_PE_LENGTH = 32768


def normalize_name_token(value: Any, *, field: str = "token") -> str:
    """Normalize one token for stable filesystem-friendly model names."""

    token = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower())
    token = re.sub(r"_+", "_", token).strip("_")
    if not token:
        raise ValueError(f"export.naming.{field} must resolve to a non-empty token")
    return token


def normalize_chip_token(chip_arch: Any) -> str:
    """Normalize chip names while keeping the historical xh2 artifact prefix."""

    token = normalize_name_token(chip_arch, field="chip_arch")
    if token.startswith("xh2"):
        return "xh2"
    return token


def format_context_suffix(context_max_length: int) -> str:
    if context_max_length <= 0:
        raise ValueError(f"context_max_length must be positive, got {context_max_length!r}")
    if context_max_length % 1024 == 0:
        return f"{context_max_length // 1024}k"
    return str(context_max_length)


def resolve_quant_method(quant_cfg: Mapping[str, Any] | None, naming_cfg: Mapping[str, Any] | None = None) -> str:
    naming_cfg = naming_cfg or {}
    explicit_method = naming_cfg.get("quant_method")
    if explicit_method:
        return normalize_name_token(explicit_method, field="quant_method")
    if quant_cfg is None:
        return "base"

    algorithm = str(quant_cfg.get("algorithm") or "").lower().replace("-", "_")
    method = str(quant_cfg.get("method") or "").lower().replace("-", "_")
    if method:
        return normalize_name_token(method, field="quant.method")
    if algorithm in {"gptqmodel"}:
        return "gptq"
    if algorithm in {"gptq", "autoround", "auto_round"}:
        return "autoround" if algorithm == "auto_round" else algorithm
    if algorithm == "existing_hf":
        raise ValueError("quant.algorithm='existing_hf' requires quant.method for auto model naming")
    raise ValueError(f"Cannot resolve quant method from quant.algorithm={algorithm!r}")


def _quant_type(export_model_cfg: Mapping[str, Any]) -> str:
    quant_scheme = export_model_cfg.get("quant_scheme")
    if not isinstance(quant_scheme, Mapping):
        raise ValueError("export.model.quant_scheme must be a mapping for auto model naming")
    quant_type = quant_scheme.get("quant_type")
    if not quant_type:
        raise ValueError("export.model.quant_scheme.quant_type is required for auto model naming")
    return str(quant_type)


def _parse_bits(pattern: str, text: str, *, field: str) -> int:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Cannot parse {field} bits from quant_type={text!r}")
    return int(match.group(1))


def resolve_activation_bits(export_model_cfg: Mapping[str, Any]) -> int:
    return _parse_bits(r"a(\d+)", _quant_type(export_model_cfg), field="activation")


def resolve_weight_bits(
    quant_cfg: Mapping[str, Any] | None,
    export_model_cfg: Mapping[str, Any],
    naming_cfg: Mapping[str, Any] | None = None,
) -> int:
    naming_cfg = naming_cfg or {}
    if naming_cfg.get("weight_bits") is not None:
        return int(naming_cfg["weight_bits"])
    if quant_cfg is not None and quant_cfg.get("bits") is not None:
        return int(quant_cfg["bits"])
    return _parse_bits(r"w(\d+)", _quant_type(export_model_cfg), field="weight")


def resolve_shape_contract(naming_cfg: Mapping[str, Any], export_model_cfg: Mapping[str, Any]) -> str:
    explicit_shape = naming_cfg.get("shape")
    if explicit_shape is not None:
        return normalize_name_token(explicit_shape, field="shape")

    prefill = int(export_model_cfg.get("prefill_chunk_length") or export_model_cfg.get("input_sequence_length") or 0)
    context = int(export_model_cfg.get("context_max_length") or export_model_cfg.get("max_context_length") or 0)
    if prefill <= 0 or context <= 0:
        raise ValueError(
            "export.model.prefill_chunk_length/input_sequence_length and context_max_length "
            "are required for auto model naming when export.naming.shape is not provided"
        )
    return f"{prefill}_{format_context_suffix(context)}"


def _read_hf_max_position_embeddings(hf_model_dir: Any) -> int | None:
    if not hf_model_dir:
        return None
    config_file = Path(str(hf_model_dir)) / "config.json"
    if not config_file.is_file():
        return None
    try:
        hf_config = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    candidates = [
        hf_config.get("max_position_embeddings"),
        hf_config.get("text_config", {}).get("max_position_embeddings")
        if isinstance(hf_config.get("text_config"), Mapping)
        else None,
        hf_config.get("llm_config", {}).get("max_position_embeddings")
        if isinstance(hf_config.get("llm_config"), Mapping)
        else None,
    ]
    for value in candidates:
        if value is not None:
            return int(value)
    return None


def resolve_max_pe_contract(
    *,
    hf_model_dir: str | None = None,
    max_pe_length: Any = None,
) -> str:
    resolved_max_pe_length = _read_hf_max_position_embeddings(hf_model_dir)
    if resolved_max_pe_length is None:
        resolved_max_pe_length = max_pe_length if max_pe_length is not None else DEFAULT_MAX_PE_LENGTH
    resolved_max_pe_length = int(resolved_max_pe_length)
    if resolved_max_pe_length <= 0:
        raise ValueError(f"max_pe_length must be positive, got {resolved_max_pe_length!r}")
    return f"mpe{format_context_suffix(resolved_max_pe_length)}"


def build_merak_model_name(
    *,
    chip_arch: Any,
    naming_cfg: Mapping[str, Any],
    quant_cfg: Mapping[str, Any] | None,
    export_model_cfg: Mapping[str, Any],
    hf_model_dir: str | None = None,
) -> str:
    family = normalize_name_token(naming_cfg.get("family", ""), field="family")
    variant = normalize_name_token(naming_cfg.get("variant", ""), field="variant")
    profile = normalize_name_token(naming_cfg.get("profile", ""), field="profile")
    method = resolve_quant_method(quant_cfg, naming_cfg)
    weight_bits = resolve_weight_bits(quant_cfg, export_model_cfg, naming_cfg)
    activation_bits = resolve_activation_bits(export_model_cfg)
    shape = resolve_shape_contract(naming_cfg, export_model_cfg)
    max_pe = resolve_max_pe_contract(
        hf_model_dir=hf_model_dir or export_model_cfg.get("hf_model"),
        max_pe_length=export_model_cfg.get("max_pe_length"),
    )

    return "_".join(
        [
            normalize_chip_token(chip_arch),
            family,
            variant,
            profile,
            method,
            f"w{weight_bits}a{activation_bits}",
            shape,
            max_pe,
        ]
    )


def resolve_auto_model_name(workflow_config: WorkflowConfig, *, hf_model_dir: str | None = None) -> WorkflowConfig:
    """Resolve ``export.model.model_name: auto`` in a workflow config."""

    export_cfg = workflow_config.export
    model_cfg = export_cfg.get("model")
    if not isinstance(model_cfg, Mapping):
        raise ValueError("workflow export.model must be a mapping for auto model naming")

    current_name = str(model_cfg.get("model_name") or "")
    if current_name != AUTO_MODEL_NAME:
        return workflow_config

    naming_cfg = export_cfg.get("naming")
    if not isinstance(naming_cfg, Mapping):
        raise ValueError("export.naming must be provided when export.model.model_name is 'auto'")

    resolved_name = build_merak_model_name(
        chip_arch=model_cfg.get("chip_arch"),
        naming_cfg=naming_cfg,
        quant_cfg=workflow_config.quant,
        export_model_cfg=model_cfg,
        hf_model_dir=hf_model_dir,
    )
    data = copy.deepcopy(workflow_config.data)
    data["export"]["model"]["model_name"] = resolved_name
    return WorkflowConfig(data=data, source=workflow_config.source)
