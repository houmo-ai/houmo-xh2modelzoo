"""Stable Qwen3.5/Qwen3.6 workflow defaults and recommended YAML discovery."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONFIG_ROOTS = (
    _REPO_ROOT / "configs_merak/workflows/xh2a/llm_models/qwen3_5",
    _REPO_ROOT / "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe",
)

DEFAULT_QUANT_CONFIG: dict[str, Any] = {
    "algorithm": "gptqmodel",
    "method": "autoround",
    "output_format": "gptqmodel_hf",
    "artifact_format": "gptqmodel_hf",
    "bits": 4,
    "group_size": 64,
    "rotation": False,
    "sym": True,
    "iters": 200,
    "seed": 42,
    "quant_nontext_module": False,
    "format": "auto_gptq",
    "calibration": {
        "dataset": "NeelNanda/pile-10k",
        "nsamples": 128,
        "seqlen": 2048,
    },
    "runtime": {
        "batch_size": 8,
        "trust_remote_code": True,
        "low_gpu_mem_usage": True,
    },
}

QUANT_CONFIG_TEMPLATE: dict[str, Any] = {
    **DEFAULT_QUANT_CONFIG,
    "existing_hf": {
        "algorithm": "existing_hf",
        "method": "autoround",
        "artifact_format": "gptqmodel_hf",
        "existing_hf_model_dir": "weights/<existing-gptqmodel-hf-dir>",
    },
}

EXPORT_CONFIG_TEMPLATE: dict[str, Any] = {
    "variants": ["full", "mtp", "dflash", "visual_only"],
    "visual_sizes": [448, 896],
    "full": {
        "naming": {
            "family": "qwen3_5",
            "variant": "<model-size>",
            "profile": "full",
        },
        "model": {
            "chip_arch": "XH2a",
            "model_type": "Qwen3_5ForConditionalGeneration",
            "hf_model": "weights/<hf-model-dir>",
            "model_name": "auto",
            "context_max_length": 2048,
            "prefill_chunk_length": 256,
            "max_pe_length": 262144,
            "use_cache": True,
            "linear_attention_mode": "auto",
            "linear_chunk_size": 64,
            "split_conv_cache": True,
            "quant_scheme": {"quant_type": "w8a8h1_sefp", "ops": {}},
            "visual_config": {
                "max_size_w": 448,
                "max_size_h": 448,
                "quant_scheme": {"quant_type": "w8a8h1_sefp", "ops": {}},
            },
        }
    },
    "mtp": {
        "naming": {
            "family": "qwen3_5",
            "variant": "<model-size>",
            "profile": "full_mtp",
        },
        "model": {
            "model_type": "Qwen3_5ForConditionalGeneration",
            "model_name": "auto",
            "mtp_config": {
                "num_nextn_predict_layers": 1,
                "output_hidden_state_indices": [30],
            },
        }
    },
    "dflash": {
        "naming": {
            "family": "qwen3_5",
            "variant": "<model-size>",
            "profile": "full_dflash",
        },
        "model": {
            "model_type": "Qwen3_5ForConditionalGeneration",
            "model_name": "auto",
            "dflash_config": {
                "target_model_dir": None,
            },
        }
    },
    "visual_only": {
        "naming": {
            "family": "qwen3_5",
            "variant": "<model-size>",
            "profile": "visual_only",
            "shape": "448",
        },
        "model": {
            "chip_arch": "XH2a",
            "model_type": "Qwen3_5ForConditionalGeneration_visual",
            "hf_model": "weights/<hf-model-dir>",
            "model_name": "auto",
            "max_size_w": 448,
            "max_size_h": 448,
            "quant_scheme": {"quant_type": "w8a8h1_sefp", "ops": {}},
            "fuse_gdr_ops": False,
        }
    },
}


def repo_root() -> Path:
    return _REPO_ROOT


def recommended_config_paths() -> list[Path]:
    paths: list[Path] = []
    for root in _CONFIG_ROOTS:
        paths.extend(root.rglob("*.yaml"))
        paths.extend(root.rglob("*.yml"))
    return sorted(paths, key=lambda path: path.relative_to(_REPO_ROOT).as_posix())


def list_recommended_configs() -> list[dict[str, Any]]:
    """Return stable, repo-relative metadata for all recommended Qwen3.5/Qwen3.6 YAMLs."""
    configs: list[dict[str, Any]] = []
    for path in recommended_config_paths():
        data = _load_yaml(path)
        quant = data.get("quant") or {}
        model = data["export"]["model"]
        configs.append(
            {
                "name": path.stem,
                "family": _family_for_path(path),
                "model_size": path.parent.name,
                "variant": _variant_for_path(path),
                "quant_method": str(quant.get("method", "")).lower(),
                "visual_size": _visual_size(model),
                "config_path": path.relative_to(_REPO_ROOT).as_posix(),
                "model_type": model["model_type"],
                "model_name": model["model_name"],
            }
        )
    return configs


def get_recommended_config_path(
    *,
    name: str | None = None,
    family: str | None = None,
    model_size: str | None = None,
    variant: str | None = None,
    quant_method: str | None = None,
    visual_size: int | None = None,
) -> Path:
    """Return the unique recommended workflow YAML matching the selector."""
    matches = _select_recommended_configs(
        name=name,
        family=family,
        model_size=model_size,
        variant=variant,
        quant_method=quant_method,
        visual_size=visual_size,
    )
    if len(matches) != 1:
        selector = {
            "name": name,
            "family": family,
            "model_size": model_size,
            "variant": variant,
            "quant_method": quant_method,
            "visual_size": visual_size,
        }
        available = [item["name"] for item in list_recommended_configs()]
        raise ValueError(
            f"Expected exactly one Qwen3.5/Qwen3.6 recommended config for {selector}, "
            f"got {len(matches)}. Available names: {available}"
        )
    return _REPO_ROOT / matches[0]["config_path"]


def get_default_workflow_config(
    *,
    name: str | None = None,
    family: str | None = None,
    model_size: str | None = None,
    variant: str | None = None,
    quant_method: str | None = None,
    visual_size: int | None = None,
) -> dict[str, Any]:
    """Return a deep copy of one checked-in recommended workflow YAML."""
    return copy.deepcopy(
        _load_yaml(
            get_recommended_config_path(
                name=name,
                family=family,
                model_size=model_size,
                variant=variant,
                quant_method=quant_method,
                visual_size=visual_size,
            )
        )
    )


def get_default_quant_config() -> dict[str, Any]:
    """Return the default Qwen3.5/Qwen3.6 AutoRound/GPTQModel quant section."""
    return copy.deepcopy(DEFAULT_QUANT_CONFIG)


def get_default_export_config(
    *,
    name: str | None = None,
    family: str | None = None,
    model_size: str | None = None,
    variant: str | None = None,
    quant_method: str | None = None,
    visual_size: int | None = None,
) -> dict[str, Any]:
    """Return the export section from one recommended workflow YAML."""
    workflow_config = get_default_workflow_config(
        name=name,
        family=family,
        model_size=model_size,
        variant=variant,
        quant_method=quant_method,
        visual_size=visual_size,
    )
    return copy.deepcopy(workflow_config["export"])


def dump_yaml_template(template: dict[str, Any], output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        yaml.safe_dump(copy.deepcopy(template), fout, allow_unicode=True, sort_keys=False)
    return str(path)


def quant_config_template() -> dict[str, Any]:
    return copy.deepcopy(QUANT_CONFIG_TEMPLATE)


def export_config_template() -> dict[str, Any]:
    return copy.deepcopy(EXPORT_CONFIG_TEMPLATE)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fin:
        data = yaml.safe_load(fin) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Workflow config must be a YAML mapping: {path}")
    return data


def _select_recommended_configs(
    *,
    name: str | None = None,
    family: str | None = None,
    model_size: str | None = None,
    variant: str | None = None,
    quant_method: str | None = None,
    visual_size: int | None = None,
) -> list[dict[str, Any]]:
    expected_family = _normalize_family(family) if family is not None else None
    expected_model_size = model_size.lower() if model_size is not None else None
    expected_variant = variant.lower() if variant is not None else None
    expected_name = name.lower() if name is not None else None
    expected_quant_method = quant_method.lower() if quant_method is not None else None

    matches: list[dict[str, Any]] = []
    for item in list_recommended_configs():
        if expected_name is not None and item["name"].lower() != expected_name:
            continue
        if expected_family is not None and item["family"] != expected_family:
            continue
        if expected_model_size is not None and item["model_size"].lower() != expected_model_size:
            continue
        if expected_variant is not None and item["variant"] != expected_variant:
            continue
        if expected_quant_method is not None and item["quant_method"] != expected_quant_method:
            continue
        if visual_size is not None and item["visual_size"] != int(visual_size):
            continue
        matches.append(item)
    if expected_name is None and expected_quant_method is None and len(matches) > 1:
        autoround_matches = [item for item in matches if item["quant_method"] == "autoround"]
        if autoround_matches:
            return autoround_matches
    return matches


def _normalize_family(family: str) -> str:
    value = family.lower()
    if value in {"dense", "qwen3_5", "qwen35", "qwen3.5"}:
        return "qwen3_5"
    if value in {"moe", "qwen3_5_moe", "qwen35_moe", "qwen3.5_moe"}:
        return "qwen3_5_moe"
    raise ValueError("family must be one of: dense/qwen3_5 or moe/qwen3_5_moe")


def _family_for_path(path: Path) -> str:
    parts = path.relative_to(_REPO_ROOT).parts
    if "qwen3_5_moe" in parts:
        return "qwen3_5_moe"
    return "qwen3_5"


def _variant_for_path(path: Path) -> str:
    name = path.stem
    if name.endswith("_gptq"):
        name = name[: -len("_gptq")]
    if name.endswith("_autoround"):
        name = name[: -len("_autoround")]
    if "visual_only" in name:
        return "visual_only"
    if name.endswith("_full_mtp"):
        return "mtp"
    if name.endswith("_full_dflash"):
        return "dflash"
    if name.endswith("_full"):
        return "full"
    raise ValueError(f"Unsupported Qwen3.5 workflow variant in {path}")


def _visual_size(model: dict[str, Any]) -> int | None:
    width = model.get("max_size_w")
    height = model.get("max_size_h")
    if width is None or height is None:
        visual = model.get("visual_config")
        if isinstance(visual, dict):
            width = visual.get("max_size_w")
            height = visual.get("max_size_h")
    if width is None or height is None:
        return None
    if int(width) != int(height):
        raise ValueError(f"Expected square visual size, got width={width}, height={height}")
    return int(width)


__all__ = [
    "DEFAULT_QUANT_CONFIG",
    "EXPORT_CONFIG_TEMPLATE",
    "QUANT_CONFIG_TEMPLATE",
    "dump_yaml_template",
    "export_config_template",
    "get_default_export_config",
    "get_default_quant_config",
    "get_default_workflow_config",
    "get_recommended_config_path",
    "list_recommended_configs",
    "quant_config_template",
    "recommended_config_paths",
    "repo_root",
]
