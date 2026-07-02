"""Thin GPTQModel quant adapter for Qwen3.5/Qwen3.6 workflows."""
from __future__ import annotations

import importlib
import importlib.util
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...workflows.result import QuantResult


_DENSE_CALIBRATION_JSONL = "gptqmodel://quantization/calibration/dense_ivsg/gen_data/Qwen3.5-27B.jsonl"
_MOE_CALIBRATION_JSONL = "gptqmodel://quantization/calibration/moe_ebss/gen_data/Qwen3-Next-80B-A3B-Instruct.jsonl"
DEFAULT_AUTOROUND_DATASET = "data/calib_data/NeelNanda-pile-10k.jsonl"
_LEGACY_AUTOROUND_DATASETS = {"NeelNanda/pile-10k", "pile-10k"}


def quantize_with_autoround_api(
    *,
    model_dir: str,
    output_dir: str,
    device: str,
    quant_cfg: Mapping[str, Any],
    export_model_cfg: Mapping[str, Any],
    workflow_seed: int,
) -> QuantResult:
    kwargs = build_qwen35_autoround_kwargs(
        hf_model_dir=model_dir,
        output_dir=output_dir,
        device=device,
        quant_cfg=quant_cfg,
        export_model_cfg=export_model_cfg,
        workflow_seed=workflow_seed,
    )
    recipe = _load_callable(
        "gptqmodel.recipes.qwen35_autoround",
        "quantize_qwen35_autoround",
        "Qwen3.5 AutoRound quant requires GPTQModel with qwen35_autoround recipe API.",
    )
    result = recipe(**kwargs)
    quanted_model_dir = _result_output_dir(result, kwargs["output_dir"])
    return QuantResult(
        raw_model_dir=model_dir,
        quanted_model_dir=_normalize_path(quanted_model_dir),
    )


def quantize_with_gptqmodel_api(
    *,
    model_dir: str,
    output_dir: str,
    device: str,
    quant_cfg: Mapping[str, Any],
    export_model_cfg: Mapping[str, Any],
    workflow_seed: int,
) -> QuantResult:
    kwargs = build_qwen35_gptqmodel_kwargs(
        hf_model_dir=model_dir,
        output_dir=output_dir,
        device=device,
        quant_cfg=quant_cfg,
        export_model_cfg=export_model_cfg,
        workflow_seed=workflow_seed,
    )
    recipe = _load_callable(
        "gptqmodel.recipes.qwen35",
        "quantize_qwen35",
        "Qwen3.5 GPTQModel quant requires GPTQModel with qwen35 recipe API.",
    )
    result = recipe(**kwargs)
    quanted_model_dir = _result_output_dir(result, kwargs["output_dir"])
    return QuantResult(
        raw_model_dir=model_dir,
        quanted_model_dir=_normalize_path(quanted_model_dir),
    )


def build_qwen35_autoround_kwargs(
    *,
    hf_model_dir: str,
    output_dir: str,
    device: str,
    quant_cfg: Mapping[str, Any],
    export_model_cfg: Mapping[str, Any],
    workflow_seed: int,
) -> dict[str, Any]:
    _validate_artifact_format(quant_cfg)
    group_size = _group_size(quant_cfg)
    validate_rotation_mtp_compatibility(quant_cfg, export_model_cfg)
    topology = _topology(export_model_cfg)
    runtime_cfg = _section(quant_cfg, "runtime")
    calibration_cfg = _section(quant_cfg, "calibration")
    moe_cfg = _section(quant_cfg, "moe")
    if _rotation_value(quant_cfg):
        raise ValueError("Qwen3.5 AutoRound mode1 is no-rotation; set quant.rotation=false")

    default_batch_size = 8
    default_device_map = "0" if topology == "moe" else None
    default_low_gpu_mem_usage = True if topology == "moe" else None
    default_gradient_accumulate_steps = 1 if topology == "moe" else None

    kwargs: dict[str, Any] = {
        "model_dir": hf_model_dir,
        "output_dir": _quant_output_path(output_dir, quant_cfg),
        "topology": topology,
        "mode": str(quant_cfg.get("mode", quant_cfg.get("preset", "llm-only"))).replace("_", "-"),
        "bits": int(quant_cfg.get("bits", 4)),
        "group_size": group_size,
        "sym": _as_bool(quant_cfg.get("sym", True)),
        "iters": int(quant_cfg.get("iters", 200)),
        "nsamples": int(calibration_cfg.get("nsamples", quant_cfg.get("nsamples", 128))),
        "seqlen": int(calibration_cfg.get("seqlen", quant_cfg.get("seqlen", 2048))),
        "batch_size": int(runtime_cfg.get("batch_size", quant_cfg.get("batch_size", default_batch_size))),
        "dataset": _resolve_autoround_dataset_value(
            calibration_cfg.get(
                "dataset",
                calibration_cfg.get("jsonl", quant_cfg.get("dataset", DEFAULT_AUTOROUND_DATASET)),
            )
        ),
        "device": device,
        "device_map": runtime_cfg.get("device_map", quant_cfg.get("device_map", default_device_map)),
        "low_gpu_mem_usage": _optional_bool(
            runtime_cfg.get("low_gpu_mem_usage", quant_cfg.get("low_gpu_mem_usage", default_low_gpu_mem_usage))
        ),
        "gradient_accumulate_steps": runtime_cfg.get(
            "gradient_accumulate_steps",
            quant_cfg.get("gradient_accumulate_steps", default_gradient_accumulate_steps),
        ),
        "format": str(quant_cfg.get("format", quant_cfg.get("autoround_format", "auto_gptq"))),
        "seed": int(runtime_cfg.get("seed", quant_cfg.get("seed", workflow_seed))),
        "deterministic": _as_bool(runtime_cfg.get("deterministic", quant_cfg.get("deterministic", False))),
        "save_rotated": quant_cfg.get("save_rotated"),
        "extra_args": quant_cfg.get("extra_args", runtime_cfg.get("extra_args")),
    }
    if topology == "moe":
        kwargs.update(
            {
                "attn_bits": _optional_int(moe_cfg.get("attn_bits", moe_cfg.get("self_attn_bits", 8))),
                "shared_expert_bits": _optional_int(moe_cfg.get("shared_expert_bits", 8)),
                "expert_bits": _optional_int(moe_cfg.get("expert_bits")),
                "expert_up_gate_bits": _optional_int(moe_cfg.get("expert_up_gate_bits")),
                "expert_down_bits": _optional_int(moe_cfg.get("expert_down_bits")),
            }
        )
        kwargs["format"] = str(quant_cfg.get("format", quant_cfg.get("autoround_format", "auto_round:gptqmodel")))
    return {key: value for key, value in kwargs.items() if value is not None}


def build_qwen35_gptqmodel_kwargs(
    *,
    hf_model_dir: str,
    output_dir: str,
    device: str,
    quant_cfg: Mapping[str, Any],
    export_model_cfg: Mapping[str, Any],
    workflow_seed: int,
) -> dict[str, Any]:
    _validate_artifact_format(quant_cfg)
    group_size = _group_size(quant_cfg)
    validate_rotation_mtp_compatibility(quant_cfg, export_model_cfg)
    topology = _topology(export_model_cfg)
    runtime_cfg = _section(quant_cfg, "runtime")
    calibration_cfg = _section(quant_cfg, "calibration")
    validation_cfg = _section(quant_cfg, "validation")
    moe_cfg = _section(quant_cfg, "moe")
    rotation = _rotation_value(quant_cfg)

    default_jsonl = _MOE_CALIBRATION_JSONL if topology == "moe" else _DENSE_CALIBRATION_JSONL
    nsamples_default = 512 if topology == "moe" else 256
    batch_size = runtime_cfg.get("gptqmodel_batch_size", quant_cfg.get("gptqmodel_batch_size", quant_cfg.get("batch_size", 1)))
    nsamples = quant_cfg.get("nsamples", nsamples_default)
    seqlen = quant_cfg.get("seqlen", 1024)
    if _calibration_overrides_gptqmodel_defaults(calibration_cfg):
        nsamples = calibration_cfg.get("nsamples", nsamples)
        seqlen = calibration_cfg.get("seqlen", seqlen)
    kwargs: dict[str, Any] = {
        "model_dir": hf_model_dir,
        "output_dir": _quant_output_path(output_dir, quant_cfg),
        "method": _gptqmodel_method(quant_cfg),
        "preset": str(quant_cfg.get("preset", "full_vlm")),
        "rotation": rotation,
        "artifact_format": "gptqmodel_hf",
        "topology": topology,
        "bits": int(quant_cfg.get("bits", 4)),
        "group_size": group_size,
        "device": device,
        "device_map": runtime_cfg.get("device_map", quant_cfg.get("device_map", "auto")),
        "trust_remote_code": _as_bool(runtime_cfg.get("trust_remote_code", quant_cfg.get("trust_remote_code", True))),
        "batch_size": int(batch_size),
        "offload_to_disk": _as_bool(runtime_cfg.get("offload_to_disk", quant_cfg.get("offload_to_disk", False))),
        "offload_path": runtime_cfg.get("offload_path", quant_cfg.get("offload_path")),
        "calibration_jsonl": _resolve_calibration_value(
            calibration_cfg.get("jsonl", calibration_cfg.get("calibration_jsonl", default_jsonl))
        ),
        "calibration_text_key": calibration_cfg.get("text_key", calibration_cfg.get("calibration_text_key", "text")),
        "calibration_dataset": quant_cfg.get("calibration_dataset"),
        "nsamples": int(nsamples),
        "seqlen": int(seqlen),
        "check_quant_vision_demo": _as_bool(
            validation_cfg.get("check_quant_vision_demo", quant_cfg.get("check_quant_vision_demo", True))
        ),
        "check_rotation_ppl": _as_bool(validation_cfg.get("check_rotation_ppl", quant_cfg.get("check_rotation_ppl", False))),
        "max_quant_layers": validation_cfg.get("max_quant_layers", quant_cfg.get("max_quant_layers")),
        "run_inference": _optional_bool(validation_cfg.get("run_inference", quant_cfg.get("run_inference"))),
        "skip_inference": _optional_bool(validation_cfg.get("skip_inference", quant_cfg.get("skip_inference"))),
        "seed": int(runtime_cfg.get("seed", quant_cfg.get("seed", workflow_seed))),
        "hessian_mse": _as_bool(quant_cfg.get("hessian_mse", True)),
        "wait_for_submodule_finalizers": _as_bool(
            runtime_cfg.get("wait_for_submodule_finalizers", quant_cfg.get("wait_for_submodule_finalizers", True))
        ),
    }
    if topology == "moe":
        kwargs.update(
            {
                "self_attn_bits": _optional_int(moe_cfg.get("self_attn_bits", moe_cfg.get("attn_bits", 8))),
                "shared_expert_bits": _optional_int(moe_cfg.get("shared_expert_bits", 8)),
                "expert_bits": _optional_int(moe_cfg.get("expert_bits", 4)),
                "expert_down_bits": _optional_int(moe_cfg.get("expert_down_bits", 5)),
                "moe": {"routing": moe_cfg.get("routing", moe_cfg.get("moe_routing", "bypass"))},
            }
        )
    return {key: value for key, value in kwargs.items() if value is not None}


def validate_rotation_mtp_compatibility(quant_cfg: Mapping[str, Any], export_model_cfg: Mapping[str, Any]) -> None:
    if not _rotation_value(quant_cfg):
        return
    mtp_keys = (
        "use_mtp",
        "enable_mtp",
        "mtp",
        "mtp_config",
        "spec_decode",
        "spec_decode_mode",
        "speculative_decode",
        "dflash_config",
    )
    for key in mtp_keys:
        value = export_model_cfg.get(key)
        if isinstance(value, str):
            value = value.strip().lower()
        if value and value not in {"false", "none", "off", "disable", "disabled"}:
            raise ValueError("Qwen3.5 GPTQModel rotation is incompatible with MTP/spec-decode export settings")


def _gptqmodel_method(quant_cfg: Mapping[str, Any]) -> str:
    method = str(quant_cfg.get("gptqmodel_method", quant_cfg.get("method", "gptq")))
    if method == "mode1":
        return "gptq"
    return method


def _calibration_overrides_gptqmodel_defaults(calibration_cfg: Mapping[str, Any]) -> bool:
    """Return whether ``quant.calibration`` should shape GPTQModel CLI args.

    The shared Qwen3.5 YAMLs are AutoRound-first and carry AutoRound defaults
    under ``quant.calibration`` (pile-10k, 128 samples, seqlen 2048).
    When users switch only ``quant.algorithm`` to ``gptqmodel`` we must keep the
    README-verified GPTQModel defaults instead of inheriting those AutoRound
    values.  However explicit GPTQModel calibration overrides still need to be
    honored, including CLI overrides such as ``quant.calibration.nsamples=16``.
    """

    if any(key in calibration_cfg for key in ("jsonl", "calibration_jsonl", "text_key", "calibration_text_key")):
        return True
    dataset = calibration_cfg.get("dataset")
    if dataset is None:
        return "nsamples" in calibration_cfg or "seqlen" in calibration_cfg
    if str(dataset) not in _LEGACY_AUTOROUND_DATASETS | {DEFAULT_AUTOROUND_DATASET}:
        return True
    return (
        calibration_cfg.get("nsamples", 128) != 128
        or calibration_cfg.get("seqlen", 2048) != 2048
    )


def _load_callable(module_name: str, callable_name: str, message: str):
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(message) from exc
    try:
        fn = getattr(module, callable_name)
    except AttributeError as exc:
        raise ImportError(message) from exc
    return fn


def _result_output_dir(result: Any, fallback: str) -> str:
    if result is None:
        return fallback
    if isinstance(result, Mapping):
        return str(result.get("output_dir") or result.get("quanted_model_dir") or fallback)
    return str(getattr(result, "output_dir", None) or getattr(result, "quanted_model_dir", None) or fallback)


def _validate_artifact_format(quant_cfg: Mapping[str, Any]) -> None:
    artifact_format = quant_cfg.get("artifact_format")
    output_format = quant_cfg.get("output_format")
    if artifact_format is not None and output_format is not None and artifact_format != output_format:
        raise ValueError("quant.artifact_format and quant.output_format must match when both are provided")
    resolved = artifact_format or output_format or "gptqmodel_hf"
    if resolved != "gptqmodel_hf":
        raise ValueError(f"Qwen3.5 quant requires artifact_format='gptqmodel_hf', got {resolved!r}")


def _group_size(quant_cfg: Mapping[str, Any]) -> int:
    group_size = int(quant_cfg.get("group_size", 64))
    if group_size != 64:
        raise ValueError(f"Qwen3.5/Qwen3.6 quant group_size must be 64, got {group_size!r}")
    return group_size


def _topology(export_model_cfg: Mapping[str, Any]) -> str:
    model_type = str(export_model_cfg.get("model_type", ""))
    if "Moe" in model_type or "moe" in model_type.lower():
        return "moe"
    return "dense"


def _rotation_value(quant_cfg: Mapping[str, Any]) -> str | bool | None:
    value = quant_cfg.get("rotation", False)
    if value in (None, False):
        return False
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "false"}:
        return False
    return value


def _section(quant_cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = quant_cfg.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"quant.{name} must be a mapping when provided")
    return value


def _quant_output_path(output_dir: str, quant_cfg: Mapping[str, Any]) -> str:
    return _normalize_path(quant_cfg.get("save_path") or quant_cfg.get("output_dir") or output_dir)


def _resolve_calibration_value(value: Any) -> str:
    text = os.path.expanduser(os.path.expandvars(str(value)))
    if text.startswith("gptqmodel://"):
        return _resolve_gptqmodel_resource(text)
    return text


def _resolve_autoround_dataset_value(value: Any) -> str:
    text = os.path.expanduser(os.path.expandvars(str(value)))
    if text in _LEGACY_AUTOROUND_DATASETS:
        text = DEFAULT_AUTOROUND_DATASET
    if _is_existing_local_path(text):
        return str(Path(text).resolve())
    if _is_path_like_value(text):
        raise FileNotFoundError(
            "Qwen3.5 AutoRound calibration dataset path does not exist: "
            f"{text!r}. Download the prepared Artifactory archive and place "
            f"it at {DEFAULT_AUTOROUND_DATASET}."
        )
    return text


def _is_existing_local_path(value: str) -> bool:
    return Path(value).expanduser().is_file()


def _is_path_like_value(value: str) -> bool:
    return (
        "$" in value
        or value.startswith(("/", "./", "../", "~"))
        or value.endswith((".json", ".jsonl", ".txt"))
    )


def _resolve_gptqmodel_resource(uri: str) -> str:
    relative_path = uri.removeprefix("gptqmodel://").lstrip("/")
    spec = importlib.util.find_spec("gptqmodel")
    package_locations = list(spec.submodule_search_locations or []) if spec and spec.submodule_search_locations else []
    for package_root in package_locations:
        candidate = Path(package_root) / relative_path
        if candidate.is_file():
            return str(candidate.resolve())
    return uri


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return _as_bool(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _normalize_path(path: str | os.PathLike[str]) -> str:
    return os.path.abspath(os.path.normpath(str(path)))


__all__ = [
    "build_qwen35_autoround_kwargs",
    "build_qwen35_gptqmodel_kwargs",
    "quantize_with_autoround_api",
    "quantize_with_gptqmodel_api",
    "validate_rotation_mtp_compatibility",
]
