from __future__ import annotations

import importlib
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ...workflows.result import QuantResult
from .gptqmodel_compat import laguna_gptqmodel_loader


DEFAULT_AUTOROUND_DATASET = "xh2modelzoo://data/calib_data/NeelNanda-pile-10k.jsonl"


def quantize_with_autoround_api(
    *,
    model_dir: str,
    output_dir: str,
    device: str,
    quant_cfg: Mapping[str, Any],
    workflow_seed: int,
) -> QuantResult:
    """Run GPTQModel's maintained Laguna QuaRot + AutoRound recipe."""

    kwargs = build_laguna_autoround_kwargs(
        hf_model_dir=model_dir,
        output_dir=output_dir,
        device=device,
        quant_cfg=quant_cfg,
        workflow_seed=workflow_seed,
    )
    recipe = _load_callable(
        "gptqmodel.recipes.laguna_autoround",
        "quantize_laguna_autoround",
        "Laguna AutoRound quant requires GPTQModel with the laguna_autoround recipe API.",
    )
    result = recipe(**kwargs)
    output_path = _result_output_dir(result, kwargs["output_dir"])
    return QuantResult(
        raw_model_dir=model_dir,
        quanted_model_dir=_normalize_path(output_path),
        meta=_result_provenance(result),
    )


def build_laguna_autoround_kwargs(
    *,
    hf_model_dir: str,
    output_dir: str,
    device: str,
    quant_cfg: Mapping[str, Any],
    workflow_seed: int,
) -> dict[str, Any]:
    """Translate Laguna workflow YAML into the maintained script contract."""

    _validate_artifact_format(quant_cfg)
    method = str(quant_cfg.get("method", "autoround")).lower().replace("-", "_")
    if method == "auto_round":
        method = "autoround"
    if method != "autoround":
        raise ValueError(f"Laguna AutoRound adapter requires method='autoround', got {method!r}")

    runtime_cfg = _section(quant_cfg, "runtime")
    calibration_cfg = _section(quant_cfg, "calibration")
    moe_cfg = _section(quant_cfg, "moe")
    rest_bits = int(quant_cfg.get("bits", quant_cfg.get("rest_bits", 8)))
    expert_bits = int(moe_cfg.get("expert_bits", quant_cfg.get("expert_bits", 4)))
    group_size = int(quant_cfg.get("group_size", 64))
    sym = _as_bool(quant_cfg.get("sym", True))
    if (rest_bits, expert_bits, group_size, sym) != (8, 4, 64, True):
        raise ValueError(
            "Laguna AutoRound supports only rest W8, routed experts W4, group_size=64, and symmetric quantization"
        )

    autoround_format = str(quant_cfg.get("format", quant_cfg.get("autoround_format", "auto_round")))
    if autoround_format != "auto_round":
        raise ValueError(f"Laguna mixed-bit AutoRound requires format='auto_round', got {autoround_format!r}")

    rotation = _normalize_rotation(quant_cfg.get("rotation", "hadamard"))
    dataset = calibration_cfg.get(
        "dataset",
        calibration_cfg.get("jsonl", quant_cfg.get("dataset", DEFAULT_AUTOROUND_DATASET)),
    )
    python = runtime_cfg.get("python", os.environ.get("GPTQMODEL_PYTHON"))
    extra_args = runtime_cfg.get("extra_args", quant_cfg.get("extra_args"))
    if extra_args is not None and (not isinstance(extra_args, Sequence) or isinstance(extra_args, (str, bytes))):
        raise TypeError("quant.runtime.extra_args must be a sequence when provided")
    env = runtime_cfg.get("env", quant_cfg.get("env"))
    if env is not None and not isinstance(env, Mapping):
        raise TypeError("quant.runtime.env must be a mapping when provided")

    save_rotated = quant_cfg.get("save_rotated", runtime_cfg.get("save_rotated"))
    kwargs: dict[str, Any] = {
        "model_dir": str(hf_model_dir),
        "output_dir": _normalize_path(quant_cfg.get("save_path") or quant_cfg.get("output_dir") or output_dir),
        "rest_bits": rest_bits,
        "expert_bits": expert_bits,
        "group_size": group_size,
        "sym": sym,
        "iters": int(quant_cfg.get("iters", 200)),
        "nsamples": int(calibration_cfg.get("nsamples", quant_cfg.get("nsamples", 128))),
        "seqlen": int(calibration_cfg.get("seqlen", quant_cfg.get("seqlen", 2048))),
        "batch_size": int(runtime_cfg.get("batch_size", quant_cfg.get("batch_size", 1))),
        "gradient_accumulate_steps": int(
            runtime_cfg.get(
                "gradient_accumulate_steps",
                quant_cfg.get("gradient_accumulate_steps", 1),
            )
        ),
        "dataset": _resolve_autoround_dataset(dataset),
        "device_map": str(runtime_cfg.get("device_map", quant_cfg.get("device_map", "0,1,2,3"))),
        "format": autoround_format,
        "seed": int(runtime_cfg.get("seed", quant_cfg.get("seed", workflow_seed))),
        "rotation": rotation,
        "rotation_device": str(runtime_cfg.get("rotation_device", quant_cfg.get("rotation_device", device))),
        "rotation_headwise_v_o": _as_bool(
            runtime_cfg.get(
                "rotation_headwise_v_o",
                quant_cfg.get("rotation_headwise_v_o", True),
            )
        ),
        "rotation_chunk_rows": int(runtime_cfg.get("rotation_chunk_rows", quant_cfg.get("rotation_chunk_rows", 1024))),
        "rotation_chunk_columns": int(
            runtime_cfg.get(
                "rotation_chunk_columns",
                quant_cfg.get("rotation_chunk_columns", 1024),
            )
        ),
        "rotation_matrix_batch": int(
            runtime_cfg.get("rotation_matrix_batch", quant_cfg.get("rotation_matrix_batch", 4))
        ),
        "rotation_compute_dtype": str(
            runtime_cfg.get(
                "rotation_compute_dtype",
                quant_cfg.get("rotation_compute_dtype", "float64"),
            )
        ),
        "low_gpu_mem_usage": _as_bool(runtime_cfg.get("low_gpu_mem_usage", quant_cfg.get("low_gpu_mem_usage", True))),
        "low_cpu_mem_usage": _as_bool(runtime_cfg.get("low_cpu_mem_usage", quant_cfg.get("low_cpu_mem_usage", True))),
        "quant_lm_head": _as_bool(quant_cfg.get("quant_lm_head", False)),
        "save_rotated": None if save_rotated in (None, "") else _normalize_path(save_rotated),
        "python": None if python in (None, "") else os.path.expanduser(os.path.expandvars(str(python))),
        "dry_run": _as_bool(runtime_cfg.get("dry_run", quant_cfg.get("dry_run", False))),
        "extra_args": None if extra_args is None else [str(item) for item in extra_args],
        "env": None if env is None else {str(key): str(value) for key, value in env.items()},
    }
    return {key: value for key, value in kwargs.items() if value is not None}


def quantize_with_gptqmodel_api(
    *,
    model_dir: str,
    output_dir: str,
    device: str,
    quant_cfg: Mapping[str, Any],
) -> QuantResult:
    """Optionally create a GPTQModel-compatible HF checkpoint for Laguna."""

    from gptqmodel import GPTQModel, QuantizeConfig

    bits = int(quant_cfg.get("bits", 4))
    group_size = int(quant_cfg.get("group_size", 64))
    save_path = Path(output_dir) / str(quant_cfg.get("output_name", f"{Path(model_dir).name}-gptqmodel-{bits}bit"))
    calibration = _load_calibration(quant_cfg)
    quant_config = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        sym=_as_bool(quant_cfg.get("sym", True)),
        desc_act=_as_bool(quant_cfg.get("desc_act", False)),
        hessian_mse=_as_bool(quant_cfg.get("hessian_mse", True)),
        offload_to_disk=_as_bool(quant_cfg.get("offload_to_disk", True)),
        offload_to_disk_path=quant_cfg.get("offload_path"),
    )

    with laguna_gptqmodel_loader(model_dir):
        model = GPTQModel.load(
            model_dir,
            quant_config,
            device=device,
            device_map=quant_cfg.get("device_map"),
            trust_remote_code=True,
        )
        model.quantize(
            calibration,
            batch_size=int(quant_cfg.get("batch_size", 1)),
            calibration_concat_size=quant_cfg.get("calibration_concat_size"),
        )
        model.save(str(save_path), max_shard_size=quant_cfg.get("max_shard_size", "4GB"))

    return QuantResult(raw_model_dir=model_dir, quanted_model_dir=str(save_path.resolve()))


def _load_calibration(quant_cfg: Mapping[str, Any]) -> list[str]:
    calibration = quant_cfg.get("calibration")
    if isinstance(calibration, Mapping):
        texts = calibration.get("texts")
        jsonl = calibration.get("jsonl") or calibration.get("dataset")
        text_key = str(calibration.get("text_key", "text"))
        nsamples = int(calibration.get("nsamples", 128))
    else:
        texts = None
        jsonl = None
        text_key = "text"
        nsamples = 128

    if isinstance(texts, Sequence) and not isinstance(texts, (str, bytes)):
        result = [str(text) for text in texts if str(text).strip()]
        if result:
            return result[:nsamples]

    if jsonl:
        import json

        path = Path(str(jsonl)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Laguna GPTQ calibration JSONL does not exist: {path}")
        result = []
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                text = record.get(text_key) if isinstance(record, Mapping) else None
                if isinstance(text, str) and text.strip():
                    result.append(text)
                if len(result) >= nsamples:
                    break
        if result:
            return result

    raise ValueError(
        "Laguna GPTQModel quantization requires quant.calibration.texts or "
        "quant.calibration.jsonl/dataset; the default HF floating-point workflow does not require this."
    )


def _validate_artifact_format(quant_cfg: Mapping[str, Any]) -> None:
    artifact_format = quant_cfg.get("artifact_format")
    output_format = quant_cfg.get("output_format")
    if artifact_format is not None and output_format is not None and artifact_format != output_format:
        raise ValueError("quant.artifact_format and quant.output_format must match when both are provided")
    resolved = artifact_format or output_format or "gptqmodel_hf"
    if resolved != "gptqmodel_hf":
        raise ValueError(f"Laguna quant requires artifact_format='gptqmodel_hf', got {resolved!r}")


def _normalize_rotation(value: Any) -> str:
    if value in (None, False):
        return "none"
    text = str(value).strip().lower().replace("-", "_")
    if text in {"", "none", "null", "false", "off"}:
        return "none"
    if text != "hadamard":
        raise ValueError(f"Laguna AutoRound rotation must be 'none' or 'hadamard', got {value!r}")
    return text


def _section(quant_cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = quant_cfg.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"quant.{name} must be a mapping when provided")
    return value


def _resolve_autoround_dataset(value: Any) -> str:
    text = os.path.expanduser(os.path.expandvars(str(value)))
    if text.startswith(("xh2modelzoo://", "repo://")):
        relative_path = text.split("://", 1)[1].lstrip("/")
        candidate = _repo_root() / relative_path
        if not candidate.is_file():
            raise FileNotFoundError(f"Laguna AutoRound calibration dataset does not exist: {candidate}")
        return str(candidate.resolve())
    path = Path(text)
    if path.is_file():
        return str(path.resolve())
    if text.startswith(("/", "./", "../", "~")) or text.endswith((".json", ".jsonl")):
        raise FileNotFoundError(f"Laguna AutoRound calibration dataset does not exist: {text}")
    return text


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs_merak").is_dir() and (parent / "xhmodel_merak").is_dir():
            return parent
    return Path(__file__).resolve().parents[4]


def _load_callable(module_name: str, callable_name: str, message: str):
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(message) from exc
    try:
        return getattr(module, callable_name)
    except AttributeError as exc:
        raise ImportError(message) from exc


def _result_output_dir(result: Any, fallback: str) -> str:
    if result is None:
        return fallback
    if isinstance(result, Mapping):
        return str(result.get("output_dir") or result.get("quanted_model_dir") or fallback)
    return str(getattr(result, "output_dir", None) or getattr(result, "quanted_model_dir", None) or fallback)


def _result_provenance(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        provenance = result.get("provenance")
    else:
        provenance = getattr(result, "provenance", None)
    return dict(provenance) if isinstance(provenance, Mapping) else {}


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _normalize_path(path: str | os.PathLike[str]) -> str:
    return os.path.abspath(os.path.normpath(str(path)))


__all__ = [
    "build_laguna_autoround_kwargs",
    "quantize_with_autoround_api",
    "quantize_with_gptqmodel_api",
]
