"""Thin GPTQModel quantization adapter for Gemma4 Series workflows.

This module keeps the xh2modelzoo side deliberately small: it validates the
workflow-level contract, translates YAML into a stable GPTQModel recipe call,
records provenance, and returns ``QuantResult``.  GPTQ internals,
modality-specific sidecars, and MoE bit policies belong to the
GPTQModel Gemma4 recipe.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...workflows.result import QuantResult
from .export_plan import build_gemma4_series_export_plan


_SUPPORTED_GPTQMODEL_METHODS = {"gptq"}
_SUPPORTED_AUTOROUND_PRESETS = {"mode1", "llm_only"}
_SUPPORTED_GPTQMODEL_PRESETS = {
    "llm_only",
    "full_vlm",
    "full_vlm_rotate",
    "full_multimodal",
    "rotated_fp_only",
}
_SUPPORTED_ROTATIONS = {None}
_DEFAULT_RECIPE_ENTRYPOINT = "gptqmodel.recipes.gemma4:quantize_gemma4"
DEFAULT_DENSE_CALIBRATION_JSONL = (
    "gptqmodel://quantization/calibration/dense_ivsg/shared-dense-text.jsonl"
)
DEFAULT_MOE_CALIBRATION_JSONL = (
    "gptqmodel://quantization/calibration/moe_ebss/gen_data/Qwen3-Next-80B-A3B-Instruct.jsonl"
)
DEFAULT_AUTOROUND_DATASET_PATH = "data/calib_data/NeelNanda-pile-10k.jsonl"
DEFAULT_AUTOROUND_DATASET = f"xh2modelzoo://{DEFAULT_AUTOROUND_DATASET_PATH}"
_LEGACY_AUTOROUND_DATASETS = {"NeelNanda/pile-10k", "pile-10k"}
_REPO_RESOURCE_PREFIXES = ("xh2modelzoo://", "repo://")
_GPTQMODEL_RESOURCE_ALIASES = {
    "quantization/calibration/dense_ivsg/shared-dense-text.jsonl": (
        "quantization/calibration/dense_ivsg/gen_data/Qwen3.5-27B.jsonl"
    ),
}


def quantize_with_autoround_mode1(
    *,
    model_dir: str,
    output_dir: str,
    device: str,
    quant_cfg: Mapping[str, Any],
    export_model_cfg: Mapping[str, Any],
    effective_config_file: str | None,
    workflow_seed: int,
) -> QuantResult:
    """Run the maintained AutoRound Gemma4 mode1 scripts through workflow config."""

    command, quanted_model_dir, algorithm = build_autoround_mode1_command(
        hf_model_dir=model_dir,
        output_dir=output_dir,
        device=device,
        quant_cfg=quant_cfg,
        export_model_cfg=export_model_cfg,
        workflow_seed=workflow_seed,
    )
    if not _as_bool(quant_cfg.get("dry_run", False)):
        _preflight_autoround_command_inputs(command)
        run_env = os.environ.copy()
        runtime_cfg = _section(quant_cfg, "runtime")
        env_cfg = runtime_cfg.get("env", quant_cfg.get("env", {}))
        if isinstance(env_cfg, Mapping):
            run_env.update({str(key): str(value) for key, value in env_cfg.items()})
        subprocess.run(command, cwd=str(_autoround_repo_root(quant_cfg)), env=run_env, check=True)
    return QuantResult(
        raw_model_dir=model_dir,
        quanted_model_dir=_normalize_path(quanted_model_dir),
    )


def build_autoround_mode1_command(
    *,
    hf_model_dir: str,
    output_dir: str,
    device: str,
    quant_cfg: Mapping[str, Any],
    export_model_cfg: Mapping[str, Any],
    workflow_seed: int,
) -> tuple[list[str], str, str]:
    """Translate Gemma4 AutoRound mode1 YAML into the checked-in script command."""

    _validate_artifact_format(quant_cfg)
    plan = build_gemma4_series_export_plan(hf_model_dir=hf_model_dir, export_model_cfg=export_model_cfg)

    preset = str(quant_cfg.get("preset") or "mode1").lower().replace("-", "_")
    if preset not in _SUPPORTED_AUTOROUND_PRESETS:
        raise ValueError(f"Gemma4 AutoRound supports preset/mode 'mode1' only, got {preset!r}")
    rotation = quant_cfg.get("rotation")
    if rotation not in {None, "", "none"}:
        raise ValueError("Gemma4 AutoRound mode1 is no-rotation; set quant.rotation to null")

    if plan.variant.topology == "moe":
        return _build_autoround_moe_mode1_command(
            hf_model_dir=hf_model_dir,
            output_dir=output_dir,
            device=device,
            quant_cfg=quant_cfg,
            workflow_seed=workflow_seed,
        )

    runtime_cfg = _section(quant_cfg, "runtime")
    calibration_cfg = _section(quant_cfg, "calibration")
    quanted_model_dir = _quant_output_path(output_dir, quant_cfg)
    script = _autoround_repo_root(quant_cfg) / "scripts_gemma4" / "quantize.py"
    if not script.exists():
        raise FileNotFoundError(f"Gemma4 AutoRound mode1 script not found: {script}")

    command = [
        sys.executable,
        str(script),
        "--model",
        _normalize_path(hf_model_dir),
        "--mode",
        "llm-only",
        "--device",
        str(device),
        "--output_dir",
        _normalize_path(quanted_model_dir),
        "--llm_bits",
        str(int(quant_cfg.get("bits", quant_cfg.get("llm_bits", 4)))),
        "--llm_group_size",
        str(int(quant_cfg.get("group_size", quant_cfg.get("llm_group_size", 64)))),
        "--iters",
        str(int(quant_cfg.get("iters", 200))),
        "--nsamples",
        str(int(calibration_cfg.get("nsamples", quant_cfg.get("nsamples", 128)))),
        "--seqlen",
        str(int(calibration_cfg.get("seqlen", quant_cfg.get("seqlen", 2048)))),
        "--batch_size",
        str(int(runtime_cfg.get("batch_size", quant_cfg.get("batch_size", 8)))),
        "--dataset",
        str(
            _resolve_autoround_dataset_value(
                calibration_cfg.get(
                    "dataset",
                    calibration_cfg.get("jsonl", quant_cfg.get("dataset", DEFAULT_AUTOROUND_DATASET)),
                )
            )
        ),
        "--seed",
        str(int(runtime_cfg.get("seed", quant_cfg.get("seed", workflow_seed)))),
        "--format",
        str(quant_cfg.get("format", quant_cfg.get("autoround_format", "auto_gptq"))),
    ]
    if _as_bool(quant_cfg.get("sym", True)):
        command.append("--sym")
    save_rotated = quant_cfg.get("save_rotated")
    if save_rotated:
        command.extend(["--save_rotated", _normalize_path(save_rotated)])
    extra_args = quant_cfg.get("extra_args") or runtime_cfg.get("extra_args")
    if extra_args:
        if not isinstance(extra_args, (list, tuple)):
            raise TypeError("Gemma4 AutoRound quant.extra_args must be a list/tuple when provided")
        command.extend(str(item) for item in extra_args)
    return command, quanted_model_dir, "autoround:mode1"


def _build_autoround_moe_mode1_command(
    *,
    hf_model_dir: str,
    output_dir: str,
    device: str,
    quant_cfg: Mapping[str, Any],
    workflow_seed: int,
) -> tuple[list[str], str, str]:
    """Build the Gemma4 26B-A4B MoE AutoRound command.

    The MoE script owns expert splitting/reload fixes; workflow only validates
    the stable public knobs and keeps the CLI mapping explicit.
    """

    runtime_cfg = _section(quant_cfg, "runtime")
    calibration_cfg = _section(quant_cfg, "calibration")
    validation_cfg = _section(quant_cfg, "validation")
    quanted_model_dir = _quant_output_path(output_dir, quant_cfg)
    script = _autoround_repo_root(quant_cfg) / "scripts_gemma4_moe" / "quantize_moe.py"
    if not script.exists():
        raise FileNotFoundError(f"Gemma4 AutoRound MoE script not found: {script}")

    command = [
        sys.executable,
        str(script),
        "--model",
        _normalize_path(hf_model_dir),
        "--device",
        str(device),
        "--output_dir",
        _normalize_path(quanted_model_dir),
        "--dtype",
        str(runtime_cfg.get("dtype", quant_cfg.get("dtype", "bfloat16"))),
        "--llm_bits",
        str(int(quant_cfg.get("bits", quant_cfg.get("llm_bits", 4)))),
        "--llm_group_size",
        str(int(quant_cfg.get("group_size", quant_cfg.get("llm_group_size", 64)))),
        "--iters",
        str(int(quant_cfg.get("iters", 200))),
        "--nsamples",
        str(int(calibration_cfg.get("nsamples", quant_cfg.get("nsamples", 128)))),
        "--seqlen",
        str(int(calibration_cfg.get("seqlen", quant_cfg.get("seqlen", 2048)))),
        "--batch_size",
        str(int(runtime_cfg.get("batch_size", quant_cfg.get("batch_size", 8)))),
        "--dataset",
        str(
            _resolve_autoround_dataset_value(
                calibration_cfg.get("dataset", calibration_cfg.get("jsonl", DEFAULT_AUTOROUND_DATASET))
            )
        ),
        "--seed",
        str(int(runtime_cfg.get("seed", quant_cfg.get("seed", workflow_seed)))),
        "--format",
        str(quant_cfg.get("format", quant_cfg.get("autoround_format", "auto_gptq"))),
        "--prompt",
        str(validation_cfg.get("prompt", quant_cfg.get("prompt", "你是谁"))),
        "--max_new_tokens",
        str(int(validation_cfg.get("max_new_tokens", quant_cfg.get("max_new_tokens", 128)))),
    ]
    if _as_bool(quant_cfg.get("sym", True)):
        command.append("--sym")
    if _as_bool(quant_cfg.get("quantize_router", False)):
        command.append("--quantize_router")
    ignore_layers = quant_cfg.get("ignore_layers")
    if ignore_layers:
        if isinstance(ignore_layers, (list, tuple)):
            ignore_layers = ",".join(str(item) for item in ignore_layers)
        command.extend(["--ignore_layers", str(ignore_layers)])
    extra_args = quant_cfg.get("extra_args") or runtime_cfg.get("extra_args")
    if extra_args:
        if not isinstance(extra_args, (list, tuple)):
            raise TypeError("Gemma4 AutoRound quant.extra_args must be a list/tuple when provided")
        command.extend(str(item) for item in extra_args)
    return command, quanted_model_dir, "autoround:mode1_moe"


def quantize_with_gptqmodel_recipe(
    *,
    model_dir: str,
    output_dir: str,
    device: str,
    quant_cfg: Mapping[str, Any],
    export_model_cfg: Mapping[str, Any],
    effective_config_file: str | None,
    workflow_seed: int,
) -> QuantResult:
    """Run GPTQModel's Gemma4 recipe and return a workflow ``QuantResult``."""

    recipe_kwargs = build_gptqmodel_recipe_kwargs(
        hf_model_dir=model_dir,
        output_dir=output_dir,
        device=device,
        quant_cfg=quant_cfg,
        export_model_cfg=export_model_cfg,
        workflow_seed=workflow_seed,
    )
    _preflight_gptqmodel_recipe_inputs(recipe_kwargs)
    recipe_entrypoint = str(quant_cfg.get("recipe_entrypoint") or _DEFAULT_RECIPE_ENTRYPOINT)
    recipe_result = _call_gptqmodel_recipe(recipe_entrypoint, recipe_kwargs)
    quanted_model_dir = _result_output_dir(recipe_result, recipe_kwargs["output_dir"])
    return QuantResult(
        raw_model_dir=model_dir,
        quanted_model_dir=_normalize_path(quanted_model_dir),
    )


def build_gptqmodel_recipe_kwargs(
    *,
    hf_model_dir: str,
    output_dir: str,
    device: str,
    quant_cfg: Mapping[str, Any],
    export_model_cfg: Mapping[str, Any],
    workflow_seed: int,
) -> dict[str, Any]:
    """Translate Gemma4 workflow YAML into stable GPTQModel recipe arguments."""

    _validate_artifact_format(quant_cfg)
    group_size = int(quant_cfg.get("group_size", 64))
    if group_size != 64:
        raise ValueError(f"Gemma4 Series quant group_size must be 64 when provided, got {group_size!r}")

    algorithm = str(quant_cfg.get("algorithm") or "gptqmodel").lower()
    method = str(quant_cfg.get("method") or _legacy_method_from_algorithm(algorithm)).lower()
    if method not in _SUPPORTED_GPTQMODEL_METHODS:
        raise ValueError(f"quant.method must be one of {sorted(_SUPPORTED_GPTQMODEL_METHODS)}, got {method!r}")

    preset = str(quant_cfg.get("preset") or "full_multimodal").lower()
    if preset not in _SUPPORTED_GPTQMODEL_PRESETS:
        raise ValueError(f"quant.preset must be one of {sorted(_SUPPORTED_GPTQMODEL_PRESETS)}, got {preset!r}")

    rotation = quant_cfg.get("rotation")
    if rotation == "":
        rotation = None
    if rotation not in _SUPPORTED_ROTATIONS:
        raise ValueError("Gemma4 GPTQModel quant does not support rotation; set quant.rotation to null")

    runtime_cfg = _section(quant_cfg, "runtime")
    calibration_cfg = _section(quant_cfg, "calibration")
    validation_cfg = _section(quant_cfg, "validation")
    moe_cfg = _section(quant_cfg, "moe")
    audio_cfg = _section(quant_cfg, "audio")
    offload_to_disk = _as_bool(runtime_cfg.get("offload_to_disk", quant_cfg.get("offload_to_disk", False)))
    plan = build_gemma4_series_export_plan(hf_model_dir=hf_model_dir, export_model_cfg=export_model_cfg)
    seed = int(runtime_cfg.get("seed", quant_cfg.get("seed", workflow_seed)))
    calibration_jsonl = _resolve_calibration_jsonl(plan.variant.topology, calibration_cfg)
    normalized_moe_cfg = _normalize_moe_config(moe_cfg, topology=plan.variant.topology)

    recipe_kwargs: dict[str, Any] = {
        "model_dir": _normalize_path(hf_model_dir),
        "output_dir": _quant_output_path(output_dir, quant_cfg),
        "method": method,
        "preset": preset,
        "rotation": rotation,
        "artifact_format": "gptqmodel_hf",
        "variant": plan.variant.name,
        "topology": plan.variant.topology,
        "capabilities": dict(plan.capabilities),
        "export_subgraphs": dict(plan.to_log_dict()["exports"]),
        "context_max_length": int(export_model_cfg.get("context_max_length", 2048)),
        "prefill_chunk_length": int(export_model_cfg.get("prefill_chunk_length", 320)),
        "group_size": group_size,
        "bits": int(quant_cfg.get("bits", 4)),
        "sym": _as_bool(quant_cfg.get("sym", True)),
        "iters": int(quant_cfg.get("iters", 200)),
        "seed": seed,
        "quant_nontext_module": _as_bool(quant_cfg.get("quant_nontext_module", False)),
        "device": device,
        "device_map": runtime_cfg.get("device_map"),
        "trust_remote_code": _as_bool(runtime_cfg.get("trust_remote_code", quant_cfg.get("trust_remote_code", True))),
        "batch_size": int(runtime_cfg.get("batch_size", quant_cfg.get("batch_size", 1))),
        "low_gpu_mem_usage": _optional_bool(runtime_cfg.get("low_gpu_mem_usage")),
        "offload_to_disk": offload_to_disk,
        "offload_path": runtime_cfg.get("offload_path"),
        "calibration_jsonl": calibration_jsonl,
        "calibration_text_key": calibration_cfg.get("text_key", calibration_cfg.get("calibration_text_key", "text")),
        "calibration_dataset": None if calibration_jsonl else calibration_cfg.get("dataset"),
        "calibration_split": None if calibration_jsonl else calibration_cfg.get("split"),
        "nsamples": int(calibration_cfg.get("nsamples", quant_cfg.get("nsamples", 256))),
        "seqlen": int(calibration_cfg.get("seqlen", quant_cfg.get("seqlen", quant_cfg.get("seq_len", 2048)))),
        "check_quant_text_demo": _as_bool(validation_cfg.get("check_quant_text_demo", True)),
        "check_quant_image_demo": _as_bool(validation_cfg.get("check_quant_image_demo", True)),
        "check_quant_video_demo": _as_bool(validation_cfg.get("check_quant_video_demo", plan.variant.has_video)),
        "check_quant_audio_demo": _as_bool(validation_cfg.get("check_quant_audio_demo", plan.variant.has_audio)),
        "check_rotation_ppl": _as_bool(validation_cfg.get("check_rotation_ppl", False)),
        "max_quant_layers": validation_cfg.get("max_quant_layers", quant_cfg.get("max_quant_layers")),
        "dry_run": _optional_bool(validation_cfg.get("dry_run", quant_cfg.get("dry_run"))),
        "skip_inference": _optional_bool(validation_cfg.get("skip_inference", quant_cfg.get("skip_inference"))),
        "run_inference": _optional_bool(validation_cfg.get("run_inference", quant_cfg.get("run_inference"))),
        "moe": normalized_moe_cfg,
        "self_attn_bits": _optional_int(normalized_moe_cfg.get("self_attn_bits", normalized_moe_cfg.get("attn_bits"))),
        "shared_expert_bits": _optional_int(normalized_moe_cfg.get("shared_expert_bits")),
        "expert_bits": _optional_int(normalized_moe_cfg.get("expert_bits")),
        "expert_down_bits": _optional_int(normalized_moe_cfg.get("expert_down_bits")),
        "audio": dict(audio_cfg),
        "audio_max_duration_seconds": _optional_int(audio_cfg.get("max_duration_seconds")),
    }
    return {key: value for key, value in recipe_kwargs.items() if value is not None}


def _call_gptqmodel_recipe(recipe_entrypoint: str, recipe_kwargs: dict[str, Any]) -> Any:
    recipe = _load_recipe_callable(recipe_entrypoint)
    signature = inspect.signature(recipe)
    try:
        signature.bind(**recipe_kwargs)
    except TypeError as exc:
        raise TypeError(
            "Gemma4 ModelZoo/GPTQModel recipe contract drift for "
            f"{recipe_entrypoint!r}: {exc}. Update both sides in the same change."
        ) from exc
    return recipe(**recipe_kwargs)


def _load_recipe_callable(recipe_entrypoint: str):
    if ":" not in recipe_entrypoint:
        raise ValueError(
            "quant.recipe_entrypoint must be '<module>:<callable>', "
            f"got {recipe_entrypoint!r}"
        )
    module_name, callable_name = recipe_entrypoint.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            "Gemma4SeriesWorkflow quant.algorithm='gptqmodel' requires "
            f"{_DEFAULT_RECIPE_ENTRYPOINT}. Update/provide GPTQModel with the Gemma4 recipe API, "
            "or use quant.algorithm='existing_hf' to reuse an already quantized GPTQModel HF directory, "
            "or config_overrides={'quant': None} for base validation."
        ) from exc
    try:
        recipe = getattr(module, callable_name)
    except AttributeError as exc:
        raise ImportError(
            f"GPTQModel recipe entrypoint {recipe_entrypoint!r} was imported but the callable is missing."
        ) from exc
    if not callable(recipe):
        raise TypeError(f"GPTQModel recipe entrypoint {recipe_entrypoint!r} is not callable")
    return recipe


def _autoround_repo_root(quant_cfg: Mapping[str, Any]) -> Path:
    root = (
        quant_cfg.get("autoround_repo")
        or quant_cfg.get("script_root")
        or os.environ.get("AUTOROUND_REPO")
        or _default_autoround_repo_root()
    )
    return Path(root).expanduser().resolve()


def _default_autoround_repo_root() -> Path:
    spec = importlib.util.find_spec("gptqmodel")
    package_locations = list(spec.submodule_search_locations or []) if spec and spec.submodule_search_locations else []
    if package_locations:
        return Path(package_locations[0]).resolve().parent / "third_party" / "auto-round"
    return Path("third_party/auto-round")


def _resolve_calibration_jsonl(topology: str, calibration_cfg: Mapping[str, Any]) -> str | None:
    explicit = calibration_cfg.get("jsonl") or calibration_cfg.get("calibration_jsonl")
    if explicit:
        return _resolve_calibration_value(explicit)

    dataset = calibration_cfg.get("dataset")
    if dataset and str(dataset).strip().lower() not in {"wikitext", "wikitext2", "wikitext-2-raw-v1"}:
        return None

    default_jsonl = DEFAULT_MOE_CALIBRATION_JSONL if topology == "moe" else DEFAULT_DENSE_CALIBRATION_JSONL
    return _resolve_calibration_value(default_jsonl)


def _resolve_calibration_value(value: Any) -> str:
    expanded = _expand_path_like_value(value)
    if expanded.startswith("gptqmodel://"):
        return _resolve_gptqmodel_resource(expanded)
    return expanded


def _resolve_autoround_dataset_value(value: Any) -> str:
    expanded = _expand_path_like_value(value)
    if expanded.startswith("gptqmodel://"):
        return _resolve_gptqmodel_resource(expanded)
    if expanded in _LEGACY_AUTOROUND_DATASETS:
        expanded = DEFAULT_AUTOROUND_DATASET
    if _is_repo_resource(expanded):
        return _resolve_repo_resource(expanded)
    if _is_existing_local_path(expanded):
        return str(Path(expanded).resolve())
    if _is_path_like_value(expanded):
        raise FileNotFoundError(
            "Gemma4 AutoRound calibration dataset path does not exist: "
            f"{expanded!r}. Download the prepared Artifactory archive and "
            f"place it at {DEFAULT_AUTOROUND_DATASET_PATH}."
        )
    return expanded


def _resolve_gptqmodel_resource(uri: str) -> str:
    relative_path = uri.removeprefix("gptqmodel://").lstrip("/")
    relative_path = _GPTQMODEL_RESOURCE_ALIASES.get(relative_path, relative_path)
    resolved_uri = f"gptqmodel://{relative_path}"
    spec = importlib.util.find_spec("gptqmodel")
    package_locations = list(spec.submodule_search_locations or []) if spec and spec.submodule_search_locations else []
    for package_root in package_locations:
        candidate = Path(package_root) / relative_path
        if candidate.is_file():
            return str(candidate.resolve())
    return resolved_uri


def _is_repo_resource(value: str) -> bool:
    return any(value.startswith(prefix) for prefix in _REPO_RESOURCE_PREFIXES)


def _resolve_repo_resource(uri: str) -> str:
    relative_path = uri
    for prefix in _REPO_RESOURCE_PREFIXES:
        if relative_path.startswith(prefix):
            relative_path = relative_path.removeprefix(prefix)
            break
    relative_path = relative_path.lstrip("/")
    for candidate in _repo_resource_candidates(relative_path):
        if candidate.is_file():
            return str(candidate.resolve())
    raise FileNotFoundError(
        "Gemma4 calibration repo resource does not exist: "
        f"{uri!r}. Use xh2modelzoo://data/calib_data/NeelNanda-pile-10k.jsonl, "
        "or set XH2MODELZOO_DATA_ROOT to the directory containing calibration data, "
        "or set XH2MODELZOO_ROOT to the repository root."
    )


def _repo_resource_candidates(relative_path: str) -> list[Path]:
    candidates: list[Path] = []
    data_root = os.environ.get("XH2MODELZOO_DATA_ROOT")
    if data_root:
        root = Path(data_root).expanduser()
        candidates.append(root / relative_path)
        if relative_path.startswith("data/"):
            candidates.append(root / relative_path.removeprefix("data/"))

    env_root = os.environ.get("XH2MODELZOO_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser() / relative_path)

    candidates.append(Path.cwd() / relative_path)
    candidates.append(_repo_root() / relative_path)
    return candidates


def _repo_root() -> Path:
    env_root = os.environ.get("XH2MODELZOO_ROOT")
    if env_root:
        root = Path(env_root).expanduser()
        if root.is_dir():
            return root.resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs_merak").is_dir() and (parent / "xhmodel_merak").is_dir():
            return parent
    return Path(__file__).resolve().parents[4]


def _preflight_gptqmodel_recipe_inputs(recipe_kwargs: Mapping[str, Any]) -> None:
    calibration_jsonl = recipe_kwargs.get("calibration_jsonl")
    if calibration_jsonl:
        _require_existing_local_file(
            calibration_jsonl,
            field="quant.calibration.jsonl",
            hint=(
                "Install/provide GPTQModel with its Gemma4 calibration resources, "
                "or override quant.calibration.jsonl with a readable JSONL file. "
                "Use quant.calibration.dataset only when the GPTQModel recipe should load "
                "a named dataset instead of a local JSONL."
            ),
        )


def _preflight_autoround_command_inputs(command: list[str]) -> None:
    if "--dataset" not in command:
        return
    dataset = command[command.index("--dataset") + 1]
    _require_existing_local_file(
        dataset,
        field="quant.calibration.dataset/jsonl",
        hint=(
            "Provide a readable local calibration path, or provide "
            "a supported AutoRound dataset name."
        ),
        only_if_path_like=True,
    )


def _expand_path_like_value(value: Any) -> str:
    return os.path.expanduser(os.path.expandvars(str(value)))


def _is_existing_local_path(value: str) -> bool:
    return Path(value).expanduser().is_file()


def _is_path_like_value(value: str) -> bool:
    return (
        "$" in value
        or value.startswith(("/", "./", "../", "~"))
        or value.endswith((".json", ".jsonl", ".txt"))
    )


def _require_existing_local_file(
    value: Any,
    *,
    field: str,
    hint: str,
    only_if_path_like: bool = False,
) -> None:
    raw_value = str(value)
    expanded = _expand_path_like_value(raw_value)
    unresolved_env = "$" in expanded
    path = Path(expanded)
    path_like = (
        unresolved_env
        or path.is_absolute()
        or raw_value.startswith(("./", "../", "~"))
        or raw_value.endswith((".jsonl", ".json", ".txt"))
    )
    if only_if_path_like and not path_like:
        return
    if unresolved_env or not path.is_file():
        raise FileNotFoundError(
            f"Gemma4 quant preflight failed: {field}={raw_value!r} "
            f"does not resolve to a readable local file. {hint}"
        )


def _normalize_moe_config(moe_cfg: Mapping[str, Any], *, topology: str) -> dict[str, Any]:
    normalized = dict(moe_cfg)
    if topology != "moe":
        return normalized

    routing = normalized.get("routing", normalized.get("moe_routing"))
    if routing is None:
        normalized["routing"] = "bypass"
        return normalized
    if isinstance(routing, bool):
        normalized["routing"] = "bypass" if routing else "none"
        return normalized
    normalized["routing"] = str(routing).strip().lower().replace("-", "_")
    return normalized


def _result_output_dir(recipe_result: Any, fallback: str) -> str:
    if recipe_result is None:
        return fallback
    if isinstance(recipe_result, Mapping):
        return str(recipe_result.get("output_dir") or recipe_result.get("quanted_model_dir") or fallback)
    output_dir = getattr(recipe_result, "output_dir", None)
    quanted_model_dir = getattr(recipe_result, "quanted_model_dir", None)
    return str(output_dir or quanted_model_dir or fallback)


def _validate_artifact_format(quant_cfg: Mapping[str, Any]) -> None:
    artifact_format = quant_cfg.get("artifact_format")
    output_format = quant_cfg.get("output_format")
    if artifact_format is not None and output_format is not None and artifact_format != output_format:
        raise ValueError(
            "quant.artifact_format and quant.output_format must match when both are provided; "
            f"got artifact_format={artifact_format!r}, output_format={output_format!r}"
        )
    resolved = artifact_format or output_format or "gptqmodel_hf"
    if resolved != "gptqmodel_hf":
        raise ValueError(f"Gemma4 GPTQModel quant requires artifact_format='gptqmodel_hf', got {resolved!r}")


def _legacy_method_from_algorithm(algorithm: str) -> str:
    if algorithm in _SUPPORTED_GPTQMODEL_METHODS:
        return algorithm
    return "gptq"


def _section(quant_cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = quant_cfg.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"quant.{name} must be a mapping when provided")
    return value


def _quant_output_path(output_dir: str, quant_cfg: Mapping[str, Any]) -> str:
    configured = quant_cfg.get("save_path") or quant_cfg.get("output_dir")
    return _normalize_path(configured or output_dir)


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
    "DEFAULT_DENSE_CALIBRATION_JSONL",
    "DEFAULT_MOE_CALIBRATION_JSONL",
    "build_autoround_mode1_command",
    "build_gptqmodel_recipe_kwargs",
    "quantize_with_autoround_mode1",
    "quantize_with_gptqmodel_recipe",
]
