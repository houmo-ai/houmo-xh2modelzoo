from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from xhquant.api import ConfigDict, PrecisionMode, ptq_quantize

# Component implementations still use the legacy lifecycle enum while the
# model package is registered under xh_llm.  Keep this explicit compatibility
# import until those component lifecycles are migrated to LLMModelState.
from xhmodel_merak.xh_other_model.eval_model_type import EvalModelType


def quant_config(target_device: str, component_cfg: Mapping[str, Any]) -> ConfigDict:
    from xhquant.api import DeviceType, QuantScheme, create_quant_config

    if target_device != "XH2a":
        raise ValueError(f"MiniCPM-o-4.5 supports target_device='XH2a', got {target_device!r}")
    scheme_kwargs: dict[str, Any] = {
        "target_device": DeviceType.XH2a,
        "quant_type": str(component_cfg["quant_type"]),
    }
    if component_cfg.get("nodes"):
        scheme_kwargs["nodes"] = deepcopy(dict(component_cfg["nodes"]))
    return ConfigDict(create_quant_config(QuantScheme(**scheme_kwargs)))


def build_component_model(model_dir: str, component_cfg: Mapping[str, Any]) -> Any:
    from ...register import XH_LLM_MODELS as MODELS

    model_cfg = {
        "type": component_cfg["model_type"],
        "hf_model": model_dir,
        "wrap_cfg": ConfigDict(deepcopy(dict(component_cfg.get("wrap_cfg") or {}))),
        "quant_config": ConfigDict(),
        "frontend_type": "TorchFX",
        "export_cfg": ConfigDict(
            {
                "input_names": list(component_cfg["input_names"]),
                "output_names": list(component_cfg["output_names"]),
            }
        ),
    }
    if component_cfg["model_type"] in {"MiniCPMO45VisionModel", "MiniCPMO45AudioModel"}:
        wrap_cfg = {"image_slice_max_size": list(component_cfg.get("image_slice_max_size", [40, 40]))}
        if component_cfg["model_type"] == "MiniCPMO45AudioModel":
            wrap_cfg["cache_capacity"] = int((component_cfg.get("streaming") or {}).get("cache_capacity", 1500))
        model_cfg["wrap_cfg"] = ConfigDict(wrap_cfg)
    model_cfg["quant_config"] = quant_config("XH2a", component_cfg)
    return MODELS.build(model_cfg)


def quantize_and_export(
    model: Any,
    data: dict[str, Any] | tuple[torch.Tensor, ...],
    output_dir: Path,
    prefix: str,
    device: str,
    *,
    calibrate: bool = False,
    calib_data: dict[str, Any] | tuple[torch.Tensor, ...] | None = None,
) -> str:
    """Export one component graph through the xhquant TorchFX pipeline.

    ``ptq_quantize`` is always required here: the export chain asserts a fixed
    quant graph (``convert_to_export_graph`` -> ``is_fixed()``), and the weight
    static quantization + fixed step lives inside ``ptq_quantize``. Activation
    scales are computed online per forward by the hardware operators, so the
    calibration data only drives shape propagation (``MetaInfoPro``) and is not
    used to freeze activation scales.

    - ``calibrate=False`` (vision/audio): shape-only calibration inputs derived
      from the export dummy data. Sufficient because activation quantization is
      dynamic; mirrors the zero-calibration static export of pi05/qwen3_asr
      encoder components while staying on the wrap-model export chain.
    - ``calibrate=True`` (llm/tts): use ``calib_data`` real inputs and enable the
      standard xhquant calibration entry point, matching pi05 LLM / qwen3_tts
      talker calibration.
    """
    model.convert_to_fronted_graph(data)
    model.convert_to_quant_graph("XH2a")
    if calibrate:
        model.change_eval_type(EvalModelType.CALIBRATION)
        model.to(device)
        model.enable_calibration()
        inputs = _flatten_inputs(model.prepare_inputs(calib_data if calib_data is not None else data))
    else:
        inputs = model._get_export_dummy_data(data)
    ptq_quantize(model.quanted_model, [inputs], PrecisionMode.ALIGNED, [torch.device(device)])
    if calibrate:
        model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    model.fixed()
    model.to("cpu")
    return model.to_export_onnx(data, str(output_dir), prefix)[0]


def _flatten_inputs(inputs: Any) -> list[Any]:
    flattened: list[Any] = []
    for arg in inputs:
        if isinstance(arg, (list, tuple)):
            flattened.extend(arg)
        else:
            flattened.append(arg)
    return flattened


def relative_artifact_path(work_dir: Path, artifact_path: Path) -> str:
    if artifact_path.is_absolute() or artifact_path.is_relative_to(work_dir):
        return str(artifact_path.resolve().relative_to(work_dir.resolve()))
    return str(artifact_path)


def normalize_component_artifacts(work_dir: Path, result: dict[str, object]) -> dict[str, object]:
    result["graphs"] = {role: relative_artifact_path(work_dir, Path(path)) for role, path in result["graphs"].items()}
    for artifact_name in ("rand_noise_file", "phase_noise_file", "source_noise_file"):
        if artifact_name in result:
            result[artifact_name] = relative_artifact_path(work_dir, Path(result[artifact_name]))
    stream_contract = result.get("stream_contract")
    if isinstance(stream_contract, dict):
        for artifact_name in ("phase_noise_file", "source_noise_file", "initial_source_cache_file"):
            if artifact_name in stream_contract:
                stream_contract[artifact_name] = relative_artifact_path(work_dir, Path(stream_contract[artifact_name]))
        if "base_cache_files" in stream_contract:
            stream_contract["base_cache_files"] = {
                cache_name: relative_artifact_path(work_dir, Path(cache_path))
                for cache_name, cache_path in stream_contract["base_cache_files"].items()
            }
    return result


__all__ = [
    "build_component_model",
    "normalize_component_artifacts",
    "quant_config",
    "quantize_and_export",
    "relative_artifact_path",
]
