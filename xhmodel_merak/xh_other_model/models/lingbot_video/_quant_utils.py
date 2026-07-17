from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from xhquant.api import (
    ConfigDict,
    DeviceType,
    FrontendType,
    PrecisionMode,
    QuantScheme,
    create_quant_config,
    ptq_quantize,
    to_export_graph,
    to_export_hmonnx_v2,
    to_frontend_graph,
    to_quant_graph,
)


def require_xh2a(target_device: str) -> DeviceType:
    if str(target_device) != "XH2a":
        raise ValueError(
            f"LingBot Video Merak workflow currently supports target_device='XH2a' only, got {target_device!r}."
        )
    return DeviceType.XH2a


def build_quant_scheme(
    target_device: str,
    component_cfg: Mapping[str, Any],
) -> QuantScheme:
    quant_type = str(component_cfg.get("quant_type", "w8a8h1_sefp"))
    kwargs: dict[str, Any] = {
        "target_device": require_xh2a(target_device),
        "quant_type": quant_type,
    }
    if component_cfg.get("nodes"):
        kwargs["nodes"] = dict(component_cfg["nodes"])
    if component_cfg.get("ops"):
        kwargs["ops"] = dict(component_cfg["ops"])
    return QuantScheme(**kwargs)


def build_quant_config(
    target_device: str,
    component_cfg: Mapping[str, Any],
) -> ConfigDict:
    return ConfigDict(create_quant_config(build_quant_scheme(target_device, component_cfg)))


def export_torch_component(
    *,
    model: nn.Module,
    inputs: Sequence[torch.Tensor],
    input_names: Sequence[str],
    output_names: Sequence[str],
    output_file: Path,
    target_device: str,
    component_cfg: Mapping[str, Any],
    exec_device: str,
    frontend_type: str = "TorchFX",
) -> dict[str, Any]:
    """Run the Merak frontend -> static-weight PTQ -> export pipeline."""
    if frontend_type not in {"TorchFX", "TorchExport"}:
        raise ValueError(f"Unsupported frontend_type: {frontend_type}")
    frontend_enum = FrontendType.TorchFX if frontend_type == "TorchFX" else FrontendType.TorchExport
    tensor_inputs = list(inputs)
    frontend_graph = to_frontend_graph(model.eval(), frontend_enum, tensor_inputs)
    quant_graph = to_quant_graph(
        frontend_graph,
        require_xh2a(target_device),
        build_quant_config(target_device, component_cfg),
    )
    ptq_quantize(
        quant_graph,
        [tensor_inputs],
        PrecisionMode.ALIGNED,
        [exec_device],
        auto_release_unused_parameters=True,
    )
    quant_graph.fixed()
    quant_graph.to("cpu")
    cpu_inputs = [tensor.detach().cpu() for tensor in tensor_inputs]
    calibration_file = output_file.parent / "calibration_inputs.pt"
    calibration_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cpu_inputs, calibration_file)
    export_graph = to_export_graph(quant_graph, cpu_inputs)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    exported_file = to_export_hmonnx_v2(
        export_graph,
        cpu_inputs,
        str(output_file),
        {
            "input_names": list(input_names),
            "output_names": list(output_names),
        },
    )
    return {
        "hmonnx_file": str(exported_file),
        "quant_type": str(component_cfg.get("quant_type", "w8a8h1_sefp")),
        "frontend_type": frontend_type,
        "input_names": list(input_names),
        "output_names": list(output_names),
        "input_shapes": [list(tensor.shape) for tensor in tensor_inputs],
        "input_dtypes": [str(tensor.dtype).removeprefix("torch.") for tensor in tensor_inputs],
        "calibration_inputs": str(calibration_file),
        "calibration_method": "shape_sample_dynamic_sefp_no_dataset_statistics",
    }
