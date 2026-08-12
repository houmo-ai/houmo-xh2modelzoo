# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: I001

# pyright: reportMissingImports=false

from __future__ import annotations

import json
import shutil
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import onnx
import torch

from xhmodel_merak.xh_other_model.workflows.base import BaseOtherModelWorkflow
from xhmodel_merak.xh_other_model.workflows.result import ExportResult, QuantResult
from xhquant.api import HMONNXGoldenInference

from .wan2_2_converter import WAN_EXPORT_COMPONENTS, Wan22ConvertConfig, Wan22Converter


class Wan22Workflow(BaseOtherModelWorkflow):
    """Merak workflow for Wan2.2 component export and golden dumping."""

    def quant(
        self,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> QuantResult:
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        if workflow_config.quant is not None:
            raise NotImplementedError("Wan2.2 does not use a standalone Merak quant stage; set quant: null")
        return QuantResult(
            raw_model_dir=self.model_dir,
            skipped=True,
            meta={"reason": "Wan2.2 quant/export is integrated in the export stage"},
        )

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        from .release_layout import build_release_directory

        workflow_config = self.workflow_config.with_overrides(config_overrides)
        export_cfg = workflow_config.build_export_dict()
        model_dir = self._resolve_export_model_dir(quant_result)
        output_dir_path = Path(output_dir).expanduser().resolve()
        if output_dir_path.exists():
            raise FileExistsError(f"Export directory already exists: {output_dir_path}")
        output_dir_path.mkdir(parents=True)
        release_staging_dir = output_dir_path / ".release"
        try:
            with tempfile.TemporaryDirectory(
                prefix=".wan2_2_export_staging_",
                dir=output_dir_path,
            ) as staging_value:
                work_dir = Path(staging_value)
                config_file = workflow_config.dump(str(work_dir / f"{workflow_config.name}.yaml"))
                convert_config = _build_convert_config(export_cfg)
                converter = Wan22Converter(model_dir, convert_config)
                legacy_meta_path = converter.export(str(work_dir), device=device)
                legacy_meta = json.loads(Path(legacy_meta_path).read_text(encoding="utf-8"))
                meta = {
                    "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "config": str(Path(config_file).relative_to(work_dir)),
                    "model_type": export_cfg["model"]["type"],
                    "source_model_dir": model_dir,
                    "target_device": str(export_cfg["target_device"]),
                    "components": list(convert_config.export_components),
                    "legacy_meta": str(Path(legacy_meta_path).relative_to(work_dir)),
                    "wan2_2": legacy_meta,
                }
                meta_file = work_dir / "export_meta_info.json"
                meta_file.write_text(json.dumps(_jsonable(meta), ensure_ascii=False, indent=2), encoding="utf-8")
                build_release_directory(work_dir, release_staging_dir)

            for path in release_staging_dir.iterdir():
                path.rename(output_dir_path / path.name)
            release_staging_dir.rmdir()
            release_dir = output_dir_path
        except Exception:
            shutil.rmtree(output_dir_path, ignore_errors=True)
            raise

        release_meta_file = release_dir / "export_meta_info.json"
        release_meta = json.loads(release_meta_file.read_text(encoding="utf-8"))
        release_config = release_dir / str(release_meta["config"])
        return ExportResult(work_dir=str(release_dir), config_file=str(release_config), meta=release_meta)

    def dump_golden(
        self,
        export_result: ExportResult,
        device: str,
        input_messages: Any = None,
    ) -> str:
        work_dir = Path(export_result.work_dir).expanduser().resolve()
        meta_file = work_dir / "export_meta_info.json"
        if not meta_file.is_file():
            raise FileNotFoundError(f"Wan2.2 export metadata not found: {meta_file}")
        export_meta = json.loads(meta_file.read_text(encoding="utf-8"))
        legacy_meta_file = work_dir / export_meta.get("legacy_meta", "wan2_2_export_meta.json")
        legacy_meta = json.loads(legacy_meta_file.read_text(encoding="utf-8"))
        components = tuple(legacy_meta.get("export_components", ()))
        if isinstance(input_messages, Mapping) and input_messages.get("components"):
            components = _normalize_components(input_messages["components"])

        execution_device = torch.device(device)
        base_seed = int(legacy_meta.get("base_seed", 0))
        component_results = {}
        for component in components:
            component_meta_file = _resolve_component_meta_file(work_dir, legacy_meta, component)
            component_meta = json.loads(component_meta_file.read_text(encoding="utf-8"))
            hmonnx_file = _resolve_hmonnx_file(work_dir, component_meta_file, component_meta, component)
            inputs, input_specs = _build_hmonnx_golden_inputs(
                hmonnx_file,
                component_meta,
                execution_device,
                seed=base_seed + WAN_EXPORT_COMPONENTS.index(component),
            )
            golden_dir = work_dir / component / "golden" / hmonnx_file.stem
            golden_dir.mkdir(parents=True, exist_ok=True)

            session = HMONNXGoldenInference(str(hmonnx_file))
            session.save_golden = True
            session.exec_device = execution_device
            session.golden_dir = str(golden_dir)
            with torch.no_grad():
                session.forward(*inputs)

            component_results[component] = {
                "hmonnx_file": _path_for_meta(hmonnx_file, work_dir),
                "golden_dir": _path_for_meta(golden_dir, work_dir),
                "inputs": input_specs,
            }
            del inputs, session
            if execution_device.type == "cuda":
                torch.cuda.empty_cache()

        golden_meta = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "work_dir": str(work_dir),
            "device": device,
            "base_seed": base_seed,
            "golden_components": list(components),
            "components": component_results,
        }
        golden_meta_file = work_dir / "golden_meta_info.json"
        golden_meta_file.write_text(json.dumps(_jsonable(golden_meta), ensure_ascii=False, indent=2), encoding="utf-8")
        return str(golden_meta_file)


def _build_convert_config(export_cfg: Mapping[str, Any]) -> Wan22ConvertConfig:
    cfg = export_cfg.get("wan2_2") or {}
    if not isinstance(cfg, Mapping):
        raise TypeError("export.wan2_2 must be a mapping")
    export_components = _normalize_components(cfg.get("components", ("t5",)))
    quant_types = dict(cfg.get("quant_types") or {})
    missing_quant_types = [component for component in export_components if component not in quant_types]
    if missing_quant_types:
        raise ValueError(
            "Missing export.wan2_2.quant_types for components "
            f"{missing_quant_types}; configured quant_types={quant_types}"
        )
    convert_config = Wan22ConvertConfig(
        quant_types=quant_types,
        task=str(cfg.get("task", "i2v-A14B")),
        size=tuple(cfg.get("size", (832, 480))),
        frame_num=int(cfg.get("frame_num", 81)),
        sample_steps=int(cfg.get("sample_steps", 4)),
        sample_shift=float(cfg.get("sample_shift", 5.0)),
        sample_guide_scale=float(cfg.get("sample_guide_scale", 5.0)),
        prompt=str(cfg.get("prompt", "A calm seaside scene with gentle waves.")),
        negative_prompt=str(cfg.get("negative_prompt", "")),
        export_components=export_components,
        golden_components=(),
        torch_dtype=_parse_torch_dtype(str(cfg.get("torch_dtype", "float16"))),
        base_seed=int(cfg.get("base_seed", 0)),
        use_resolved_float_loader=bool(cfg.get("use_resolved_float_loader", False)),
        release_dit_fp16_weights=bool(cfg.get("release_dit_fp16_weights", True)),
    )
    target_device = str(export_cfg.get("target_device", "XH2a"))
    if hasattr(convert_config.quant_scheme, "target_device"):
        convert_config.quant_scheme.target_device = target_device
    return convert_config


def _resolve_component_meta_file(
    work_dir: Path,
    legacy_meta: Mapping[str, Any],
    component: str,
) -> Path:
    meta_name = legacy_meta.get(f"{component}_meta", f"{component}_meta.json")
    meta_file = work_dir / str(meta_name)
    if not meta_file.is_file():
        raise FileNotFoundError(f"Wan2.2 {component} metadata not found: {meta_file}")
    return meta_file


def _resolve_hmonnx_file(
    work_dir: Path,
    component_meta_file: Path,
    component_meta: Mapping[str, Any],
    component: str,
) -> Path:
    hmonnx_value = component_meta.get("hmonnx_file")
    if not hmonnx_value:
        raise KeyError(f"Wan2.2 {component} metadata has no hmonnx_file: {component_meta_file}")

    hmonnx_path = Path(str(hmonnx_value))
    candidates = []
    if hmonnx_path.is_absolute():
        candidates.append(hmonnx_path)
    else:
        candidates.extend(
            (
                component_meta_file.parent / hmonnx_path,
                work_dir / hmonnx_path,
                work_dir / component / hmonnx_path,
            )
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Wan2.2 {component} HMONNX not found; metadata={component_meta_file}, "
        f"hmonnx_file={hmonnx_value!r}, candidates={[str(path) for path in candidates]}"
    )


_ONNX_TO_TORCH_DTYPE = {
    onnx.TensorProto.FLOAT: torch.float32,
    onnx.TensorProto.FLOAT16: torch.float16,
    onnx.TensorProto.BFLOAT16: torch.bfloat16,
    onnx.TensorProto.DOUBLE: torch.float64,
    onnx.TensorProto.INT8: torch.int8,
    onnx.TensorProto.INT16: torch.int16,
    onnx.TensorProto.INT32: torch.int32,
    onnx.TensorProto.INT64: torch.int64,
    onnx.TensorProto.UINT8: torch.uint8,
    onnx.TensorProto.BOOL: torch.bool,
}


def _build_hmonnx_golden_inputs(
    hmonnx_file: Path,
    component_meta: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    # HMONNX 权重通常位于 external_data；这里只读取图和输入协议，不加载权重文件。
    model = onnx.load(str(hmonnx_file), load_external_data=False)
    initializer_names = {value.name for value in model.graph.initializer}
    graph_input_values = [value for value in model.graph.input if value.name not in initializer_names]
    graph_inputs = {value.name: value for value in graph_input_values}
    input_names = [value.name for value in graph_input_values]
    metadata_input_names = list(component_meta.get("input_names") or ())
    if metadata_input_names and metadata_input_names != input_names:
        raise ValueError(
            f"HMONNX input order differs from component metadata: graph={input_names}, "
            f"metadata={metadata_input_names}, model={hmonnx_file}"
        )
    fallback_shapes = component_meta.get("sample_input_shapes") or {}
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    inputs = []
    specs = []
    for input_name in input_names:
        value_info = graph_inputs.get(input_name)
        if value_info is None:
            raise KeyError(f"HMONNX input {input_name!r} is not present in graph: {hmonnx_file}")
        tensor_type = value_info.type.tensor_type
        dtype = _ONNX_TO_TORCH_DTYPE.get(tensor_type.elem_type)
        if dtype is None:
            raise TypeError(f"Unsupported ONNX input dtype {tensor_type.elem_type} for {input_name!r}: {hmonnx_file}")

        fallback_shape = fallback_shapes.get(input_name)
        shape = []
        for dim_index, dim in enumerate(tensor_type.shape.dim):
            if dim.HasField("dim_value") and dim.dim_value > 0:
                shape.append(int(dim.dim_value))
            elif fallback_shape is not None and dim_index < len(fallback_shape):
                shape.append(int(fallback_shape[dim_index]))
            else:
                raise ValueError(
                    f"Cannot resolve static shape for HMONNX input {input_name!r} dim {dim_index}: {hmonnx_file}"
                )

        if dtype.is_floating_point:
            if "mask" in input_name.lower():
                tensor = torch.zeros(shape, device=device, dtype=dtype)
            else:
                tensor = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        elif dtype == torch.bool:
            tensor = torch.zeros(shape, device=device, dtype=dtype)
        else:
            tensor = torch.zeros(shape, device=device, dtype=dtype)
        inputs.append(tensor)
        specs.append(
            {
                "name": input_name,
                "shape": shape,
                "dtype": str(dtype).replace("torch.", ""),
            }
        )
    return inputs, specs


def _path_for_meta(path: Path, work_dir: Path) -> str:
    try:
        return str(path.relative_to(work_dir))
    except ValueError:
        return str(path)


def _normalize_components(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        raw = [item.strip() for item in value.split(",") if item.strip()]
    else:
        raw = list(value)
    aliases = {
        "vae_encoder": "vae_encode",
        "vae_decoder": "vae_decode",
        "low": "low_noise_model",
        "high": "high_noise_model",
    }
    result = []
    for item in raw:
        name = aliases.get(str(item).strip().lower(), str(item).strip().lower())
        if name not in WAN_EXPORT_COMPONENTS:
            raise ValueError(f"Unsupported Wan2.2 component: {item}, expected one of {WAN_EXPORT_COMPONENTS}")
        if name not in result:
            result.append(name)
    return tuple(result)


def _parse_torch_dtype(dtype_name: str) -> torch.dtype:
    dtype_name = dtype_name.replace("torch.", "")
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return dtype


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
