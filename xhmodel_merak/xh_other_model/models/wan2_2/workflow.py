# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from xhmodel_merak.xh_other_model.workflows.base import BaseOtherModelWorkflow
from xhmodel_merak.xh_other_model.workflows.result import ExportResult, QuantResult

from .wan2_2_converter import Wan22ConvertConfig, Wan22Converter, WAN_EXPORT_COMPONENTS


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
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        export_cfg = workflow_config.build_export_dict()
        model_dir = self._resolve_export_model_dir(quant_result)
        work_dir = Path(output_dir).expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        config_file = workflow_config.dump(str(work_dir / f"{workflow_config.name}.yaml"))

        convert_config = _build_convert_config(export_cfg)
        converter = Wan22Converter(model_dir, convert_config)
        legacy_meta_path = converter.export(str(work_dir))
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
        return ExportResult(work_dir=str(work_dir), config_file=config_file, meta=meta)

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

        cfg = _config_from_legacy_meta(legacy_meta, golden_components=components)
        converter = Wan22Converter(str(legacy_meta["checkpoint_dir"]), cfg)
        converter.export(str(work_dir))

        golden_meta = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "work_dir": str(work_dir),
            "device": device,
            "golden_components": list(components),
        }
        golden_meta_file = work_dir / "golden_meta_info.json"
        golden_meta_file.write_text(json.dumps(_jsonable(golden_meta), ensure_ascii=False, indent=2), encoding="utf-8")
        return str(golden_meta_file)


def _build_convert_config(export_cfg: Mapping[str, Any]) -> Wan22ConvertConfig:
    cfg = export_cfg.get("wan2_2") or {}
    if not isinstance(cfg, Mapping):
        raise TypeError("export.wan2_2 must be a mapping")
    quant_cfg = export_cfg.get("quant_scheme") or cfg.get("quant_scheme") or {}
    convert_config = Wan22ConvertConfig(
        task=str(cfg.get("task", "i2v-A14B")),
        size=tuple(cfg.get("size", (832, 480))),
        frame_num=int(cfg.get("frame_num", 81)),
        sample_steps=int(cfg.get("sample_steps", 4)),
        sample_shift=float(cfg.get("sample_shift", 5.0)),
        sample_guide_scale=float(cfg.get("sample_guide_scale", 5.0)),
        prompt=str(cfg.get("prompt", "A calm seaside scene with gentle waves.")),
        negative_prompt=str(cfg.get("negative_prompt", "")),
        export_components=_normalize_components(cfg.get("components", ("t5",))),
        golden_components=(),
        torch_dtype=_parse_torch_dtype(str(cfg.get("torch_dtype", "float16"))),
        base_seed=int(cfg.get("base_seed", 0)),
            use_resolved_float_loader=bool(cfg.get("use_resolved_float_loader", False)),
    )
    for key, value in quant_cfg.items():
        if hasattr(convert_config.quant_scheme, key):
            setattr(convert_config.quant_scheme, key, value)
    target_device = str(export_cfg.get("target_device", "XH2a"))
    if hasattr(convert_config.quant_scheme, "target_device"):
        convert_config.quant_scheme.target_device = target_device
    return convert_config


def _config_from_legacy_meta(legacy_meta: Mapping[str, Any], golden_components: tuple[str, ...]) -> Wan22ConvertConfig:
    cfg = Wan22ConvertConfig(
        task=str(legacy_meta.get("task", "i2v-A14B")),
        size=tuple(legacy_meta.get("size", (832, 480))),
        frame_num=int(legacy_meta.get("frame_num", 81)),
        sample_steps=int(legacy_meta.get("sample_steps", 4)),
        sample_shift=float(legacy_meta.get("sample_shift", 5.0)),
        sample_guide_scale=float(legacy_meta.get("sample_guide_scale", 5.0)),
        prompt=str(legacy_meta.get("prompt", "A calm seaside scene with gentle waves.")),
        export_components=_normalize_components(legacy_meta.get("export_components", ())),
        golden_components=golden_components,
            use_resolved_float_loader=bool(legacy_meta.get("use_resolved_float_loader", False)),
    )
    quant_scheme = legacy_meta.get("quant_scheme") or {}
    for key, value in quant_scheme.items():
        if hasattr(cfg.quant_scheme, key):
            setattr(cfg.quant_scheme, key, value)
    return cfg


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
