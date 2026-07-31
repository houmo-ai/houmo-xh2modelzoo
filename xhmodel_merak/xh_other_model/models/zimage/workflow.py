import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from diffusers import ZImagePipeline

from xhmodel_merak.xh_other_model.workflows.base import BaseOtherModelWorkflow
from xhmodel_merak.xh_other_model.workflows.result import ExportResult, QuantResult

from .dit_converter import Dit_ConverterXH2a
from .qwen3_convert_config import Qwen3LegacyConvertConfig
from .qwen3_converter import Qwen3LegacyConverterXH2a
from .vae_converter import VAE_ConverterXH2a


class ZImageWorkflow(BaseOtherModelWorkflow):
    def quant(
        self,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> QuantResult:
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        if workflow_config.quant is not None:
            raise NotImplementedError("ZImage does not support a separate quant stage; set quant: null")
        return QuantResult(raw_model_dir=self.model_dir, skipped=True)

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        model_dir = self._resolve_export_model_dir(quant_result)
        export_cfg = workflow_config.build_export_dict()
        zimage_cfg = export_cfg.get("zimage") or {}
        if not isinstance(zimage_cfg, Mapping):
            raise TypeError("export.zimage must be a mapping")

        work_dir = Path(output_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        config_file = workflow_config.dump(str(work_dir / f"{workflow_config.name}.yaml"))

        component_dirs = {}
        components = tuple(zimage_cfg.get("components", ("text_encoder", "vae", "dit")))
        if "text_encoder" in components:
            text_dir = work_dir / "text_encoder"
            text_dir.mkdir(parents=True, exist_ok=True)
            text_cfg = _build_convert_config(zimage_cfg.get("text_encoder") or zimage_cfg)
            pipe = _load_pipe(model_dir, zimage_cfg, device)
            Qwen3LegacyConverterXH2a(text_cfg)._convert(pipe.text_encoder.half(), text_dir)
            component_dirs["text_encoder"] = str(text_dir.relative_to(work_dir))
        if "vae" in components:
            vae_dir = work_dir / "vae"
            vae_dir.mkdir(parents=True, exist_ok=True)
            vae_cfg = _build_convert_config(zimage_cfg.get("vae") or zimage_cfg)
            pipe = _load_pipe(model_dir, zimage_cfg, device)
            pipe.text_encoder = None
            pipe.transformer = None
            VAE_ConverterXH2a(vae_cfg)._convert(pipe.vae.half(), vae_dir, pipe.image_processor.postprocess)
            component_dirs["vae"] = str(vae_dir.relative_to(work_dir))
        if "dit" in components:
            dit_dir = work_dir / "dit"
            dit_dir.mkdir(parents=True, exist_ok=True)
            dit_cfg = _build_convert_config(zimage_cfg.get("dit") or zimage_cfg)
            pipe = _load_pipe(model_dir, zimage_cfg, device)
            Dit_ConverterXH2a(dit_cfg)._convert(pipe.transformer.half(), dit_dir, pipe.image_processor.postprocess)
            component_dirs["dit"] = str(dit_dir.relative_to(work_dir))

        meta = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "config": str(Path(config_file).relative_to(work_dir)),
            "model_type": export_cfg["model"]["type"],
            "source_model_dir": model_dir,
            "target_device": str(export_cfg.get("target_device", "XH2a")),
            "components": list(component_dirs),
            "zimage": component_dirs,
        }
        meta_file = work_dir / "export_meta_info.json"
        meta_file.write_text(json.dumps(_jsonable(meta), ensure_ascii=False, indent=4), encoding="utf-8")
        return ExportResult(work_dir=str(work_dir), config_file=config_file, meta=meta)

    def dump_golden(
        self,
        export_result: ExportResult,
        device: str,
        input_messages: Any = None,
    ) -> str:
        return str(Path(export_result.work_dir))


def _build_convert_config(cfg: Mapping[str, Any]) -> Qwen3LegacyConvertConfig:
    from xhquant.api import DeviceType, QuantScheme

    return Qwen3LegacyConvertConfig(
        batch_size=int(cfg.get("batch_size", 1)),
        context_length=int(cfg.get("context_length", 2048)),
        input_sequence_length=int(cfg.get("input_sequence_length", 256)),
        quant_scheme=QuantScheme(target_device=getattr(DeviceType, str(cfg.get("target_device", "XH2a"))), quant_type=str(cfg.get("quant_type", "w8a8h1_sefp"))),
        quant_weight=cfg.get("quant_weight"),
    )


def _load_pipe(model_dir: str, cfg: Mapping[str, Any], device: str) -> ZImagePipeline:
    pipe = ZImagePipeline.from_pretrained(
        model_dir,
        torch_dtype=_torch_dtype(cfg.get("torch_dtype", "float16")),
        low_cpu_mem_usage=False,
    )
    return pipe.to(device)


def _torch_dtype(name: str):
    import torch

    dtype = getattr(torch, name.replace("torch.", ""), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype: {name}")
    return dtype


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return value
