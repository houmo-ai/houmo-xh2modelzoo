"""Merak workflow for MiniCPM-V-4.6."""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xhmodel_merak.xh_llm.workflows.base import BaseLLMWorkflow
from xhmodel_merak.xh_llm.workflows.result import ExportResult, QuantResult


class MiniCPMV46Workflow(BaseLLMWorkflow):
    """Export the two MiniCPM Vision profiles and Qwen3.5 text graphs."""

    SUPPORTED_COMPONENTS = {"vision_4x", "vision_16x", "llm"}

    def quant(
        self,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> QuantResult:
        del output_dir, device
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        if workflow_config.quant is not None:
            raise NotImplementedError(
                "MiniCPMV46Workflow expects quant=null; each component is quantized during HMONNX export"
            )
        return QuantResult(raw_model_dir=self.model_dir, skipped=True)

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        import torch

        from xhmodel_merak.xh_llm.models.qwen3_5.xh_qwen3_5_config import (
            XHQwen3_5ModelConfig,
        )
        from xhquant.api import xhquant_init

        from .model import XHMiniCPMV46Model
        from .vision_export import export_vision_profile

        workflow_config = self.workflow_config.with_overrides(config_overrides)
        model_dir = self._resolve_export_model_dir(quant_result)
        export_cfg = workflow_config.build_export_dict()
        model_cfg = export_cfg.get("model") or {}
        if not isinstance(model_cfg, Mapping):
            raise TypeError("MiniCPM export.model must be a mapping")
        target_device = str(model_cfg["chip_arch"])
        if target_device.lower() != "xh2a":
            raise ValueError("MiniCPM-V-4.6 has only been validated for export.model.chip_arch=XH2a")

        work_dir = Path(output_dir).resolve()
        if work_dir.exists() and any(work_dir.iterdir()):
            raise FileExistsError(
                f"MiniCPM export directory is not empty: {work_dir}. "
                "Remove it explicitly or use the example's --overwrite option."
            )
        work_dir.mkdir(parents=True, exist_ok=True)
        xhquant_init(work_dir / "convert.log", debug=self.debug)
        torch.manual_seed(self.seed)
        config_file = workflow_config.dump(str(work_dir / f"{workflow_config.name}.yaml"))

        components_cfg = export_cfg.get("components") or {}
        if not isinstance(components_cfg, Mapping):
            raise TypeError("MiniCPM export.components must be a mapping")
        unsupported = sorted(set(components_cfg) - self.SUPPORTED_COMPONENTS)
        if unsupported:
            raise ValueError(f"Unsupported MiniCPM component(s): {unsupported}")

        meta: dict[str, Any] = {
            "schema_version": 1,
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "model_type": "MiniCPM-V-4.6",
            "hf_model": model_dir,
            "target_device": target_device,
            "config": str(Path(config_file).relative_to(work_dir)),
            "components": [],
            "vision": {},
        }

        vision_cfg = export_cfg.get("vision") or {}
        if not isinstance(vision_cfg, Mapping):
            raise TypeError("MiniCPM export.vision must be a mapping")
        token_capacity = int(vision_cfg.get("token_capacity", 1536))
        keep_onnx = bool(vision_cfg.get("keep_onnx", False))
        for component_name, mode in (("vision_4x", "4x"), ("vision_16x", "16x")):
            component_cfg = components_cfg.get(component_name, False)
            if not _is_enabled(component_cfg):
                continue
            if not isinstance(component_cfg, Mapping):
                raise TypeError(f"export.components.{component_name} must be a mapping or false")
            validation_target = _pair(
                component_cfg.get("validation_target"),
                field=f"export.components.{component_name}.validation_target",
            )
            profile = export_vision_profile(
                model_dir=model_dir,
                work_dir=work_dir,
                target_device=target_device,
                quant_type=str(component_cfg.get("quant_type", "w8a8h1_sefp")),
                token_capacity=token_capacity,
                downsample_mode=mode,
                validation_target=validation_target,
                device=device,
                keep_onnx=keep_onnx,
            )
            meta["vision"][mode] = profile
            meta["components"].append(component_name)

        llm_component_cfg = components_cfg.get("llm", False)
        if _is_enabled(llm_component_cfg):
            if not isinstance(llm_component_cfg, Mapping):
                raise TypeError("export.components.llm must be a mapping or false")
            llm_model_cfg = copy.deepcopy(dict(model_cfg))
            llm_quant_type = str(llm_component_cfg.get("quant_type", "w8a8h1_sefp"))
            configured_name = str(llm_model_cfg.get("model_name", "minicpm_v_4_6_qwen3_5_0_8b"))
            llm_model_cfg.update(
                {
                    "hf_model": model_dir,
                    "chip_arch": target_device,
                    "model_name": f"{configured_name}_{target_device}_{llm_quant_type}",
                    "visual_config": None,
                    "quant_scheme": {"quant_type": llm_quant_type},
                }
            )
            llm_output_dir = work_dir / "llm"
            llm_model = XHMiniCPMV46Model(XHQwen3_5ModelConfig(**llm_model_cfg))
            llm_model.export_hmonnx(str(llm_output_dir))
            artifact_dirs = sorted(llm_output_dir.glob("hmquant_*"))
            if len(artifact_dirs) != 1:
                raise RuntimeError(
                    "Expected exactly one MiniCPM LLM artifact directory, "
                    f"found {len(artifact_dirs)} under {llm_output_dir}"
                )
            artifact_dir = artifact_dirs[0]
            child_meta_file = artifact_dir / "golden_meta_info.json"
            child_meta = json.loads(child_meta_file.read_text(encoding="utf-8"))

            def child_path(value: str) -> str:
                return str((artifact_dir / value).relative_to(work_dir))

            meta["llm"] = {
                "quant_type": llm_quant_type,
                "artifact_dir": str(artifact_dir.relative_to(work_dir)),
                "metadata": str(child_meta_file.relative_to(work_dir)),
                "prefill_hmonnx": child_path(child_meta["prefill_hmonnx"]),
                "decode_hmonnx": child_path(child_meta["decode_hmonnx"]),
                "quant_embedding": child_path(child_meta["quant_embedding"]),
                "hf_config": child_path(child_meta["hf_config"]),
                "context_max_length": int(llm_model_cfg["context_max_length"]),
                "prefill_chunk_length": int(llm_model_cfg["prefill_chunk_length"]),
            }
            meta["components"].append("llm")

        if not meta["components"]:
            raise ValueError("MiniCPM export must enable at least one component")

        meta_file = work_dir / "export_meta_info.json"
        meta_file.write_text(
            json.dumps(meta, ensure_ascii=False, indent=4) + "\n",
            encoding="utf-8",
        )
        return ExportResult(
            work_dir=str(work_dir),
            config_file=config_file,
            meta=meta,
        )

    def dump_golden(
        self,
        export_result: ExportResult,
        device: str,
        input_messages: Any = None,
    ) -> str:
        from .inference import dump_minicpm_v46_golden

        return dump_minicpm_v46_golden(
            work_dir=Path(export_result.work_dir),
            device=device,
            input_messages=input_messages,
        )


def _is_enabled(config: Any) -> bool:
    return config is not False and (not isinstance(config, Mapping) or bool(config.get("enabled", True)))


def _pair(value: Any, *, field: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field} must contain [height, width]")
    return int(value[0]), int(value[1])


__all__ = ["MiniCPMV46Workflow"]
