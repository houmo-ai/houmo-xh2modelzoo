"""Merak workflow for MiniCPM-V-4.5."""

from __future__ import annotations

import copy
import json
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xhmodel_merak.xh_llm.workflows.base import BaseLLMWorkflow
from xhmodel_merak.xh_llm.workflows.result import ExportResult, QuantResult

from .vision import DEFAULT_GROUP_CAPACITY, DEFAULT_PATCH_CAPACITY


class MiniCPMV45Workflow(BaseLLMWorkflow):
    """Export the MiniCPM-V-4.5 Vision profiles and Qwen3-8B text graphs."""

    SUPPORTED_COMPONENTS = {"vision", "vision_video", "llm"}

    @staticmethod
    def visual_config_from_model(model_cfg: Mapping[str, Any]) -> dict[str, Any] | None:
        """Read the repository-standard ``export.model.visual_config`` nest.

        Returns a normalized dict with the keys consumed by this workflow's
        vision export (``patch_capacity``, ``group_capacity`` and the complete
        visual ``quant_scheme``), or ``None`` when the nested visual config is
        absent.  ``patch_capacity`` is the static graph capacity and is kept
        separate from ``max_size_w/h``, which are the processor's per-slice
        scale-resolution hints.  The resolution-to-capacity mapping remains
        as a compatibility fallback for older configs that do not declare
        ``patch_capacity`` explicitly.
        """
        visual_cfg = model_cfg.get("visual_config")
        if visual_cfg is None:
            return None
        if not isinstance(visual_cfg, Mapping):
            raise TypeError("export.model.visual_config must be a mapping")
        patch_size = int(visual_cfg.get("patch_size", 14))
        max_size_w = int(visual_cfg.get("max_size_w", 448))
        max_size_h = int(visual_cfg.get("max_size_h", max_size_w))
        if "patch_capacity" in visual_cfg:
            patch_capacity = int(visual_cfg["patch_capacity"])
            if patch_capacity <= 0:
                raise ValueError(f"visual_config.patch_capacity must be positive, got {patch_capacity}")
        else:
            if max_size_w != max_size_h:
                raise ValueError(
                    "MiniCPM-V-4.5 compatibility configs without patch_capacity require a square slice; "
                    f"visual_config.max_size_w and max_size_h must match, got {(max_size_h, max_size_w)}"
                )
            if max_size_w % patch_size != 0:
                raise ValueError(f"visual_config.max_size_w {max_size_w} must be divisible by patch_size {patch_size}")
            side = max_size_w // patch_size
            patch_capacity = side * side
        quant_scheme = visual_cfg.get("quant_scheme")
        if quant_scheme is None:
            quant_scheme = {"quant_type": "w8a8h1_sefp"}
        if not isinstance(quant_scheme, Mapping):
            raise TypeError("export.model.visual_config.quant_scheme must be a mapping")
        quant_scheme = copy.deepcopy(dict(quant_scheme))
        quant_type = str(quant_scheme.get("quant_type", "w8a8h1_sefp"))
        quant_scheme["quant_type"] = quant_type
        return {
            "patch_capacity": patch_capacity,
            "group_capacity": int(visual_cfg.get("max_size_t", 6)),
            "quant_type": quant_type,
            "quant_scheme": quant_scheme,
        }

    def quant(
        self,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> QuantResult:
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        if workflow_config.quant is None:
            return QuantResult(raw_model_dir=self.model_dir, skipped=True)

        from .quant_llm import quantize_minicpm_llm_gptq

        quant_cfg = workflow_config.quant
        if not isinstance(quant_cfg, Mapping):
            raise TypeError("MiniCPM quant must be a mapping")
        dequant_dir = quantize_minicpm_llm_gptq(
            model_dir=self.model_dir,
            output_dir=output_dir,
            quant_cfg=quant_cfg,
            device=device,
        )
        return QuantResult(raw_model_dir=self.model_dir, quanted_model_dir=dequant_dir)

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        import torch

        from xhmodel_merak.xh_llm.models.qwen3.qwen3_model import (
            XHQwen3ModelConfig,
        )
        from xhquant.api import xhquant_init

        from .model import XHMiniCPMV45Model
        from .vision_export import export_video_group_profile, export_vision_profile

        workflow_config = self.workflow_config.with_overrides(config_overrides)
        # The MiniCPM host (vision + llm) always loads from the raw model dir;
        # a GPTQ quant only replaces the LLM backbone weights.
        model_dir = self._resolve_export_model_dir(quant_result)
        if not quant_result.skipped:
            model_dir = quant_result.raw_model_dir
        export_cfg = workflow_config.build_export_dict()
        model_cfg = export_cfg.get("model") or {}
        if not isinstance(model_cfg, Mapping):
            raise TypeError("MiniCPM export.model must be a mapping")
        target_device = str(model_cfg["chip_arch"])
        if target_device.lower() != "xh2a":
            raise ValueError("MiniCPM-V-4.5 has only been validated for export.model.chip_arch=XH2a")

        components_cfg = export_cfg.get("components") or {}
        if not isinstance(components_cfg, Mapping):
            raise TypeError("MiniCPM export.components must be a mapping")
        unsupported = sorted(set(components_cfg) - self.SUPPORTED_COMPONENTS)
        if unsupported:
            raise ValueError(f"Unsupported MiniCPM component(s): {unsupported}")
        _reject_component_quant_types(components_cfg)
        if not _is_enabled(components_cfg.get("llm", False)):
            raise ValueError(
                "MiniCPM-V-4.5 export requires export.components.llm.enabled=true; "
                "the root golden metadata is produced by the LLM export"
            )

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

        meta: dict[str, Any] = {
            "schema_version": 1,
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "model_type": "MiniCPM-V-4.5",
            "hf_model": model_dir,
            "target_device": target_device,
            "config": str(Path(config_file).relative_to(work_dir)),
            "components": [],
            "vision": {},
        }

        vision_cfg = export_cfg.get("vision") or {}
        if not isinstance(vision_cfg, Mapping):
            raise TypeError("MiniCPM export.vision must be a mapping")
        # Repository VLM convention: `export.model.visual_config` is the
        # standard nesting (qwen2_vl).  When present it derives the capacity /
        # group capacity / vision quant scheme.  An explicit legacy
        # `export.vision.patch_capacity` wins over the model-level value.
        aligned_visual = self.visual_config_from_model(model_cfg)
        vision_quant_scheme = {"quant_type": "w8a8h1_sefp"}
        group_capacity = int(vision_cfg.get("group_capacity", DEFAULT_GROUP_CAPACITY))
        if aligned_visual is not None:
            if "patch_capacity" not in vision_cfg:
                patch_capacity = int(aligned_visual["patch_capacity"])
            else:
                patch_capacity = int(vision_cfg["patch_capacity"])
            if "group_capacity" not in vision_cfg:
                group_capacity = int(aligned_visual["group_capacity"])
            vision_quant_scheme = copy.deepcopy(aligned_visual["quant_scheme"])
        else:
            patch_capacity = int(vision_cfg.get("patch_capacity", DEFAULT_PATCH_CAPACITY))
            configured_quant_scheme = vision_cfg.get("quant_scheme", vision_quant_scheme)
            if not isinstance(configured_quant_scheme, Mapping):
                raise TypeError("export.vision.quant_scheme must be a mapping")
            vision_quant_scheme = copy.deepcopy(dict(configured_quant_scheme))
        vision_quant_type = str(vision_quant_scheme.get("quant_type", "w8a8h1_sefp"))
        vision_quant_scheme["quant_type"] = vision_quant_type
        keep_onnx = bool(vision_cfg.get("keep_onnx", False))
        release_date = time.strftime("%Y%m%d")
        model_quant_scheme = model_cfg.get("quant_scheme")
        if model_quant_scheme is None:
            model_quant_scheme = {"quant_type": "w8a8h1_sefp"}
        if not isinstance(model_quant_scheme, Mapping):
            raise TypeError("export.model.quant_scheme must be a mapping")
        model_quant_scheme = copy.deepcopy(dict(model_quant_scheme))
        model_default_quant_type = str(model_quant_scheme.get("quant_type", "w8a8h1_sefp"))
        model_quant_scheme["quant_type"] = model_default_quant_type
        modelcope_name = str(model_cfg.get("model_name", "minicpm_v_4_5"))
        llm_quant_type_planned = model_default_quant_type
        # 目录名与模型命名对齐 minicpm_v_4_6：
        # hmquant_<model_name>_<device>_<quant_type>_<date>。
        llm_model_name = f"{modelcope_name}_{target_device}_{llm_quant_type_planned}"

        llm_component_cfg = components_cfg.get("llm", False)
        if _is_enabled(llm_component_cfg):
            if not isinstance(llm_component_cfg, Mapping):
                raise TypeError("export.components.llm must be a mapping or false")
            llm_model_cfg = copy.deepcopy(dict(model_cfg))
            # export.model.quant_scheme 是 LLM 量化方案的唯一来源；完整
            # nodes/ops 配置必须原样传给 Qwen3 导出。
            llm_quant_type = model_default_quant_type
            llm_quant_scheme = copy.deepcopy(model_quant_scheme)
            llm_model_cfg.update(
                {
                    "hf_model": model_dir,
                    "chip_arch": target_device,
                    "model_name": llm_model_name,
                    "quant_scheme": llm_quant_scheme,
                }
            )
            if not quant_result.skipped and quant_result.quanted_model_dir:
                llm_model_cfg["quant_weight"] = str(Path(quant_result.quanted_model_dir) / "pytorch_model.bin")
            llm_output_dir = work_dir / "llm"
            llm_model = XHMiniCPMV45Model(XHQwen3ModelConfig(**llm_model_cfg))
            llm_model.export_hmonnx(str(llm_output_dir))
            artifact_dirs = sorted(llm_output_dir.glob("hmquant_*"))
            if len(artifact_dirs) != 1:
                raise RuntimeError(
                    "Expected exactly one MiniCPM LLM artifact directory, "
                    f"found {len(artifact_dirs)} under {llm_output_dir}"
                )
            artifact_dir = artifact_dirs[0]
            # 主目录名对齐 minicpm_v_4_6（hmquant_<model_name>_<device>_<quant_type>_<date>/），
            # 不附加分辨率后缀：MiniCPM-V-4.5 是动态切片 + 容量化导出，
            # 任意分辨率输入都支持，写死分辨率会误导使用者。
            main_dir_name = artifact_dir.name
            main_dir = work_dir / main_dir_name
            shutil.move(str(artifact_dir), str(main_dir))
            shutil.rmtree(llm_output_dir)
            child_meta_file = main_dir / "golden_meta_info.json"
            child_meta = json.loads(child_meta_file.read_text(encoding="utf-8"))

            meta["llm"] = {
                "quant_type": llm_quant_type,
                "artifact_dir": main_dir.name,
                "metadata": "golden_meta_info.json",
                "prefill_hmonnx": child_meta["prefill_hmonnx"],
                "decode_hmonnx": child_meta["decode_hmonnx"],
                "quant_embedding": child_meta["quant_embedding"],
                "hf_config": child_meta["hf_config"],
                "context_max_length": int(llm_model_cfg["context_max_length"]),
                "prefill_chunk_length": int(llm_model_cfg["prefill_chunk_length"]),
            }
            meta["components"].append("llm")
            work_dir = main_dir

        vision_component_cfg = components_cfg.get("vision", False)
        if _is_enabled(vision_component_cfg):
            if not isinstance(vision_component_cfg, Mapping):
                raise TypeError("export.components.vision must be a mapping or false")
            validation_target = _pair(
                vision_component_cfg.get("validation_target"),
                field="export.components.vision.validation_target",
            )
            profile = export_vision_profile(
                model_dir=model_dir,
                work_dir=work_dir,
                target_device=target_device,
                quant_scheme=vision_quant_scheme,
                patch_capacity=patch_capacity,
                validation_target=validation_target,
                device=device,
                keep_onnx=keep_onnx,
                # 发布目录命名 vision_1x：x1 = 单帧图像。视觉输入是容量化
                # 张量（patch_capacity），448 = processor scale_resolution。
                profile_dir=work_dir / "vision_1x",
            )
            meta["vision"] = profile
            meta["components"].append("vision")

        video_group_cfg = components_cfg.get("vision_video", False)
        if _is_enabled(video_group_cfg):
            if not isinstance(video_group_cfg, Mapping):
                raise TypeError("export.components.vision_video must be a mapping or false")
            validation_target = _pair(
                video_group_cfg.get("validation_target"),
                field="export.components.vision_video.validation_target",
            )
            profile = export_video_group_profile(
                model_dir=model_dir,
                work_dir=work_dir,
                target_device=target_device,
                quant_scheme=vision_quant_scheme,
                patch_capacity=patch_capacity,
                group_capacity=group_capacity,
                validation_target=validation_target,
                validation_frames=int(video_group_cfg.get("validation_frames", 3)),
                device=device,
                keep_onnx=keep_onnx,
                # x{group_capacity} = 时间组容量（视频路径，最多 6 帧合并
                # 为一个 3D-Resampler 组 → 64 token），与 x1 单帧图区分。
                profile_dir=work_dir / f"vision_{group_capacity}x",
            )
            meta["vision_video"] = profile
            meta["components"].append("vision_video")

        if not meta["components"]:
            raise ValueError("MiniCPM export must enable at least one component")

        # 对齐仓库 VLM 惯例：golden_meta_info.json 为唯一外部入口
        # （VLLMModelMeta 语义：LLMModelMeta + visual_config）。
        # LLM 导出已生成基础 LLMModelMeta，这里注入视觉 meta 与模型类型。
        golden_meta_file = work_dir / "golden_meta_info.json"
        if not golden_meta_file.is_file():
            raise RuntimeError(f"Missing LLM golden_meta_info.json under {work_dir}")
        golden_meta = json.loads(golden_meta_file.read_text(encoding="utf-8"))
        golden_meta["model_type"] = "MiniCPM-V-4.5"
        if meta.get("vision"):
            golden_meta["visual_config"] = _visual_meta_to_dict(meta["vision"])
        if meta.get("vision_video"):
            golden_meta["visual_video_config"] = _visual_meta_to_dict(meta["vision_video"])
        golden_meta_file.write_text(
            json.dumps(golden_meta, ensure_ascii=False, indent=4) + "\n",
            encoding="utf-8",
        )
        # 发布规范附加产物：独立推理脚本 + 转换日志（<llm_model_name>_hmonnx.py 等）。
        _write_release_aux_files(work_dir, llm_model_name, release_date)
        return ExportResult(
            work_dir=str(work_dir),
            config_file=config_file,
            meta=golden_meta,
        )

    def dump_golden(
        self,
        export_result: ExportResult,
        device: str,
        input_messages: Any = None,
    ) -> str:
        from .inference import dump_minicpm_v45_golden

        return dump_minicpm_v45_golden(
            work_dir=Path(export_result.work_dir),
            device=device,
            input_messages=input_messages,
        )


def _is_enabled(config: Any) -> bool:
    return config is not False and (not isinstance(config, Mapping) or bool(config.get("enabled", True)))


def _reject_component_quant_types(components_cfg: Mapping[str, Any]) -> None:
    """Reject the obsolete component-level quant_type shadow settings."""
    duplicates = sorted(
        name for name, config in components_cfg.items() if isinstance(config, Mapping) and "quant_type" in config
    )
    if duplicates:
        joined = ", ".join(f"export.components.{name}.quant_type" for name in duplicates)
        raise ValueError(
            f"{joined} is not supported; use export.model.quant_scheme.quant_type for LLM "
            "and export.model.visual_config.quant_scheme.quant_type for vision/video"
        )


def _visual_meta_to_dict(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Convert an export vision profile to a VisualModelMeta-style dict.

    Keeps the fields consumed by the MiniCPM runtime (hmonnx path, input
    names/shapes, patch capacity, embed dim, validation) under keys prefixed
    consistently with the qwen2_vl VisualModelMeta naming.
    """
    result: dict[str, Any] = {}
    for key in (
        "patch_capacity",
        "positions_per_side",
        "embed_dim",
        "hmonnx",
        "input_names",
        "input_shapes",
        "output_shape",
        "quant_type",
        "quant_scheme",
        "validation_target_size",
        "validation",
    ):
        if key in profile:
            result[key] = profile[key]
    if "group_capacity" in profile:
        result["group_capacity"] = profile["group_capacity"]
    return result


def _write_release_aux_files(work_dir: Path, model_name: str, release_date: str) -> None:
    """Write the release auxiliary files next to the artifact directory."""
    prefix = f"hmquant_{model_name}_{release_date}"

    demo_source = Path(__file__).resolve().parents[4] / "examples_merak" / "llm" / "minicpm_v_4_5" / "hmonnx_demo.py"
    if demo_source.is_file():
        target = work_dir / f"{prefix}_hmonnx.py"
        target.write_text(
            demo_source.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    convert_log = work_dir / "convert.log"
    if convert_log.is_file():
        target = work_dir / f"{prefix}_hmonnx_debug_xhquant.log"
        if not target.exists():
            shutil.copy2(convert_log, target)


def _pair(value: Any, *, field: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field} must contain [height, width]")
    return int(value[0]), int(value[1])


__all__ = ["MiniCPMV45Workflow"]
