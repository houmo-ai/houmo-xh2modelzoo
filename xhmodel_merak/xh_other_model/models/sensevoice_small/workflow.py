# Copyright 2025 HOUMO AI
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xhmodel_merak.xh_other_model.workflows.base import BaseOtherModelWorkflow
from xhmodel_merak.xh_other_model.workflows.result import ExportResult, QuantResult


class SenseVoiceSmallWorkflow(BaseOtherModelWorkflow):
    """Export SenseVoiceSmall to a static FP32 ONNX and an XH2a HMONNX."""

    def quant(
        self,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> QuantResult:
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        if workflow_config.quant is not None:
            raise NotImplementedError(
                "SenseVoiceSmall does integrated PTQ while exporting ONNX to HMONNX; "
                "the standalone workflow quant stage must remain null"
            )
        return QuantResult(
            raw_model_dir=self.model_dir,
            skipped=True,
            meta={"reason": "PTQ is integrated into the HMONNX export stage"},
        )

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        from .export_utils import export_fp32_onnx, export_hmonnx

        workflow_config = self.workflow_config.with_overrides(config_overrides)
        model_dir = self._resolve_export_model_dir(quant_result)
        _validate_model_dir(Path(model_dir))
        export_cfg = workflow_config.build_export_dict()
        work_dir = Path(output_dir).expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        config_file = workflow_config.dump(str(work_dir / f"{workflow_config.name}.yaml"))

        target_device = str(export_cfg["target_device"])
        if target_device != "XH2a":
            raise ValueError(f"SenseVoiceSmall workflow currently supports target_device='XH2a', got {target_device!r}")
        model_cfg = export_cfg.get("model") or {}
        if model_cfg.get("type") != "XHSenseVoiceSmallModel":
            raise ValueError("SenseVoiceSmall workflow requires export.model.type='XHSenseVoiceSmallModel'")

        onnx_cfg = export_cfg.get("onnx") or {}
        hmonnx_cfg = export_cfg.get("hmonnx") or {}
        if not isinstance(onnx_cfg, Mapping) or not isinstance(hmonnx_cfg, Mapping):
            raise TypeError("SenseVoiceSmall export.onnx and export.hmonnx must be mappings")
        export_onnx_enabled = bool(onnx_cfg.get("enabled", True))
        export_hmonnx_enabled = bool(hmonnx_cfg.get("enabled", True))
        if not export_onnx_enabled:
            raise ValueError("SenseVoiceSmall export currently requires export.onnx.enabled=true")

        assets_dir = _copy_runtime_assets(Path(model_dir), work_dir / "assets")
        onnx_meta = export_fp32_onnx(
            model_dir=model_dir,
            work_dir=work_dir,
            device=str(onnx_cfg.get("device", "cpu")),
            max_seq_len=int(onnx_cfg.get("max_seq_len", 512)),
            opset=int(onnx_cfg.get("opset", 14)),
            static=bool(onnx_cfg.get("static", True)),
            simplify=bool(onnx_cfg.get("simplify", True)),
            layer_norm_scale=float(onnx_cfg.get("layer_norm_scale", 32.0)),
            verbose=bool(onnx_cfg.get("verbose", False)),
            seed=int(self.seed),
        )
        onnx_file = Path(onnx_meta["onnx"])

        hmonnx_meta: dict[str, Any] | None = None
        if export_hmonnx_enabled:
            calibration_cfg = hmonnx_cfg.get("calibration") or {}
            if not isinstance(calibration_cfg, Mapping):
                raise TypeError("SenseVoiceSmall export.hmonnx.calibration must be a mapping")
            hmonnx_meta = export_hmonnx(
                onnx_path=onnx_file,
                model_dir=model_dir,
                work_dir=work_dir,
                target_device=target_device,
                quant_type=str(hmonnx_cfg.get("quant_type", "w8a8h1_sefp")),
                calib_metric=str(hmonnx_cfg.get("calib_metric", "minmax")),
                force_fp32_ops=hmonnx_cfg.get("force_fp32_ops", ["LayerNorm"]),
                calibration_cfg=calibration_cfg,
                execution_device=device,
                debug=bool(self.debug),
            )

        component_meta: dict[str, Any] = {
            "onnx_file": str(onnx_file.relative_to(work_dir)),
            "input_names": onnx_meta["input_names"],
            "output_names": onnx_meta["output_names"],
            "input_shapes": onnx_meta["input_shapes"],
            "max_seq_len": onnx_meta["max_seq_len"],
            "opset": onnx_meta["opset"],
            "static": onnx_meta["static"],
            "layer_norm_scale": onnx_meta["layer_norm_scale"],
        }
        if hmonnx_meta is not None:
            component_meta.update(
                {
                    "hmonnx_file": str(Path(hmonnx_meta["hmonnx"]).relative_to(work_dir)),
                    "quant_type": hmonnx_meta["quant_type"],
                    "calib_metric": hmonnx_meta["calib_metric"],
                    "force_fp32_ops": hmonnx_meta["force_fp32_ops"],
                    "calibration_source": hmonnx_meta["calibration_source"],
                    "calibration_samples": hmonnx_meta["calibration_samples"],
                    "calibration": hmonnx_meta["calibration"],
                }
            )

        meta: dict[str, Any] = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "config": str(Path(config_file).relative_to(work_dir)),
            "model_type": model_cfg["type"],
            "source_model_dir": model_dir,
            "assets_dir": str(assets_dir.relative_to(work_dir)),
            "target_device": target_device,
            "components": ["sensevoice"],
            "sensevoice": component_meta,
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
        import numpy as np
        import torch

        from xhquant.api import HMONNXGoldenInference

        from . import runtime

        work_dir, meta = runtime.load_export_meta(export_result.work_dir)
        component_meta = meta["sensevoice"]
        hmonnx_value = component_meta.get("hmonnx_file")
        if not hmonnx_value:
            raise ValueError("Cannot dump SenseVoice golden data because HMONNX export was disabled")
        hmonnx_file = work_dir / hmonnx_value
        assets_dir = work_dir / meta["assets_dir"]

        audio_value: Any = assets_dir / "example" / "en.mp3"
        language = "auto"
        textnorm = "woitn"
        if isinstance(input_messages, Mapping):
            audio_value = input_messages.get("audio") or input_messages.get("audio_file") or audio_value
            language = str(input_messages.get("language", language))
            textnorm = str(input_messages.get("textnorm", textnorm))
        elif isinstance(input_messages, (str, Path)):
            audio_value = input_messages
        elif input_messages is not None:
            raise TypeError("SenseVoice golden input_messages must be a path, a mapping, or None")

        torch.manual_seed(int(self.seed))
        np.random.seed(int(self.seed))
        frontend = runtime.build_frontend(assets_dir)
        audio_path = Path(str(audio_value)).expanduser().resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"SenseVoice golden audio file not found: {audio_path}")
        sample = runtime.Sample(audio=str(audio_path), text="", language=language, textnorm=textnorm)
        waveform = runtime.load_audio_any(sample, target_sr=int(frontend.cfg.fs))
        feat, feat_len = runtime.extract_features(frontend, waveform)
        inputs = runtime.make_inputs_for_sample(feat, feat_len, language, textnorm)

        torch_device = _select_torch_device(device)
        session = HMONNXGoldenInference(str(hmonnx_file))
        session.to(torch_device)
        input_info = runtime.read_hmonnx_input_info(hmonnx_file)
        aligned_inputs = runtime.align_hmonnx_inputs(input_info, inputs, torch_device)
        golden_dir = hmonnx_file.parent / "golden"
        if golden_dir.exists():
            shutil.rmtree(golden_dir)
        golden_dir.mkdir(parents=True, exist_ok=True)
        session.save_golden = True
        session.golden_dir = str(golden_dir)
        session.step = 0
        session(*aligned_inputs)

        golden_meta = {
            "work_dir": str(work_dir),
            "device": torch_device,
            "audio_file": str(audio_path),
            "language": language,
            "textnorm": textnorm,
            "feature_length": int(feat_len),
            "hmonnx_file": str(hmonnx_file.relative_to(work_dir)),
            "golden_dir": str(golden_dir.relative_to(work_dir)),
        }
        golden_meta_file = work_dir / "golden_meta_info.json"
        golden_meta_file.write_text(
            json.dumps(_jsonable(golden_meta), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(golden_meta_file)


def _validate_model_dir(model_dir: Path) -> None:
    required = ["config.yaml", "am.mvn", "model.pt"]
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"SenseVoiceSmall model directory {model_dir} is missing: {missing}")


def _copy_runtime_assets(model_dir: Path, assets_dir: Path) -> Path:
    assets_dir.mkdir(parents=True, exist_ok=True)
    required = ["config.yaml", "am.mvn"]
    optional = ["configuration.json", "tokens.json", "chn_jpn_yue_eng_ko_spectok.bpe.model"]
    for filename in required + optional:
        source = model_dir / filename
        if source.is_file():
            shutil.copy2(source, assets_dir / filename)
        elif filename in required:
            raise FileNotFoundError(f"Required SenseVoice runtime asset not found: {source}")

    example_dir = model_dir / "example"
    if example_dir.is_dir():
        destination = assets_dir / "example"
        destination.mkdir(parents=True, exist_ok=True)
        for audio_file in sorted(example_dir.iterdir()):
            if audio_file.is_file():
                shutil.copy2(audio_file, destination / audio_file.name)
    return assets_dir


def _select_torch_device(device: str) -> str:
    import torch

    if str(device).startswith("cuda") and torch.cuda.is_available():
        return str(device)
    return "cpu"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


__all__ = ["SenseVoiceSmallWorkflow"]
