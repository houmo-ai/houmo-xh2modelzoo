from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ...workflows.base import BaseOtherModelWorkflow
from ...workflows.result import ExportResult, QuantResult
from .graph import RATE_CONFIG, specialize_model


class SileroVADWorkflow(BaseOtherModelWorkflow):
    def quant(
        self,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> QuantResult:
        config = self.workflow_config.with_overrides(config_overrides)
        if config.quant is not None:
            raise NotImplementedError("Silero VAD performs ONNX PTQ during HMONNX export; quant must be null")
        return QuantResult(raw_model_dir=self.model_dir, skipped=True)

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        config = self.workflow_config.with_overrides(config_overrides)
        model_dir = Path(self._resolve_export_model_dir(quant_result))
        export_cfg = config.build_export_dict()
        if export_cfg["model"].get("type") != "XHSileroVADModel":
            raise ValueError("Silero workflow requires export.model.type='XHSileroVADModel'")
        target = str(export_cfg["target_device"])
        if target != "XH2a":
            raise ValueError("Silero workflow currently supports XH2a only")
        source_onnx = _find_source_onnx(model_dir)
        variants = export_cfg.get("variants")
        if not isinstance(variants, list) or not variants:
            raise ValueError("Silero export.variants must be a non-empty list")

        work_dir = Path(output_dir).expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        config_file = config.dump(str(work_dir / f"{config.name}.yaml"))
        onnx_dir = work_dir / "onnx"
        hmonnx_dir = work_dir / "hmonnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        hmonnx_dir.mkdir(parents=True, exist_ok=True)

        components: dict[str, object] = {}
        for variant in variants:
            sample_rate = int(variant["sample_rate"])
            if sample_rate not in RATE_CONFIG:
                raise ValueError(f"unsupported Silero rate: {sample_rate}")
            quant_type = str(variant["quant_type"])
            static_path = specialize_model(
                source_onnx,
                onnx_dir / f"silero_vad_{sample_rate}_b1_ifless.onnx",
                sample_rate,
                int(export_cfg.get("check_steps", 5)),
            )
            hmonnx_path = hmonnx_dir / (f"silero_vad_{sample_rate}_b1_ifless_{target}_{quant_type}.onnx")
            _convert_hmonnx(
                static_path,
                hmonnx_path,
                target,
                quant_type,
                debug=self.debug,
            )
            components[str(sample_rate)] = {
                "sample_rate": sample_rate,
                "frame_samples": RATE_CONFIG[sample_rate]["frame"],
                "context_samples": RATE_CONFIG[sample_rate]["context"],
                "onnx_file": str(static_path.relative_to(work_dir)),
                "hmonnx_file": str(hmonnx_path.relative_to(work_dir)),
                "quant_type": quant_type,
                "onnx_sha256": _sha256(static_path),
                "hmonnx_sha256": _sha256(hmonnx_path),
            }

        meta = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": "XHSileroVADModel",
            "source_project": "https://github.com/snakers4/silero-vad",
            "source_model": str(source_onnx),
            "source_sha256": _sha256(source_onnx),
            "source_commit": _git_commit(model_dir),
            "target_device": target,
            "config": str(Path(config_file).relative_to(work_dir)),
            "components": components,
        }
        (work_dir / "export_meta_info.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return ExportResult(work_dir=str(work_dir), config_file=config_file, meta=meta)

    def dump_golden(
        self,
        export_result: ExportResult,
        device: str,
        input_messages: Any = None,
    ) -> str:
        import torch

        from xhquant.api import HMONNXGoldenInference

        from .runtime import load_audio

        work_dir = Path(export_result.work_dir)
        meta = json.loads((work_dir / "export_meta_info.json").read_text(encoding="utf-8"))
        audio_path = _resolve_audio(self.model_dir, input_messages)
        golden_root = work_dir / "golden"
        if golden_root.exists():
            shutil.rmtree(golden_root)
        device_value = device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
        golden_meta: dict[str, object] = {}
        for rate, component in meta["components"].items():
            sample_rate = int(rate)
            config = RATE_CONFIG[sample_rate]
            if audio_path is None:
                waveform = np.zeros(config["frame"], dtype=np.float32)
            else:
                waveform = load_audio(audio_path, sample_rate, seconds=1.0)[: config["frame"]]
                waveform = np.pad(waveform, (0, max(config["frame"] - waveform.size, 0)))
            model_input = np.concatenate(
                [
                    np.zeros(config["context"], dtype=np.float32),
                    waveform,
                ]
            ).reshape(1, -1)
            state = np.zeros((2, 1, 128), dtype=np.float32)
            hmonnx = work_dir / component["hmonnx_file"]
            golden_dir = golden_root / rate
            golden_dir.mkdir(parents=True)
            session = HMONNXGoldenInference(str(hmonnx))
            session.to(device_value)
            session.save_golden = True
            session.golden_dir = str(golden_dir)
            session.step = 0
            session(
                torch.from_numpy(model_input).to(torch.float16).to(device_value),
                torch.from_numpy(state).to(torch.float16).to(device_value),
            )
            golden_meta[rate] = {
                "hmonnx_file": component["hmonnx_file"],
                "golden_dir": str(golden_dir.relative_to(work_dir)),
                "audio": None if audio_path is None else str(audio_path),
            }
        result = golden_root / "manifest.json"
        result.write_text(
            json.dumps(golden_meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return str(result)


def _find_source_onnx(model_dir: Path) -> Path:
    candidates = [
        model_dir / "src/silero_vad/data/silero_vad.onnx",
        model_dir / "silero_vad.onnx",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Silero model directory {model_dir} has no silero_vad.onnx")


def _convert_hmonnx(
    onnx_path: Path,
    output_path: Path,
    target: str,
    quant_type: str,
    *,
    debug: bool,
) -> None:
    import onnx
    import torch

    from xhquant.api import (
        DeviceType,
        QuantScheme,
        convert_onnx_to_hmonnx,
        create_quant_config,
        xhquant_init,
    )

    model = onnx.load(onnx_path, load_external_data=False)
    args = []
    for value in model.graph.input:
        shape = [int(dim.dim_value) for dim in value.type.tensor_type.shape.dim]
        args.append(torch.zeros(shape, dtype=torch.float32))
    xhquant_init(str(output_path.parent / f"{output_path.stem}.log"), debug)
    device_type = DeviceType.XH2a
    quant_config = create_quant_config(QuantScheme(target_device=device_type, quant_type=quant_type))
    convert_onnx_to_hmonnx(
        str(onnx_path),
        args,
        device_type,
        str(output_path),
        quant_config=quant_config,
        input_names=[value.name for value in model.graph.input],
        output_names=[value.name for value in model.graph.output],
        simplify=False,
    )


def _resolve_audio(model_dir: str | Path, input_messages: Any) -> Path | None:
    value = None
    if isinstance(input_messages, Mapping):
        value = input_messages.get("audio")
    elif isinstance(input_messages, (str, Path)):
        value = input_messages
    elif input_messages is not None:
        raise TypeError("Silero golden input must be a path, mapping, or None")
    if value is not None:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidate = Path(model_dir) / "tests/data/test.wav"
    return candidate if candidate.is_file() else None


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


__all__ = ["SileroVADWorkflow"]
