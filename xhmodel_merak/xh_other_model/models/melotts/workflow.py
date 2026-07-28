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
import onnx

from ...workflows.base import BaseOtherModelWorkflow
from ...workflows.result import ExportResult, QuantResult
from .assets import resolve_model_assets
from .graph import export_static_graphs
from .runtime import OrtRunner, prepare_decoder_inputs


class MeloTTSWorkflow(BaseOtherModelWorkflow):
    def quant(
        self,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> QuantResult:
        config = self.workflow_config.with_overrides(config_overrides)
        if config.quant is not None:
            raise NotImplementedError("MeloTTS performs ONNX PTQ during HMONNX export; quant must be null")
        return QuantResult(raw_model_dir=self.model_dir, skipped=True)

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        import torch

        config = self.workflow_config.with_overrides(config_overrides)
        model_dir = Path(self._resolve_export_model_dir(quant_result))
        export_cfg = config.build_export_dict()
        if export_cfg["model"].get("type") != "XHMeloTTSModel":
            raise ValueError("MeloTTS requires export.model.type='XHMeloTTSModel'")
        target = str(export_cfg["target_device"])
        if target != "XH2a":
            raise ValueError("MeloTTS workflow currently supports XH2a only")
        lmax = int(export_cfg["text_max_length"])
        tmax = int(export_cfg["acoustic_max_length"])
        lang_id = int(export_cfg.get("language_id", 3))
        speaker_id = int(export_cfg.get("speaker_id", 1))
        opset = int(export_cfg.get("opset", 17))
        components_cfg = export_cfg["components"]
        assets = resolve_model_assets(model_dir)
        work_dir = Path(output_dir).expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        config_file = config.dump(str(work_dir / f"{config.name}.yaml"))
        onnx_dir = work_dir / "onnx"
        hmonnx_dir = work_dir / "hmonnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        hmonnx_dir.mkdir(parents=True, exist_ok=True)
        encoder_path, decoder_path, strict_validation = export_static_graphs(
            melo_root=assets.source_root,
            config_path=assets.config,
            checkpoint_path=assets.checkpoint,
            release_onnx=assets.release_package / "model.onnx",
            output_dir=onnx_dir,
            lmax=lmax,
            tmax=tmax,
            lang_id=lang_id,
            speaker_id=speaker_id,
            opset=opset,
            validation_lengths=tuple(int(value) for value in export_cfg.get("validation_lengths", [3, 5, 9])),
        )
        source_graphs = {
            "encoder": encoder_path,
            "decoder": decoder_path,
        }
        components = {}
        for role, onnx_path in source_graphs.items():
            quant_type = str(components_cfg[role]["quant_type"])
            hmonnx_path = hmonnx_dir / (f"{onnx_path.stem}_{target}_{quant_type}.onnx")
            _convert_hmonnx(
                onnx_path,
                hmonnx_path,
                quant_type,
                debug=self.debug,
            )
            components[role] = {
                "onnx_file": str(onnx_path.relative_to(work_dir)),
                "hmonnx_file": str(hmonnx_path.relative_to(work_dir)),
                "quant_type": quant_type,
                "onnx_sha256": _sha256(onnx_path),
                "hmonnx_sha256": _sha256(hmonnx_path),
            }
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        meta = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": "XHMeloTTSModel",
            "source_project": "https://github.com/myshell-ai/MeloTTS",
            "source_commit": _git_commit(assets.source_root),
            "source_root": str(assets.source_root),
            "config_file": str(assets.config),
            "config_sha256": _sha256(assets.config),
            "checkpoint_file": str(assets.checkpoint),
            "checkpoint_sha256": _sha256(assets.checkpoint),
            "release_model": str(assets.release_package / "model.onnx"),
            "release_model_sha256": _sha256(assets.release_package / "model.onnx"),
            "target_device": target,
            "workflow_config": str(Path(config_file).relative_to(work_dir)),
            "sample_rate": 44100,
            "hop_length": 512,
            "text_max_length": lmax,
            "acoustic_max_length": tmax,
            "language_id": lang_id,
            "speaker_id": speaker_id,
            "components": components,
            "strict_pytorch_validation": strict_validation,
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

        work_dir = Path(export_result.work_dir)
        meta = json.loads((work_dir / "export_meta_info.json").read_text(encoding="utf-8"))
        tokens, tones = _golden_tokens(input_messages)
        lmax = int(meta["text_max_length"])
        if tokens.size > lmax:
            raise ValueError(f"golden token length {tokens.size} exceeds Lmax={lmax}")
        x = np.zeros((1, lmax), dtype=np.int32)
        tone_input = np.zeros_like(x)
        x[0, : tokens.size] = tokens
        tone_input[0, : tones.size] = tones
        encoder_feed = {
            "x": x,
            "x_lengths": np.asarray([tokens.size], dtype=np.int32),
            "tones": tone_input,
            "sid": np.asarray([meta["speaker_id"]], dtype=np.int32),
        }
        encoder_onnx = work_dir / meta["components"]["encoder"]["onnx_file"]
        encoded = OrtRunner(encoder_onnx).run(encoder_feed)
        decoder_feed, acoustic_length, durations = prepare_decoder_inputs(encoded, int(meta["acoustic_max_length"]))
        golden_root = work_dir / "golden"
        if golden_root.exists():
            shutil.rmtree(golden_root)
        device_value = device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
        _run_golden_component(
            work_dir / meta["components"]["encoder"]["hmonnx_file"],
            encoder_feed,
            golden_root / "encoder",
            device_value,
        )
        _run_golden_component(
            work_dir / meta["components"]["decoder"]["hmonnx_file"],
            decoder_feed,
            golden_root / "decoder",
            device_value,
        )
        result = golden_root / "manifest.json"
        result.write_text(
            json.dumps(
                {
                    "text_length": int(tokens.size),
                    "tokens": tokens.tolist(),
                    "tones": tones.tolist(),
                    "acoustic_length": acoustic_length,
                    "valid_samples": acoustic_length * 512,
                    "durations": durations.tolist(),
                    "graphs": {
                        role: {
                            "hmonnx_file": component["hmonnx_file"],
                            "golden_dir": role,
                        }
                        for role, component in meta["components"].items()
                    },
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return str(result)


def _golden_tokens(
    input_messages: Any,
) -> tuple[np.ndarray, np.ndarray]:
    if input_messages is None:
        return (
            np.asarray(
                [0, 62, 0, 40, 0, 37, 0, 16, 0, 107, 0],
                dtype=np.int32,
            ),
            np.asarray(
                [0, 3, 0, 3, 0, 3, 0, 3, 0, 0, 0],
                dtype=np.int32,
            ),
        )
    if not isinstance(input_messages, Mapping):
        raise TypeError("MeloTTS golden input must be a mapping or None")
    tokens = np.asarray(input_messages["tokens"], dtype=np.int32).reshape(-1)
    tones = np.asarray(input_messages["tones"], dtype=np.int32).reshape(-1)
    if tokens.shape != tones.shape or tokens.size == 0:
        raise ValueError("golden tokens/tones must have equal nonzero length")
    return tokens, tones


def _run_golden_component(
    hmonnx_path: Path,
    feed: dict[str, np.ndarray],
    golden_dir: Path,
    device: str,
) -> None:
    import torch

    from xhquant.api import HMONNXGoldenInference

    proto = onnx.load(hmonnx_path, load_external_data=False)
    tensors = []
    for value in proto.graph.input:
        array = np.asarray(feed[value.name])
        tensor = (
            torch.from_numpy(array.astype(np.int32))
            if np.issubdtype(array.dtype, np.integer)
            else torch.from_numpy(array.astype(np.float16))
        )
        tensors.append(tensor.to(device))
    golden_dir.mkdir(parents=True, exist_ok=True)
    session = HMONNXGoldenInference(str(hmonnx_path))
    session.to(device)
    session.save_golden = True
    session.golden_dir = str(golden_dir)
    session.step = 0
    session(*tensors)


def _make_zero_torch_inputs(model: onnx.ModelProto) -> list[Any]:
    import torch

    values = []
    for value in model.graph.input:
        shape = [int(dimension.dim_value) for dimension in value.type.tensor_type.shape.dim]
        dtype = torch.int32 if value.type.tensor_type.elem_type == onnx.TensorProto.INT32 else torch.float32
        values.append(torch.zeros(shape, dtype=dtype))
    return values


def _convert_hmonnx(
    onnx_path: Path,
    output_path: Path,
    quant_type: str,
    *,
    debug: bool,
) -> None:
    from xhquant.api import (
        DeviceType,
        QuantScheme,
        convert_onnx_to_hmonnx,
        create_quant_config,
        xhquant_init,
    )

    model = onnx.load(onnx_path, load_external_data=False)
    xhquant_init(str(output_path.parent / f"{output_path.stem}.log"), debug)
    device_type = DeviceType.XH2a
    quant_config = create_quant_config(QuantScheme(target_device=device_type, quant_type=quant_type))
    convert_onnx_to_hmonnx(
        str(onnx_path),
        _make_zero_torch_inputs(model),
        device_type,
        str(output_path),
        quant_config=quant_config,
        input_names=[value.name for value in model.graph.input],
        output_names=[value.name for value in model.graph.output],
        simplify=False,
    )


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


__all__ = ["MeloTTSWorkflow"]
