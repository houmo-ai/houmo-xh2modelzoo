from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import onnx

from ...workflows.base import BaseOtherModelWorkflow
from ...workflows.result import ExportResult, QuantResult
from .graph import specialize_model


class StreamingZipformerWorkflow(BaseOtherModelWorkflow):
    def quant(
        self,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> QuantResult:
        config = self.workflow_config.with_overrides(config_overrides)
        if config.quant is not None:
            raise NotImplementedError("Zipformer performs ONNX PTQ during HMONNX export; quant must be null")
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
        if export_cfg["model"].get("type") != "XHStreamingZipformerModel":
            raise ValueError("Zipformer requires export.model.type='XHStreamingZipformerModel'")
        target = str(export_cfg["target_device"])
        if target != "XH2a":
            raise ValueError("Zipformer workflow currently supports XH2a only")
        quant_type = str(export_cfg["quant_type"])
        source = _find_source_encoder(model_dir)
        work_dir = Path(output_dir).expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        config_file = config.dump(str(work_dir / f"{config.name}.yaml"))
        onnx_dir = work_dir / "onnx"
        hmonnx_dir = work_dir / "hmonnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        hmonnx_dir.mkdir(parents=True, exist_ok=True)
        static_path, equivalence = specialize_model(
            source,
            onnx_dir / "zipformer_encoder_b1_layer_cache.onnx",
            int(export_cfg.get("check_steps", 3)),
        )
        hmonnx_path = hmonnx_dir / (f"zipformer_encoder_b1_layer_cache_{target}_{quant_type}.onnx")
        _convert_hmonnx(
            static_path,
            hmonnx_path,
            target,
            quant_type,
            debug=self.debug,
        )
        static_model = onnx.load(static_path, load_external_data=False)
        package_files = {}
        for filename in (
            "decoder-epoch-99-avg-1.onnx",
            "joiner-epoch-99-avg-1.onnx",
            "tokens.txt",
        ):
            path = model_dir / filename
            if path.is_file():
                package_files[filename] = {
                    "path": str(path),
                    "sha256": _sha256(path),
                }
        meta = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": "XHStreamingZipformerModel",
            "source_project": "https://github.com/k2-fsa/sherpa-onnx",
            "source_model": str(source),
            "source_sha256": _sha256(source),
            "target_device": target,
            "quant_type": quant_type,
            "config": str(Path(config_file).relative_to(work_dir)),
            "onnx_file": str(static_path.relative_to(work_dir)),
            "hmonnx_file": str(hmonnx_path.relative_to(work_dir)),
            "onnx_sha256": _sha256(static_path),
            "hmonnx_sha256": _sha256(hmonnx_path),
            "public_input_count": len(static_model.graph.input),
            "public_output_count": len(static_model.graph.output),
            "layer_cache_input_count": 84,
            "layer_counts": [2, 3, 2, 2, 3],
            "source_equivalence": equivalence,
            "package_files": package_files,
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

        work_dir = Path(export_result.work_dir)
        meta = json.loads((work_dir / "export_meta_info.json").read_text(encoding="utf-8"))
        static_path = work_dir / meta["onnx_file"]
        hmonnx_path = work_dir / meta["hmonnx_file"]
        contract = onnx.load(static_path, load_external_data=False)
        values = _make_zero_torch_inputs(contract)
        if isinstance(input_messages, Mapping) and "features" in input_messages:
            features = np.load(input_messages["features"]).astype(np.float32)
            if features.shape != (1, 39, 80):
                raise ValueError(f"golden features must be [1,39,80], got {features.shape}")
            values[0] = torch.from_numpy(features)
        golden_dir = work_dir / "golden"
        if golden_dir.exists():
            shutil.rmtree(golden_dir)
        golden_dir.mkdir(parents=True)
        device_value = device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
        session = HMONNXGoldenInference(str(hmonnx_path))
        session.to(device_value)
        session.save_golden = True
        session.golden_dir = str(golden_dir)
        session.step = 0
        golden_values = [
            (value.to(torch.float16) if value.is_floating_point() else value).to(device_value) for value in values
        ]
        session(*golden_values)
        result = golden_dir / "manifest.json"
        result.write_text(
            json.dumps(
                {
                    "hmonnx_file": meta["hmonnx_file"],
                    "input_count": len(values),
                    "input_shape": [1, 39, 80],
                    "cache_initialized_to_zero": True,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return str(result)


def _find_source_encoder(model_dir: Path) -> Path:
    candidates = [
        model_dir / "encoder-epoch-99-avg-1.onnx",
        model_dir / "sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23" / "encoder-epoch-99-avg-1.onnx",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(model_dir.glob("**/encoder-epoch-99-avg-1.onnx"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"{model_dir} must contain the official 14M Zipformer encoder")


def _make_zero_torch_inputs(
    model: onnx.ModelProto,
) -> list[Any]:
    import torch

    values = []
    for value in model.graph.input:
        shape = [int(dimension.dim_value) for dimension in value.type.tensor_type.shape.dim]
        element_type = value.type.tensor_type.elem_type
        dtype = torch.int32 if element_type == onnx.TensorProto.INT32 else torch.float32
        values.append(torch.zeros(shape, dtype=dtype))
    return values


def _convert_hmonnx(
    onnx_path: Path,
    output_path: Path,
    target: str,
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
    args = _make_zero_torch_inputs(model)
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


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["StreamingZipformerWorkflow"]
