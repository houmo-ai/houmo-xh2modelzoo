"""Convert FunASR-Nano exported ONNX modules to HMONNX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import onnx
import torch

from xhquant.api import DeviceType, QuantScheme, convert_onnx_to_hmonnx, create_quant_config, xhquant_init


def _onnx_io_names(onnx_path: Path) -> tuple[list[str], list[str]]:
    model = onnx.load(str(onnx_path), load_external_data=False)
    init_names = {i.name for i in model.graph.initializer}
    inputs = [vi.name for vi in model.graph.input if vi.name not in init_names]
    outputs = [vi.name for vi in model.graph.output]
    return inputs, outputs


def _shape_from_value_info(value_info: Any) -> tuple[int, ...]:
    shape = []
    for dim in value_info.type.tensor_type.shape.dim:
        shape.append(dim.dim_value if dim.HasField("dim_value") and dim.dim_value > 0 else 1)
    return tuple(shape)


def _dummy_inputs(onnx_path: Path, input_names: list[str]) -> list[torch.Tensor]:
    model = onnx.load(str(onnx_path), load_external_data=False)
    by_name = {vi.name: vi for vi in model.graph.input}
    result = []
    for name in input_names:
        vi = by_name[name]
        shape = _shape_from_value_info(vi)
        elem_type = vi.type.tensor_type.elem_type
        if elem_type in (6, 7):  # INT32 / INT64
            dtype = torch.int32 if elem_type == 6 else torch.int64
            value = torch.ones(shape, dtype=dtype)
        else:
            value = torch.randn(shape, dtype=torch.float32)
        result.append(value)
    return result


def _static_input_shapes(onnx_path: Path, input_names: list[str]) -> dict[str, list[int]]:
    model = onnx.load(str(onnx_path), load_external_data=False)
    by_name = {vi.name: vi for vi in model.graph.input}
    shapes: dict[str, list[int]] = {}
    for name in input_names:
        vi = by_name[name]
        dims = []
        for dim in vi.type.tensor_type.shape.dim:
            if not dim.HasField("dim_value") or dim.dim_value <= 0:
                dims = []
                break
            dims.append(int(dim.dim_value))
        if dims:
            shapes[name] = dims
    return shapes


def _simplify_static_onnx(onnx_file: Path, input_names: list[str]) -> Path:
    shapes = _static_input_shapes(onnx_file, input_names)
    if not shapes:
        return onnx_file
    simplified_file = onnx_file.with_name(f"{onnx_file.stem}_static_simplified.onnx")
    if simplified_file.exists():
        return simplified_file
    try:
        from onnxsim import simplify

        model = onnx.load(str(onnx_file))
        simplified, check = simplify(model, overwrite_input_shapes=shapes)
        if check:
            onnx.save(simplified, str(simplified_file))
            return simplified_file
    except Exception as exc:
        print(f"Warning: simplify {onnx_file} failed: {exc}")
    return onnx_file


def _convert_one(onnx_file: Path, out_dir: Path, quant_type: str, debug: bool) -> Path:
    input_names, output_names = _onnx_io_names(onnx_file)
    onnx_file = _simplify_static_onnx(onnx_file, input_names)
    input_names, output_names = _onnx_io_names(onnx_file)
    inputs = _dummy_inputs(onnx_file, input_names)
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)
    if hasattr(quant_config, "ops_cfg"):
        quant_config.ops_cfg.setdefault("LayerNorm", dict(force_fp32=True))

    hmonnx_dir = out_dir / "hmonnx"
    hmonnx_dir.mkdir(parents=True, exist_ok=True)
    hmonnx_file = hmonnx_dir / f"{onnx_file.stem}_xh2a_{quant_type}.onnx"
    convert_onnx_to_hmonnx(
        str(onnx_file),
        inputs,
        DeviceType.XH2a,
        str(hmonnx_file),
        quant_config=quant_config,
        input_names=input_names,
        output_names=output_names,
    )
    return hmonnx_file


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--work-dir", default="work_dirs/funasr_nano_xh2a")
    parser.add_argument("--quant-type", default="w8a8h1_sefp")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    work_dir = Path(args.work_dir).expanduser().resolve()
    meta_file = work_dir / "export_meta_info.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    xhquant_init(str(work_dir / "convert_hmonnx.log"), debug=bool(args.debug))

    key_map = {
        "encoder_onnx_file": "encoder_hmonnx_file",
        "audio_adaptor_onnx_file": "audio_adaptor_hmonnx_file",
        "ctc_onnx_file": "ctc_hmonnx_file",
        "prefill_onnx_file": "prefill_hmonnx_file",
        "decode_onnx_file": "decode_hmonnx_file",
    }
    for onnx_key, hmonnx_key in key_map.items():
        if meta.get(hmonnx_key):
            print(f"Skip {onnx_key}: {hmonnx_key} already exists in metadata")
            continue
        rel = meta.get(onnx_key)
        if not rel:
            continue
        onnx_file = (work_dir / rel).resolve()
        if not onnx_file.exists():
            print(f"Skip missing {onnx_file}")
            continue
        hmonnx_file = _convert_one(onnx_file, onnx_file.parent, args.quant_type, args.debug)
        meta[hmonnx_key] = str(hmonnx_file.relative_to(work_dir))
        print(f"Converted {onnx_file} -> {hmonnx_file}")

    meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
