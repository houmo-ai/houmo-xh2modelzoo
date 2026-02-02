import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
from xhquant.api import (
    DeviceType,
    HMONNXGoldenInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    get_root_logger,
    xhquant_init,
)

from export_utils import resolve_decoder_onnx, resolve_manifest
from input_utils import build_decoder_inputs, ensure_file, ensure_dir, get_onnx_input_specs, resolve_path


def _to_torch_tensors(arrays: List[np.ndarray]) -> List[torch.Tensor]:
    return [torch.from_numpy(arr) for arr in arrays]


def _select_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _cast_float_inputs(inputs: List[np.ndarray], dtype: str) -> List[np.ndarray]:
    if dtype in {"fp16", "float16"}:
        target = np.float16
    elif dtype in {"fp32", "float32"}:
        target = np.float32
    else:
        raise ValueError(f"unsupported input dtype: {dtype}")

    casted: List[np.ndarray] = []
    for arr in inputs:
        if np.issubdtype(arr.dtype, np.floating):
            casted.append(arr.astype(target))
        else:
            casted.append(arr)
    return casted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="", help="decoder ONNX path")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="directory containing exported decoder onnx",
    )
    parser.add_argument(
        "--export-manifest",
        type=str,
        default="",
        help="export manifest json path (default: examples/audio/funasr/exports/latest.json if exists)",
    )
    parser.add_argument(
        "--inputs-dir",
        type=str,
        default="examples/audio/funasr/inputs",
        help="directory containing prepared decoder inputs",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="",
        help="output root directory (default: work_dirs/<onnx_stem>)",
    )
    parser.add_argument("--debug", action="store_true", help="enable debug logging")
    parser.add_argument("--quant-type", default="w8a8", help="quant type")
    parser.add_argument(
        "--input-dtype",
        type=str,
        default="fp16",
        help="float dtype for inputs: fp16/fp32",
    )
    parser.add_argument(
        "--max-token-len",
        type=int,
        default=100,
        help="pad/trim length for pre_acoustic_embeds and pre_token_mask",
    )
    parser.add_argument(
        "--skip-golden",
        action="store_true",
        help="skip golden inference",
    )
    parser.add_argument(
        "--golden-device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="device for golden inference",
    )
    args = parser.parse_args()

    manifest_path = resolve_manifest(args.export_manifest)
    onnx_path = resolve_decoder_onnx(args.onnx, args.model_dir, manifest_path)
    ensure_file(onnx_path, "onnx model")
    inputs_dir = resolve_path(args.inputs_dir)

    work_dir = resolve_path(args.work_dir) if args.work_dir else Path("work_dirs") / onnx_path.stem
    ensure_dir(work_dir)

    xhquant_init(None, debug=args.debug)
    logger = get_root_logger()

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=args.quant_type)
    quant_config = create_quant_config(quant_scheme)
    quant_config.ops_cfg = dict(
        LayerNorm=dict(force_fp32=True),
    )

    input_dict = build_decoder_inputs(onnx_path, inputs_dir, max_token_len=args.max_token_len)
    input_specs = get_onnx_input_specs(onnx_path)
    input_arrays = [input_dict[name] for name, _ in input_specs]
    input_arrays = _cast_float_inputs(input_arrays, args.input_dtype)
    input_tensors = _to_torch_tensors(input_arrays)

    out_hmonnx_file = work_dir / "hmonnx" / f"{onnx_path.stem}_{args.quant_type}_{DeviceType.XH2a}.onnx"
    ensure_dir(out_hmonnx_file.parent)

    logger.info("Converting decoder ONNX to hmonnx...")
    convert_onnx_to_hmonnx(
        str(onnx_path),
        input_tensors,
        DeviceType.XH2a,
        str(out_hmonnx_file),
        quant_config=quant_config,
    )
    logger.info("hmonnx saved: %s", out_hmonnx_file)

    if args.skip_golden:
        return

    device = _select_device(args.golden_device)
    session = HMONNXGoldenInference(str(out_hmonnx_file))
    session.to(device)
    session.save_golden = True
    session.golden_dir = work_dir / f"hmonnx/golden_{args.quant_type}"

    input_tensors = [tensor.to(device) for tensor in input_tensors]
    session(*input_tensors)
    logger.info("Golden outputs saved to: %s", session.golden_dir)


if __name__ == "__main__":
    main()
