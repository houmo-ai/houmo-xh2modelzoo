import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
from xhquant.api import DeviceType, HMONNXGoldenInference, QuantScheme, convert_onnx_to_hmonnx, create_quant_config, get_root_logger, xhquant_init

from export_utils import resolve_decoder_onnx, resolve_manifest
from input_utils import build_decoder_inputs_from_arrays, ensure_dir, ensure_file, get_onnx_input_specs, load_npy, resolve_path


def _to_torch_tensors(arrays: List[np.ndarray]) -> List[torch.Tensor]:
    return [torch.from_numpy(array) for array in arrays]


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
    outputs = []
    for value in inputs:
        if np.issubdtype(value.dtype, np.floating):
            outputs.append(value.astype(target))
        else:
            outputs.append(value)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="")
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--export-manifest", default="")
    parser.add_argument("--inputs-dir", default="examples/audio/paraformer/inputs")
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--quant-type", default="w8a8")
    parser.add_argument("--input-dtype", default="fp16")
    parser.add_argument("--skip-golden", action="store_true")
    parser.add_argument("--golden-device", default="auto", choices=["auto", "cpu", "cuda"])
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
    quant_config.ops_cfg = {"LayerNorm": {"force_fp32": True}}

    enc = load_npy(inputs_dir / "enc.npy")
    enc_len = load_npy(inputs_dir / "enc_len.npy")
    acoustic_embeds = load_npy(inputs_dir / "acoustic_embeds.npy")
    acoustic_embeds_len = load_npy(inputs_dir / "acoustic_embeds_len.npy")
    caches = [load_npy(inputs_dir / f"in_cache_{index}.npy") for index in range(16)]
    input_dict = build_decoder_inputs_from_arrays(onnx_path, enc, enc_len, acoustic_embeds, acoustic_embeds_len, caches=caches)
    input_specs = get_onnx_input_specs(onnx_path)
    input_arrays = _cast_float_inputs([input_dict[name] for name, _ in input_specs], args.input_dtype)
    input_tensors = _to_torch_tensors(input_arrays)

    out_hmonnx_file = work_dir / "hmonnx" / f"{onnx_path.stem}_{args.quant_type}_{DeviceType.XH2a}.onnx"
    ensure_dir(out_hmonnx_file.parent)

    logger.info("Converting decoder ONNX to hmonnx...")
    convert_onnx_to_hmonnx(str(onnx_path), input_tensors, DeviceType.XH2a, str(out_hmonnx_file), quant_config=quant_config)
    logger.info("hmonnx saved: %s", out_hmonnx_file)

    if args.skip_golden:
        return

    device = _select_device(args.golden_device)
    session = HMONNXGoldenInference(str(out_hmonnx_file))
    session.to(device)
    session.save_golden = True
    session.golden_dir = work_dir / f"hmonnx/golden_{args.quant_type}"
    session(*[tensor.to(device) for tensor in input_tensors])
    logger.info("Golden outputs saved to: %s", session.golden_dir)


if __name__ == "__main__":
    main()