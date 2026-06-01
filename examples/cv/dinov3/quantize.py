# Copyright 2025 HOUMO AI
#
# File: examples/cv/dinov3/quantize.py
# Description:
#   Quantize DINOv3 ONNX models to HMONNX format for XH2a deployment.
#   Supports both ViT and ConvNeXt backbones with configurable quantization schemes.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Quantize DINOv3 ONNX to HMONNX format.

Usage:
    # Quantize ViT-S/16 ONNX to w8a8_sefp HMONNX
    python quantize.py --onnx onnx/facebook_dinov3-vits16-pretrain-lvd1689m_is224.onnx \\
        --image-size 224 --hidden-size 384

    # Quantize ConvNeXt Tiny ONNX to w8a8_sefp HMONNX
    python quantize.py --onnx onnx/facebook_dinov3-convnext-tiny-pretrain-lvd1689m_is224.onnx \\
        --image-size 224 --arch convnext

    # Quantize with int8 scheme
    python quantize.py --onnx <onnx_path> --quant-type w8a8_sefp --image-size 224
"""

import argparse
import shutil
from pathlib import Path

import onnx
import torch
from onnx import version_converter
from xhquant.api import (
    DeviceType,
    HMONNXGoldenInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    get_root_logger,
    xhquant_init,
)


def _ensure_supported_opset(onnx_file: str, work_dirs: Path, min_opset: int = 11) -> str:
    """Ensure ONNX model has opset >= min_opset for conversion compatibility."""
    model = onnx.load(onnx_file, load_external_data=True)
    default_opsets = [opset.version for opset in model.opset_import if opset.domain in ("", "ai.onnx")]
    if not default_opsets or default_opsets[0] >= min_opset:
        return onnx_file

    converted_model = version_converter.convert_version(model, min_opset)
    onnx.checker.check_model(converted_model)
    converted_onnx_file = work_dirs / f"{Path(onnx_file).stem}_opset{min_opset}.onnx"
    onnx.save(converted_model, converted_onnx_file)
    return str(converted_onnx_file)


def _detect_input_info(onnx_file: str) -> tuple[str, tuple]:
    """Detect input name and shape from ONNX model."""
    model = onnx.load(onnx_file, load_external_data=True)
    graph = model.graph
    first_input = graph.input[0]
    input_name = first_input.name
    shape = []
    for dim in first_input.type.tensor_type.shape.dim:
        shape.append(dim.dim_value if dim.dim_value else 1)
    return input_name, tuple(shape)


def main(args):
    onnx_file = args.onnx
    onnx_name = Path(onnx_file).stem
    work_dirs = Path("work_dirs") / onnx_name
    work_dirs.mkdir(exist_ok=True, parents=True)

    onnx_file = _ensure_supported_opset(onnx_file, work_dirs)

    # Detect input name and shape from ONNX
    input_name, input_shape = _detect_input_info(onnx_file)
    # Override shape if user specifies image_size
    if args.image_size is not None:
        input_shape = (input_shape[0], input_shape[1], args.image_size, args.image_size)

    target_device = DeviceType.XH2a
    quant_type = args.quant_type
    out_hmonnx_file = work_dirs / "hmonnx" / f"{onnx_name}_{quant_type}_{target_device}.onnx"
    out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)
    out_hmonnx_file_str = str(out_hmonnx_file)

    xhquant_init(None, debug=args.debug)
    logger = get_root_logger()

    logger.info(f"Input: name={input_name}, shape={input_shape}")
    logger.info(f"Quant type: {quant_type}, target device: {target_device}")

    quant_scheme = QuantScheme(target_device=target_device, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)
    sample_input = torch.randn(*input_shape, dtype=torch.float32)

    convert_onnx_to_hmonnx(
        onnx_file,
        [sample_input],
        target_device,
        out_hmonnx_file_str,
        quant_config=quant_config,
        input_names=[input_name],
        output_names=["last_hidden_state", "pooler_output"],
    )

    logger.info(f"Convert onnx to hmonnx success, out file: {out_hmonnx_file_str}")

    if args.skip_golden:
        return

    # Run golden inference for sanity check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    golden_dir = out_hmonnx_file.parent / f"golden_{quant_type}"
    shutil.rmtree(golden_dir, ignore_errors=True)
    golden_dir.mkdir(parents=True, exist_ok=True)

    session = HMONNXGoldenInference(out_hmonnx_file_str)
    session.to(device)
    session.save_golden = True
    session.golden_dir = golden_dir
    session.step = 0
    session(sample_input.to(torch.float16).to(device))
    logger.info(f"Golden generated at: {golden_dir}")

    # Also save sample input for accuracy testing
    torch.save(sample_input, golden_dir / "sample_input.pt")
    logger.info(f"Sample input saved to: {golden_dir / 'sample_input.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize DINOv3 ONNX to HMONNX")
    parser.add_argument("--onnx", type=str, required=True,
                        help="Path to ONNX model file")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Input image size (square). Auto-detected from ONNX if not specified.")
    parser.add_argument("--quant-type", default="w8a8_sefp",
                        help="Quantization type (default: w8a8_sefp)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    parser.add_argument("--skip-golden", action="store_true",
                        help="Skip golden inference sanity check")
    args = parser.parse_args()
    main(args)
