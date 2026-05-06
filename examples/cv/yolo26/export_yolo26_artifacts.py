#!/usr/bin/env python3
# Copyright 2025 HOUMO AI
#
# File: export_yolo26_artifacts.py
# Description:
#   Export utilities for yolo26 in HOUMO AI xh2modelzoo.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Export yolo26m.onnx to hmonnx and golden artifacts.

This script intentionally lives next to the model file so the whole workflow
can stay self-contained under ``examples/cv/yolo26``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import onnx
import torch
from onnx import TensorProto
from torch import Tensor
from xhquant.api import (
    DeviceType,
    HMONNXGoldenInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    get_root_logger,
    xhquant_init,
)

SCRIPT_DIR = Path(__file__).resolve().parent

TENSOR_TYPE_TO_TORCH_TYPE = {
    int(TensorProto.FLOAT): torch.float32,
    int(TensorProto.UINT8): torch.uint8,
    int(TensorProto.INT8): torch.int8,
    int(TensorProto.INT16): torch.int16,
    int(TensorProto.INT32): torch.int32,
    int(TensorProto.INT64): torch.int64,
    int(TensorProto.BOOL): torch.bool,
    int(TensorProto.FLOAT16): torch.float16,
    int(TensorProto.DOUBLE): torch.float64,
}


def _iter_real_inputs(model: onnx.ModelProto) -> Iterable[onnx.ValueInfoProto]:
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    for value_info in model.graph.input:
        if value_info.name not in initializer_names:
            yield value_info


def _shape_of(value_info: onnx.ValueInfoProto) -> list[int]:
    dims: list[int] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.dim_value > 0:
            dims.append(dim.dim_value)
        else:
            dims.append(1)
    return dims


def _build_sample_tensor(value_info: onnx.ValueInfoProto) -> Tensor:
    elem_type = int(value_info.type.tensor_type.elem_type)
    dtype = TENSOR_TYPE_TO_TORCH_TYPE.get(elem_type)
    if dtype is None:
        raise NotImplementedError(
            f"Unsupported ONNX input dtype {TensorProto.DataType.Name(elem_type)} "
            f"for input {value_info.name!r}"
        )

    shape = _shape_of(value_info)
    if dtype in (torch.float16, torch.float32, torch.float64):
        return torch.randn(*shape, dtype=torch.float32)
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        return torch.randint(low=0, high=10, size=shape, dtype=dtype)
    if dtype is torch.bool:
        return torch.randint(low=0, high=2, size=shape, dtype=torch.bool)
    raise NotImplementedError(f"Unsupported torch dtype {dtype} for input {value_info.name!r}")


def _build_golden_tensor(tensor: Tensor, device: str) -> Tensor:
    if tensor.dtype in (torch.float16, torch.float32, torch.float64):
        return tensor.to(dtype=torch.float16, device=device)
    return tensor.to(device=device)


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    onnx_path = Path(args.onnx).resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    model = onnx.load(str(onnx_path))
    input_infos = list(_iter_real_inputs(model))
    output_names = [output.name for output in model.graph.output]
    input_names = [value_info.name for value_info in input_infos]

    onnx_name = onnx_path.stem
    work_dir = Path(args.work_dir).resolve() if args.work_dir else SCRIPT_DIR / "work_dirs" / onnx_name
    hmonnx_dir = work_dir / "hmonnx"
    hmonnx_dir.mkdir(parents=True, exist_ok=True)

    target_device = DeviceType.XH2a
    out_hmonnx_path = hmonnx_dir / f"{onnx_name}_{args.quant_type}_{target_device.name}.onnx"
    golden_dir = hmonnx_dir / f"golden_{args.quant_type}"

    xhquant_init(None, debug=args.debug)
    logger = get_root_logger()

    calibration_inputs = [_build_sample_tensor(value_info) for value_info in input_infos]
    quant_scheme = QuantScheme(target_device=target_device, quant_type=args.quant_type)
    quant_config = create_quant_config(quant_scheme)

    convert_onnx_to_hmonnx(
        str(onnx_path),
        calibration_inputs,
        target_device,
        str(out_hmonnx_path),
        quant_config=quant_config,
        input_names=input_names,
        output_names=output_names,
    )
    logger.info(f"Converted ONNX to HMONNX: {out_hmonnx_path}")

    manifest = {
        "onnx": str(onnx_path),
        "hmonnx": str(out_hmonnx_path),
        "golden_dir": str(golden_dir),
        "input_names": input_names,
        "output_names": output_names,
        "quant_type": args.quant_type,
        "target_device": target_device.name,
    }

    if not args.skip_golden:
        runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
        session = HMONNXGoldenInference(str(out_hmonnx_path))
        session.to(runtime_device)
        session.save_golden = True
        session.golden_dir = golden_dir
        session.step = 0
        golden_inputs = [_build_golden_tensor(tensor, runtime_device) for tensor in calibration_inputs]
        session(*golden_inputs)
        logger.info(f"Saved golden tensors to: {golden_dir}")

    manifest_path = work_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info(f"Wrote manifest to: {manifest_path}")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export yolo26m.onnx to hmonnx and golden artifacts.")
    parser.add_argument("--onnx", type=str, default=str(SCRIPT_DIR / "yolo26m.onnx"))
    parser.add_argument("--quant-type", type=str, default="w8a8h1_sefp")
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-golden", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Enable xhquant debug mode.")
    main(parser.parse_args())
