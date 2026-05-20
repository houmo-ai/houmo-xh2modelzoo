# Copyright 2025 HOUMO AI
#
# File: vest_cls_export.py
# Description:
#   Example script: cv/big_helmet/vest_cls_export.py
#   Export vest_cls_20260403.onnx (feature extractor, 1536-dim embedding)
#   to HMONNX for XH2a deployment.
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

DEFAULT_ONNX = Path(__file__).with_name("vest_cls_20260403.onnx")
INPUT_SHAPE = (1, 3, 128, 128)
INPUT_NAME = "onnx::Add_0"
OUTPUT_NAME = "703"


def _ensure_supported_opset(onnx_file: str, work_dirs: Path, min_opset: int = 11) -> str:
    model = onnx.load(onnx_file, load_external_data=True)
    default_opsets = [opset.version for opset in model.opset_import if opset.domain in ("", "ai.onnx")]
    if not default_opsets or default_opsets[0] >= min_opset:
        return onnx_file

    converted_model = version_converter.convert_version(model, min_opset)
    onnx.checker.check_model(converted_model)
    converted_onnx_file = work_dirs / f"{Path(onnx_file).stem}_opset{min_opset}.onnx"
    onnx.save(converted_model, converted_onnx_file)
    return str(converted_onnx_file)


def main(args):
    onnx_file = args.onnx
    onnx_name = Path(onnx_file).stem
    work_dirs = Path("work_dirs") / onnx_name
    work_dirs.mkdir(exist_ok=True, parents=True)
    onnx_file = _ensure_supported_opset(onnx_file, work_dirs)
    target_device = DeviceType.XH2a
    quant_type = args.quant_type
    out_hmonnx_file = work_dirs / "hmonnx" / f"{onnx_name}_{quant_type}_{target_device}.onnx"
    out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)
    out_hmonnx_file_str = str(out_hmonnx_file)

    xhquant_init(None, debug=args.debug)
    logger = get_root_logger()

    quant_scheme = QuantScheme(target_device=target_device, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)
    sample_input = torch.randn(*INPUT_SHAPE, dtype=torch.float32)

    convert_onnx_to_hmonnx(
        onnx_file,
        [sample_input],
        target_device,
        out_hmonnx_file_str,
        quant_config=quant_config,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
    )

    logger.info(f"Convert onnx to hmonnx success, out hmonnx file to: {out_hmonnx_file_str}")

    if args.skip_golden:
        return

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx",
        type=str,
        default=str(DEFAULT_ONNX),
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--quant-type", default="w8a8_sefp", help="quant type, default is w8a8_sefp")
    parser.add_argument("--skip-golden", action="store_true", help="skip golden inference sanity check")
    args = parser.parse_args()
    main(args)
