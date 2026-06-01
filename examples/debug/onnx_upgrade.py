# Copyright 2025 HOUMO AI
#
# File: onnx_upgrade.py
# Description:
#   Example script: debug/onnx_upgrade.py
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
from pathlib import Path

import onnx
from loguru import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="onnx/onnx_model.onnx")
    parser.add_argument("--target-version", type=int, default=11, help="target onnx version")
    args = parser.parse_args()
    onnx_model = onnx.load(args.onnx)
    target_onnx_model = onnx.version_converter.convert_version(onnx_model, args.target_version)

    # const_initializer_names = [init.name for init in target_onnx_model.graph.initializer]
    # remove_inputs = []
    # for input in target_onnx_model.graph.input:
    #     if input.name in const_initializer_names:
    #         remove_inputs.append(input)
    # for input in remove_inputs:
    #     target_onnx_model.graph.input.remove(input)

    out_onnx_file = args.onnx.replace(".onnx", f"_v{args.target_version}.onnx")
    onnx.save(
        target_onnx_model,
        out_onnx_file,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{Path(out_onnx_file).stem}_external_data",
    )

    logger.info(f"Save to {out_onnx_file}")
