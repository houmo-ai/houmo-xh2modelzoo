# Copyright 2025 HOUMO AI
#
# File: modify_onnx.py
# Description:
#   Example script: cv/rtdetr/scripts/modify_onnx.py
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
import json

import numpy as np
import onnx
import onnx_graphsurgeon as gs


def process_graph(graph):
    print(len(graph.inputs))
    tensors = graph.tensors()
    tensor = tensors["Concat.180"]
    tensor.outputs.clear()
    graph.outputs[1] = tensor

    tensor_scores = tensors["sigmoid_20.tmp_0"]
    tensor_scores.outputs.clear()
    graph.outputs[0] = tensor_scores

    graph.outputs[0].name = "scores"
    graph.outputs[1].name = "boxes"

    del graph.inputs[2]
    del graph.inputs[0]
    return graph


def main():
    parser = argparse.ArgumentParser(description="Modify DCNv2 plugin node into ONNX model")
    parser.add_argument(
        "-i",
        "--input",
        help="Path to ONNX model with 'Plugin' node to replace with DCNv2_TRT",
        default="data/models/rtdetr/rtdetr_hgnetv2_l_6x_coco-sim.onnx",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output ONNX model with 'DCNv2_TRT' node",
        default="data/models/rtdetr/rtdetr_hgnetv2_l_6x_coco-no-post_process.onnx",
    )

    args, _ = parser.parse_known_args()
    graph = gs.import_onnx(onnx.load(args.input))
    graph = process_graph(graph)

    # 删除现在游离的子图
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), args.output)


if __name__ == "__main__":
    main()
