# Copyright 2025 HOUMO AI
#
# File: big_helmet_export.py
# Description:
#   Example script: cv/big_helmet/big_helmet_export.py
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
import torch
from onnx import numpy_helper
from xhquant.api import (
    DeviceType,
    HMONNXGoldenInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    get_root_logger,
    xhquant_init,
)


def _rewrite_external_data_location(onnx_file: str, location: str) -> None:
    model = onnx.load(onnx_file, load_external_data=False)
    for tensor in onnx.external_data_helper._get_all_tensors(model):
        if not onnx.external_data_helper.uses_external_data(tensor):
            continue
        for entry in tensor.external_data:
            if entry.key == "location":
                entry.value = location
    onnx.save(model, onnx_file)


def _materialize_rope_rank4_constants(hmonnx_file: str) -> int:
    model = onnx.load(hmonnx_file, load_external_data=True)
    graph = model.graph
    initializer_map = {initializer.name: initializer for initializer in graph.initializer}
    producer_map = {output: node for node in graph.node for output in node.output}
    materialized_names = {}
    removable_initializer_names = set()
    rewritten_rope_nodes = 0

    def materialize_rank4_initializer(tensor_name: str) -> str | None:
        if tensor_name in materialized_names:
            return materialized_names[tensor_name]

        unsqueeze_node = producer_map.get(tensor_name)
        if unsqueeze_node is None or unsqueeze_node.op_type != "Unsqueeze" or len(unsqueeze_node.input) != 2:
            return None

        source_name, axes_name = unsqueeze_node.input
        source_initializer = initializer_map.get(source_name)
        axes_initializer = initializer_map.get(axes_name)
        if source_initializer is None or axes_initializer is None:
            return None

        source_array = numpy_helper.to_array(source_initializer)
        axes_array = numpy_helper.to_array(axes_initializer)
        axes = [int(axis) for axis in axes_array.reshape(-1).tolist()]
        if source_array.ndim != 2 or sorted(axes) != [0, 1]:
            return None

        materialized_name = f"{tensor_name}_materialized"
        suffix = 0
        while materialized_name in initializer_map:
            suffix += 1
            materialized_name = f"{tensor_name}_materialized_{suffix}"

        materialized_array = source_array.reshape(1, 1, *source_array.shape)
        materialized_initializer = numpy_helper.from_array(materialized_array, name=materialized_name)
        graph.initializer.extend([materialized_initializer])
        initializer_map[materialized_name] = materialized_initializer
        materialized_names[tensor_name] = materialized_name
        removable_initializer_names.update({source_name, axes_name})
        return materialized_name

    for node in graph.node:
        if node.op_type != "Rope" or len(node.input) < 3:
            continue

        node_rewritten = False
        for input_index in (1, 2):
            materialized_name = materialize_rank4_initializer(node.input[input_index])
            if materialized_name is None:
                continue
            node.input[input_index] = materialized_name
            node_rewritten = True

        if node_rewritten:
            rewritten_rope_nodes += 1

    if rewritten_rope_nodes == 0:
        return 0

    graph_output_names = {output.name for output in graph.output}
    used_tensor_names = {input_name for node in graph.node for input_name in node.input if input_name}

    kept_nodes = []
    removed_output_names = set()
    for node in graph.node:
        if node.op_type == "Unsqueeze" and any(output in materialized_names for output in node.output):
            if all(output not in used_tensor_names and output not in graph_output_names for output in node.output):
                removed_output_names.update(node.output)
                continue
        kept_nodes.append(node)

    del graph.node[:]
    graph.node.extend(kept_nodes)

    if removed_output_names:
        kept_value_infos = [value_info for value_info in graph.value_info if value_info.name not in removed_output_names]
        del graph.value_info[:]
        graph.value_info.extend(kept_value_infos)

    used_tensor_names = {input_name for node in graph.node for input_name in node.input if input_name}
    used_tensor_names.update(output.name for output in graph.output)
    used_tensor_names.update(input_info.name for input_info in graph.input)
    kept_initializers = [
        initializer
        for initializer in graph.initializer
        if initializer.name not in removable_initializer_names or initializer.name in used_tensor_names
    ]
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)

    onnx.checker.check_model(model)

    hmonnx_path = Path(hmonnx_file)
    external_data_path = hmonnx_path.with_name(f"{hmonnx_path.stem}_external_data")
    temp_hmonnx_path = hmonnx_path.with_suffix(f"{hmonnx_path.suffix}.tmp")
    temp_external_data_path = external_data_path.with_name(f"{external_data_path.name}.tmp")
    temp_hmonnx_path.unlink(missing_ok=True)
    temp_external_data_path.unlink(missing_ok=True)
    onnx.save_model(
        model,
        str(temp_hmonnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=temp_external_data_path.name,
        size_threshold=1024,
    )
    _rewrite_external_data_location(str(temp_hmonnx_path), external_data_path.name)
    external_data_path.unlink(missing_ok=True)
    temp_external_data_path.replace(external_data_path)
    temp_hmonnx_path.replace(hmonnx_path)
    return rewritten_rope_nodes


def main(args):
    onnx_file = args.onnx
    onnx_name = Path(onnx_file).stem
    work_dirs = Path("work_dirs") / onnx_name
    work_dirs.mkdir(exist_ok=True, parents=True)
    target_device = DeviceType.XH2a
    quant_type = args.quant_type
    out_hmonnx_file = work_dirs / "hmonnx" / f"{onnx_name}_{quant_type}_{target_device}.onnx"
    out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)
    out_hmonnx_file_str = str(out_hmonnx_file)

    xhquant_init(None, debug=args.debug)
    logger = get_root_logger()

    quant_scheme = QuantScheme(target_device=target_device, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)

    convert_onnx_to_hmonnx(
        onnx_file,
        [torch.randn(1, 3, 256, 256, dtype=torch.float32)],
        target_device,
        out_hmonnx_file_str,
        quant_config=quant_config,
        input_names=["input"],
        output_names=["features"],
    )

    if not args.disable_rope_rank_fix:
        rewritten_rope_nodes = _materialize_rope_rank4_constants(out_hmonnx_file_str)
        if rewritten_rope_nodes > 0:
            logger.info(
                f"Rewrote {rewritten_rope_nodes} Rope node(s) to use materialized rank-4 cos/sin constants for compiler compatibility."
            )

    logger.info(f"Convert onnx to hmonnx success, out hmonnx file to: {out_hmonnx_file_str}")

    if args.skip_golden:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    golden_dir = out_hmonnx_file.parent / f"golden_{quant_type}"
    golden_dir.mkdir(parents=True, exist_ok=True)

    session = HMONNXGoldenInference(out_hmonnx_file_str)
    session.to(device)
    session.save_golden = True
    session.golden_dir = golden_dir
    session.step = 0
    session(torch.randn(1, 3, 256, 256, dtype=torch.float16).to(device))
    logger.info(f"Golden generated at: {golden_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx",
        type=str,
        default="examples/cv/big_helmet/big_model_vest_20260313_sim.onnx",
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--quant-type", default="w8a8_sefp", help="quant type, default is w8a8_sefp")
    parser.add_argument(
        "--disable-rope-rank-fix",
        default=True,
        help="disable post-export Rope rank fix for compiler compatibility",
    )
    parser.add_argument("--skip-golden", action="store_true", help="skip golden inference sanity check")
    args = parser.parse_args()
    main(args)
