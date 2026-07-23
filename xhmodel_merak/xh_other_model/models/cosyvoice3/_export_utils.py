# -*- coding: utf-8 -*-
# Copyright 2025 HOUMO AI. All rights reserved.
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
# File: _export_utils.py
# Description:
#   Onnx transform / quant-config / golden helpers for CosyVoice3 export.
#   Ported from the legacy examples/audio/Cosyvoice3/*_export_hmonnx.py and
#   speech_tokenizer_v3_convert.py scripts so the migrated model package does
#   not depend on ./examples or ./xh_model_zoo at runtime.

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
from onnx import TensorProto, helper, shape_inference
from onnxsim import simplify


# ---------------------------------------------------------------------------
# Generic onnx helpers
# ---------------------------------------------------------------------------


def inspect_onnx(model_path: str) -> None:
    """Print onnx input/output nodes. Debug helper."""
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    print("模型输入节点:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.type}, shape={inp.shape}")
    print("模型输出节点:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.type}, shape={out.shape}")


def fix_input_shape(model: onnx.ModelProto, fixed_dims: dict[str, int]) -> onnx.ModelProto:
    """Replace symbolic input dims with fixed values declared in ``fixed_dims``."""
    for input_node in model.graph.input:
        dims = []
        for d in input_node.type.tensor_type.shape.dim:
            if d.dim_value > 0:
                dims.append(d.dim_value)
            else:
                dims.append(d.dim_param)

        new_dims = []
        for dim in dims:
            if dim in fixed_dims:
                new_dims.append(fixed_dims[dim])
            elif isinstance(dim, int):
                new_dims.append(dim)
            else:
                raise ValueError(f"未处理的动态维度: {dim}")

        input_node.type.tensor_type.shape.dim.clear()
        for val in new_dims:
            input_node.type.tensor_type.shape.dim.add(dim_value=val)

    return model


def simplify_model(model: onnx.ModelProto, save_path: str | None = None) -> onnx.ModelProto:
    """Run onnx-simplify. Optionally save to ``save_path``."""
    print("Running onnx-simplify ...")
    model_simp, check = simplify(model)
    assert check, "onnx-simplify check failed"
    print("onnx-simplify done")
    if save_path is not None:
        onnx.save(model_simp, save_path)
    return model_simp


# ---------------------------------------------------------------------------
# hift transforms (ported from hift_export_hmonnx.py)
# ---------------------------------------------------------------------------


def hift_fix_pad_mode_constant(model: onnx.ModelProto) -> onnx.ModelProto:
    """Switch Pad nodes to ``constant`` mode (hift compatibility)."""
    for node in model.graph.node:
        if node.op_type == "Pad":
            for a in node.attribute:
                if a.name == "mode":
                    a.s = b"constant"
    return model


def hift_replace_resize_with_sizes(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace the hift ``/m_source/l_sin_gen/Resize`` sizes input with a constant."""
    graph = model.graph

    sizes = onnx.numpy_helper.from_array(
        np.array([1, 9, 1024], dtype=np.int64),
        name="sizes",
    )

    graph.initializer.append(sizes)

    for node in graph.node:
        if node.name == "/m_source/l_sin_gen/Resize":
            node.input[2] = ""
            node.input.append("sizes")
    return model


def hift_transform_scatter_add(
    model_path: str,
    output_path: str,
    win: int = 16,
    hop: int = 4,
) -> str:
    """Decompose hift ScatterElements(reduction=add) into grouped ScatterElements+Add.

    Ported verbatim from the legacy ``hift_export_hmonnx.py``.  The runtime does
    not support ScatterElements with ``reduction=add`` directly, so the node is
    split into ``K = win // hop`` non-reduction ScatterElements whose outputs are
    summed.  Only constant indices are supported (which holds for the hift graph).
    """
    model = onnx.load(model_path)
    graph = model.graph
    K = win // hop

    const_map = {init.name: onnx.numpy_helper.to_array(init) for init in graph.initializer}

    new_graph_nodes = []
    transform_count = 0

    for node in graph.node:
        if node.op_type == "ScatterElements":
            reduction = "none"
            axis = 0
            for attr in node.attribute:
                if attr.name == "reduction":
                    reduction = attr.s.decode() if attr.s else "none"
                elif attr.name == "axis":
                    axis = attr.i

            if reduction == "add":
                print(f"\n分解节点: {node.name}")
                transform_count += 1

                data = node.input[0]
                indices_name = node.input[1]
                updates = node.input[2]
                output = node.output[0]

                print(f"  data: {data}")
                print(f"  indices: {indices_name}")
                print(f"  updates: {updates}")

                if indices_name in const_map:
                    indices = const_map[indices_name]
                else:
                    print("  警告: indices 不是常量，尝试追溯...")
                    new_graph_nodes.append(node)
                    continue

                print(f"  indices shape: {indices.shape}")
                print(f"  indices dtype: {indices.dtype}")

                if indices.ndim == 2 and indices.shape[0] == 1:
                    indices = indices[0]
                    print(f"  展平后 shape: {indices.shape}")

                total_len = len(indices)
                num_frames = total_len // win

                print(f"  总长度: {total_len}")
                print(f"  每帧长度 (win): {win}")
                print(f"  帧数: {num_frames}")
                print(f"  分组数 K: {K}")

                zero_tensor = f"{output}_zero"
                shape_name = f"{output}_shape"

                new_graph_nodes.append(
                    helper.make_node(
                        "Shape",
                        [data],
                        [shape_name],
                        f"{output}_Shape",
                    )
                )

                zero_value = onnx.numpy_helper.from_array(np.array([0.0], dtype=np.float32))
                const_of_shape_node = helper.make_node(
                    "ConstantOfShape",
                    [shape_name],
                    [zero_tensor],
                    f"{output}_ConstShape",
                )
                const_of_shape_node.attribute.append(helper.make_attribute("value", zero_value))
                new_graph_nodes.append(const_of_shape_node)

                outs = []

                for q in range(K):
                    print(f"  处理组 q={q}")

                    indices_q_list = []
                    for t in range(num_frames):
                        for r in range(hop):
                            idx = (t + q) * hop + r
                            indices_q_list.append(idx)

                    indices_q = np.array(indices_q_list, dtype=np.int64)
                    indices_q = indices_q.reshape(1, -1)
                    indices_q_name = f"{output}_indices_q{q}"
                    graph.initializer.append(onnx.numpy_helper.from_array(indices_q, name=indices_q_name))

                    updates_q_name = f"{output}_updates_q{q}"

                    reshape_name = f"{output}_reshape_q{q}"
                    reshape_shape = f"{output}_reshape_shape_q{q}"

                    graph.initializer.append(
                        helper.make_tensor(reshape_shape, TensorProto.INT64, [2], [num_frames, win])
                    )

                    new_graph_nodes.append(
                        helper.make_node(
                            "Reshape",
                            [updates, reshape_shape],
                            [reshape_name],
                            f"{output}_Reshape_q{q}",
                        )
                    )

                    slice_out_name = f"{output}_slice_q{q}"
                    graph.initializer.extend(
                        [
                            helper.make_tensor(f"{output}_s{q}_0", TensorProto.INT64, [2], [0, q * hop]),
                            helper.make_tensor(f"{output}_e{q}_0", TensorProto.INT64, [2], [num_frames, (q + 1) * hop]),
                            helper.make_tensor(f"{output}_a{q}_0", TensorProto.INT64, [2], [0, 1]),
                        ]
                    )

                    new_graph_nodes.append(
                        helper.make_node(
                            "Slice",
                            [reshape_name, f"{output}_s{q}_0", f"{output}_e{q}_0", f"{output}_a{q}_0"],
                            [slice_out_name],
                            f"{output}_Slice_q{q}",
                        )
                    )

                    shape_flat_name = f"{output}_shape_flat_q{q}"
                    graph.initializer.append(helper.make_tensor(shape_flat_name, TensorProto.INT64, [2], [1, -1]))

                    new_graph_nodes.append(
                        helper.make_node(
                            "Reshape",
                            [slice_out_name, shape_flat_name],
                            [updates_q_name],
                            f"{output}_Reshape_back_q{q}",
                        )
                    )

                    out_q_name = f"{output}_out_q{q}"
                    scatter = helper.make_node(
                        "ScatterElements",
                        [zero_tensor, indices_q_name, updates_q_name],
                        [out_q_name],
                        f"{output}_Sct_q{q}",
                        axis=axis,
                    )
                    scatter.attribute.append(helper.make_attribute("reduction", "none"))
                    new_graph_nodes.append(scatter)
                    outs.append(out_q_name)

                if K == 1:
                    new_graph_nodes.append(
                        helper.make_node(
                            "Identity",
                            outs,
                            [output],
                            f"{output}_Id",
                        )
                    )
                elif K == 2:
                    new_graph_nodes.append(
                        helper.make_node(
                            "Add",
                            outs,
                            [output],
                            f"{output}_Add",
                        )
                    )
                else:
                    current = outs[0]
                    for i in range(1, len(outs)):
                        next_out = f"{output}_add_{i}"
                        new_graph_nodes.append(
                            helper.make_node(
                                "Add",
                                [current, outs[i]],
                                [next_out],
                                f"{output}_Add_{i}",
                            )
                        )
                        current = next_out
                    new_graph_nodes[-1].output[0] = output
            else:
                new_graph_nodes.append(node)
        else:
            new_graph_nodes.append(node)

    graph.ClearField("node")
    graph.node.extend(new_graph_nodes)

    onnx.save(model, output_path)

    try:
        onnx.checker.check_model(output_path)
        print("✓ 模型检查通过")
    except Exception as e:
        print(f"✗ 模型检查失败: {e}")

    return output_path


def hift_prepare_onnx(
    source_onnx: str,
    fixed_dims: dict[str, int] | None = None,
    intermediate_dir: Path | None = None,
) -> str:
    """Apply the full hift onnx transform pipeline.

    Returns the path to the transformed onnx ready for hmonnx conversion.
    Mirrors the legacy ``main`` flow: fix shape -> simplify -> pad constant ->
    simplify -> resize->sizes -> simplify -> scatter_add decomposition.
    """
    fixed_dims = fixed_dims or {"batch_size": 1, "seq_len": 1024}
    intermediate_dir = Path(intermediate_dir) if intermediate_dir else Path(source_onnx).parent
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    model = onnx.load(source_onnx)
    inspect_onnx(source_onnx)
    model = fix_input_shape(model, fixed_dims)
    simplified = simplify_model(model, str(intermediate_dir / "hift_simplify.onnx"))

    simplified = hift_fix_pad_mode_constant(simplified)
    reflect = simplify_model(simplified, str(intermediate_dir / "hift_simplify_reflect.onnx"))

    reflect = hift_replace_resize_with_sizes(onnx.load(str(intermediate_dir / "hift_simplify_reflect.onnx")))
    simplify_model(reflect, str(intermediate_dir / "hift_simplify_final.onnx"))

    final_1 = intermediate_dir / "hift_simplify_final_1.onnx"
    hift_transform_scatter_add(str(intermediate_dir / "hift_simplify_final.onnx"), str(final_1))
    return str(final_1)


# ---------------------------------------------------------------------------
# speech_tokenizer_v3 transforms (ported from speech_tokenizer_v3_convert.py)
# ---------------------------------------------------------------------------

_ST_FIXED_T = 3000
_ST_FEAT_LENGTH_NAME = "feats_length"
_ST_FIXED_FEAT_LENGTH = 3000
_ST_BLOCK_IDS = list(range(6))
_ST_MASK_INPUT_NAME = "mask"
_ST_MASK_SHAPE = [1, 20, 750, 750]
_ST_MASK_DTYPE = TensorProto.FLOAT
_ST_MASK1_INPUT_NAME = "mask1"
_ST_MASK1_SHAPE = [1, 750, 1280]
_ST_MASK1_DTYPE = TensorProto.FLOAT


def _st_add_mask_input(model: onnx.ModelProto, name: str, shape, dtype) -> None:
    existing = {i.name for i in model.graph.input}
    if name in existing:
        print(f"Model already has input '{name}', skip adding.")
        return
    vi = helper.make_tensor_value_info(name, dtype, shape)
    model.graph.input.append(vi)
    print(f"Added model input '{name}' with shape {shape}.")


def _st_insert_add_before_softmax(model: onnx.ModelProto, mask_name: str) -> None:
    """Insert ``Add(x, mask)`` before every Softmax op (first mask input)."""
    softmax_nodes = [n for n in model.graph.node if n.op_type == "Softmax"]
    count = 0
    for node in softmax_nodes:
        original_in = node.input[0]
        new_in = original_in + "_masked"
        add_node = helper.make_node(
            "Add",
            inputs=[original_in, mask_name],
            outputs=[new_in],
            name=f"Add_mask_{count}",
        )
        node.input[0] = new_in
        idx = list(model.graph.node).index(node)
        model.graph.node.insert(idx, add_node)
        count += 1
    print(f"Inserted {count} Add nodes before Softmax ops.")


def _st_find_target_add_nodes(graph, block_id: int):
    name1 = f"/blocks.{block_id}/attn/value/Add"
    name2 = f"/blocks.{block_id}/attn/Add"
    return [node for node in graph.node if node.name in (name1, name2)]


def _st_insert_mul_after_node(graph, target_node, mask_name: str) -> bool:
    if len(target_node.output) == 0:
        raise RuntimeError(f"Target node {target_node.name} has no outputs")
    old_output = target_node.output[0]
    downstream_nodes = [node for node in graph.node if old_output in node.input]
    mul_inserted = False
    for node in downstream_nodes:
        new_mul_output = old_output + f"_mul_{node.name}"
        mul_name = f"{target_node.name}_Mul_{node.name}"
        mul_node = helper.make_node(
            "Mul",
            inputs=[old_output, mask_name],
            outputs=[new_mul_output],
            name=mul_name,
        )
        try:
            idx = list(graph.node).index(target_node)
        except ValueError:
            graph.node.append(mul_node)
        else:
            graph.node.insert(idx + 1, mul_node)
        for i, inp in enumerate(node.input):
            if inp == old_output:
                node.input[i] = new_mul_output
        mul_inserted = True
        print(f"Inserted Mul '{mul_name}' for branch '{node.name}'.")
    if mul_inserted:
        for out_vi in graph.output:
            if out_vi.name == old_output:
                out_vi.name = old_output + "_mul"
    return mul_inserted


def _st_fix_reducemean_axes_to_input(graph) -> None:
    for node in graph.node:
        if node.op_type == "ReduceMean":
            axes_attr = next((a for a in node.attribute if a.name == "axes"), None)
            if axes_attr is None:
                continue
            axes_name = node.name + "_axes"
            axes_tensor = helper.make_tensor(
                name=axes_name,
                data_type=TensorProto.INT64,
                dims=[len(axes_attr.ints)],
                vals=np.array(axes_attr.ints, dtype=np.int64),
            )
            graph.initializer.append(axes_tensor)
            node.input.append(axes_name)
            node.attribute.remove(axes_attr)
            print(f"Fixed ReduceMean '{node.name}': axes attr -> input")


def convert_speech_tokenizer_v3_onnx(source_onnx: str, output_onnx: str) -> str:
    """Transform the upstream ``speech_tokenizer_v3.onnx`` into the fixed-shape,
    mask-instrumented graph consumed by the hmonnx export.

    Steps (ported from ``speech_tokenizer_v3_convert.py``):
      1. fix input shape T=3000 and simplify
      2. replace the dynamic ``feats_length`` input with a constant
      3. add ``mask`` input and ``Add`` before every Softmax, then simplify
      4. add ``mask1`` input and ``Mul`` after each block attn Add, then simplify
    """
    # Step 1: fix shape + simplify
    model = onnx.load(source_onnx)
    fixed_d = {"T": _ST_FIXED_T}
    for input_node in model.graph.input:
        dims = [d.dim_value if d.dim_value != 0 else d.dim_param for d in input_node.type.tensor_type.shape.dim]
        new_dims = []
        for dim in dims:
            if dim in fixed_d:
                new_dims.append(fixed_d[dim])
            else:
                new_dims.append(dim if isinstance(dim, int) else 0)
        input_node.type.tensor_type.shape.dim.clear()
        for dim_val in new_dims:
            input_node.type.tensor_type.shape.dim.add(dim_value=dim_val)
    slimmed_model, _ = onnxsim.simplify(model)
    print("✅ 固定形状完成")

    # Step 2: replace feats_length input with a Constant node
    model = slimmed_model
    graph = model.graph
    feat_length_input = next((inp for inp in graph.input if inp.name == _ST_FEAT_LENGTH_NAME), None)
    if feat_length_input is not None:
        graph.input.remove(feat_length_input)
        const_node = helper.make_node(
            op_type="Constant",
            inputs=[],
            outputs=[_ST_FEAT_LENGTH_NAME],
            value=helper.make_tensor(
                name="fixed_feat_val",
                data_type=TensorProto.INT32,
                dims=[1],
                vals=[_ST_FIXED_FEAT_LENGTH],
            ),
        )
        graph.node.insert(0, const_node)
    simplified_model, check = simplify(model, check_n=0, skip_fuse_bn=False, dynamic_input_shape=False)
    assert check, "Simplified model is invalid!"
    print("✅ 替换feat_length完成")

    # Step 3: add mask input + Add before softmax + simplify
    _st_add_mask_input(simplified_model, _ST_MASK_INPUT_NAME, _ST_MASK_SHAPE, _ST_MASK_DTYPE)
    _st_insert_add_before_softmax(simplified_model, _ST_MASK_INPUT_NAME)
    masked_model = simplify_model(simplified_model)
    print("✅ 添加mask输入完成")

    # Step 4: add mask1 input + Mul after block attn Add + simplify
    graph = masked_model.graph
    _st_add_mask_input(masked_model, _ST_MASK1_INPUT_NAME, _ST_MASK1_SHAPE, _ST_MASK1_DTYPE)
    total_inserted = 0
    for bid in _ST_BLOCK_IDS:
        matched = _st_find_target_add_nodes(graph, bid)
        print(f"[block {bid}] found {len(matched)} target nodes")
        for node in matched:
            _st_insert_mul_after_node(graph, node, _ST_MASK1_INPUT_NAME)
            total_inserted += 1
    print(f"Total Mul nodes inserted: {total_inserted}")
    inferred_model = shape_inference.infer_shapes(onnx.helper.make_model(graph, producer_name="mask_inserter"))
    _st_fix_reducemean_axes_to_input(inferred_model.graph)
    final_model, check = simplify(inferred_model)
    if not check:
        raise RuntimeError("onnx-simplify check failed for speech_tokenizer_v3 final model")
    Path(output_onnx).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(final_model, output_onnx)
    print("Done! speech_tokenizer_v3 final model saved to:", output_onnx)
    return output_onnx


# ---------------------------------------------------------------------------
# quant config + hmonnx conversion + golden
# ---------------------------------------------------------------------------


def to_xh_device_type(target_device: str):
    """Map the workflow ``target_device`` string to ``xhquant.api.DeviceType``."""
    from xhquant.api import DeviceType

    if target_device != "XH2a":
        raise ValueError(f"CosyVoice3 workflow currently supports target_device='XH2a', got {target_device!r}")
    return DeviceType.XH2a


def build_quant_config(target_device: str, quant_type: str, **overrides: Any) -> Any:
    """Build a quant config for the given device + quant_type string."""
    from xhquant.api import QuantScheme, create_quant_config

    quant_scheme = QuantScheme(target_device=to_xh_device_type(target_device), quant_type=quant_type)
    config = create_quant_config(quant_scheme)
    for key, value in overrides.items():
        config[key] = value
    return config


def convert_onnx_to_hmonnx(
    onnx_file: str,
    dummy_inputs: Sequence[Any],
    hmonnx_file: str,
    target_device: str,
    quant_type: str,
    *,
    quant_config_overrides: Mapping[str, Any] | None = None,
) -> str:
    """Convert an onnx file to a quantized hmonnx file."""
    from xhquant.api import convert_onnx_to_hmonnx

    Path(hmonnx_file).parent.mkdir(parents=True, exist_ok=True)
    qc = build_quant_config(target_device, quant_type)
    if quant_config_overrides:
        qc.update(quant_config_overrides)
    convert_onnx_to_hmonnx(
        onnx_file,
        [x.cpu() for x in dummy_inputs],
        to_xh_device_type(target_device),
        hmonnx_file,
        quant_config=qc,
    )
    return hmonnx_file


def run_hmonnx_golden(hmonnx_file: str, golden_dir: str, device: str, inputs: Sequence[Any]) -> None:
    """Run a hmonnx graph and dump its golden tensors into ``golden_dir``.

    Clears ``golden_dir`` first to keep generation repeatable.
    """
    import shutil

    from xhquant.api import HMONNXGoldenInference

    golden_path = Path(golden_dir)
    if golden_path.exists():
        shutil.rmtree(golden_path)
    golden_path.mkdir(parents=True, exist_ok=True)
    session = HMONNXGoldenInference(hmonnx_file)
    session.initialize()
    session.to(device)
    session.save_golden = True
    session.golden_dir = str(golden_dir)
    session.step = 0
    session(*inputs)
