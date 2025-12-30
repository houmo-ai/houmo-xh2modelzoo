import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto, helper


def verify_decomposition(
    original_node: onnx.NodeProto,
    decomposed_nodes: list[onnx.NodeProto],
    decomposed_initializers: list[onnx.TensorProto],
    full_graph: onnx.GraphProto,
    model_opsets,
):
    """
    Verifies that the decomposed subgraph behaves identically to the original MHA node.
    """
    print(f"    Verifying decomposition for node {original_node.name}...")

    # --- 1. Gather all necessary inputs and initializers ---
    all_graph_initializers = {init.name: init for init in full_graph.initializer}
    all_graph_inputs = {inp.name: inp for inp in full_graph.input}
    all_graph_value_infos = {vi.name: vi for vi in full_graph.value_info}

    node_input_names = list(original_node.input)
    node_inputs_vi = []

    for name in node_input_names:
        if name in all_graph_initializers or name == "":
            continue

        vi = None
        if name in all_graph_inputs:
            vi = all_graph_inputs[name]
        elif name in all_graph_value_infos:
            vi = all_graph_value_infos[name]

        if vi and vi.type.tensor_type.HasField("shape"):
            node_inputs_vi.append(vi)
        else:
            # Fallback for query/key/value if not in graph inputs/value_info or shape is missing
            shape = ["batch_size", 100, 256]
            new_vi = helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
            node_inputs_vi.append(new_vi)

    unique_node_inputs_dict = {vi.name: vi for vi in node_inputs_vi}
    unique_node_inputs_vi = list(unique_node_inputs_dict.values())

    output_name = original_node.output[0]
    output_vi = []
    if output_name in all_graph_value_infos and all_graph_value_infos[output_name].type.tensor_type.HasField("shape"):
        output_vi = [all_graph_value_infos[output_name]]
    else:
        shape = ["batch_size", 100, 256]
        output_vi = [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, shape)]

    # --- 2. Create the "original" model ---
    original_initializers = [
        all_graph_initializers[name] for name in node_input_names if name in all_graph_initializers
    ]
    graph_orig = helper.make_graph(
        [original_node], "original_mha", unique_node_inputs_vi, output_vi, initializer=original_initializers
    )
    model_orig = helper.make_model(graph_orig, opset_imports=list(model_opsets))

    # --- 3. Create the "decomposed" model ---
    decomposed_all_initializers = original_initializers + decomposed_initializers
    graph_decomp = helper.make_graph(
        decomposed_nodes, "decomposed_mha", unique_node_inputs_vi, output_vi, initializer=decomposed_all_initializers
    )
    model_decomp = helper.make_model(graph_decomp, opset_imports=list(model_opsets))

    try:
        onnx.checker.check_model(model_orig)
        onnx.checker.check_model(model_decomp)
    except Exception as e:
        print(f"    Verification failed: Sub-model check failed. Error: {e}")
        # onnx.save(model_orig, f"debug_original_{original_node.name.replace('/', '_')}.onnx")
        # onnx.save(model_decomp, f"debug_decomposed_{original_node.name.replace('/', '_')}.onnx")
        return

    # --- 4. Run inference and compare ---
    try:
        sess_orig = onnxruntime.InferenceSession(model_orig.SerializeToString())
        sess_decomp = onnxruntime.InferenceSession(model_decomp.SerializeToString())

        input_feed = {}
        for vi in unique_node_inputs_vi:
            shape = [
                dim.dim_value if (isinstance(dim, onnx.TensorShapeProto.Dimension) and dim.dim_value > 0) else 1
                for dim in vi.type.tensor_type.shape.dim
            ]
            input_feed[vi.name] = np.random.rand(*shape).astype(np.float32)

        output_orig = sess_orig.run(None, input_feed)[0]
        output_decomp = sess_decomp.run(None, input_feed)[0]

        if np.allclose(output_orig, output_decomp, atol=1e-5):
            print(f"    ✅ Verification PASSED for node {original_node.name}")
        else:
            print(f"    ❌ Verification FAILED for node {original_node.name}")
            print(f"       Max difference: {np.max(np.abs(output_orig - output_decomp))}")
    except Exception as e:
        print(f"    ❌ Verification FAILED for node {original_node.name} during ONNX Runtime execution.")
        print(f"       Error: {e}")


def manually_decompose_mha(node: onnx.NodeProto, graph: onnx.GraphProto, num_heads: int = 8):
    """
    Manually decomposes a MultiHeadAttention node for (Seq, Batch, Dim) layout.
    """
    print(f"  - Decomposing node: {node.name} with {num_heads} heads...")

    (
        query_name,
        key_name,
        value_name,
        q_weight_name,
        q_bias_name,
        k_weight_name,
        k_bias_name,
        v_weight_name,
        v_bias_name,
        o_weight_name,
        o_bias_name,
    ) = node.input

    output_name = node.output[0]

    initializers = {init.name: init for init in graph.initializer}
    if q_weight_name not in initializers:
        raise ValueError(f"Weight '{q_weight_name}' not found in initializers.")

    q_weight_tensor = initializers[q_weight_name]
    embed_dim = q_weight_tensor.dims[0]
    head_dim = embed_dim // num_heads
    if embed_dim % num_heads != 0:
        raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")

    new_nodes = []
    new_initializers = []

    def unique_name(name):
        return f"{node.name}_{name}"

    # 1. Project Q, K, V
    for type_char, input_name, weight_name, bias_name in [
        ("q", query_name, q_weight_name, q_bias_name),
        ("k", key_name, k_weight_name, k_bias_name),
        ("v", value_name, v_weight_name, v_bias_name),
    ]:
        proj_matmul_out = unique_name(f"{type_char}_proj_matmul")
        proj_add_out = unique_name(f"{type_char}_proj")
        weight_t_name = unique_name(f"{type_char}_weight_t")
        new_nodes.append(helper.make_node("Transpose", [weight_name], [weight_t_name], name=weight_t_name, perm=[1, 0]))
        new_nodes.append(
            helper.make_node("MatMul", [input_name, weight_t_name], [proj_matmul_out], name=proj_matmul_out)
        )
        new_nodes.append(helper.make_node("Add", [proj_matmul_out, bias_name], [proj_add_out], name=proj_add_out))

    # 2. Split heads for Q, K, V --
    # 输入形状: (S, B, E) -> Reshape -> (S, B, H, D) -> Transpose -> (B, H, S, D)
    shape_split_heads_name = unique_name("shape_split_heads")
    # -1 for SeqLen, 0 for Batch (copy from input)
    shape_split_heads = np.array([-1, 0, num_heads, head_dim], dtype=np.int64)
    new_initializers.append(helper.make_tensor(shape_split_heads_name, TensorProto.INT64, [4], shape_split_heads))

    for type_char in ["q", "k", "v"]:
        proj_out = unique_name(f"{type_char}_proj")
        reshaped_out = unique_name(f"{type_char}_reshaped")
        transposed_out = unique_name(f"{type_char}_transposed")
        new_nodes.extend(
            [
                helper.make_node("Reshape", [proj_out, shape_split_heads_name], [reshaped_out], name=reshaped_out),
                # Transpose (S, B, H, D) -> (B, H, S, D)
                helper.make_node("Transpose", [reshaped_out], [transposed_out], name=transposed_out, perm=[1, 2, 0, 3]),
            ]
        )

    # 3. Scaled Dot-Product Attention
    k_transposed_for_matmul_name, qk_matmul_name = unique_name("k_transposed_for_matmul"), unique_name("qk_matmul")
    # Transpose K: (B, H, S_k, D) -> (B, H, D, S_k)
    new_nodes.append(
        helper.make_node(
            "Transpose",
            [unique_name("k_transposed")],
            [k_transposed_for_matmul_name],
            name=k_transposed_for_matmul_name,
            perm=[0, 1, 3, 2],
        )
    )
    new_nodes.append(
        helper.make_node(
            "MatMul", [unique_name("q_transposed"), k_transposed_for_matmul_name], [qk_matmul_name], name=qk_matmul_name
        )
    )

    scaler_name, scaled_qk_name = unique_name("scaler"), unique_name("scaled_qk")
    new_initializers.append(helper.make_tensor(scaler_name, TensorProto.FLOAT, [], [1.0 / np.sqrt(head_dim)]))
    new_nodes.append(helper.make_node("Mul", [qk_matmul_name, scaler_name], [scaled_qk_name], name=scaled_qk_name))

    softmax_name, attention_out_name = unique_name("softmax"), unique_name("attention_out")
    new_nodes.append(helper.make_node("Softmax", [scaled_qk_name], [softmax_name], name=softmax_name, axis=-1))
    new_nodes.append(
        helper.make_node(
            "MatMul", [softmax_name, unique_name("v_transposed")], [attention_out_name], name=attention_out_name
        )
    )

    # 4. Combine Heads --
    # 输入形状: (B, H, S_q, D) -> Transpose -> (S_q, B, H, D) -> Reshape -> (S_q, B, E)
    attention_transposed_name = unique_name("attention_transposed")
    # Transpose (B, H, S_q, D) -> (S_q, B, H, D)
    new_nodes.append(
        helper.make_node(
            "Transpose",
            [attention_out_name],
            [attention_transposed_name],
            name=attention_transposed_name,
            perm=[2, 0, 1, 3],
        )
    )

    shape_combine_heads_name = unique_name("shape_combine_heads")
    # -1 for SeqLen_q, 0 for Batch
    shape_combine_heads = np.array([-1, 0, embed_dim], dtype=np.int64)
    new_initializers.append(helper.make_tensor(shape_combine_heads_name, TensorProto.INT64, [3], shape_combine_heads))
    combined_heads_name = unique_name("combined_heads")
    new_nodes.append(
        helper.make_node(
            "Reshape",
            [attention_transposed_name, shape_combine_heads_name],
            [combined_heads_name],
            name=combined_heads_name,
        )
    )

    # 5. Final Output Projection
    o_weight_t_name, output_matmul_name = unique_name("o_weight_t"), unique_name("output_matmul")
    new_nodes.append(
        helper.make_node("Transpose", [o_weight_name], [o_weight_t_name], name=o_weight_t_name, perm=[1, 0])
    )
    new_nodes.append(
        helper.make_node(
            "MatMul", [combined_heads_name, o_weight_t_name], [output_matmul_name], name=output_matmul_name
        )
    )
    new_nodes.append(
        helper.make_node("Add", [output_matmul_name, o_bias_name], [output_name], name=unique_name("output_add"))
    )

    print(f"  - Successfully decomposed node {node.name} into {len(new_nodes)} basic nodes.")
    return new_nodes, new_initializers


def prepare_detr_for_quantization_manual(input_path: str, output_path: str, verify: bool = False):
    """
    Manually prepares a DETR ONNX model... (docstring unchanged)
    """
    print(f"--- Starting manual preparation for model: {input_path} ---")

    try:
        model = onnx.load(input_path)
        graph = model.graph
        print("1. Loading original ONNX model... Successful.")

        print("2. Decomposing MultiHeadAttention operators...")

        final_nodes = []
        new_initializers = []
        mha_found = False

        for node in graph.node:
            if node.op_type == "MultiHeadAttention":
                mha_found = True
                decomposed_nodes, decomp_initializers = manually_decompose_mha(node, graph, num_heads=8)

                if verify:
                    verify_decomposition(node, decomposed_nodes, decomp_initializers, graph, model.opset_import)

                final_nodes.extend(decomposed_nodes)
                new_initializers.extend(decomp_initializers)
            else:
                final_nodes.append(node)

        if mha_found:
            graph.ClearField("node")
            graph.node.extend(final_nodes)
            graph.initializer.extend(new_initializers)
            print("   Decomposition complete.")
        else:
            print("   No MultiHeadAttention nodes found to decompose.")

        print("3. Cleaning, checking, and saving the final model...")
        onnx.checker.check_model(model)
        onnx.save(model, output_path)
        print(f"   Model check passed! Fully prepared model saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred during model preparation: {e}")
        import traceback

        traceback.print_exc()
        raise

    print("\n--- Model preparation finished successfully! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Manually prepare a DETR ONNX model by patching and decomposing MultiHeadAttention nodes.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input-onnx",
        type=str,
        default="data/models/detr/detr_1312.onnx",
        help="Path to the original DETR ONNX model.",
    )
    parser.add_argument(
        "--output-onnx",
        type=str,
        default="data/models/detr/detr_1312_prepared.onnx",
        help="Path to save the prepared ONNX model.",
    )
    # 别用，目前有bug
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Enable verification of each MHA decomposition against the original. Requires onnxruntime.",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output_onnx).parent.mkdir(exist_ok=True, parents=True)
    prepare_detr_for_quantization_manual(args.input_onnx, args.output_onnx, args.verify)
