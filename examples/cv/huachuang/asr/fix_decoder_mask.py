import onnx
from onnx import helper, TensorProto
from collections import deque


def prune_by_outputs(model: onnx.ModelProto) -> onnx.ModelProto:
    """从 graph outputs 反向追溯，删掉不可达 node/init/input（让 speech_lengths 那条链自动清掉）"""
    g = model.graph

    producer = {}
    for n in g.node:
        for out in n.output:
            if out:
                producer[out] = n

    needed = set(o.name for o in g.output)
    q = deque(list(needed))

    while q:
        t = q.popleft()
        n = producer.get(t)
        if n is None:
            continue
        for inp in n.input:
            if inp and inp not in needed:
                needed.add(inp)
                q.append(inp)

    kept_nodes = [n for n in g.node if any(o in needed for o in n.output if o)]
    used_inputs = set()
    for n in kept_nodes:
        for inp in n.input:
            if inp:
                used_inputs.add(inp)

    kept_inits = [i for i in g.initializer if i.name in used_inputs]
    kept_init_names = {i.name for i in kept_inits}
    kept_graph_inputs = [i for i in g.input if (i.name in used_inputs and i.name not in kept_init_names)]

    del g.node[:]
    g.node.extend(kept_nodes)
    del g.initializer[:]
    g.initializer.extend(kept_inits)
    del g.input[:]
    g.input.extend(kept_graph_inputs)
    g.ClearField("value_info")
    return model


def replace_cast3_with_external_mask(
    in_onnx="in.onnx",
    out_onnx="out_mask_input.onnx",
    cast_node_name="/encoder/make_pad_mask/Cast_3",
    mask_tensor_name="/encoder/make_pad_mask/Cast_3_output_0",
    new_mask_input_name="encoder_masks",
    mask_shape=(1, 334),   # 你要 1*334 就用这个；如果后面期望 1*1*334 就改成 (1,1,334)
    do_prune=True
):
    model = onnx.load(in_onnx)
    g = model.graph

    # 1) 找到并删除 Cast_3 节点（避免“同名tensor既是input又被node产生”的冲突）
    target_idx = None
    for i, n in enumerate(g.node):
        if n.name == cast_node_name:
            target_idx = i
            # 顺便校验输出名确实匹配
            if len(n.output) != 1 or n.output[0] != mask_tensor_name:
                raise RuntimeError(
                    f"找到 {cast_node_name}，但输出不是期望的 {mask_tensor_name}，实际输出={n.output}"
                )
            break
    if target_idx is None:
        raise RuntimeError(f"找不到节点：{cast_node_name}")

    del g.node[target_idx]

    # 2) 把 new_mask_input_name 加为 graph input（float32，[1,334]）
    #    若已存在同名 input，先移除再加
    kept_inputs = [
        inp for inp in g.input if inp.name not in (mask_tensor_name, new_mask_input_name)
    ]
    del g.input[:]
    g.input.extend(kept_inputs)

    if new_mask_input_name != mask_tensor_name:
        for n in g.node:
            for j, inp in enumerate(n.input):
                if inp == mask_tensor_name:
                    n.input[j] = new_mask_input_name

    g.input.append(
        helper.make_tensor_value_info(new_mask_input_name, TensorProto.FLOAT, list(mask_shape))
    )

    # 3) 可选剪枝：把 speech_lengths -> Range/Less/... 那条链清掉
    if do_prune:
        model = prune_by_outputs(model)

    # 4) 检查并保存
    onnx.checker.check_model(model)
    onnx.save(model, out_onnx)
    print("[OK] saved:", out_onnx)
    print("     new input:", new_mask_input_name, "dtype=float32", "shape=", list(mask_shape))


if __name__ == "__main__":
    replace_cast3_with_external_mask(
        in_onnx="weights/huachuang/decoder.onnx",
        out_onnx="weights/huachuang/decoder_fix_mask.onnx",
        cast_node_name="/decoder/make_pad_mask/Cast_3",
        mask_tensor_name="/decoder/make_pad_mask/Cast_3_output_0",
        new_mask_input_name="pre_token_mask",
        mask_shape=(1, 92),
        do_prune=True,
    )   

    replace_cast3_with_external_mask(
        in_onnx="weights/huachuang/decoder_fix_mask.onnx",
        out_onnx="weights/huachuang/decoder_fix_mask2.onnx",
        cast_node_name="/decoder/Sub",
        mask_tensor_name="/decoder/Sub_output_0",
        new_mask_input_name="enc_mask",
        mask_shape=(1, 1, 1, 334),
        do_prune=True,
    )   
