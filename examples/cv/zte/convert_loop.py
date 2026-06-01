import argparse
from copy import deepcopy
from typing import Optional
import onnx
from onnx import helper, numpy_helper, shape_inference


def _uniq(name: str, prefix: str) -> str:
    return f"{prefix}{name}"


def _clone_initializer(init, prefix: str):
    ni = deepcopy(init)
    ni.name = _uniq(init.name, prefix)
    return ni


def _clone_node(node, prefix: str, tensor_rename: dict):
    nn = deepcopy(node)
    nn.name = _uniq(node.name, prefix) if node.name else ""
    # rename inputs
    nn.input[:] = [tensor_rename.get(x, x) for x in node.input]
    # rename outputs
    nn.output[:] = [_uniq(x, prefix) for x in node.output]
    return nn


def _replace_all_graph_inputs_outputs(g, repl: dict):
    # Replace node inputs
    for n in g.node:
        n.input[:] = [repl.get(x, x) for x in n.input]
    # Replace graph outputs
    for o in g.output:
        if o.name in repl:
            o.name = repl[o.name]


def _make_int64_scalar_initializer(name: str, value: int):
    import numpy as np
    arr = np.array(value, dtype=np.int64)
    return numpy_helper.from_array(arr, name=name)


def _topo_sort_graph(graph: onnx.GraphProto):
    # Kahn's algorithm over node dependencies (inputs produced by other nodes)
    producers = {}
    for idx, node in enumerate(graph.node):
        for out in node.output:
            if out and out not in producers:
                producers[out] = idx

    deps = []
    rev = [[] for _ in graph.node]
    for idx, node in enumerate(graph.node):
        dep = set()
        for inp in node.input:
            if inp in producers:
                dep.add(producers[inp])
        deps.append(dep)
        for d in dep:
            rev[d].append(idx)

    ready = [i for i, d in enumerate(deps) if not d]
    ordered = []
    head = 0
    while head < len(ready):
        n = ready[head]
        head += 1
        ordered.append(n)
        for m in rev[n]:
            deps[m].discard(n)
            if not deps[m]:
                ready.append(m)

    if len(ordered) != len(graph.node):
        # fallback to original order if cycle or unresolved deps
        return

    new_nodes = [graph.node[i] for i in ordered]
    graph.ClearField("node")
    graph.node.extend(new_nodes)


def unroll_loop(model: onnx.ModelProto, loop_name: Optional[str], trip_count: int):
    g = model.graph

    # 1) find Loop node
    loop_idx = None
    for i, n in enumerate(g.node):
        if n.op_type == "Loop" and (loop_name is None or n.name == loop_name):
            loop_idx = i
            break
    if loop_idx is None:
        # fallback: first Loop
        for i, n in enumerate(g.node):
            if n.op_type == "Loop":
                loop_idx = i
                break
    if loop_idx is None:
        raise RuntimeError("No Loop node found in the graph.")

    loop_node = g.node[loop_idx]

    # 2) get body graph
    body_attr = None
    for a in loop_node.attribute:
        if a.name == "body":
            body_attr = a
            break
    if body_attr is None or not body_attr.g:
        raise RuntimeError("Loop node has no body graph attribute.")

    body = body_attr.g

    body_inputs = [vi.name for vi in body.input]
    body_outputs = [vi.name for vi in body.output]

    if len(body_inputs) < 2:
        raise RuntimeError("Unexpected Loop body inputs (<2).")

    # ONNX Loop body signature:
    # inputs:  iter, cond, loop_vars..., scan_inputs...
    # outputs: cond_out, loop_vars_out..., scan_outputs...
    iter_name = body_inputs[0]
    cond_name = body_inputs[1]

    # Loop node signature:
    # inputs: trip_count, cond, loop_vars..., scan_inputs...
    # outputs: loop_vars_out..., scan_outputs...
    #
    # IMPORTANT: tf2onnx sometimes packs extra counters into loop_vars;
    # we will follow *body graph* I/O count to decide split.

    # Determine how many loop_vars there are from the body outputs.
    # body_outputs = [cond_out] + loop_vars_out + scan_outputs
    # loop_node.outputs = loop_vars_out + scan_outputs
    n_loop_out = len(loop_node.output)
    n_body_out = len(body_outputs)

    if n_body_out < 1 or n_body_out != (1 + n_loop_out):
        # Not always exact in exotic exports, but for tf2onnx it's usually true.
        # We'll still try best-effort mapping by aligning from the end.
        pass

    # 3) Prepare to inline: we will create new nodes and new initializers
    new_nodes = []
    new_inits = []
    new_value_infos = []

    # We'll use constants for iter/cond to make the unroll deterministic.
    # cond = True always (or 1 if they cast-bool to int)
    # BUT: body expects the *type* of cond_name. We don't have easy type here
    # without loading the original model; tf2onnx often uses bool cond.
    # We'll generate BOTH and pick one by checking how cond is used:
    # - If cond_name is consumed by Less/Cast patterns, could be int64.
    # We'll inspect first consumer op_types quickly.
    cond_is_int64 = False
    for n in body.node:
        for inp in n.input:
            if inp == cond_name:
                # if it's used by Add/Gather as index, it's int64-ish
                if n.op_type in ("Add", "Gather", "Cast"):
                    cond_is_int64 = True
                break

    # Create per-iteration scalar indices for iter + "cond"
    # (If cond is really bool, value True; if int64, value 1)
    cond_const_value = 1 if cond_is_int64 else True

    # NOTE: ONNX initializer can't store Python bool directly via numpy_helper
    # using from_array with dtype=bool works. We'll implement both.
    import numpy as np

    def make_bool_scalar(name: str, v: bool):
        arr = np.array(v, dtype=np.bool_)
        return numpy_helper.from_array(arr, name=name)

    def make_int64_scalar(name: str, v: int):
        arr = np.array(v, dtype=np.int64)
        return numpy_helper.from_array(arr, name=name)

    # 4) Mapping for loop-carried vars:
    # body inputs after (iter, cond) correspond to loop_node inputs after (trip_count, cond)
    # until we've matched what the body expects.
    loop_node_inputs = list(loop_node.input)

    if len(loop_node_inputs) < 2:
        raise RuntimeError("Loop node has <2 inputs unexpectedly.")

    trip_in = loop_node_inputs[0]
    cond_in = loop_node_inputs[1]
    rest_in = loop_node_inputs[2:]

    body_rest_in = body_inputs[2:]  # loop_vars + scan_inputs
    # We don't know the split between loop_vars and scan_inputs; we just map by position.
    if len(rest_in) != len(body_rest_in):
        # Best effort: allow mismatch if tf2onnx inserted something; align from the end.
        # Most commonly they match.
        if len(rest_in) < len(body_rest_in):
            raise RuntimeError(
                f"Loop inputs mismatch: Loop has {len(rest_in)} vars after first two, "
                f"but body expects {len(body_rest_in)}."
            )

    # initial state mapping for iteration 0
    state_map = {}
    # Map body iter/cond later per-iteration
    for i, bin_name in enumerate(body_rest_in):
        state_map[bin_name] = rest_in[i]

    # We will keep track of the "current" values of body inputs
    cur_inputs = dict(state_map)

    # 5) Also inline body initializers into main graph (with per-iteration prefix to avoid collisions)
    # (yes, it's redundant across 9 iters, but safe & simple; onnxsim can fold later)
    body_inits = list(body.initializer)

    # Map body output name -> producer node for scan list detection
    body_output_producer = {}
    for n in body.node:
        for out in n.output:
            if out in body_outputs:
                body_output_producer[out] = n
    body_value_info = {v.name: v for v in body.value_info}

    # Outputs produced by TensorListSetItem/TensorArrayV2Write or Concat(prev_list, item)
    # are scan-list updates; we will keep only the last item and drop concat nodes.
    scan_list_output_indices = set()
    list_output_item_input = {}
    for i, out_name in enumerate(body_outputs[1:]):
        prod = body_output_producer.get(out_name)
        if not prod:
            continue
        if prod.op_type in ("TensorListSetItem", "TensorArrayV2Write"):
            scan_list_output_indices.add(i)
            if prod.input:
                list_output_item_input[out_name] = prod.input[-1]
            continue
        # tf2onnx often lowers TensorListSetItem to Concat(prev_list, item)
        if prod.op_type == "Concat" and prod.input:
            scan_list_output_indices.add(i)
            # prefer the input with dim0 == 1 (likely [1, batch, hidden])
            chosen = None
            for inp in prod.input:
                vi = body_value_info.get(inp)
                if not vi or not vi.type.tensor_type.HasField("shape"):
                    continue
                dims = vi.type.tensor_type.shape.dim
                if dims and dims[0].HasField("dim_value") and dims[0].dim_value == 1:
                    chosen = inp
                    break
            list_output_item_input[out_name] = chosen or prod.input[-1]

    # 6) Unroll iterations
    # We will capture the last iteration's mapped body outputs to rewire the original Loop outputs.
    last_body_outputs_renamed = None
    body_node_outputs = {o for n in body.node for o in n.output}

    for t in range(trip_count):
        prefix = f"unroll_t{t}/"

        # add per-iter iter/cond initializers
        iter_const_name = _uniq("__iter", prefix)
        if cond_is_int64:
            cond_const_name = _uniq("__cond_i64", prefix)
            new_inits.append(make_int64_scalar(iter_const_name, t))
            new_inits.append(make_int64_scalar(cond_const_name, int(cond_const_value)))
        else:
            cond_const_name = _uniq("__cond_bool", prefix)
            new_inits.append(make_int64_scalar(iter_const_name, t))
            new_inits.append(make_bool_scalar(cond_const_name, True))

        # build input rename map for this iteration:
        # - body inputs (iter/cond/vars...) -> actual tensors
        tensor_rename = {}
        tensor_rename[iter_name] = iter_const_name
        tensor_rename[cond_name] = cond_const_name

        for bin_name in body_rest_in:
            tensor_rename[bin_name] = cur_inputs[bin_name]

        # clone body initializers with prefix and map their names
        init_rename = {init.name: _uniq(init.name, prefix) for init in body_inits}
        tensor_rename.update(init_rename)
        for init in body_inits:
            new_inits.append(_clone_initializer(init, prefix))

        # rename internal tensors produced in the body
        internal_rename = {name: _uniq(name, prefix) for name in body_node_outputs}
        tensor_rename.update(internal_rename)

        # clone body nodes
        # outputs must be uniquely renamed by prefix
        for n in body.node:
            # skip scan-list concat nodes; we'll use the last item directly
            if n.op_type == "Concat" and n.output and n.output[0] in list_output_item_input:
                continue
            nn = _clone_node(n, prefix, tensor_rename)
            new_nodes.append(nn)

        # body outputs (original names) -> renamed names
        # if output is passthrough of a body input, reuse the mapped input tensor
        renamed_outs = []
        for x in body_outputs:
            if x in list_output_item_input:
                item_name = list_output_item_input[x]
                renamed_outs.append(tensor_rename.get(item_name, item_name))
            else:
                renamed_outs.append(tensor_rename.get(x, _uniq(x, prefix)))
        last_body_outputs_renamed = renamed_outs

        # update cur_inputs for next iteration:
        # body outputs: [cond_out] + updated(loop_vars...) + scan_outputs...
        # body_rest_in corresponds to loop_vars + scan_inputs (scan_inputs do not update)
        # loop_vars count is typically len(loop_node.output) for tf2onnx export (no explicit scan outputs in Loop output list),
        # BUT safest is to update as many as possible from body outputs excluding cond_out.
        updated_vals = renamed_outs[1:]  # skip cond_out

        # Update only for the portion that correspond to loop-carried vars (not scan_inputs).
        # Heuristic: loop-carried vars are the ones that are NOT constant "scan inputs".
        # In tf2onnx LSTM, the scan input is usually the last one (sequence tensor).
        # We'll assume scan_inputs are the tail that never appear in body outputs update.
        n_update = min(len(body_rest_in), len(updated_vals))
        for i in range(n_update):
            if i in scan_list_output_indices:
                continue
            cur_inputs[body_rest_in[i]] = updated_vals[i]

        # If there are extra scan_inputs beyond updated vars, keep them unchanged automatically.

    # 7) Rewire: replace Loop node outputs with last iteration values
    # Loop outputs list aligns with body outputs excluding cond_out.
    if last_body_outputs_renamed is None:
        raise RuntimeError("Unroll failed: no iterations executed.")

    replacement = {}
    last_updated = last_body_outputs_renamed[1:]  # skip cond_out

    # Map as many as we can (usually exact)
    for i, out_name in enumerate(loop_node.output):
        if i < len(last_updated):
            replacement[out_name] = last_updated[i]
        else:
            # If mismatch, keep original name (rare). Better than crashing.
            replacement[out_name] = out_name

    # 8) Remove original Loop node; insert new nodes
    # We'll insert at the original Loop position to keep topo order friendly.
    new_node_list = list(g.node)
    del new_node_list[loop_idx]
    new_node_list[loop_idx:loop_idx] = new_nodes
    g.ClearField("node")
    g.node.extend(new_node_list)

    # 9) Append new initializers (avoid name collisions)
    existing_init_names = {x.name for x in g.initializer}
    for init in new_inits:
        if init.name not in existing_init_names:
            g.initializer.append(init)
            existing_init_names.add(init.name)

    # 10) Replace all uses of old loop outputs
    _replace_all_graph_inputs_outputs(g, replacement)

    # 11) Topologically sort nodes after inlining
    _topo_sort_graph(g)

    # 11) (Optional) shape inference to clean up
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        # shape inference can fail on some graphs; it's ok.
        pass
    # Remove stale value_info that may have fixed batch dims from the Loop body.
    model.graph.ClearField("value_info")

    # Make graph outputs static by fixing batch dim to match input batch, if known.
    batch_dim = None
    if model.graph.input:
        in_shape = model.graph.input[0].type.tensor_type.shape
        if in_shape.dim and in_shape.dim[0].HasField("dim_value"):
            batch_dim = in_shape.dim[0].dim_value
    if batch_dim:
        for out in model.graph.output:
            t = out.type.tensor_type
            if t.HasField("shape") and t.shape.dim:
                d0 = t.shape.dim[0]
                if not d0.HasField("dim_value") or d0.dim_value == 0:
                    d0.dim_value = batch_dim

    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input .onnx")
    ap.add_argument("--out", dest="out", required=True, help="output .onnx")
    ap.add_argument("--loop_name", default="while_loop", help="Loop node name (default while_loop). If not found, uses first Loop.")
    ap.add_argument("--trip_count", type=int, default=9, help="unroll count, default 9")
    args = ap.parse_args()

    m = onnx.load(args.inp)
    m2 = unroll_loop(m, args.loop_name, args.trip_count)
    onnx.checker.check_model(m2)
    onnx.save(m2, args.out)
    print(f"[OK] saved unrolled model to: {args.out}")


if __name__ == "__main__":
    main()
