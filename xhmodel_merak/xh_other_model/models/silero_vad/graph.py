from __future__ import annotations

import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.tools.onnx_model_utils import (
    fix_output_shapes,
    make_input_shape_fixed,
)
from onnxsim import simplify


RATE_CONFIG = {
    8000: {"frame": 256, "context": 32},
    16000: {"frame": 512, "context": 64},
}


def _rename_graph_values(graph: onnx.GraphProto, mapping: dict[str, str]) -> None:
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name in mapping:
                node.input[index] = mapping[name]
        for index, name in enumerate(node.output):
            if name in mapping:
                node.output[index] = mapping[name]
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                _rename_graph_values(attr.g, mapping)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attr.graphs:
                    _rename_graph_values(subgraph, mapping)
    for value in list(graph.input) + list(graph.output) + list(graph.value_info):
        if value.name in mapping:
            value.name = mapping[value.name]


def extract_rate_branch(model: onnx.ModelProto, sample_rate: int) -> onnx.ModelProto:
    if sample_rate not in RATE_CONFIG:
        raise ValueError("sample_rate must be 8000 or 16000")
    result = deepcopy(model)
    if_index = next(
        (index for index, node in enumerate(result.graph.node) if node.op_type == "If"),
        None,
    )
    if if_index is None:
        raise ValueError("Silero source graph has no sample-rate selector If")
    selector = result.graph.node[if_index]
    branch_name = "then_branch" if sample_rate == 16000 else "else_branch"
    branch = deepcopy(next(attr.g for attr in selector.attribute if attr.name == branch_name))
    _rename_graph_values(
        branch,
        {branch.output[index].name: selector.output[index] for index in range(len(selector.output))},
    )
    post_if_nodes = list(result.graph.node)[if_index + 1 :]
    del result.graph.node[:]
    result.graph.node.extend(branch.node)
    result.graph.node.extend(post_if_nodes)
    result.graph.initializer.extend(branch.initializer)
    for index in reversed(range(len(result.graph.input))):
        if result.graph.input[index].name == "sr":
            del result.graph.input[index]

    config = RATE_CONFIG[sample_rate]
    make_input_shape_fixed(
        result.graph,
        "input",
        [1, config["frame"] + config["context"]],
    )
    make_input_shape_fixed(result.graph, "state", [2, 1, 128])
    fix_output_shapes(result)
    onnx.checker.check_model(result)
    return result


def _test_feed(sample_rate: int, *, random: bool) -> dict[str, np.ndarray]:
    config = RATE_CONFIG[sample_rate]
    rng = np.random.default_rng(20260728)
    input_shape = (1, config["frame"] + config["context"])
    if random:
        return {
            "input": rng.standard_normal(input_shape).astype(np.float32),
            "state": rng.standard_normal((2, 1, 128)).astype(np.float32),
        }
    return {
        "input": np.zeros(input_shape, dtype=np.float32),
        "state": np.zeros((2, 1, 128), dtype=np.float32),
    }


def _evaluate_if_conditions(
    model: onnx.ModelProto,
    feeds: list[dict[str, np.ndarray]],
    temp_dir: Path,
) -> dict[str, bool]:
    if_nodes = [node for node in model.graph.node if node.op_type == "If"]
    if not if_nodes:
        return {}
    debug_model = deepcopy(model)
    existing = {value.name for value in debug_model.graph.output}
    for node in if_nodes:
        condition = node.input[0]
        if condition not in existing:
            debug_model.graph.output.append(helper.make_tensor_value_info(condition, TensorProto.BOOL, None))
            existing.add(condition)
    debug_path = temp_dir / "evaluate_if.onnx"
    onnx.save(debug_model, debug_path)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(debug_path),
        options,
        providers=["CPUExecutionProvider"],
    )
    output_names = [node.input[0] for node in if_nodes]
    evaluations = [[bool(np.asarray(value).item()) for value in session.run(output_names, feed)] for feed in feeds]
    if any(values != evaluations[0] for values in evaluations[1:]):
        raise RuntimeError("Silero If depends on tensor data instead of fixed shapes")
    return {node.name: evaluations[0][index] for index, node in enumerate(if_nodes)}


def _inline_if_pass(model: onnx.ModelProto, choices: dict[str, bool]) -> None:
    new_nodes: list[onnx.NodeProto] = []
    new_initializers: list[onnx.TensorProto] = []
    for node in model.graph.node:
        if node.op_type != "If":
            new_nodes.append(node)
            continue
        branch_name = "then_branch" if choices[node.name] else "else_branch"
        branch = deepcopy(next(attr.g for attr in node.attribute if attr.name == branch_name))
        _rename_graph_values(
            branch,
            {branch.output[index].name: node.output[index] for index in range(len(node.output))},
        )
        new_nodes.extend(branch.node)
        new_initializers.extend(branch.initializer)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(new_initializers)


def inline_shape_ifs(model: onnx.ModelProto, sample_rate: int) -> onnx.ModelProto:
    result = deepcopy(model)
    feeds = [
        _test_feed(sample_rate, random=False),
        _test_feed(sample_rate, random=True),
    ]
    with tempfile.TemporaryDirectory(prefix="silero-if-") as temp:
        temp_dir = Path(temp)
        rounds = 0
        while any(node.op_type == "If" for node in result.graph.node):
            _inline_if_pass(
                result,
                _evaluate_if_conditions(result, feeds, temp_dir),
            )
            rounds += 1
            if rounds > 16:
                raise RuntimeError("too many nested Silero If rounds")
    onnx.checker.check_model(result)
    return result


def lift_reflect_pad(model: onnx.ModelProto) -> int:
    """Rewrite rank-2 reflect Pad as equivalent rank-3 Pad for xhquant."""

    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    new_nodes: list[onnx.NodeProto] = []
    rewritten = 0
    for node in model.graph.node:
        mode = next(
            (helper.get_attribute_value(attr) for attr in node.attribute if attr.name == "mode"),
            b"constant",
        )
        if node.op_type != "Pad" or mode != b"reflect":
            new_nodes.append(node)
            continue
        pads = initializers.get(node.input[1])
        if pads is None or pads.tolist() not in ([0, 0, 0, 32], [0, 0, 0, 64]):
            raise RuntimeError(f"unexpected Silero reflect pads: {pads}")
        prefix = f"{node.name}_xhquant_rank3"
        unsqueeze_axes = f"{prefix}_unsqueeze_axes"
        squeeze_axes = f"{prefix}_squeeze_axes"
        rank3_pads = f"{prefix}_pads"
        pad_input = f"{prefix}_input"
        pad_output = f"{prefix}_output"
        public_output = node.output[0]
        model.graph.initializer.extend(
            [
                numpy_helper.from_array(np.asarray([1], dtype=np.int64), name=unsqueeze_axes),
                numpy_helper.from_array(np.asarray([1], dtype=np.int64), name=squeeze_axes),
                numpy_helper.from_array(
                    np.asarray(
                        [
                            int(pads[0]),
                            0,
                            int(pads[1]),
                            int(pads[2]),
                            0,
                            int(pads[3]),
                        ],
                        dtype=np.int64,
                    ),
                    name=rank3_pads,
                ),
            ]
        )
        new_nodes.append(
            helper.make_node(
                "Unsqueeze",
                [node.input[0], unsqueeze_axes],
                [pad_input],
                name=f"{prefix}_unsqueeze",
            )
        )
        node.input[0] = pad_input
        node.input[1] = rank3_pads
        node.output[0] = pad_output
        new_nodes.extend(
            [
                node,
                helper.make_node(
                    "Squeeze",
                    [pad_output, squeeze_axes],
                    [public_output],
                    name=f"{prefix}_squeeze",
                ),
            ]
        )
        rewritten += 1
    if rewritten:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
    return rewritten


def add_metadata(model: onnx.ModelProto, sample_rate: int) -> None:
    config = RATE_CONFIG[sample_rate]
    values = {
        "model_type": "silero-vad-v5-static-ifless",
        "sample_rate": str(sample_rate),
        "batch_size": "1",
        "frame_samples": str(config["frame"]),
        "context_samples": str(config["context"]),
        "input_samples": str(config["frame"] + config["context"]),
        "state_contract": "stateN from call k feeds state at call k+1",
        "context_contract": "Host prepends previous context to each new frame",
    }
    del model.metadata_props[:]
    for key, value in values.items():
        item = model.metadata_props.add()
        item.key = key
        item.value = value


def compare_streaming(
    source_path: Path,
    static_path: Path,
    sample_rate: int,
    steps: int = 5,
) -> float:
    source = ort.InferenceSession(str(source_path), providers=["CPUExecutionProvider"])
    static = ort.InferenceSession(str(static_path), providers=["CPUExecutionProvider"])
    config = RATE_CONFIG[sample_rate]
    rng = np.random.default_rng(sample_rate)
    source_state = np.zeros((2, 1, 128), dtype=np.float32)
    static_state = source_state.copy()
    max_error = 0.0
    for _ in range(steps):
        waveform = rng.standard_normal((1, config["frame"] + config["context"])).astype(np.float32) * 0.05
        source_output, source_state = source.run(
            None,
            {
                "input": waveform,
                "state": source_state,
                "sr": np.asarray(sample_rate, dtype=np.int64),
            },
        )
        static_output, static_state = static.run(
            None,
            {"input": waveform, "state": static_state},
        )
        max_error = max(
            max_error,
            float(np.max(np.abs(source_output - static_output))),
            float(np.max(np.abs(source_state - static_state))),
        )
    if max_error > 1e-6:
        raise AssertionError(f"Silero {sample_rate} specialization error: {max_error}")
    return max_error


def specialize_model(
    source_path: str | Path,
    output_path: str | Path,
    sample_rate: int,
    check_steps: int = 5,
) -> Path:
    source_path = Path(source_path)
    output_path = Path(output_path)
    selected = extract_rate_branch(onnx.load(source_path), sample_rate)
    ifless = inline_shape_ifs(selected, sample_rate)
    simplified, checked = simplify(ifless, check_n=3)
    if not checked:
        raise RuntimeError(f"Silero {sample_rate} onnxsim check failed")
    if any(node.op_type == "If" for node in simplified.graph.node):
        raise RuntimeError(f"Silero {sample_rate} still contains If")
    if lift_reflect_pad(simplified) != 1:
        raise RuntimeError("expected exactly one Silero reflect Pad rewrite")
    add_metadata(simplified, sample_rate)
    onnx.checker.check_model(simplified)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(simplified, output_path)
    compare_streaming(source_path, output_path, sample_rate, check_steps)
    return output_path


__all__ = [
    "RATE_CONFIG",
    "compare_streaming",
    "specialize_model",
]
