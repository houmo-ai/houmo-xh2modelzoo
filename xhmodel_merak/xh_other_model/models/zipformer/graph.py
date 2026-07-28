from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from onnxruntime.tools.onnx_model_utils import (
    fix_output_shapes,
    make_dim_param_fixed,
)
from onnxsim import simplify


STACK_LAYER_COUNTS = (2, 3, 2, 2, 3)
CACHE_FAMILIES = (
    "len",
    "avg",
    "key",
    "val",
    "val2",
    "conv1",
    "conv2",
)
CACHE_INPUT_PATTERN = re.compile(r"^cached_(len|avg|key|val|val2|conv1|conv2)_([0-4])$")


def _shape(value: onnx.ValueInfoProto) -> list[int]:
    result = []
    for dimension in value.type.tensor_type.shape.dim:
        if not dimension.HasField("dim_value"):
            raise ValueError(f"{value.name} still has a dynamic shape")
        result.append(int(dimension.dim_value))
    return result


def _dtype(value: onnx.ValueInfoProto) -> int:
    return int(value.type.tensor_type.elem_type)


def _concrete_shape(meta: ort.NodeArg) -> tuple[int, ...]:
    return tuple(int(dimension) for dimension in meta.shape)


def _numpy_dtype(type_name: str) -> np.dtype:
    if type_name == "tensor(int64)":
        return np.dtype(np.int64)
    if type_name == "tensor(int32)":
        return np.dtype(np.int32)
    return np.dtype(np.float32)


def _initial_feed(
    session: ort.InferenceSession,
) -> dict[str, np.ndarray]:
    return {
        value.name: np.zeros(_concrete_shape(value), dtype=_numpy_dtype(value.type)) for value in session.get_inputs()
    }


def convert_cached_len_interface_to_int32(
    model: onnx.ModelProto,
) -> int:
    """Keep public position states INT32 and cast to INT64 inside the graph."""

    input_names = {value.name for value in model.graph.input if value.name.startswith("cached_len_")}
    output_names = {value.name for value in model.graph.output if value.name.startswith("new_cached_len_")}
    if len(input_names) != 5 or len(output_names) != 5:
        raise RuntimeError("expected five cached_len inputs and five new_cached_len outputs")

    prefix_casts = []
    suffix_casts = []
    for public_name in sorted(input_names):
        internal_name = f"{public_name}__int64"
        for node in model.graph.node:
            for index, name in enumerate(node.input):
                if name == public_name:
                    node.input[index] = internal_name
        prefix_casts.append(
            helper.make_node(
                "Cast",
                [public_name],
                [internal_name],
                name=f"{public_name}_public_int32_to_int64",
                to=TensorProto.INT64,
            )
        )
        value = next(item for item in model.graph.input if item.name == public_name)
        value.type.tensor_type.elem_type = TensorProto.INT32

    for public_name in sorted(output_names):
        internal_name = f"{public_name}__int64"
        for node in model.graph.node:
            for index, name in enumerate(node.input):
                if name == public_name:
                    node.input[index] = internal_name
            for index, name in enumerate(node.output):
                if name == public_name:
                    node.output[index] = internal_name
        suffix_casts.append(
            helper.make_node(
                "Cast",
                [internal_name],
                [public_name],
                name=f"{public_name}_internal_int64_to_int32",
                to=TensorProto.INT32,
            )
        )
        value = next(item for item in model.graph.output if item.name == public_name)
        value.type.tensor_type.elem_type = TensorProto.INT32

    original_nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(prefix_casts)
    model.graph.node.extend(original_nodes)
    model.graph.node.extend(suffix_casts)
    return len(prefix_casts) + len(suffix_casts)


def layer_cache_name(cache_name: str, layer_index: int) -> str:
    return f"{cache_name}_layer{layer_index}"


def split_layer_cache_interface(model: onnx.ModelProto) -> int:
    """Expose each encoder layer cache as an independent leading-dim-1 I/O."""

    original_inputs = list(model.graph.input)
    original_outputs = list(model.graph.output)
    input_by_name = {value.name: value for value in original_inputs}
    prefix_nodes = []
    suffix_nodes = []
    split_initializers = []
    public_input_count = 0

    new_inputs: list[onnx.ValueInfoProto] = []
    for value in original_inputs:
        match = CACHE_INPUT_PATTERN.match(value.name)
        if match is None:
            new_inputs.append(value)
            continue
        stack_index = int(match.group(2))
        layer_count = STACK_LAYER_COUNTS[stack_index]
        shape = _shape(value)
        if shape[0] != layer_count:
            raise ValueError(
                f"{value.name} leading dimension {shape[0]} does not match stack layer count {layer_count}"
            )
        layer_shape = [1, *shape[1:]]
        layer_names = []
        for layer_index in range(layer_count):
            name = layer_cache_name(value.name, layer_index)
            layer_names.append(name)
            new_inputs.append(helper.make_tensor_value_info(name, _dtype(value), layer_shape))
            public_input_count += 1
        prefix_nodes.append(
            helper.make_node(
                "Concat",
                layer_names,
                [value.name],
                name=f"{value.name}_join_layers",
                axis=0,
            )
        )

    new_outputs: list[onnx.ValueInfoProto] = []
    for value in original_outputs:
        if not value.name.startswith("new_"):
            new_outputs.append(value)
            continue
        cache_name = value.name.removeprefix("new_")
        match = CACHE_INPUT_PATTERN.match(cache_name)
        if match is None:
            new_outputs.append(value)
            continue
        input_value = input_by_name[cache_name]
        stack_index = int(match.group(2))
        layer_count = STACK_LAYER_COUNTS[stack_index]
        layer_shape = [1, *_shape(input_value)[1:]]
        layer_names = [layer_cache_name(value.name, layer_index) for layer_index in range(layer_count)]
        sizes_name = f"{value.name}_layer_split_sizes"
        split_initializers.append(
            helper.make_tensor(
                sizes_name,
                TensorProto.INT64,
                [layer_count],
                [1] * layer_count,
            )
        )
        suffix_nodes.append(
            helper.make_node(
                "Split",
                [value.name, sizes_name],
                layer_names,
                name=f"{value.name}_split_layers",
                axis=0,
            )
        )
        new_outputs.extend(
            helper.make_tensor_value_info(name, _dtype(input_value), layer_shape) for name in layer_names
        )

    if public_input_count != sum(STACK_LAYER_COUNTS) * len(CACHE_FAMILIES):
        raise RuntimeError(f"expected 84 layer-cache inputs, got {public_input_count}")
    original_nodes = list(model.graph.node)
    del model.graph.input[:]
    model.graph.input.extend(new_inputs)
    del model.graph.output[:]
    model.graph.output.extend(new_outputs)
    del model.graph.node[:]
    model.graph.node.extend(prefix_nodes)
    model.graph.node.extend(original_nodes)
    model.graph.node.extend(suffix_nodes)
    model.graph.initializer.extend(split_initializers)
    return public_input_count


def join_split_state(outputs: dict[str, np.ndarray], cache_name: str) -> np.ndarray:
    match = CACHE_INPUT_PATTERN.match(cache_name)
    if match is None:
        raise ValueError(cache_name)
    layer_count = STACK_LAYER_COUNTS[int(match.group(2))]
    return np.concatenate(
        [outputs[layer_cache_name(f"new_{cache_name}", index)] for index in range(layer_count)],
        axis=0,
    )


def compare_streaming(
    source_path: str | Path,
    split_path: str | Path,
    steps: int = 3,
    atol: float = 1e-4,
) -> dict[str, float | int]:
    source = ort.InferenceSession(str(source_path), providers=["CPUExecutionProvider"])
    split = ort.InferenceSession(str(split_path), providers=["CPUExecutionProvider"])
    source_feed = {}
    for value in source.get_inputs():
        shape = tuple(1 if dimension == "N" else int(dimension) for dimension in value.shape)
        source_feed[value.name] = np.zeros(shape, dtype=_numpy_dtype(value.type))
    split_feed = _initial_feed(split)
    source_output_names = [value.name for value in source.get_outputs()]
    split_output_names = [value.name for value in split.get_outputs()]
    rng = np.random.default_rng(20260728)
    maximum_error = 0.0

    for _ in range(steps):
        features = rng.standard_normal((1, 39, 80)).astype(np.float32)
        source_feed["x"] = features
        split_feed["x"] = features
        source_values = source.run(None, source_feed)
        split_values = split.run(None, split_feed)
        source_outputs = dict(zip(source_output_names, source_values, strict=True))
        split_outputs = dict(zip(split_output_names, split_values, strict=True))
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(source_outputs["encoder_out"] - split_outputs["encoder_out"]))),
        )
        for cache_name in list(source_feed):
            if cache_name == "x":
                continue
            expected = source_outputs[f"new_{cache_name}"]
            observed = join_split_state(split_outputs, cache_name)
            if np.issubdtype(expected.dtype, np.integer):
                if not np.array_equal(expected.astype(np.int32), observed):
                    raise AssertionError(f"integer state mismatch: {cache_name}")
            else:
                maximum_error = max(
                    maximum_error,
                    float(np.max(np.abs(expected - observed))),
                )
            source_feed[cache_name] = expected
        for cache_name in list(split_feed):
            if cache_name != "x":
                split_feed[cache_name] = split_outputs[f"new_{cache_name}"]

    if maximum_error > atol:
        raise AssertionError(f"Zipformer maximum error {maximum_error} exceeds {atol}")
    return {
        "streaming_steps": steps,
        "maximum_float_error": maximum_error,
    }


def add_metadata(model: onnx.ModelProto) -> None:
    metadata = {item.key: item.value for item in model.metadata_props}
    metadata.update(
        {
            "fixed_batch_size": "1",
            "input_contract": ("x=[1,39,80] fbank window; advance by 32 frames"),
            "state_contract": (
                "84 leading-dim-1 layer cache inputs; every new_cached_* "
                "layer output feeds its same-named cached_* layer input"
            ),
            "cache_layer_counts": "2,3,2,2,3",
            "integer_contract": ("public cached_len states are INT32; internal position arithmetic remains INT64"),
        }
    )
    del model.metadata_props[:]
    for key, value in metadata.items():
        item = model.metadata_props.add()
        item.key = key
        item.value = value


def specialize_model(
    source_path: str | Path,
    output_path: str | Path,
    check_steps: int = 3,
) -> tuple[Path, dict[str, float | int]]:
    model = onnx.load(str(source_path))
    make_dim_param_fixed(model.graph, "N", 1)
    fix_output_shapes(model)
    model, check = simplify(model, check_n=3)
    if not check:
        raise RuntimeError("onnxsim equivalence check failed")
    if convert_cached_len_interface_to_int32(model) != 10:
        raise RuntimeError("failed to convert all cached_len interfaces")
    if split_layer_cache_interface(model) != 84:
        raise RuntimeError("failed to create 84 layer-cache interfaces")
    add_metadata(model)
    onnx.checker.check_model(model)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    metrics = compare_streaming(source_path, output, check_steps)
    return output, metrics


__all__ = [
    "CACHE_FAMILIES",
    "STACK_LAYER_COUNTS",
    "add_metadata",
    "compare_streaming",
    "convert_cached_len_interface_to_int32",
    "join_split_state",
    "layer_cache_name",
    "specialize_model",
    "split_layer_cache_interface",
]
