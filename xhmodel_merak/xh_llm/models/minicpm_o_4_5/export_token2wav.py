from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper, shape_inference
from onnx.reference import ReferenceEvaluator
from torch import Tensor, nn

from xhquant.api import DeviceType, QuantScheme, convert_onnx_to_hmonnx, create_quant_config

from .minicpmo_token2wav_modules import (
    DeterministicHiftForwardWrapper,
    FlowDecoderWrapper,
    FlowEstimatorStepWrapper,
    FlowFrontendWrapper,
    FlowStreamingFrontendWrapper,
    _normalize_embedding,
)
from .streaming_token2wav import (
    FLOW_CACHE_NAMES,
    HiftStreamFinalWrapper,
    HiftStreamWrapper,
)


_RANDOM_ONNX_OPERATORS = frozenset({"RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike"})


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_flow_decoder_noise(template: Tensor, *, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(template.shape, dtype=template.dtype, generator=generator)


def constantize_dynamic_expand_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    """Fold constant-rooted Expand shape vectors into initializers, in place.

    The official Flow streaming internals build many Expand target shapes
    through chains of shape arithmetic whose leaves are all constants
    (`ConstantOfShape`/`Equal`/`Where`/...). ``torch.onnx.export`` leaves these
    as non-initializer tensors, so xhquant's Expand parser must resolve a
    dynamic shape even though it is provably constant. This pass evaluates any
    shape chain with no graph-input dependence up front, rewrites the Expand to
    consume an initializer, and drops the now-dead shape nodes. Semantics are
    unchanged (the folded value is the exact evaluated shape vector).
    """
    graph = model.graph
    node_list = list(graph.node)
    output_map: dict[str, onnx.NodeProto] = {}
    for node in node_list:
        for output in node.output:
            output_map.setdefault(output, node)
    initializer_map = {initializer.name: initializer for initializer in graph.initializer}
    graph_input_names = {value.name for value in graph.input}

    def _collect_constant_chain(shape_name: str):
        reachable: set[int] = set()
        leaves: set[str] = set()
        dynamic = [False]
        seen: set[str] = set()

        def walk(name: str) -> None:
            if name in seen:
                return
            seen.add(name)
            if name in initializer_map or name in graph_input_names:
                leaves.add(name)
                if name in graph_input_names:
                    dynamic[0] = True
                return
            node = output_map.get(name)
            if node is None:
                leaves.add(name)
                return
            if node.op_type == "Constant":
                leaves.add(name)
                return
            if id(node) in reachable:
                return
            reachable.add(id(node))
            for source in node.input:
                if source:
                    walk(source)

        walk(shape_name)
        return reachable, leaves, dynamic[0]

    shape_suffix = 0
    for expand in list(graph.node):
        if expand.op_type != "Expand" or len(expand.input) < 2:
            continue
        shape_name = expand.input[1]
        if shape_name in initializer_map:
            continue
        reachable, leaves, dynamic = _collect_constant_chain(shape_name)
        if dynamic or not reachable:
            continue
        topo_nodes = [candidate for candidate in node_list if id(candidate) in reachable]
        sub_initializers: list[onnx.TensorProto] = []
        for leaf in leaves:
            if leaf in initializer_map:
                sub_initializers.append(initializer_map[leaf])
            else:
                constant_node = output_map.get(leaf)
                if constant_node is not None and constant_node.op_type == "Constant":
                    value_attr = next((attr for attr in constant_node.attribute if attr.name == "value"), None)
                    if value_attr is not None:
                        sub_initializers.append(
                            copy.deepcopy(numpy_helper.from_array(numpy_helper.to_array(value_attr.t), leaf))
                        )
        sub_graph = onnx.helper.make_graph(
            topo_nodes,
            "expand_shape_fold",
            [],
            [onnx.helper.make_tensor_value_info(shape_name, onnx.TensorProto.INT64, None)],
            initializer=sub_initializers,
        )
        sub_model = onnx.helper.make_model(
            sub_graph,
            opset_imports=[onnx.helper.make_opsetid("", 17)],
            ir_version=model.ir_version,
        )
        reference = ReferenceEvaluator(sub_model)
        try:
            evaluated = reference.run(None, {})[0]
        except Exception:
            continue
        if evaluated.ndim != 1 or evaluated.dtype != np.int64:
            continue
        folded_name = f"expand_shape_folded_{shape_suffix}"
        shape_suffix += 1
        graph.initializer.append(numpy_helper.from_array(np.asarray(evaluated, dtype=np.int64), folded_name))
        expand.input[1] = folded_name
        for redundant in reachable:
            node = next(candidate for candidate in node_list if id(candidate) == redundant)
            for output in node.output:
                output_map.pop(output, None)
        for candidate in node_list:
            if id(candidate) in reachable:
                graph.node.remove(candidate)
    return model


def staticize_known_shape_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace ``Shape`` nodes whose source tensor has a fixed inferred shape.

    Streaming Flow is exported with fixed token and cache capacities. The
    official cache-buffer stores nevertheless lower their fixed extents as
    ``Shape(runtime_tensor)`` because the stores are expressed using runtime
    indexing. This pass specializes only the shape metadata; tensor values,
    cache reads, cache writes, and valid-length inputs remain graph inputs and
    therefore retain their runtime data dependence.
    """
    inferred = shape_inference.infer_shapes(model)
    shape_by_name: dict[str, tuple[int, ...]] = {}

    def add_shape(value: onnx.ValueInfoProto) -> None:
        tensor = value.type.tensor_type
        if not tensor.HasField("shape"):
            return
        dimensions = tuple(dimension.dim_value for dimension in tensor.shape.dim)
        if dimensions and all(dimension > 0 for dimension in dimensions):
            shape_by_name[value.name] = dimensions

    for value in inferred.graph.input:
        add_shape(value)
    for value in inferred.graph.value_info:
        add_shape(value)
    for value in inferred.graph.output:
        add_shape(value)
    for initializer in model.graph.initializer:
        shape_by_name[initializer.name] = tuple(initializer.dims)

    existing_initializers = {initializer.name for initializer in model.graph.initializer}
    shape_index = 0
    replacements: dict[str, str] = {}
    removable: set[str] = set()
    for node in model.graph.node:
        if node.op_type != "Shape" or not node.output:
            continue
        source_shape = shape_by_name.get(node.input[0])
        if source_shape is None:
            continue
        replacement = f"static_shape_{shape_index}"
        while replacement in existing_initializers:
            shape_index += 1
            replacement = f"static_shape_{shape_index}"
        shape_index += 1
        model.graph.initializer.append(
            numpy_helper.from_array(np.asarray(source_shape, dtype=np.int64), name=replacement)
        )
        existing_initializers.add(replacement)
        replacements[node.output[0]] = replacement
        removable.add(node.name or node.output[0])

    if not replacements:
        return model
    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            if input_name in replacements:
                node.input[index] = replacements[input_name]
    kept_nodes = [node for node in model.graph.node if (node.name or node.output[0]) not in removable]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    return model


def remove_identity_expands(model: onnx.ModelProto) -> onnx.ModelProto:
    inferred = shape_inference.infer_shapes(model)
    shapes: dict[str, tuple[int, ...]] = {}
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        tensor = value.type.tensor_type
        if not tensor.HasField("shape"):
            continue
        dimensions = tuple(dimension.dim_value for dimension in tensor.shape.dim)
        if dimensions and all(dimension > 0 for dimension in dimensions):
            shapes[value.name] = dimensions
    rewrites: dict[str, str] = {}
    removable: set[str] = set()
    for node in model.graph.node:
        if node.op_type != "Expand" or shapes.get(node.input[0]) != shapes.get(node.output[0]):
            continue
        rewrites[node.output[0]] = node.input[0]
        removable.add(node.name or node.output[0])
    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            if input_name in rewrites:
                node.input[index] = rewrites[input_name]
    kept_nodes = [node for node in model.graph.node if (node.name or node.output[0]) not in removable]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    return model


def remove_static_broadcast_expands(model: onnx.ModelProto) -> onnx.ModelProto:
    inferred = shape_inference.infer_shapes(model)
    shapes: dict[str, tuple[int, ...]] = {}
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        tensor = value.type.tensor_type
        if not tensor.HasField("shape"):
            continue
        dimensions = tuple(dimension.dim_value for dimension in tensor.shape.dim)
        if dimensions and all(dimension > 0 for dimension in dimensions):
            shapes[value.name] = dimensions
    initializer_values = {
        initializer.name: tuple(int(value) for value in numpy_helper.to_array(initializer).reshape(-1).tolist())
        for initializer in model.graph.initializer
    }
    rewrites: dict[str, str] = {}
    removable: set[str] = set()
    for node in model.graph.node:
        if node.op_type != "Expand" or node.input[1] not in initializer_values:
            continue
        source_shape = shapes.get(node.input[0])
        if source_shape is None:
            continue
        target_shape = initializer_values[node.input[1]]
        if len(target_shape) < len(source_shape):
            continue
        aligned_source = (1,) * (len(target_shape) - len(source_shape)) + source_shape
        if aligned_source != target_shape:
            continue
        rewrites[node.output[0]] = node.input[0]
        removable.add(node.name or node.output[0])
    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            if input_name in rewrites:
                node.input[index] = rewrites[input_name]
    kept_nodes = [node for node in model.graph.node if (node.name or node.output[0]) not in removable]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    return model


@contextmanager
def _exportable_normalize():
    original = torch.nn.functional.normalize
    torch.nn.functional.normalize = _normalize_embedding
    try:
        yield
    finally:
        torch.nn.functional.normalize = original


def make_stream_export_flow(flow: nn.Module, *, cache_capacity: int, append_capacity: int = 56) -> nn.Module:
    export_flow = copy.deepcopy(flow)
    decoder = export_flow.decoder
    estimator = decoder.estimator
    decoder_att_cache = decoder.att_cache_buffer
    estimator_att_cache = estimator.att_cache_buffer
    decoder.register_buffer(
        "att_cache_buffer",
        decoder_att_cache.new_zeros(
            (*decoder_att_cache.shape[:-2], cache_capacity + append_capacity, decoder_att_cache.shape[-1])
        ),
        persistent=False,
    )
    estimator.register_buffer(
        "att_cache_buffer",
        estimator_att_cache.new_zeros(
            (*estimator_att_cache.shape[:-2], cache_capacity + append_capacity, estimator_att_cache.shape[-1])
        ),
        persistent=False,
    )
    if hasattr(estimator, "use_cuda_graph"):
        estimator.use_cuda_graph = False
    return export_flow


def create_hift_random_inputs(
    *,
    harmonic_count: int,
    source_length: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    phase_noise = torch.rand((1, harmonic_count), generator=generator)
    phase_noise[:, 0] = 0
    source_noise = torch.randn((1, source_length, harmonic_count), generator=generator)
    return phase_noise, source_noise


def require_deterministic_onnx(model: onnx.ModelProto) -> None:
    random_operators = sorted({node.op_type for node in model.graph.node if node.op_type in _RANDOM_ONNX_OPERATORS})
    if random_operators:
        raise RuntimeError(f"HiFT ONNX contains random operators: {', '.join(random_operators)}")


def load_token2wav_modules(model_dir: str) -> tuple[nn.Module, nn.Module]:
    import stepaudio2.token2wav as token2wav
    from hyperpyyaml import load_hyperpyyaml

    token2wav._setup_cosyvoice2_alias()
    asset_dir = Path(model_dir) / "assets" / "token2wav"
    with (asset_dir / "flow.yaml").open(encoding="utf-8") as stream:
        flow = load_hyperpyyaml(stream)["flow"]
    flow.load_state_dict(torch.load(asset_dir / "flow.pt", map_location="cpu", weights_only=True), strict=True)
    flow.eval()
    hift = token2wav.HiFTGenerator()
    state = torch.load(asset_dir / "hift.pt", map_location="cpu", weights_only=True)
    hift.load_state_dict({key.removeprefix("generator."): value for key, value in state.items()}, strict=True)
    hift.eval()
    return flow, hift


def _convert(
    module: nn.Module,
    inputs: Sequence[Tensor],
    names: tuple[Sequence[str], Sequence[str]],
    output_file: Path,
    target_device: str,
    quant_type: str,
    prepare: Callable[[onnx.ModelProto], onnx.ModelProto] | None = None,
) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    onnx_dir = output_file.parent / "onnx"
    hmonnx_dir = output_file.parent / "hmonnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    hmonnx_dir.mkdir(parents=True, exist_ok=True)
    source_file = onnx_dir / f"{output_file.stem}_source.onnx"
    hmonnx_file = hmonnx_dir / output_file.name
    with _exportable_normalize():
        torch.onnx.export(
            module,
            tuple(inputs),
            str(source_file),
            input_names=list(names[0]),
            output_names=list(names[1]),
            opset_version=17,
            do_constant_folding=True,
        )
    folded_model = staticize_known_shape_nodes(onnx.load(str(source_file)))
    folded_model = constantize_dynamic_expand_shapes(folded_model)
    folded_model = remove_identity_expands(folded_model)
    folded_model = remove_static_broadcast_expands(folded_model)
    if prepare is not None:
        folded_model = prepare(folded_model)
    onnx.save(folded_model, str(source_file))
    onnx.checker.check_model(str(source_file))
    if target_device != "XH2a":
        raise RuntimeError(f"Unsupported Token2Wav target device: {target_device}")
    quant_config = create_quant_config(QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type))
    convert_onnx_to_hmonnx(
        str(source_file),
        [value.cpu() for value in inputs],
        DeviceType.XH2a,
        str(hmonnx_file),
        quant_config=quant_config,
        input_names=list(names[0]),
        output_names=list(names[1]),
    )
    return hmonnx_file


def prepare_hift_onnx(model: onnx.ModelProto, *, frame_capacity: int) -> onnx.ModelProto:
    sizes_name = "token2wav_hift_source_sizes"
    model.graph.initializer.append(
        numpy_helper.from_array(np.array([1, 9, frame_capacity], dtype=np.int64), name=sizes_name)
    )
    for node in model.graph.node:
        if node.name == "/m_source/l_sin_gen/Resize":
            while len(node.input) < 4:
                node.input.append("")
            node.input[2] = ""
            node.input[3] = sizes_name
    return model


def prepare_hift_stream_onnx(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace scale-based Resize with fixed sizes for the streaming HiFT graphs.

    The stochastic sine generation downscales/upscales by a fractional scale factor
    ``1/480`` that loses precision in fp16: for the streaming source length 31680 the
    backend floors the downsampled length to 65 (-> 31200 samples) instead of 66
    (-> 31680). Pinning each Resize to explicit integer sizes keeps the fixed-shape
    contract exact so the HMONNX graph runs on the simulator.
    """
    infer_model = shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    del model.graph.value_info[:]
    model.graph.value_info.extend(infer_model.graph.value_info)
    shapes: dict[str, list[int]] = {}
    for value in list(infer_model.graph.value_info) + list(infer_model.graph.input):
        tensor = value.type.tensor_type
        if tensor.HasField("shape"):
            shapes[value.name] = [dim.dim_value for dim in tensor.shape.dim]
    for value in infer_model.graph.output:
        tensor = value.type.tensor_type
        if tensor.HasField("shape"):
            shapes[value.name] = [dim.dim_value for dim in tensor.shape.dim]
    constants: dict[str, np.ndarray] = {}
    for initializer in model.graph.initializer:
        constants[initializer.name] = numpy_helper.to_array(initializer)
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attribute in node.attribute:
                if attribute.name == "value":
                    constants[node.output[0]] = numpy_helper.to_array(attribute.t)

    # torch.onnx may leave the intermediate shape of the three HiFT source
    # Resizes unavailable during the first export pass.  In that case the
    # scale-to-size rewrite below would silently skip the node, and the
    # HMONNX frontend can round 30720 samples down to 30240 in fp16.  The
    # source-noise input gives us the fixed physical length and harmonic
    # count, so use the known HiFT source path as a deterministic fallback.
    source_noise = next((value for value in model.graph.input if value.name == "source_noise"), None)
    source_length = 0
    harmonic_count = 0
    if source_noise is not None:
        dimensions = source_noise.type.tensor_type.shape.dim
        if len(dimensions) >= 3:
            source_length = int(dimensions[1].dim_value)
            harmonic_count = int(dimensions[2].dim_value)
    for node in model.graph.node:
        if node.op_type != "Resize" or len(node.input) < 3 or not node.input[2]:
            continue
        scales_name = node.input[2]
        scales = constants.get(scales_name)
        input_shape = shapes.get(node.input[0])
        if scales is None:
            continue
        scales = scales.reshape(-1)
        if scales.size != 3:
            continue
        # If shape inference cannot provide a fully static input shape, use the
        # fixed source-noise contract. This covers both the dynamo exporter
        # (missing intermediate value_info) and legacy exporter graphs whose
        # inferred dimensions contain zeros/dynamic parameters.
        if input_shape is None or any(dimension <= 0 for dimension in input_shape):
            if not source_length or not harmonic_count:
                continue
            scale = float(scales[-1])
            if scale <= 0:
                continue
            # The f0 branch starts from a mel-rate tensor and its Resize
            # scale is 480; its target is the already-fixed source length.
            # The two source branches, in contrast, use 1/480 then 480.
            is_f0_resize = "/f0_upsamp/" in node.name or "Unsqueeze" in node.input[0]
            target = source_length if is_f0_resize or scale > 1.0 else int(round(source_length * scale))
            if target <= 0:
                continue
            channels = 1 if is_f0_resize else harmonic_count
            sizes = [1, channels, target]
        else:
            sizes = []
            for axis, dimension in enumerate(input_shape):
                factor = float(scales[axis]) if axis < len(scales) else 1.0
                target = int(round(dimension * factor))
                if target <= 0:
                    break
                sizes.append(target)
            if len(sizes) != len(input_shape):
                continue
        sizes_name = f"{node.name}_stream_sizes"
        model.graph.initializer.append(numpy_helper.from_array(np.asarray(sizes, dtype=np.int64), name=sizes_name))
        while len(node.input) < 4:
            node.input.append("")
        node.input[2] = ""
        node.input[3] = sizes_name
    constant_values: dict[str, np.ndarray] = {}
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    for initializer in model.graph.initializer:
        constant_values[initializer.name] = numpy_helper.to_array(initializer)
    for node in model.graph.node:
        if node.op_type == "Constant":
            value = next((attribute.t for attribute in node.attribute if attribute.name == "value"), None)
            if value is not None:
                constant_values[node.output[0]] = numpy_helper.to_array(value)
    producers = {output: node for node in model.graph.node for output in node.output}

    def evaluate_constant(name: str) -> np.ndarray | None:
        if name in constant_values:
            return constant_values[name]
        node = producers.get(name)
        if node is None:
            return None
        values = [evaluate_constant(input_name) for input_name in node.input if input_name]
        if any(value is None for value in values):
            return None
        resolved = [value for value in values if value is not None]
        if node.op_type == "Cast":
            result = resolved[0].astype(np.int64)
        elif node.op_type == "Concat":
            axis = next(attribute.i for attribute in node.attribute if attribute.name == "axis")
            result = np.concatenate(resolved, axis=axis)
        elif node.op_type == "Unsqueeze":
            axes = resolved[1].reshape(-1).tolist()
            result = resolved[0]
            for axis in sorted((int(axis) for axis in axes)):
                result = np.expand_dims(result, axis=axis)
        elif node.op_type == "ConstantOfShape":
            shape = tuple(int(item) for item in resolved[0].reshape(-1))
            attribute = next((attribute for attribute in node.attribute if attribute.name == "value"), None)
            fill = numpy_helper.to_array(attribute.t).item() if attribute is not None else 0.0
            result = np.full(shape, fill)
        elif node.op_type == "Reshape":
            result = resolved[0].reshape(tuple(int(item) for item in resolved[1].reshape(-1)))
        elif node.op_type == "Transpose":
            permutation = next((list(attribute.ints) for attribute in node.attribute if attribute.name == "perm"), None)
            result = np.transpose(resolved[0], axes=permutation)
        elif node.op_type == "Slice":
            starts = resolved[1].reshape(-1)
            ends = resolved[2].reshape(-1)
            axes = resolved[3].reshape(-1) if len(resolved) > 3 else np.arange(starts.size)
            steps = resolved[4].reshape(-1) if len(resolved) > 4 else np.ones(starts.size)
            slices = [slice(None)] * resolved[0].ndim
            for start, end, axis, step in zip(starts, ends, axes, steps, strict=True):
                slices[int(axis)] = slice(int(start), int(end), int(step))
            result = resolved[0][tuple(slices)]
        else:
            return None
        constant_values[name] = result
        return result

    slice_index = 0
    for node in model.graph.node:
        if node.op_type != "Slice":
            continue
        for input_index in range(1, min(len(node.input), 5)):
            name = node.input[input_index]
            if not name or name in initializer_names:
                continue
            value = evaluate_constant(name)
            if value is None:
                continue
            initializer_name = f"hift_stream_slice_{slice_index}_{input_index}"
            slice_index += 1
            model.graph.initializer.append(numpy_helper.from_array(np.asarray(value, dtype=np.int64), initializer_name))
            initializer_names.add(initializer_name)
            constant_values[initializer_name] = np.asarray(value, dtype=np.int64)
            node.input[input_index] = initializer_name
    for node in model.graph.node:
        if (
            node.op_type not in {"CumSum", "Reshape", "Squeeze", "Unsqueeze"}
            or len(node.input) < 2
            or node.input[1] in initializer_names
        ):
            continue
        value = evaluate_constant(node.input[1])
        if value is None:
            continue
        initializer_name = f"hift_stream_unsqueeze_{slice_index}"
        slice_index += 1
        model.graph.initializer.append(numpy_helper.from_array(np.asarray(value, dtype=np.int64), initializer_name))
        initializer_names.add(initializer_name)
        constant_values[initializer_name] = np.asarray(value, dtype=np.int64)
        node.input[1] = initializer_name
    for node in model.graph.node:
        if node.op_type != "Pad" or len(node.input) < 2 or node.input[1] in initializer_names:
            continue
        value = evaluate_constant(node.input[1])
        if value is None:
            continue
        initializer_name = f"hift_stream_pad_{slice_index}"
        slice_index += 1
        model.graph.initializer.append(numpy_helper.from_array(np.asarray(value, dtype=np.int64), initializer_name))
        initializer_names.add(initializer_name)
        constant_values[initializer_name] = np.asarray(value, dtype=np.int64)
        node.input[1] = initializer_name
    for node in model.graph.node:
        if node.op_type != "STFT":
            continue
        for input_index in range(1, min(len(node.input), 4)):
            name = node.input[input_index]
            if name in initializer_names:
                continue
            value = evaluate_constant(name)
            if value is None:
                continue
            initializer_name = f"hift_stream_stft_{slice_index}_{input_index}"
            slice_index += 1
            array = np.asarray(value)
            model.graph.initializer.append(numpy_helper.from_array(array, initializer_name))
            initializer_names.add(initializer_name)
            constant_values[initializer_name] = array
            node.input[input_index] = initializer_name
    for node in model.graph.node:
        if node.op_type != "Pow" or node.input[1] in initializer_names:
            continue
        value = evaluate_constant(node.input[1])
        if value is None:
            continue
        initializer_name = f"hift_stream_pow_{slice_index}"
        slice_index += 1
        array = np.asarray(value)
        model.graph.initializer.append(numpy_helper.from_array(array, initializer_name))
        initializer_names.add(initializer_name)
        constant_values[initializer_name] = array
        node.input[1] = initializer_name
    for node in model.graph.node:
        if node.op_type not in {"ReduceMean", "ReduceSum"} or len(node.input) < 2 or node.input[1] in initializer_names:
            continue
        value = evaluate_constant(node.input[1])
        if value is None:
            continue
        initializer_name = f"hift_stream_reduce_{slice_index}"
        slice_index += 1
        array = np.asarray(value, dtype=np.int64)
        model.graph.initializer.append(numpy_helper.from_array(array, initializer_name))
        initializer_names.add(initializer_name)
        constant_values[initializer_name] = array
        node.input[1] = initializer_name
    if source_noise is not None:
        dimensions = source_noise.type.tensor_type.shape.dim
        source_length = int(dimensions[1].dim_value) if len(dimensions) >= 2 else 0
        if source_length > 0:
            output_shapes = {"raw_waveform": (1, source_length), "full_source": (1, 1, source_length)}
            for output in model.graph.output:
                shape = output_shapes.get(output.name)
                if shape is None:
                    continue
                output.type.tensor_type.shape.ClearField("dim")
                for size in shape:
                    output.type.tensor_type.shape.dim.add().dim_value = size
    return model


def export_flow_frontend(
    flow: nn.Module,
    component_dir: Path,
    config: Mapping[str, object],
    target_device: str,
    name_prefix: str = "",
) -> dict[str, object]:
    token_capacity = int(config["token_capacity"])
    mel_capacity = int(config["mel_capacity"])
    quant_type = str(config["quant_type"])
    inputs = (
        torch.zeros((1, token_capacity), dtype=torch.int32),
        torch.tensor([token_capacity], dtype=torch.int32),
        torch.zeros((1, mel_capacity, 80), dtype=torch.float32),
        torch.tensor([mel_capacity // 2], dtype=torch.int32),
        torch.zeros((1, 192), dtype=torch.float32),
    )
    graph = _convert(
        FlowFrontendWrapper(flow, mel_capacity),
        inputs,
        (("tokens", "token_length", "prompt_feat", "prompt_feat_length", "embedding"), ("mu", "mask", "spks", "cond")),
        component_dir / "Offline" / f"{name_prefix}token2wav_flow_frontend_offline_{target_device}_{quant_type}.onnx",
        target_device,
        quant_type,
    )
    graphs = {"main": graph}
    stream_contract = None
    streaming = config.get("streaming")
    if isinstance(streaming, Mapping):
        prompt_token_capacity = int(streaming["prompt_token_capacity"])
        prompt_mel_capacity = int(streaming["prompt_mel_capacity"])
        stream_inputs = (
            torch.zeros((1, prompt_token_capacity), dtype=torch.int32),
            torch.zeros((1, prompt_mel_capacity, 80), dtype=torch.float32),
            torch.zeros((1, 192), dtype=torch.float32),
        )
        base_cache = flow.setup_cache(
            stream_inputs[0],
            stream_inputs[1],
            stream_inputs[2],
            n_timesteps=int(config.get("n_timesteps", 10)),
        )
        configured_base_shapes = {
            name: tuple(int(value) for value in streaming["base_cache_shapes"][name]) for name in FLOW_CACHE_NAMES
        }
        base_cache_files = {}
        for name in FLOW_CACHE_NAMES:
            actual_shape = tuple(int(value) for value in base_cache[name].shape)
            if actual_shape != configured_base_shapes[name]:
                raise RuntimeError(
                    f"Token2Wav base cache {name} shape {actual_shape} != configured {configured_base_shapes[name]}"
                )
            cache_file = component_dir / f"{name}_base.pt"
            torch.save(base_cache[name].detach().cpu(), cache_file)
            base_cache_files[name] = cache_file
        stream_contract = {
            "initialization_backend": "official_host",
            "base_cache_artifact_kind": "deterministic_template",
            "prompt_token_capacity": prompt_token_capacity,
            "prompt_mel_capacity": prompt_mel_capacity,
            "pre_lookahead_len": int(streaming["pre_lookahead_len"]),
            "base_cache_valid_length": int(streaming["base_cache_valid_length"]),
            "base_cache_shapes": {name: list(shape) for name, shape in configured_base_shapes.items()},
            "cache_dtypes": {name: "float32" for name in FLOW_CACHE_NAMES},
            "base_cache_files": base_cache_files,
        }
        all_cache_shapes = {
            name: tuple(int(value) for value in streaming["cache_shapes"][name]) for name in FLOW_CACHE_NAMES
        }
        cache_shapes = {name: all_cache_shapes[name] for name in ("conformer_cnn_cache", "conformer_att_cache")}
        frontend_input_names = tuple(streaming["frontend_input_names"])
        frontend_output_names = tuple(streaming["frontend_output_names"])
        chunk_tokens = int(streaming["chunk_token_capacity"])
        stream_roles = {role: dict(role_config) for role, role_config in streaming["roles"].items()}
        for role, last_chunk in (("stream_flow_frontend", False), ("stream_flow_frontend_final", True)):
            frontend_inputs = (
                torch.zeros((1, chunk_tokens), dtype=torch.int32),
                torch.tensor([chunk_tokens], dtype=torch.int32),
                torch.zeros((1, 192), dtype=torch.float32),
                torch.zeros(cache_shapes["conformer_cnn_cache"], dtype=torch.float32),
                torch.zeros(cache_shapes["conformer_att_cache"], dtype=torch.float32),
                torch.tensor([int(streaming["base_cache_valid_length"])], dtype=torch.int32),
            )
            graph_path = (
                component_dir
                / "Stream"
                / f"{name_prefix}token2wav_flow_frontend_{role}_{target_device}_{quant_type}.onnx"
            )
            graphs[role] = _convert(
                FlowStreamingFrontendWrapper(flow, last_chunk=last_chunk),
                frontend_inputs,
                (frontend_input_names, frontend_output_names),
                graph_path,
                target_device,
                quant_type,
            )
        stream_contract.update(
            {
                "frontend_input_names": list(frontend_input_names),
                "frontend_output_names": list(frontend_output_names),
                "cache_shapes": {name: list(shape) for name, shape in all_cache_shapes.items()},
                "frontend_cache_shapes": {name: list(shape) for name, shape in cache_shapes.items()},
                "chunk_token_capacity": chunk_tokens,
                "base_conformer_layers": int(streaming["base_conformer_layers"]),
                "cache_alignment": str(streaming["cache_alignment"]),
                "roles": stream_roles,
            }
        )
    result = {
        "quant_type": quant_type,
        "graphs": graphs,
        "token_capacity": token_capacity,
        "mel_capacity": mel_capacity,
        "up_rate": int(flow.up_rate),
    }
    if stream_contract is not None:
        result["stream_contract"] = stream_contract
    return result


def export_flow_decoder(
    flow: nn.Module,
    component_dir: Path,
    config: Mapping[str, object],
    target_device: str,
    name_prefix: str = "",
) -> dict[str, object]:
    mel_capacity = int(config["mel_capacity"])
    quant_type = str(config["quant_type"])
    inputs = (
        torch.zeros((2, 80, mel_capacity)),
        torch.ones((2, 1, mel_capacity)),
        torch.zeros((2, 80, mel_capacity)),
        torch.zeros((2,)),
        torch.zeros((2, 80)),
        torch.zeros((2, 80, mel_capacity)),
    )
    graph = _convert(
        FlowDecoderWrapper(flow.decoder.estimator),
        inputs,
        (("x", "mask", "mu", "t", "spks", "cond"), ("estimator_out",)),
        component_dir / "Offline" / f"{name_prefix}token2wav_flow_decoder_offline_{target_device}_{quant_type}.onnx",
        target_device,
        quant_type,
    )
    rand_noise_seed = int(config.get("noise_seed", 1024))
    rand_noise_file = component_dir / "rand_noise.pt"
    rand_noise = create_flow_decoder_noise(flow.decoder.rand_noise, seed=rand_noise_seed)
    torch.save(rand_noise, rand_noise_file)
    rand_noise_sha256 = _file_sha256(rand_noise_file)
    graphs = {"main": graph}
    stream_contract = None
    streaming = config.get("streaming")
    if isinstance(streaming, Mapping):
        step_shapes = streaming["estimator_step_cache_shapes"]
        stream_inputs = (
            torch.zeros((2, 80, int(streaming["append_capacity"]))),
            torch.zeros((2, 80, int(streaming["append_capacity"]))),
            torch.zeros((2,)),
            torch.zeros((2, 80)),
            torch.zeros((2, 80, int(streaming["append_capacity"]))),
            torch.zeros(tuple(step_shapes["input_cnn"])),
            torch.zeros(tuple(step_shapes["input_att"])),
            torch.tensor([int(streaming["base_cache_valid_length"])], dtype=torch.int32),
            torch.tensor([int(streaming["append_capacity"])], dtype=torch.int32),
        )
        input_names = tuple(streaming["estimator_step_input_names"])
        output_names = tuple(streaming["estimator_step_output_names"])
        graph_path = (
            component_dir
            / "Stream"
            / f"{name_prefix}token2wav_flow_decoder_stream_flow_estimator_step_{target_device}_{quant_type}.onnx"
        )
        graphs["stream_flow_estimator_step"] = _convert(
            FlowEstimatorStepWrapper(flow.decoder.estimator),
            stream_inputs,
            (input_names, output_names),
            graph_path,
            target_device,
            quant_type,
        )
        host_timestep_cache_banks = int(streaming.get("host_timestep_cache_banks", 10))
        stream_contract = {
            "input_names": list(input_names),
            "output_names": list(output_names),
            "estimator_step_cache_shapes": dict(step_shapes),
            "append_capacity": int(streaming["append_capacity"]),
            "base_cache_valid_length": int(streaming["base_cache_valid_length"]),
            "host_timestep_cache_banks": host_timestep_cache_banks,
            "prompt_cache_policy": dict(streaming["prompt_cache_policy"]),
            "roles": {
                "stream_flow_estimator_step": {
                    "cfg_batch_size": int(step_shapes["input_att"][1]),
                    "host_timestep_cache_banks": host_timestep_cache_banks,
                }
            },
        }
    result = {
        "quant_type": quant_type,
        "graphs": graphs,
        "mel_capacity": mel_capacity,
        "n_timesteps": int(config.get("n_timesteps", 10)),
        "cfg_rate": float(flow.decoder.inference_cfg_rate),
        "rand_noise_file": rand_noise_file,
        "rand_noise_seed": rand_noise_seed,
        "rand_noise_sha256": rand_noise_sha256,
    }
    if stream_contract is not None:
        result["stream_contract"] = stream_contract
    return result


def export_hift(
    hift: nn.Module,
    component_dir: Path,
    config: Mapping[str, object],
    target_device: str,
    name_prefix: str = "",
) -> dict[str, object]:
    frame_capacity = int(config["frame_capacity"])
    quant_type = str(config["quant_type"])
    source_length = frame_capacity * 480 - 480
    phase_noise, source_noise = create_hift_random_inputs(
        harmonic_count=int(hift.nb_harmonics) + 1,
        source_length=source_length,
        seed=int(config.get("source_seed", 1024)),
    )
    inputs = (torch.zeros((1, 80, frame_capacity)), phase_noise, source_noise)
    output_file = component_dir / "Offline" / f"{name_prefix}token2wav_hift_offline_{target_device}_{quant_type}.onnx"
    onnx_dir = output_file.parent / "onnx"
    hmonnx_dir = output_file.parent / "hmonnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    hmonnx_dir.mkdir(parents=True, exist_ok=True)
    source_file = onnx_dir / f"{output_file.stem}_source.onnx"
    torch.onnx.export(
        DeterministicHiftForwardWrapper(hift),
        inputs,
        source_file,
        input_names=["speech_feat", "phase_noise", "source_noise"],
        output_names=["waveform"],
        opset_version=17,
        do_constant_folding=True,
    )
    prepared_file = onnx_dir / f"{output_file.stem}_prepared.onnx"
    prepared_model = prepare_hift_onnx(onnx.load(source_file), frame_capacity=frame_capacity)
    require_deterministic_onnx(prepared_model)
    onnx.save(prepared_model, prepared_file)
    onnx.checker.check_model(str(prepared_file))
    if target_device != "XH2a":
        raise RuntimeError(f"Unsupported Token2Wav target device: {target_device}")
    quant_config = create_quant_config(QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type))
    hmonnx_file = hmonnx_dir / output_file.name
    convert_onnx_to_hmonnx(
        str(prepared_file),
        [value.cpu() for value in inputs],
        DeviceType.XH2a,
        str(hmonnx_file),
        quant_config=quant_config,
        input_names=["speech_feat", "phase_noise", "source_noise"],
        output_names=["waveform"],
    )
    phase_noise_file = component_dir / "phase_noise.pt"
    source_noise_file = component_dir / "source_noise.pt"
    torch.save(phase_noise, phase_noise_file)
    torch.save(source_noise, source_noise_file)
    graphs = {"main": hmonnx_file}
    stream_contract = None
    streaming = config.get("streaming")
    if isinstance(streaming, Mapping):
        stream_frame_capacity = int(streaming["frame_capacity"])
        stream_source_length = (stream_frame_capacity + int(streaming["mel_cache_length"])) * 480
        stream_phase_noise, stream_source_noise = create_hift_random_inputs(
            harmonic_count=int(hift.nb_harmonics) + 1,
            source_length=stream_source_length,
            seed=int(config.get("source_seed", 1024)),
        )
        source_cache_length = int(streaming["source_cache_length"])
        stream_inputs = (
            torch.zeros((1, 80, stream_frame_capacity)),
            torch.tensor([stream_frame_capacity], dtype=torch.int32),
            torch.zeros((1, int(streaming["mel_cache_length"]), 80)).transpose(1, 2),
            torch.tensor([0], dtype=torch.int32),
            torch.zeros((1, 1, source_cache_length)),
            torch.tensor([source_cache_length], dtype=torch.int32),
            stream_phase_noise,
            stream_source_noise,
        )
        input_names = (
            "speech_feat",
            "speech_feat_valid_length",
            "past_mel",
            "past_mel_valid_length",
            "past_source",
            "past_source_valid_length",
            "phase_noise",
            "source_noise",
        )
        output_names = ("raw_waveform", "full_source")
        role_wrappers = {"stream_hift": HiftStreamWrapper, "stream_hift_final": HiftStreamFinalWrapper}
        for role, wrapper_type in role_wrappers.items():
            role_inputs = stream_inputs
            graphs[role] = _convert(
                wrapper_type(
                    hift,
                    source_cache_length=source_cache_length,
                    mel_cache_length=int(streaming["mel_cache_length"]),
                    waveform_length=stream_source_length,
                ),
                role_inputs,
                (input_names, output_names),
                component_dir / "Stream" / f"{name_prefix}token2wav_hift_{role}_{target_device}_{quant_type}.onnx",
                target_device,
                quant_type,
                prepare=prepare_hift_stream_onnx,
            )
        stream_phase_file = component_dir / "stream_phase_noise.pt"
        stream_source_file = component_dir / "stream_source_noise.pt"
        initial_source_file = component_dir / "initial_source_cache.pt"
        torch.save(stream_phase_noise, stream_phase_file)
        torch.save(stream_source_noise, stream_source_file)
        torch.save(stream_inputs[4], initial_source_file)
        stream_contract = {
            "input_names": list(input_names),
            "output_names": list(output_names),
            "host_state_owner": "official_token2wav.stream",
            "cache_inputs": {"mel": "past_mel", "source": "past_source"},
            "source_output": "full_source",
            "frame_capacity": stream_frame_capacity,
            "combined_frame_capacity": stream_frame_capacity + int(streaming["mel_cache_length"]),
            "mel_cache_length": int(streaming["mel_cache_length"]),
            "source_cache_length": source_cache_length,
            "speech_cache_length": int(streaming["speech_cache_length"]),
            "hop_length": 480,
            "cache_axes": {"mel": 2, "source": 2},
            "graph_dtype": "float16",
            "roles": {"stream_hift": {"last_chunk": False}, "stream_hift_final": {"last_chunk": True}},
            "cache_tails": {
                "mel": {"capacity": int(streaming["mel_cache_length"]), "axis": 2},
                "source": {"capacity": source_cache_length, "axis": 2},
            },
            "first_valid_lengths": {"mel": 0, "source": 0, "speech": 0},
            "host_overlap": {
                "owner": "official_token2wav.stream",
                "speech_cache_length": int(streaming["speech_cache_length"]),
                "overlap_window_length": int(streaming["speech_cache_length"]) * 2,
                "normal": "withhold one speech-cache tail after host overlap-add",
                "final": "emit full waveform after host overlap-add",
            },
            "phase_noise_file": stream_phase_file,
            "source_noise_file": stream_source_file,
            "initial_source_cache_file": initial_source_file,
        }
    result = {
        "quant_type": quant_type,
        "graphs": graphs,
        "frame_capacity": frame_capacity,
        "hop_length": 480,
        "phase_noise_file": phase_noise_file,
        "source_noise_file": source_noise_file,
        "source_length": source_length,
    }
    if stream_contract is not None:
        result["stream_contract"] = stream_contract
    return result


def export_minicpm_o_4_5_token2wav_flow_frontend(
    *,
    work_dir: Path,
    model_dir: str,
    component_cfg: Mapping[str, object],
    target_device: str,
    device: str,
    model_name: str | None = None,
    export_basename: str | None = None,
) -> dict[str, object]:
    del device
    flow, _ = load_token2wav_modules(model_dir)
    return export_flow_frontend(
        flow,
        work_dir / "Token2WavFlowFrontend",
        component_cfg,
        target_device,
        name_prefix=f"{model_name or 'minicpm_o_4_5'}_",
    )


def export_minicpm_o_4_5_token2wav_flow_decoder(
    *,
    work_dir: Path,
    model_dir: str,
    component_cfg: Mapping[str, object],
    target_device: str,
    device: str,
    model_name: str | None = None,
    export_basename: str | None = None,
) -> dict[str, object]:
    del device
    flow, _ = load_token2wav_modules(model_dir)
    return export_flow_decoder(
        flow,
        work_dir / "Token2WavFlowDecoder",
        component_cfg,
        target_device,
        name_prefix=f"{model_name or 'minicpm_o_4_5'}_",
    )


def export_minicpm_o_4_5_token2wav_hift(
    *,
    work_dir: Path,
    model_dir: str,
    component_cfg: Mapping[str, object],
    target_device: str,
    device: str,
    model_name: str | None = None,
    export_basename: str | None = None,
) -> dict[str, object]:
    del device
    _, hift = load_token2wav_modules(model_dir)
    return export_hift(
        hift,
        work_dir / "Token2WavHiFT",
        component_cfg,
        target_device,
        name_prefix=f"{model_name or 'minicpm_o_4_5'}_",
    )
