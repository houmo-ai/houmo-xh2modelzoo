"""Deterministic finalization for production HMONNX artifacts.

Torch ONNX debug metadata and placeholder-local names are useful while an
individual graph is being exported, but they make equivalent monolithic and
low-memory exports serialize differently.  This module removes that
path-dependent information, gives executable and quant-metadata graphs stable
topological names, and streams external tensors into the same deterministic
order without loading their payloads into memory.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import heapq
import json
import os
import re
import stat
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import onnx


_GRAPH_META_KEY = "graph_meta"
_QUANT_META_KEY = "quanted_info_onnx"
_CANONICAL_VERSION_KEY = "hmonnx.deterministic_export.version"
_CANONICAL_VERSION = "4"


def _metadata_properties(model: onnx.ModelProto) -> dict[str, str]:
    return {item.key: item.value for item in model.metadata_props}


def _upsert_metadata_property(model: onnx.ModelProto, key: str, value: str) -> None:
    matches = [item for item in model.metadata_props if item.key == key]
    if len(matches) > 1:
        raise RuntimeError(f"Duplicate model metadata key {key!r}: {len(matches)} entries.")
    if matches:
        matches[0].value = value
    else:
        item = model.metadata_props.add()
        item.key = key
        item.value = value


def _decode_quant_metadata_model(
    model: onnx.ModelProto,
) -> tuple[dict[str, Any], onnx.ModelProto] | None:
    graph_meta = _metadata_properties(model).get(_GRAPH_META_KEY)
    if graph_meta is None:
        return None
    graph_meta_dict = json.loads(graph_meta)
    encoded_model = graph_meta_dict.get("quant_meta", {}).get(_QUANT_META_KEY)
    if not encoded_model:
        return None
    quant_model = onnx.ModelProto()
    quant_model.ParseFromString(base64.b64decode(encoded_model, validate=True))
    return graph_meta_dict, quant_model


def _encode_quant_metadata_model(
    model: onnx.ModelProto,
    graph_meta_dict: dict[str, Any],
    quant_model: onnx.ModelProto,
) -> None:
    graph_meta_dict["quant_meta"][_QUANT_META_KEY] = base64.b64encode(
        quant_model.SerializeToString(deterministic=True)
    ).decode("ascii")
    serialized = json.dumps(graph_meta_dict, indent=2, sort_keys=True) + "\n"
    matches = [item for item in model.metadata_props if item.key == _GRAPH_META_KEY]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {_GRAPH_META_KEY!r} metadata entry, got {len(matches)}."
        )
    matches[0].value = serialized


def _normalized_module_target(module_target: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", module_target).strip("_")


def merge_placeholder_quant_metadata(
    main_model: onnx.ModelProto,
    subgraph_model: onnx.ModelProto,
    module_target: str,
    output_names_by_index: dict[int, str],
) -> int:
    """Inline one placeholder's standalone quant metadata into the main model.

    Quant metadata is a base64-encoded ONNX graph stored in ``graph_meta``.
    The main low-memory graph contains one ``QPlaceHolderModule`` entry, while
    the standalone graph contains input/output stubs around the actual module.
    Removing those boundary stubs and rewiring positionally reproduces the
    monolithic quant graph without relying on model-family-specific operators.
    """

    main_quant_metadata = _decode_quant_metadata_model(main_model)
    subgraph_quant_metadata = _decode_quant_metadata_model(subgraph_model)
    if main_quant_metadata is None and subgraph_quant_metadata is None:
        return 0
    if main_quant_metadata is None or subgraph_quant_metadata is None:
        raise RuntimeError(
            "Main and placeholder HMONNX must either both carry quant metadata or both omit it."
        )

    graph_meta_dict, main_quant_model = main_quant_metadata
    _, subgraph_quant_model = subgraph_quant_metadata
    placeholder_name = _normalized_module_target(module_target)
    candidates = [
        node
        for node in main_quant_model.graph.node
        if node.op_type.endswith("QPlaceHolderModule") and node.name == placeholder_name
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one quant-metadata placeholder for {module_target!r} "
            f"({placeholder_name!r}), got {len(candidates)}."
        )
    placeholder = candidates[0]
    subgraph = subgraph_quant_model.graph
    if len(subgraph.input) != len(placeholder.input):
        raise RuntimeError(
            f"Quant-metadata placeholder input count mismatch for {module_target!r}: "
            f"{len(subgraph.input)} != {len(placeholder.input)}"
        )
    if len(subgraph.output) != len(placeholder.output):
        raise RuntimeError(
            f"Quant-metadata placeholder output count mismatch for {module_target!r}: "
            f"{len(subgraph.output)} != {len(placeholder.output)}"
        )
    unexpected_outputs = sorted(set(output_names_by_index) - set(range(len(subgraph.output))))
    if unexpected_outputs:
        raise RuntimeError(
            f"Quant-metadata output indices are invalid for {module_target!r}: {unexpected_outputs}"
        )

    producers = {name: node for node in subgraph.node for name in node.output if name}
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in subgraph.node:
        for name in node.input:
            if name:
                consumers[name].append(node)

    aliases: dict[str, str] = {}
    skipped_node_names: set[str] = set()
    for graph_input, placeholder_input in zip(
        subgraph.input,
        placeholder.input,
        strict=True,
    ):
        input_consumers = consumers.get(graph_input.name, [])
        if (
            len(input_consumers) == 1
            and input_consumers[0].op_type.endswith("InputStub")
            and len(input_consumers[0].output) == 1
        ):
            input_stub = input_consumers[0]
            aliases[input_stub.output[0]] = placeholder_input
            skipped_node_names.add(input_stub.name)
        else:
            aliases[graph_input.name] = placeholder_input

    for output_index, graph_output in enumerate(subgraph.output):
        # The executable HMONNX graph and its embedded quant-metadata graph
        # deliberately use different tensor namespaces.  In particular, an FX
        # TupleGetItem name from the executable graph is not necessarily the
        # value consumed after QPlaceHolderModule in QuantedInfo.  Rewiring the
        # quant graph to ``output_names_by_index`` therefore leaves its real
        # downstream value as an unbound capture.  The QPlaceHolderModule output
        # is the authoritative boundary inside this graph.
        output_name = placeholder.output[output_index]
        if not output_name:
            raise RuntimeError(
                f"Quant-metadata placeholder output {output_index} is empty for "
                f"{module_target!r}."
            )
        output_stub = producers.get(graph_output.name)
        if (
            output_stub is not None
            and output_stub.op_type.endswith("Output")
            and len(output_stub.input) == 1
        ):
            aliases[output_stub.input[0]] = output_name
            skipped_node_names.add(output_stub.name)
        else:
            aliases[graph_output.name] = output_name

    def scoped_name(name: str) -> str:
        if not name:
            return name
        if name == "module":
            return placeholder.name
        if name.startswith("module_"):
            return f"{placeholder.name}{name[len('module'):]}"
        return f"{placeholder.name}_{name}"

    def rename_tensor(name: str) -> str:
        return aliases.get(name, scoped_name(name))

    inlined_nodes = []
    for source_node in subgraph.node:
        if source_node.name in skipped_node_names:
            continue
        node = copy.deepcopy(source_node)
        # A quant node and its output often share the same FX name.  Boundary
        # aliases rewire tensors only; applying them to node.name would rename
        # the final module op to a TupleGetItem/output tensor and break its
        # semantic identity across monolithic and placeholder exports.
        node.name = scoped_name(node.name)
        for index, name in enumerate(node.input):
            node.input[index] = rename_tensor(name)
        for index, name in enumerate(node.output):
            node.output[index] = rename_tensor(name)
        inlined_nodes.append(node)

    merged_nodes = []
    inserted = False
    for node in main_quant_model.graph.node:
        if node.name == placeholder.name and node.op_type == placeholder.op_type:
            merged_nodes.extend(inlined_nodes)
            inserted = True
        else:
            merged_nodes.append(node)
    if not inserted:
        raise RuntimeError(f"Quant-metadata placeholder disappeared for {module_target!r}.")
    del main_quant_model.graph.node[:]
    main_quant_model.graph.node.extend(merged_nodes)
    _encode_quant_metadata_model(main_model, graph_meta_dict, main_quant_model)
    return len(inlined_nodes)


def _external_data_entries(tensor: onnx.TensorProto) -> dict[str, str]:
    return {item.key: item.value for item in tensor.external_data}


def _set_external_data_entries(
    tensor: onnx.TensorProto,
    *,
    location: str,
    offset: int,
    length: int,
) -> None:
    del tensor.external_data[:]
    for key, value in (
        ("location", location),
        ("offset", str(offset)),
        ("length", str(length)),
    ):
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = value
    tensor.data_location = onnx.TensorProto.EXTERNAL


def _clear_value_debug_metadata(value_info: onnx.ValueInfoProto) -> None:
    del value_info.metadata_props[:]
    value_info.doc_string = ""


def _clear_tensor_debug_metadata(tensor: onnx.TensorProto) -> None:
    del tensor.metadata_props[:]
    tensor.doc_string = ""


def _canonical_node_names(
    graph: onnx.GraphProto,
    _initializers_by_name: dict[str, onnx.TensorProto],
) -> list[str]:
    """Build names from canonical topology, never exporter-local FX paths.

    Low-memory export inlines independently exported subgraphs.  Their original
    node names carry a temporary module path and their local exporter counters
    restart at zero, whereas monolithic export uses global counters.  Inferring
    names from either source therefore makes equivalent HMONNX files differ.

    Placeholder inlining is required to preserve the executable node sequence.
    A topological position plus operator type is consequently both readable and
    invariant across regular/low-memory export.  Prefill and decode are separate
    artifacts and intentionally do not share this positional namespace.
    """

    return [
        f"node_{node_index:06d}_{_normalized_module_target(node.op_type).lower() or 'unknown'}"
        for node_index, node in enumerate(graph.node)
    ]


def _structure_hash(value: Any) -> bytes:
    """Hash a tuple made only from deterministic scalar/bytes containers."""

    return hashlib.sha256(repr(value).encode("utf-8")).digest()


def _quant_attribute_signature(attribute: onnx.AttributeProto) -> bytes:
    """Return an attribute signature without exporter-only tensor names."""

    value = copy.deepcopy(attribute)
    value.doc_string = ""
    if value.type == onnx.AttributeProto.TENSOR:
        value.t.name = ""
        _clear_tensor_debug_metadata(value.t)
    elif value.type == onnx.AttributeProto.TENSORS:
        for tensor in value.tensors:
            tensor.name = ""
            _clear_tensor_debug_metadata(tensor)
    return value.SerializeToString(deterministic=True)


def _quant_node_static_signature(node: onnx.NodeProto) -> bytes:
    """Describe a quant-metadata op without names or graph position."""

    return _structure_hash(
        (
            node.domain,
            node.op_type,
            getattr(node, "overload", ""),
            len(node.input),
            len(node.output),
            tuple(
                sorted(
                    (attribute.name, _quant_attribute_signature(attribute))
                    for attribute in node.attribute
                )
            ),
        )
    )


def _quant_initializer_signature(initializer: onnx.TensorProto) -> bytes:
    value = copy.deepcopy(initializer)
    value.name = ""
    _clear_tensor_debug_metadata(value)
    return value.SerializeToString(deterministic=True)


def _canonicalize_quant_metadata_graph(graph: onnx.GraphProto) -> None:
    """Canonicalize a QuantedInfo DAG independently of its export path.

    A monolithic FX export and a placeholder-by-placeholder export can emit
    independent nodes in different valid topological orders.  Standalone
    placeholders also give captured constants local FX names.  A lexical sort
    of those unstable names is not sufficient: it changes tensor numbering and
    can hide whether one capture is shared by several consumers.

    The forward signature below identifies every node by its producers.  The
    backward signature identifies it by all consumers and their input ports.
    Captures use their backward signature as their source identity, preserving
    sharing while discarding the path-dependent spelling.  The resulting key
    gives a stable topological order; truly symmetric ties serialize identically
    after captures and internal tensors are numbered by first use/position.
    """

    nodes = list(graph.node)
    if not nodes:
        _canonicalize_graph(graph)
        return

    producer: dict[str, tuple[int, int]] = {}
    for node_index, node in enumerate(nodes):
        for output_index, name in enumerate(node.output):
            if not name:
                continue
            if name in producer:
                raise RuntimeError(
                    f"Quant-metadata graph has duplicate producer for {name!r}."
                )
            producer[name] = (node_index, output_index)

    graph_inputs = {value.name: index for index, value in enumerate(graph.input) if value.name}
    graph_input_info = {value.name: value for value in graph.input if value.name}
    initializers = {value.name: value for value in graph.initializer if value.name}
    values = set(graph_inputs) | set(initializers) | set(producer)
    values.update(name for node in nodes for name in node.input if name)
    values.update(value.name for value in graph.output if value.name)

    consumers: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for node_index, node in enumerate(nodes):
        for input_index, name in enumerate(node.input):
            if name:
                consumers[name].append((node_index, input_index))
    graph_output_positions: dict[str, list[int]] = defaultdict(list)
    for output_index, value in enumerate(graph.output):
        if value.name:
            graph_output_positions[value.name].append(output_index)

    children: dict[int, list[int]] = defaultdict(list)
    indegree = []
    for node_index, node in enumerate(nodes):
        node_dependencies = {
            producer[name][0] for name in node.input if name in producer
        }
        if node_index in node_dependencies:
            raise RuntimeError(
                f"Quant-metadata node {node.name or node_index!r} consumes its own output."
            )
        indegree.append(len(node_dependencies))
        for dependency in node_dependencies:
            children[dependency].append(node_index)

    # Any topological order is sufficient for signature propagation.  The
    # original index is used here only as a scheduler, never as an identity.
    ready_indices = [index for index, count in enumerate(indegree) if count == 0]
    heapq.heapify(ready_indices)
    propagation_order = []
    propagation_indegree = list(indegree)
    while ready_indices:
        node_index = heapq.heappop(ready_indices)
        propagation_order.append(node_index)
        for child in children[node_index]:
            propagation_indegree[child] -= 1
            if propagation_indegree[child] == 0:
                heapq.heappush(ready_indices, child)
    if len(propagation_order) != len(nodes):
        raise RuntimeError("Quant-metadata graph is cyclic and cannot be canonicalized.")

    static_signatures = [_quant_node_static_signature(node) for node in nodes]
    node_backward: list[bytes | None] = [None] * len(nodes)
    value_backward: dict[str, bytes] = {}

    def backward_value_signature(name: str) -> bytes:
        consumer_signatures = []
        for consumer_index, input_index in consumers.get(name, []):
            signature = node_backward[consumer_index]
            if signature is None:
                raise RuntimeError(
                    f"Quant-metadata backward ordering failed at value {name!r}."
                )
            consumer_signatures.append((input_index, signature))
        return _structure_hash(
            (
                "backward_value",
                tuple(graph_output_positions.get(name, [])),
                tuple(sorted(consumer_signatures)),
            )
        )

    for node_index in reversed(propagation_order):
        node = nodes[node_index]
        output_signatures = []
        for output_index, name in enumerate(node.output):
            if name:
                value_backward[name] = backward_value_signature(name)
                output_signatures.append((output_index, value_backward[name]))
            else:
                output_signatures.append(
                    (output_index, _structure_hash(("empty_output", output_index)))
                )
        node_backward[node_index] = _structure_hash(
            (
                "backward_node",
                static_signatures[node_index],
                tuple(output_signatures),
            )
        )

    for name in values - set(producer):
        value_backward[name] = backward_value_signature(name)

    def graph_input_signature(name: str) -> bytes:
        value = copy.deepcopy(graph_input_info[name])
        value.name = ""
        _clear_value_debug_metadata(value)
        return value.SerializeToString(deterministic=True)

    value_forward: dict[str, bytes] = {}
    for name in values - set(producer):
        if name in graph_inputs:
            source = (
                "graph_input",
                graph_inputs[name],
                graph_input_signature(name),
            )
        elif name in initializers:
            source = ("initializer", _quant_initializer_signature(initializers[name]))
        else:
            # A capture's spelling is unstable across standalone FX exports;
            # its complete consumer structure and sharing pattern are stable.
            source = ("capture", value_backward[name])
        value_forward[name] = _structure_hash(source)

    node_forward: list[bytes | None] = [None] * len(nodes)
    for node_index in propagation_order:
        node = nodes[node_index]
        input_signatures = []
        for input_index, name in enumerate(node.input):
            if name:
                try:
                    signature = value_forward[name]
                except KeyError as exc:
                    raise RuntimeError(
                        f"Quant-metadata forward ordering failed at value {name!r}."
                    ) from exc
            else:
                signature = _structure_hash(("empty_input", input_index))
            input_signatures.append((input_index, signature))
        signature = _structure_hash(
            (
                "forward_node",
                static_signatures[node_index],
                tuple(input_signatures),
            )
        )
        node_forward[node_index] = signature
        for output_index, name in enumerate(node.output):
            if name:
                value_forward[name] = _structure_hash(
                    ("forward_value", output_index, signature)
                )

    node_signatures = [
        _structure_hash(("node", node_forward[index], node_backward[index]))
        for index in range(len(nodes))
    ]

    def ordering_key(node_index: int) -> tuple[Any, ...]:
        node = nodes[node_index]
        return (
            node_signatures[node_index],
            static_signatures[node_index],
            tuple(value_forward.get(name, b"") for name in node.input),
            tuple(value_backward.get(name, b"") for name in node.output),
            # This final field is observed only for structurally symmetric
            # nodes. Their canonicalized serialized forms are interchangeable.
            node_index,
        )

    canonical_indegree = list(indegree)
    ready_nodes = [
        ordering_key(index)
        for index, count in enumerate(canonical_indegree)
        if count == 0
    ]
    heapq.heapify(ready_nodes)
    canonical_order = []
    while ready_nodes:
        *_, node_index = heapq.heappop(ready_nodes)
        canonical_order.append(node_index)
        for child in children[node_index]:
            canonical_indegree[child] -= 1
            if canonical_indegree[child] == 0:
                heapq.heappush(ready_nodes, ordering_key(child))
    if len(canonical_order) != len(nodes):
        raise RuntimeError("Quant-metadata canonical ordering lost graph nodes.")

    del graph.node[:]
    graph.node.extend(copy.deepcopy(nodes[index]) for index in canonical_order)

    local_names = {value.name for value in graph.input if value.name}
    local_names.update(value.name for value in graph.output if value.name)
    local_names.update(value.name for value in graph.initializer if value.name)
    local_names.update(name for node in graph.node for name in node.output if name)
    capture_aliases: dict[str, str] = {}
    for node in graph.node:
        # Quant metadata is descriptive rather than executable.  Positional
        # names make equivalent symmetric nodes byte-identical too.
        node.name = "node"
        for input_index, name in enumerate(node.input):
            if not name or name in local_names:
                continue
            alias = capture_aliases.setdefault(
                name,
                f"capture_{len(capture_aliases)}",
            )
            node.input[input_index] = alias

    _canonicalize_graph(graph)


def _canonicalize_graph(
    graph: onnx.GraphProto,
) -> list[tuple[onnx.TensorProto, str, int, int]]:
    """Canonicalize names/order and return external source slices.

    Graph input/output names are part of the deployment contract and remain
    unchanged.  Internal tensor names follow topological position; node names
    additionally retain a deterministic layer/component scope so prefill and
    decode placement maps cannot falsely match unrelated operators.
    """

    graph.doc_string = ""
    # Torch's serialized ExportedProgram signature contains original FX names
    # and placeholder structure.  It is tracing/debug information, not part of
    # the ONNX execution contract, and is inherently mode-dependent here.
    del graph.metadata_props[:]

    nested_captures: set[str] = set()
    for node in graph.node:
        for attribute in node.attribute:
            nested_graphs = []
            if attribute.type == onnx.AttributeProto.GRAPH:
                nested_graphs.append(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                nested_graphs.extend(attribute.graphs)
            for nested_graph in nested_graphs:
                local_names = {item.name for item in nested_graph.input}
                local_names.update(name for item in nested_graph.node for name in item.output)
                nested_captures.update(
                    name
                    for item in nested_graph.node
                    for name in item.input
                    if name and name not in local_names
                )

    referenced_names = {name for node in graph.node for name in node.input if name}
    referenced_names.update(value.name for value in graph.output if value.name)
    referenced_names.update(value.name for value in graph.input if value.name)
    referenced_names.update(nested_captures)
    for annotation in graph.quantization_annotation:
        if annotation.tensor_name:
            referenced_names.add(annotation.tensor_name)
        referenced_names.update(
            value for value in annotation.quant_parameter_tensor_names.values() if value
        )

    retained_initializers = [
        initializer
        for initializer in graph.initializer
        if initializer.name in referenced_names
    ]
    del graph.initializer[:]
    graph.initializer.extend(retained_initializers)
    initializers_by_name = {initializer.name: initializer for initializer in graph.initializer}
    canonical_node_names = _canonical_node_names(graph, initializers_by_name)

    initializer_order: list[str] = []
    for node in graph.node:
        for name in node.input:
            if name in initializers_by_name and name not in initializer_order:
                initializer_order.append(name)
    for name in sorted(referenced_names):
        if name in initializers_by_name and name not in initializer_order:
            initializer_order.append(name)

    aliases = {value.name: value.name for value in graph.input if value.name}
    aliases.update({value.name: value.name for value in graph.output if value.name})
    aliases.update(
        {name: f"initializer_{index}" for index, name in enumerate(initializer_order)}
    )
    graph_output_names = {value.name for value in graph.output}
    for node_index, node in enumerate(graph.node):
        for output_index, name in enumerate(node.output):
            if not name:
                continue
            aliases.setdefault(
                name,
                name if name in graph_output_names else f"tensor_{node_index}_{output_index}",
            )

    def rewrite_nested_captures(nested_graph: onnx.GraphProto) -> None:
        local_names = {item.name for item in nested_graph.input}
        local_names.update(name for item in nested_graph.node for name in item.output)
        for nested_node in nested_graph.node:
            for input_index, name in enumerate(nested_node.input):
                if name and name not in local_names:
                    nested_node.input[input_index] = aliases.get(name, name)
            for attribute in nested_node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    rewrite_nested_captures(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for child_graph in attribute.graphs:
                        rewrite_nested_captures(child_graph)

    for node_index, node in enumerate(graph.node):
        node.name = canonical_node_names[node_index]
        node.doc_string = ""
        del node.metadata_props[:]
        for input_index, name in enumerate(node.input):
            node.input[input_index] = aliases.get(name, name)
        for output_index, name in enumerate(node.output):
            node.output[output_index] = aliases.get(name, name)
        ordered_attributes = sorted(
            (copy.deepcopy(attribute) for attribute in node.attribute),
            key=lambda attribute: attribute.name,
        )
        del node.attribute[:]
        node.attribute.extend(ordered_attributes)
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                rewrite_nested_captures(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for nested_graph in attribute.graphs:
                    rewrite_nested_captures(nested_graph)

    canonical_initializers = []
    external_sources: dict[str, tuple[str, int, int]] = {}
    for original_name in initializer_order:
        initializer = copy.deepcopy(initializers_by_name[original_name])
        entries = _external_data_entries(initializer)
        if entries:
            location = entries.get("location")
            if not location:
                raise RuntimeError(f"External initializer {original_name!r} has no location.")
            source_offset = int(entries.get("offset", 0))
            source_length = int(entries["length"]) if "length" in entries else -1
            if source_offset < 0 or source_length < -1:
                raise RuntimeError(
                    f"Invalid external range for {original_name!r}: "
                    f"offset={source_offset}, length={source_length}."
                )
            external_sources[original_name] = (location, source_offset, source_length)
        initializer.name = aliases[original_name]
        _clear_tensor_debug_metadata(initializer)
        canonical_initializers.append(initializer)
    del graph.initializer[:]
    graph.initializer.extend(canonical_initializers)

    valid_value_names = set(aliases)
    canonical_value_info: dict[str, onnx.ValueInfoProto] = {}
    for source_value in graph.value_info:
        if source_value.name not in valid_value_names:
            continue
        value = copy.deepcopy(source_value)
        value.name = aliases[source_value.name]
        _clear_value_debug_metadata(value)
        existing = canonical_value_info.get(value.name)
        if existing is not None and existing.type != value.type:
            raise RuntimeError(f"Conflicting value_info entries for {value.name!r}.")
        canonical_value_info[value.name] = value
    del graph.value_info[:]
    graph.value_info.extend(
        canonical_value_info[name] for name in sorted(canonical_value_info)
    )

    for value in graph.input:
        _clear_value_debug_metadata(value)
    for value in graph.output:
        _clear_value_debug_metadata(value)
    for annotation in graph.quantization_annotation:
        annotation.tensor_name = aliases.get(annotation.tensor_name, annotation.tensor_name)
        for item in annotation.quant_parameter_tensor_names:
            item.value = aliases.get(item.value, item.value)

    external_plan = []
    for original_name, initializer in zip(
        initializer_order,
        graph.initializer,
        strict=True,
    ):
        source = external_sources.get(original_name)
        if source is not None:
            external_plan.append((initializer, *source))
    return external_plan


def _canonicalize_model(
    model: onnx.ModelProto,
) -> list[tuple[onnx.TensorProto, str, int, int]]:
    quant_metadata = _decode_quant_metadata_model(model)
    if quant_metadata is not None:
        graph_meta_dict, quant_model = quant_metadata
        _canonicalize_quant_metadata_graph(quant_model.graph)
        _encode_quant_metadata_model(model, graph_meta_dict, quant_model)

    external_plan = _canonicalize_graph(model.graph)
    model.doc_string = ""
    ordered_opsets = sorted(
        (copy.deepcopy(item) for item in model.opset_import),
        key=lambda item: (item.domain, item.version),
    )
    del model.opset_import[:]
    model.opset_import.extend(ordered_opsets)
    _upsert_metadata_property(model, _CANONICAL_VERSION_KEY, _CANONICAL_VERSION)
    ordered_metadata = sorted(
        (copy.deepcopy(item) for item in model.metadata_props),
        key=lambda item: item.key,
    )
    del model.metadata_props[:]
    model.metadata_props.extend(ordered_metadata)
    return external_plan


def _safe_external_path(model_path: Path, location: str) -> Path:
    model_dir = model_path.parent.resolve()
    path = (model_dir / location).resolve()
    try:
        path.relative_to(model_dir)
    except ValueError as exc:
        raise RuntimeError(f"External data location escapes model directory: {location!r}") from exc
    return path


def _copy_file_slice(
    source_path: Path,
    target,
    offset: int,
    length: int,
) -> None:
    file_size = source_path.stat().st_size
    if offset < 0 or length < 0 or offset + length > file_size:
        raise RuntimeError(
            f"External data range exceeds {source_path}: "
            f"offset={offset}, length={length}, file_size={file_size}."
        )
    with source_path.open("rb") as source:
        source.seek(offset)
        remaining = length
        while remaining:
            chunk = source.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError(f"Unexpected end of external data file: {source_path}")
            target.write(chunk)
            remaining -= len(chunk)


def canonicalize_hmonnx_artifact(
    hmonnx_file: str | Path,
    logger=None,
) -> dict[str, Any]:
    """Transactionally make an HMONNX model and its external data reproducible."""

    model_path = Path(hmonnx_file).resolve()
    model_mode = stat.S_IMODE(model_path.stat().st_mode)
    model = onnx.load(str(model_path), load_external_data=False)
    if _metadata_properties(model).get(_CANONICAL_VERSION_KEY) == _CANONICAL_VERSION:
        onnx.checker.check_model(str(model_path))
        locations = {
            entries["location"]
            for initializer in model.graph.initializer
            if (entries := _external_data_entries(initializer)).get("location")
        }
        external_bytes = sum(
            _safe_external_path(model_path, location).stat().st_size
            for location in locations
        )
        return {
            "model": str(model_path),
            "model_bytes": model_path.stat().st_size,
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "external_location": next(iter(locations)) if len(locations) == 1 else None,
            "external_bytes": external_bytes,
            "nodes": len(model.graph.node),
            "initializers": len(model.graph.initializer),
            "value_info": len(model.graph.value_info),
            "already_canonical": True,
        }
    external_plan = _canonicalize_model(model)

    external_location = None
    external_target = None
    external_tmp = None
    total_external_bytes = 0
    try:
        if external_plan:
            locations = {location for _, location, _, _ in external_plan}
            # The regular exporter and the low-memory merger each produce one main
            # external file.  Keeping its location preserves the deployment layout.
            if len(locations) != 1:
                raise RuntimeError(
                    "Deterministic HMONNX finalization requires one external data "
                    f"location, got {sorted(locations)}."
                )
            external_location = next(iter(locations))
            external_target = _safe_external_path(model_path, external_location)
            if not external_target.exists():
                raise FileNotFoundError(external_target)
            external_mode = stat.S_IMODE(external_target.stat().st_mode)
            external_target.parent.mkdir(parents=True, exist_ok=True)
            resolved_plan = []
            expected_offset = 0
            external_data_already_ordered = True
            for initializer, source_location, source_offset, source_length in external_plan:
                source_path = _safe_external_path(model_path, source_location)
                if source_length < 0:
                    source_length = source_path.stat().st_size - source_offset
                if source_length < 0:
                    raise RuntimeError(
                        f"External data offset exceeds {source_path}: {source_offset}."
                    )
                resolved_plan.append(
                    (initializer, source_path, source_offset, source_length)
                )
                external_data_already_ordered &= (
                    source_path == external_target
                    and source_offset == expected_offset
                )
                _set_external_data_entries(
                    initializer,
                    location=external_location,
                    offset=expected_offset,
                    length=source_length,
                )
                expected_offset += source_length

            total_external_bytes = expected_offset
            external_data_already_ordered &= (
                total_external_bytes == external_target.stat().st_size
            )
            if not external_data_already_ordered:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f".{external_target.name}.",
                    suffix=".tmp",
                    dir=external_target.parent,
                    delete=False,
                ) as target:
                    external_tmp = Path(target.name)
                    for _, source_path, source_offset, source_length in resolved_plan:
                        _copy_file_slice(
                            source_path,
                            target,
                            source_offset,
                            source_length,
                        )
                    target.flush()
                    os.fsync(target.fileno())
                os.chmod(external_tmp, external_mode)
    except BaseException:
        if external_tmp is not None:
            external_tmp.unlink(missing_ok=True)
        raise

    serialized_model = model.SerializeToString(deterministic=True)
    model_tmp = model_path.with_name(f".{model_path.name}.{os.getpid()}.canonical.tmp")
    model_tmp.write_bytes(serialized_model)
    os.chmod(model_tmp, model_mode)
    # Path-based checking resolves external-data locations relative to the
    # model directory.  Checking the in-memory proto would incorrectly resolve
    # them against the process working directory.
    validation_tmp = model_path.with_name(
        f".{model_path.name}.{os.getpid()}.canonical.validation.tmp"
    )
    try:
        if external_tmp is None:
            onnx.checker.check_model(str(model_tmp))
        else:
            validation_model = copy.deepcopy(model)
            validation_location = external_tmp.name
            for initializer in validation_model.graph.initializer:
                entries = _external_data_entries(initializer)
                if entries.get("location") != external_location:
                    continue
                _set_external_data_entries(
                    initializer,
                    location=validation_location,
                    offset=int(entries["offset"]),
                    length=int(entries["length"]),
                )
            validation_tmp.write_bytes(
                validation_model.SerializeToString(deterministic=True)
            )
            onnx.checker.check_model(str(validation_tmp))
    except BaseException:
        model_tmp.unlink(missing_ok=True)
        validation_tmp.unlink(missing_ok=True)
        if external_tmp is not None:
            external_tmp.unlink(missing_ok=True)
        raise
    validation_tmp.unlink(missing_ok=True)

    model_backup = model_path.with_name(f".{model_path.name}.{os.getpid()}.canonical.bak")
    external_backup = (
        external_target.with_name(f".{external_target.name}.{os.getpid()}.canonical.bak")
        if external_target is not None
        else None
    )
    try:
        if external_target is not None and external_tmp is not None:
            os.replace(external_target, external_backup)
            os.replace(external_tmp, external_target)
        os.replace(model_path, model_backup)
        os.replace(model_tmp, model_path)
        onnx.checker.check_model(str(model_path))
    except BaseException:
        if model_backup.exists():
            os.replace(model_backup, model_path)
        if external_backup is not None and external_backup.exists():
            if external_target.exists():
                external_target.unlink()
            os.replace(external_backup, external_target)
        raise
    finally:
        model_tmp.unlink(missing_ok=True)
        if external_tmp is not None:
            external_tmp.unlink(missing_ok=True)
    model_backup.unlink(missing_ok=True)
    if external_backup is not None:
        external_backup.unlink(missing_ok=True)

    stats = {
        "model": str(model_path),
        "model_bytes": model_path.stat().st_size,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "external_location": external_location,
        "external_bytes": total_external_bytes,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "value_info": len(model.graph.value_info),
    }
    if logger is not None:
        logger.info("Canonicalized deterministic HMONNX artifact: %s", stats)
    return stats


__all__ = [
    "canonicalize_hmonnx_artifact",
    "merge_placeholder_quant_metadata",
]
