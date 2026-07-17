"""Convert the desensitized multimodal fusion model to a static HMONNX graph.

The original model contains an ``If`` whose condition checks whether the batch
dimension is one. XH2a does not execute ONNX control-flow nodes, so this script
fixes all input shapes, simplifies the graph, inlines the selected ``If``
branch, simplifies again, verifies the rewritten ONNX with ONNX Runtime, and
then exports HMONNX.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
import torch
from onnx import TensorProto, helper, numpy_helper

from xhquant.api import (
	DeviceType,
	HMONNXInference,
	QuantScheme,
	convert_onnx_to_hmonnx,
	create_quant_config,
	xhquant_init,
)


DEFAULT_ONNX = Path("work_dirs/multimodal_fusion_model_desensitized.onnx")


def _fixed_shapes(batch_size: int, sequence_length: int) -> dict[str, list[int]]:
	return {
		"images": [batch_size, 3, 384, 640],
		"traj_inputs": [batch_size, sequence_length, 23],
	}


def _session_feed(
	session: ort.InferenceSession, inputs: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
	return {input_info.name: inputs[input_info.name] for input_info in session.get_inputs()}


def freeze_sequence_lengths(
	model: onnx.ModelProto, batch_size: int, sequence_length: int
) -> onnx.ModelProto:
	rewritten = copy.deepcopy(model)
	rewritten.graph.initializer.append(
		numpy_helper.from_array(
			np.full((batch_size,), sequence_length, dtype=np.int64),
			name="seq_lengths",
		)
	)
	kept_inputs = [input_value for input_value in rewritten.graph.input if input_value.name != "seq_lengths"]
	del rewritten.graph.input[:]
	rewritten.graph.input.extend(kept_inputs)
	onnx.checker.check_model(rewritten)
	return rewritten


def _replace_graph_output_name(graph: onnx.GraphProto, old_name: str, new_name: str) -> None:
	for node in graph.node:
		for index, input_name in enumerate(node.input):
			if input_name == old_name:
				node.input[index] = new_name
	for output in graph.output:
		if output.name == old_name:
			output.name = new_name


def _inline_if_node(model: onnx.ModelProto, if_node: onnx.NodeProto, take_then: bool) -> None:
	branch_name = "then_branch" if take_then else "else_branch"
	branch_attribute = next(attribute for attribute in if_node.attribute if attribute.name == branch_name)
	branch = branch_attribute.g
	if len(branch.output) != len(if_node.output):
		raise ValueError(
			f"If node {if_node.name!r} has {len(if_node.output)} outputs but {branch_name} "
			f"has {len(branch.output)} outputs"
		)

	prefix = (if_node.name or "If").strip("/").replace("/", "_") + f"_{branch_name}"
	local_values = {initializer.name for initializer in branch.initializer}
	local_values.update(output for node in branch.node for output in node.output if output)
	value_mapping = {name: f"{prefix}_{name.lstrip('/').replace('/', '_')}" for name in local_values}

	for initializer in branch.initializer:
		cloned = copy.deepcopy(initializer)
		cloned.name = value_mapping[initializer.name]
		model.graph.initializer.append(cloned)

	cloned_nodes: list[onnx.NodeProto] = []
	for branch_node in branch.node:
		cloned = copy.deepcopy(branch_node)
		cloned.name = f"{prefix}_{(branch_node.name or branch_node.op_type).lstrip('/').replace('/', '_')}"
		for index, input_name in enumerate(cloned.input):
			if input_name in value_mapping:
				cloned.input[index] = value_mapping[input_name]
		for index, output_name in enumerate(cloned.output):
			if output_name in value_mapping:
				cloned.output[index] = value_mapping[output_name]
		cloned_nodes.append(cloned)

	for branch_output, if_output in zip(branch.output, if_node.output, strict=True):
		selected_output = value_mapping.get(branch_output.name, branch_output.name)
		cloned_nodes.append(
			helper.make_node(
				"Identity",
				[selected_output],
				[if_output],
				name=f"{prefix}_{if_output.lstrip('/').replace('/', '_')}_output",
			)
		)

	node_index = next(index for index, node in enumerate(model.graph.node) if node is if_node)
	del model.graph.node[node_index]
	for offset, cloned_node in enumerate(cloned_nodes):
		model.graph.node.insert(node_index + offset, cloned_node)


def eliminate_static_if(model: onnx.ModelProto, batch_size: int) -> onnx.ModelProto:
	"""Inline this model's batch-size branch after input shapes are fixed."""

	rewritten = copy.deepcopy(model)
	if_nodes = [node for node in rewritten.graph.node if node.op_type == "If"]
	for if_node in if_nodes:
		# The model condition is Shape(Transpose(...))[1] == 1. With fixed
		# inputs this is exactly the fixed batch-size predicate.
		_inline_if_node(rewritten, if_node, take_then=batch_size == 1)
	onnx.checker.check_model(rewritten)
	return rewritten


def materialize_expand_shapes(
	model: onnx.ModelProto, inputs: dict[str, np.ndarray]
) -> onnx.ModelProto:
	"""Evaluate and replace dynamic Expand shape inputs for fixed model inputs."""

	rewritten = copy.deepcopy(model)
	initializer_names = {initializer.name for initializer in rewritten.graph.initializer}
	expand_nodes = [
		node
		for node in rewritten.graph.node
		if node.op_type == "Expand" and node.input[1] not in initializer_names
	]
	if not expand_nodes:
		return rewritten

	probe = copy.deepcopy(rewritten)
	existing_outputs = {output.name for output in probe.graph.output}
	shape_names: list[str] = []
	for node in expand_nodes:
		shape_name = node.input[1]
		if shape_name not in existing_outputs:
			probe.graph.output.append(
				helper.make_tensor_value_info(shape_name, TensorProto.INT64, [None])
			)
			existing_outputs.add(shape_name)
		shape_names.append(shape_name)

	session = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
	values = session.run(shape_names, _session_feed(session, inputs))
	for index, (node, value) in enumerate(zip(expand_nodes, values, strict=True)):
		constant_name = f"{node.name or 'Expand'}_fixed_shape_{index}".replace("/", "_")
		rewritten.graph.initializer.append(
			numpy_helper.from_array(np.asarray(value, dtype=np.int64), name=constant_name)
		)
		node.input[1] = constant_name

	onnx.checker.check_model(rewritten)
	return rewritten


def materialize_shape_outputs(
	model: onnx.ModelProto, inputs: dict[str, np.ndarray]
) -> onnx.ModelProto:
	"""Replace every Shape result with an int64 initializer for fixed inputs."""

	rewritten = copy.deepcopy(model)
	shape_nodes = [node for node in rewritten.graph.node if node.op_type == "Shape"]
	if not shape_nodes:
		return rewritten
	shape_names = [node.output[0] for node in shape_nodes]
	probe = copy.deepcopy(rewritten)
	existing_outputs = {output.name for output in probe.graph.output}
	for tensor_name in shape_names:
		if tensor_name not in existing_outputs:
			probe.graph.output.append(
				helper.make_tensor_value_info(tensor_name, TensorProto.INT64, [None])
			)
			existing_outputs.add(tensor_name)
	session = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
	values = session.run(shape_names, _session_feed(session, inputs))

	shape_node_ids = {id(node) for node in shape_nodes}
	kept_nodes = [node for node in rewritten.graph.node if id(node) not in shape_node_ids]
	del rewritten.graph.node[:]
	rewritten.graph.node.extend(kept_nodes)
	for tensor_name, value in zip(shape_names, values, strict=True):
		rewritten.graph.initializer.append(
			numpy_helper.from_array(np.asarray(value, dtype=np.int64), name=tensor_name)
		)
	onnx.checker.check_model(rewritten)
	return rewritten


def materialize_resize_input_shapes(
	model: onnx.ModelProto, inputs: dict[str, np.ndarray]
) -> onnx.ModelProto:
	"""Populate concrete ValueInfo shapes required by the Resize parser."""

	rewritten = copy.deepcopy(model)
	resize_inputs = list(dict.fromkeys(node.input[0] for node in rewritten.graph.node if node.op_type == "Resize"))
	if not resize_inputs:
		return rewritten

	probe = copy.deepcopy(rewritten)
	existing_outputs = {output.name for output in probe.graph.output}
	for tensor_name in resize_inputs:
		if tensor_name not in existing_outputs:
			probe.graph.output.append(
				helper.make_tensor_value_info(tensor_name, TensorProto.FLOAT, None)
			)
			existing_outputs.add(tensor_name)
	session = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
	values = session.run(resize_inputs, _session_feed(session, inputs))

	value_infos = {value_info.name: value_info for value_info in rewritten.graph.value_info}
	for tensor_name, value in zip(resize_inputs, values, strict=True):
		replacement = helper.make_tensor_value_info(
			tensor_name,
			helper.np_dtype_to_tensor_dtype(value.dtype),
			list(value.shape),
		)
		if tensor_name in value_infos:
			value_infos[tensor_name].CopyFrom(replacement)
		else:
			rewritten.graph.value_info.append(replacement)
	return rewritten


def materialize_static_operator_parameters(
	model: onnx.ModelProto, inputs: dict[str, np.ndarray]
) -> onnx.ModelProto:
	"""Replace fixed Slice parameter tensors with initializers."""

	rewritten = copy.deepcopy(model)
	initializer_names = {initializer.name for initializer in rewritten.graph.initializer}
	parameter_inputs: list[tuple[onnx.NodeProto, int, str]] = []
	for node in rewritten.graph.node:
		if node.op_type != "Slice":
			continue
		parameter_indices = tuple(range(1, len(node.input)))
		for input_index in parameter_indices:
			input_name = node.input[input_index]
			if input_name and input_name not in initializer_names:
				parameter_inputs.append((node, input_index, input_name))
	if not parameter_inputs:
		return rewritten

	tensor_names = list(dict.fromkeys(input_name for _, _, input_name in parameter_inputs))
	probe = copy.deepcopy(rewritten)
	existing_outputs = {output.name for output in probe.graph.output}
	for tensor_name in tensor_names:
		if tensor_name not in existing_outputs:
			probe.graph.output.append(
				helper.make_tensor_value_info(tensor_name, TensorProto.INT64, None)
			)
			existing_outputs.add(tensor_name)
	session = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
	values = dict(
		zip(tensor_names, session.run(tensor_names, _session_feed(session, inputs)), strict=True)
	)

	constant_names: dict[str, str] = {}
	for parameter_index, (node, input_index, input_name) in enumerate(parameter_inputs):
		if input_name not in constant_names:
			constant_name = (
				f"{node.name or node.op_type}_fixed_parameter_{parameter_index}".replace("/", "_")
			)
			rewritten.graph.initializer.append(
				numpy_helper.from_array(np.asarray(values[input_name]), name=constant_name)
			)
			constant_names[input_name] = constant_name
		node.input[input_index] = constant_names[input_name]
	onnx.checker.check_model(rewritten)
	return rewritten


def simplify_static_model(
	source: Path,
	output: Path,
	batch_size: int,
	sequence_length: int,
	inputs: dict[str, np.ndarray],
) -> onnx.ModelProto:
	shapes = _fixed_shapes(batch_size, sequence_length)
	source_model = freeze_sequence_lengths(onnx.load(source), batch_size, sequence_length)
	simplified, success = onnxsim.simplify(
		source_model,
		overwrite_input_shapes=shapes,
		check_n=1,
		include_subgraph=True,
		skip_fuse_bn=True,
	)
	if not success:
		raise RuntimeError("onnxsim validation failed before If elimination")

	simplified = eliminate_static_if(simplified, batch_size)
	simplified, success = onnxsim.simplify(
		simplified,
		overwrite_input_shapes=shapes,
		check_n=1,
		include_subgraph=True,
		skip_fuse_bn=True,
	)
	if not success:
		raise RuntimeError("onnxsim validation failed after If elimination")
	if any(node.op_type == "If" for node in simplified.graph.node):
		raise RuntimeError("If elimination did not remove all control-flow nodes")

	simplified = materialize_shape_outputs(simplified, inputs)
	simplified, success = onnxsim.simplify(
		simplified,
		overwrite_input_shapes=shapes,
		check_n=1,
		include_subgraph=True,
		skip_fuse_bn=True,
	)
	if not success:
		raise RuntimeError("onnxsim validation failed after Shape materialization")
	simplified = materialize_expand_shapes(simplified, inputs)
	simplified = materialize_resize_input_shapes(simplified, inputs)
	simplified = materialize_static_operator_parameters(simplified, inputs)
	initializer_names = {initializer.name for initializer in simplified.graph.initializer}
	dynamic_expands = [
		node.name
		for node in simplified.graph.node
		if node.op_type == "Expand" and node.input[1] not in initializer_names
	]
	if dynamic_expands:
		raise RuntimeError(f"Dynamic Expand shape inputs remain: {dynamic_expands}")

	output.parent.mkdir(parents=True, exist_ok=True)
	onnx.save(simplified, output)
	onnx.checker.check_model(str(output))
	return simplified


def make_inputs(batch_size: int, sequence_length: int, seed: int = 20260717) -> dict[str, np.ndarray]:
	rng = np.random.default_rng(seed)
	return {
		"images": rng.normal(0, 1, (batch_size, 3, 384, 640)).astype(np.float32),
		"traj_inputs": rng.normal(0, 1, (batch_size, sequence_length, 23)).astype(np.float32),
		"seq_lengths": np.full((batch_size,), sequence_length, dtype=np.int64),
	}


def verify_onnx(original: Path, rewritten: Path, inputs: dict[str, np.ndarray]) -> None:
	original_session = ort.InferenceSession(str(original), providers=["CPUExecutionProvider"])
	rewritten_session = ort.InferenceSession(str(rewritten), providers=["CPUExecutionProvider"])
	expected = original_session.run(None, _session_feed(original_session, inputs))
	actual = rewritten_session.run(None, _session_feed(rewritten_session, inputs))
	for index, (expected_value, actual_value) in enumerate(zip(expected, actual, strict=True)):
		np.testing.assert_allclose(actual_value, expected_value, rtol=1e-5, atol=1e-6)
		cosine = float(
			np.sum(expected_value * actual_value)
			/ (np.linalg.norm(expected_value) * np.linalg.norm(actual_value))
		)
		print(f"ONNX output {index}: max_abs={np.max(np.abs(expected_value - actual_value)):.8g}, cosine={cosine:.9f}")


def export_hmonnx(
	static_onnx: Path,
	output: Path,
	inputs: dict[str, np.ndarray],
	quant_type: str,
) -> None:
	xhquant_init(None, debug=False)
	quant_config = create_quant_config(
		QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
	)
	input_names = [input_value.name for input_value in onnx.load(static_onnx).graph.input]
	input_tensors = [torch.from_numpy(inputs[name]) for name in input_names]
	output.parent.mkdir(parents=True, exist_ok=True)
	convert_onnx_to_hmonnx(
		str(static_onnx),
		input_tensors,
		DeviceType.XH2a,
		str(output),
		quant_config=quant_config,
		input_names=input_names,
		simplify=False,
	)
	try:
		onnx.checker.check_model(str(output))
	except onnx.checker.ValidationError as error:
		message = str(error)
		if "ai.houmo.xh2a::TopK" not in message or "input size 1" not in message:
			raise
		print(f"Ignore known XH2a TopK schema mismatch; runtime uses attribute k: {error}")


def verify_hmonnx(static_onnx: Path, hmonnx: Path, inputs: dict[str, np.ndarray]) -> None:
	ort_session = ort.InferenceSession(str(static_onnx), providers=["CPUExecutionProvider"])
	expected = ort_session.run(None, _session_feed(ort_session, inputs))
	session = HMONNXInference(str(hmonnx)).to("cuda" if torch.cuda.is_available() else "cpu")
	tensors = [
		torch.from_numpy(inputs[input_info.name]).to(dtype=input_info.dtype, device=session.device)
		for input_info in session.inputs
	]
	with torch.no_grad():
		actual = session(*tensors)
	actual_values = actual if isinstance(actual, tuple) else (actual,)
	for index, (expected_value, actual_value) in enumerate(zip(expected, actual_values, strict=True)):
		actual_array = actual_value.float().cpu().numpy()
		cosine = float(
			np.sum(expected_value * actual_array)
			/ (np.linalg.norm(expected_value) * np.linalg.norm(actual_array))
		)
		print(
			f"HMONNX output {index}: max_abs={np.max(np.abs(expected_value - actual_array)):.8g}, "
			f"cosine={cosine:.9f}"
		)
		if not np.isfinite(cosine) or cosine < 0.999:
			raise AssertionError(f"HMONNX output {index} cosine similarity is too low: {cosine:.9f}")


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
	parser.add_argument("--batch-size", type=int, default=1)
	parser.add_argument("--sequence-length", type=int, default=10)
	parser.add_argument("--quant-type", default="w8a8_sefp")
	parser.add_argument("--skip-hmonnx", action="store_true")
	parser.add_argument("--skip-hmonnx-verify", action="store_true")
	args = parser.parse_args()

	source = args.onnx.resolve()
	work_dir = Path("work_dirs") / source.stem
	static_onnx = work_dir / f"{source.stem}_b{args.batch_size}_s{args.sequence_length}_no_if.onnx"
	hmonnx = work_dir / "hmonnx" / f"{static_onnx.stem}_{args.quant_type}_XH2a.onnx"
	inputs = make_inputs(args.batch_size, args.sequence_length)

	model = simplify_static_model(
		source,
		static_onnx,
		args.batch_size,
		args.sequence_length,
		inputs,
	)
	print(f"Static ONNX saved to {static_onnx}; nodes={len(model.graph.node)}")
	verify_onnx(source, static_onnx, inputs)

	if not args.skip_hmonnx:
		export_hmonnx(static_onnx, hmonnx, inputs, args.quant_type)
		print(f"HMONNX saved to {hmonnx}")
		if not args.skip_hmonnx_verify:
			verify_hmonnx(static_onnx, hmonnx, inputs)


if __name__ == "__main__":
	main()
