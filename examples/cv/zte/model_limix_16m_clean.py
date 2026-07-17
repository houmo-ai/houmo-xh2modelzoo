"""Optimize limix_16m_clean and export it as a fixed-shape HMONNX model.

The source graph uses ``IsNaN`` to replace missing numeric features. XH2a does
not support that operator. This script preserves IEEE NaN semantics by replacing
``IsNaN(x)`` with ``Equal(x, x)`` and reversing every consuming ``Where`` branch:
``Where(IsNaN(x), a, b) == Where(Equal(x, x), b, a)``.
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


DEFAULT_ONNX = Path("work_dirs/limix_16m_clean.onnx")
FIXED_SHAPES = {
	"x_preprocessed": [1, 128, 1, 2],
	"y": [1, 128],
}


def _session_feed(
	session: ort.InferenceSession, inputs: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
	return {input_info.name: inputs[input_info.name] for input_info in session.get_inputs()}


def eliminate_isnan(model: onnx.ModelProto) -> onnx.ModelProto:
	"""Replace IsNaN patterns with Equal self-comparison and reversed Where branches."""

	rewritten = copy.deepcopy(model)
	consumers: dict[str, list[onnx.NodeProto]] = {}
	for node in rewritten.graph.node:
		for input_name in node.input:
			consumers.setdefault(input_name, []).append(node)

	replacements: dict[int, onnx.NodeProto] = {}
	removed_node_ids: set[int] = set()
	isnan_nodes = [node for node in rewritten.graph.node if node.op_type == "IsNaN"]
	for isnan_node in isnan_nodes:
		if len(isnan_node.input) != 1 or len(isnan_node.output) != 1:
			raise ValueError(f"Unexpected IsNaN signature at {isnan_node.name!r}")
		isnan_output = isnan_node.output[0]
		not_nan_output = f"{isnan_output}_not_nan"
		replacements[id(isnan_node)] = helper.make_node(
			"Equal",
			[isnan_node.input[0], isnan_node.input[0]],
			[not_nan_output],
			name=f"{isnan_node.name or 'IsNaN'}_as_not_nan",
		)

		for consumer in consumers.get(isnan_output, []):
			where_node = consumer
			if consumer.op_type == "Cast":
				cast_consumers = consumers.get(consumer.output[0], [])
				if len(cast_consumers) != 1 or cast_consumers[0].op_type != "Where":
					raise ValueError(
					f"Unsupported IsNaN Cast consumers at {consumer.name!r}: "
					f"{[node.op_type for node in cast_consumers]}"
				)
				where_node = cast_consumers[0]
				removed_node_ids.add(id(consumer))
			elif consumer.op_type != "Where":
				raise ValueError(
					f"Unsupported consumer {consumer.op_type!r} of IsNaN node {isnan_node.name!r}"
				)

			if where_node.input[0] not in {isnan_output, consumer.output[0]}:
				raise ValueError(f"IsNaN result is not the condition of {where_node.name!r}")
			where_node.input[0] = not_nan_output
			where_node.input[1], where_node.input[2] = where_node.input[2], where_node.input[1]

	new_nodes: list[onnx.NodeProto] = []
	for node in rewritten.graph.node:
		if id(node) in replacements:
			new_nodes.append(replacements[id(node)])
		elif id(node) not in removed_node_ids:
			new_nodes.append(node)
	del rewritten.graph.node[:]
	rewritten.graph.node.extend(new_nodes)

	if any(node.op_type == "IsNaN" for node in rewritten.graph.node):
		raise RuntimeError("IsNaN elimination did not remove every IsNaN node")
	onnx.checker.check_model(rewritten)
	return rewritten


def materialize_shape_outputs(
	model: onnx.ModelProto, inputs: dict[str, np.ndarray]
) -> onnx.ModelProto:
	"""Replace Shape results with fixed int64 initializers."""

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


def optimize_onnx(
	source: Path, output: Path, inputs: dict[str, np.ndarray]
) -> onnx.ModelProto:
	model = eliminate_isnan(onnx.load(source))
	simplified, success = onnxsim.simplify(
		model,
		overwrite_input_shapes=FIXED_SHAPES,
		check_n=1,
		skip_fuse_bn=True,
	)
	if not success:
		raise RuntimeError("onnxsim validation failed after IsNaN elimination")

	simplified = materialize_shape_outputs(simplified, inputs)
	simplified, success = onnxsim.simplify(
		simplified,
		overwrite_input_shapes=FIXED_SHAPES,
		check_n=1,
		skip_fuse_bn=True,
	)
	if not success:
		raise RuntimeError("onnxsim validation failed after Shape materialization")
	if any(node.op_type in {"IsNaN", "Shape"} for node in simplified.graph.node):
		raise RuntimeError("Static optimization left IsNaN or Shape nodes in the graph")

	output.parent.mkdir(parents=True, exist_ok=True)
	onnx.save(simplified, output)
	onnx.checker.check_model(str(output))
	return simplified


def make_inputs(seed: int = 20260717, include_nan: bool = False) -> dict[str, np.ndarray]:
	rng = np.random.default_rng(seed)
	inputs = {
		"x_preprocessed": rng.normal(size=FIXED_SHAPES["x_preprocessed"]).astype(np.float32),
		"y": rng.normal(size=FIXED_SHAPES["y"]).astype(np.float32),
	}
	if include_nan:
		inputs["x_preprocessed"].reshape(-1)[::31] = np.nan
	return inputs


def _cosine(expected: np.ndarray, actual: np.ndarray) -> float:
	expected_flat = expected.astype(np.float64).reshape(-1)
	actual_flat = actual.astype(np.float64).reshape(-1)
	return float(
		np.dot(expected_flat, actual_flat)
		/ (np.linalg.norm(expected_flat) * np.linalg.norm(actual_flat))
	)


def verify_onnx(original: Path, optimized: Path) -> None:
	original_session = ort.InferenceSession(str(original), providers=["CPUExecutionProvider"])
	optimized_session = ort.InferenceSession(str(optimized), providers=["CPUExecutionProvider"])
	for case_name, include_nan in (("finite", False), ("with_nan", True)):
		inputs = make_inputs(include_nan=include_nan)
		expected = original_session.run(None, _session_feed(original_session, inputs))
		actual = optimized_session.run(None, _session_feed(optimized_session, inputs))
		for index, (expected_value, actual_value) in enumerate(zip(expected, actual, strict=True)):
			np.testing.assert_allclose(actual_value, expected_value, rtol=1e-5, atol=1e-6)
			print(
				f"ONNX {case_name} output {index}: "
				f"max_abs={np.max(np.abs(expected_value - actual_value)):.8g}, "
				f"cosine={_cosine(expected_value, actual_value):.9f}"
			)


def export_hmonnx(
	optimized_onnx: Path,
	output: Path,
	inputs: dict[str, np.ndarray],
	quant_type: str,
) -> None:
	xhquant_init(None, debug=False)
	quant_config = create_quant_config(
		QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
	)
	input_names = [input_value.name for input_value in onnx.load(optimized_onnx).graph.input]
	input_tensors = [torch.from_numpy(inputs[name]) for name in input_names]
	output.parent.mkdir(parents=True, exist_ok=True)
	convert_onnx_to_hmonnx(
		str(optimized_onnx),
		input_tensors,
		DeviceType.XH2a,
		str(output),
		quant_config=quant_config,
		input_names=input_names,
		simplify=False,
	)
	onnx.checker.check_model(str(output))


def verify_hmonnx(
	optimized_onnx: Path, hmonnx: Path, inputs: dict[str, np.ndarray]
) -> None:
	ort_session = ort.InferenceSession(str(optimized_onnx), providers=["CPUExecutionProvider"])
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
		cosine = _cosine(expected_value, actual_array)
		print(
			f"HMONNX output {index}: max_abs={np.max(np.abs(expected_value - actual_array)):.8g}, "
			f"cosine={cosine:.9f}"
		)
		if not np.isfinite(cosine) or cosine < 0.999:
			raise AssertionError(f"HMONNX output {index} cosine similarity is too low: {cosine:.9f}")


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
	parser.add_argument("--quant-type", default="w16a16_sefp")
	parser.add_argument("--skip-hmonnx", action="store_true")
	parser.add_argument("--skip-hmonnx-verify", action="store_true")
	args = parser.parse_args()

	source = args.onnx.resolve()
	work_dir = Path("work_dirs") / source.stem
	optimized_onnx = work_dir / f"{source.stem}_fixed_no_isnan.onnx"
	hmonnx = work_dir / "hmonnx" / f"{optimized_onnx.stem}_{args.quant_type}_XH2a.onnx"
	finite_inputs = make_inputs()

	model = optimize_onnx(source, optimized_onnx, finite_inputs)
	print(f"Optimized ONNX saved to {optimized_onnx}; nodes={len(model.graph.node)}")
	verify_onnx(source, optimized_onnx)

	if not args.skip_hmonnx:
		export_hmonnx(optimized_onnx, hmonnx, finite_inputs, args.quant_type)
		print(f"HMONNX saved to {hmonnx}")
		if not args.skip_hmonnx_verify:
			verify_hmonnx(optimized_onnx, hmonnx, finite_inputs)


if __name__ == "__main__":
	main()
