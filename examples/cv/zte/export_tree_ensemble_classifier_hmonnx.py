#!/usr/bin/env python3
"""Export and verify an ONNX-ML TreeEnsembleClassifier as XH2a HMONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, numpy_helper

from xhquant.api import (
    DeviceType,
    HMONNXInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    xhquant_init,
)
from xhquant.frontend.onnx.ml.tree_ensemble_classifier import compile_tree_ensemble_classifier


def _zipmap_to_array(values, labels: list[int]) -> np.ndarray:
    return np.asarray([[row[label] for label in labels] for row in values], dtype=np.float32)


def _restore_fp32_tree_comparison(source_onnx: Path, hmonnx_path: Path, output_path: Path) -> int:
    source_model = onnx.load(source_onnx)
    classifier = next(
        node
        for node in source_model.graph.node
        if node.domain == "ai.onnx.ml" and node.op_type == "TreeEnsembleClassifier"
    )
    thresholds = compile_tree_ensemble_classifier(classifier).thresholds.astype(np.float32)

    hmonnx_model = onnx.load(hmonnx_path)
    replaced = 0
    threshold_names: set[str] = set()
    for index, initializer in enumerate(hmonnx_model.graph.initializer):
        if "thresholds" not in initializer.name.lower() or list(initializer.dims) != [thresholds.size]:
            continue
        hmonnx_model.graph.initializer[index].CopyFrom(
            numpy_helper.from_array(thresholds, name=initializer.name)
        )
        threshold_names.add(initializer.name)
        replaced += 1
    if replaced == 0:
        raise RuntimeError("No lowered tree threshold initializer was found in HMONNX")

    value_infos = [
        *hmonnx_model.graph.input,
        *hmonnx_model.graph.output,
        *hmonnx_model.graph.value_info,
    ]
    value_info_by_name = {value_info.name: value_info for value_info in value_infos}
    graph_input_names = {value_info.name for value_info in hmonnx_model.graph.input}
    for input_name in graph_input_names:
        value_info_by_name[input_name].type.tensor_type.elem_type = TensorProto.FLOAT
    for threshold_name in threshold_names:
        if threshold_name in value_info_by_name:
            value_info_by_name[threshold_name].type.tensor_type.elem_type = TensorProto.FLOAT
    for node in hmonnx_model.graph.node:
        is_feature_gather = node.op_type == "GatherElements" and node.input[0] in graph_input_names
        is_threshold_gather = node.op_type == "Gather" and node.input[0] in threshold_names
        if not is_feature_gather and not is_threshold_gather:
            continue
        for output_name in node.output:
            if output_name in value_info_by_name:
                value_info_by_name[output_name].type.tensor_type.elem_type = TensorProto.FLOAT

    external_data_name = f"{output_path.stem}_external_data"
    onnx.save_model(
        hmonnx_model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_name,
        size_threshold=1024,
        convert_attribute=False,
    )
    onnx.checker.check_model(str(output_path))
    return replaced


def _verify(source_onnx: Path, hmonnx_path: Path, feature_count: int) -> None:
    ort_session = ort.InferenceSession(str(source_onnx), providers=["CPUExecutionProvider"])
    hmonnx_session = HMONNXInference(str(hmonnx_path)).to("cuda" if torch.cuda.is_available() else "cpu")
    input_info = hmonnx_session.inputs[0]
    labels = list(range(hmonnx_session.outputs[1].shape[-1]))
    rng = np.random.default_rng(20260717)
    cases = {
        "zeros": np.zeros((1, feature_count), dtype=np.float32),
        "normal": rng.normal(0, 1, (1, feature_count)).astype(np.float32),
        "wide": rng.normal(0, 1000, (1, feature_count)).astype(np.float32),
        "large": rng.uniform(-1e5, 1e5, (1, feature_count)).astype(np.float32),
    }

    for name, array in cases.items():
        expected_label, expected_maps = ort_session.run(None, {ort_session.get_inputs()[0].name: array})
        expected_scores = _zipmap_to_array(expected_maps, labels)
        tensor = torch.from_numpy(array).to(dtype=input_info.dtype, device=hmonnx_session.device)
        with torch.no_grad():
            actual_label, actual_scores = hmonnx_session(tensor)
        actual_label = actual_label.cpu().numpy()
        actual_scores = actual_scores.float().cpu().numpy()
        cosine = float(
            np.sum(expected_scores * actual_scores)
            / (np.linalg.norm(expected_scores) * np.linalg.norm(actual_scores))
        )
        max_abs = float(np.max(np.abs(expected_scores - actual_scores)))
        labels_equal = np.array_equal(expected_label, actual_label)
        print(f"{name}: labels_equal={labels_equal}, max_abs={max_abs:.8f}, cosine={cosine:.9f}")
        if not labels_equal or cosine < 0.9999:
            raise AssertionError(f"HMONNX verification failed for {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, default=Path("work_dirs/model 2.onnx"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("work_dirs/model_2_hmonnx/model_2_XH2a_threshold_fp32.onnx"),
    )
    parser.add_argument("--quant-type", default="w8a8_sefp")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    source_onnx = args.onnx.resolve()
    output_path = args.output.resolve()
    temporary_path = output_path.with_name(f"{output_path.stem}_temporary.onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(source_onnx, load_external_data=False)
    input_shape = model.graph.input[0].type.tensor_type.shape
    feature_count = input_shape.dim[1].dim_value
    if feature_count <= 0:
        raise ValueError("TreeEnsembleClassifier export requires a static feature dimension")

    xhquant_init(None, debug=False)
    quant_config = create_quant_config(
        QuantScheme(target_device=DeviceType.XH2a, quant_type=args.quant_type)
    )
    convert_onnx_to_hmonnx(
        str(source_onnx),
        [torch.zeros((1, feature_count), dtype=torch.float32)],
        DeviceType.XH2a,
        str(temporary_path),
        quant_config=quant_config,
        input_names=[model.graph.input[0].name],
        output_names=[output.name for output in model.graph.output],
        simplify=False,
    )
    replaced = _restore_fp32_tree_comparison(source_onnx, temporary_path, output_path)
    print(f"Restored {replaced} FP32 threshold tensors")
    print(f"HMONNX saved to: {output_path}")

    if not args.skip_verify:
        _verify(source_onnx, output_path, feature_count)


if __name__ == "__main__":
    main()
