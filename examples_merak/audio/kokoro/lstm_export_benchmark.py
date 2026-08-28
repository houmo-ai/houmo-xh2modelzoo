from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper


@dataclass(frozen=True)
class LSTMPreset:
    name: str
    forward_node: str
    sequence_length: int
    input_size: int
    hidden_size: int
    direction: str


PRESETS = (
    LSTMPreset(
        name="text_t32_c512_forward_full",
        forward_node="/text_lstm/forward_lstm/LSTM",
        sequence_length=32,
        input_size=512,
        hidden_size=256,
        direction="forward",
    ),
    LSTMPreset(
        name="duration_t32_c640_forward_full",
        forward_node="/duration_predictor/forward_lstm/LSTM",
        sequence_length=32,
        input_size=640,
        hidden_size=256,
        direction="forward",
    ),
    LSTMPreset(
        name="shared_f120_c640_forward_full",
        forward_node="/shared_lstm/forward_lstm/LSTM",
        sequence_length=120,
        input_size=640,
        hidden_size=256,
        direction="forward",
    ),
)


def _node_by_name(model: onnx.ModelProto, name: str) -> onnx.NodeProto:
    for node in model.graph.node:
        if node.name == name:
            return node
    raise KeyError(f"LSTM node not found in acoustic ONNX: {name}")


def _initializer_map(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}


def _extract_direction_weights(
    node: onnx.NodeProto,
    initializers: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = tuple(np.asarray(initializers[node.input[index]], dtype=np.float32) for index in (1, 2, 3))
    if any(value.shape[0] != 1 for value in weights):
        raise ValueError(f"expected a single-direction source node: {node.name}")
    return weights


def _preset_weights(
    source: onnx.ModelProto,
    initializers: dict[str, np.ndarray],
    preset: LSTMPreset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = _extract_direction_weights(_node_by_name(source, preset.forward_node), initializers)
    combined = tuple(value.copy() for value in forward)
    expected_directions = 2 if preset.direction == "bidirectional" else 1
    if combined[0].shape != (
        expected_directions,
        4 * preset.hidden_size,
        preset.input_size,
    ):
        raise ValueError(f"unexpected W shape for {preset.name}: {combined[0].shape}")
    return combined


def _make_source_model(
    preset: LSTMPreset,
    weights: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> onnx.ModelProto:
    w, r, bias = weights
    num_directions = w.shape[0]
    initial_shape = [num_directions, 1, preset.hidden_size]
    inputs = [
        helper.make_tensor_value_info(
            "X",
            TensorProto.FLOAT,
            [preset.sequence_length, 1, preset.input_size],
        )
    ]
    node = helper.make_node(
        "LSTM",
        ["X", "W", "R", "B", "", "initial_h", "initial_c"],
        ["Y", "Y_h", "Y_c"],
        name=preset.name,
        hidden_size=preset.hidden_size,
        direction=preset.direction,
    )
    outputs = [
        helper.make_tensor_value_info(
            "Y",
            TensorProto.FLOAT,
            [preset.sequence_length, num_directions, 1, preset.hidden_size],
        ),
        helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, initial_shape),
        helper.make_tensor_value_info("Y_c", TensorProto.FLOAT, initial_shape),
    ]
    zero_state = np.zeros(initial_shape, dtype=np.float32)
    graph = helper.make_graph(
        [node],
        preset.name,
        inputs,
        outputs,
        [
            numpy_helper.from_array(w, "W"),
            numpy_helper.from_array(r, "R"),
            numpy_helper.from_array(bias, "B"),
            numpy_helper.from_array(zero_state, "initial_h"),
            numpy_helper.from_array(zero_state, "initial_c"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    onnx.checker.check_model(model)
    return model


def _make_feed(preset: LSTMPreset, seed: int) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(seed)
    feed = {
        "X": generator.standard_normal((preset.sequence_length, 1, preset.input_size), dtype=np.float32)
        * np.float32(0.25)
    }
    return feed


def _reference_outputs(
    preset: LSTMPreset,
    weights: tuple[np.ndarray, np.ndarray, np.ndarray],
    feed: dict[str, np.ndarray],
    *,
    dtype: torch.dtype,
) -> tuple[np.ndarray, ...]:
    from xhquant.ops.xh.lstm import onnx_lstm_reference

    w, r, bias = (torch.from_numpy(value).to(dtype=dtype) for value in weights)
    num_directions = w.shape[0]
    zero_state = torch.zeros((num_directions, 1, preset.hidden_size), dtype=dtype)
    outputs = onnx_lstm_reference(
        torch.from_numpy(feed["X"]).to(dtype=dtype),
        w,
        r,
        bias,
        zero_state,
        zero_state,
        hidden_size=preset.hidden_size,
        direction=preset.direction,
    )
    return tuple(value.float().cpu().numpy() for value in outputs)


def _metric(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    expected = np.asarray(reference, dtype=np.float64).reshape(-1)
    observed = np.asarray(actual, dtype=np.float64).reshape(-1)
    difference = observed - expected
    denominator = np.linalg.norm(expected) * np.linalg.norm(observed)
    cosine = float(np.dot(expected, observed) / denominator) if denominator else 1.0
    return {
        "cosine": cosine,
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
    }


def _output_metrics(reference: tuple[np.ndarray, ...], actual: tuple[np.ndarray, ...]) -> dict[str, dict[str, float]]:
    return {
        name: _metric(expected, observed)
        for name, expected, observed in zip(("Y", "Y_h", "Y_c"), reference, actual, strict=True)
    }


def _node_summary(path: Path) -> dict[str, Any]:
    model = onnx.load(path, load_external_data=False)
    histogram = Counter(f"{node.domain or 'onnx'}::{node.op_type}" for node in model.graph.node)
    related_files = [path, *sorted(path.parent.glob(f"{path.stem}*external_data*"))]
    return {
        "nodes": len(model.graph.node),
        "lstm_nodes": sum(node.op_type == "LSTM" for node in model.graph.node),
        "bytes": sum(value.stat().st_size for value in related_files),
        "top_ops": dict(histogram.most_common(12)),
    }


def _as_outputs(value: Any) -> tuple[np.ndarray, ...]:
    values = value if isinstance(value, (tuple, list)) else (value,)
    return tuple(item.detach().float().cpu().numpy() for item in values)


def _convert_and_run(
    *,
    source_path: Path,
    output_path: Path,
    feed: dict[str, np.ndarray],
    decompose_lstm: bool,
    quant_type: str,
    device: torch.device,
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    from xhquant.api import (
        DeviceType,
        HMONNXInference,
        QuantScheme,
        convert_onnx_to_hmonnx,
        create_quant_config,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_names = list(feed)
    input_values = [torch.from_numpy(feed[name]) for name in input_names]
    quant_config = create_quant_config(QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type))
    started = time.perf_counter()
    convert_onnx_to_hmonnx(
        str(source_path),
        input_values,
        DeviceType.XH2a,
        str(output_path),
        quant_config=quant_config,
        input_names=input_names,
        output_names=["Y", "Y_h", "Y_c"],
        simplify=False,
        decompose_lstm=decompose_lstm,
    )
    conversion_seconds = time.perf_counter() - started

    runner = HMONNXInference(str(output_path)).to(device)
    runtime_inputs = []
    for name in input_names:
        tensor = torch.from_numpy(feed[name])
        if tensor.is_floating_point():
            tensor = tensor.half()
        else:
            tensor = tensor.to(dtype=torch.int32)
        runtime_inputs.append(tensor.to(device))
    started = time.perf_counter()
    outputs = _as_outputs(runner(*runtime_inputs))
    inference_seconds = time.perf_counter() - started
    return outputs, {
        **_node_summary(output_path),
        "conversion_seconds": conversion_seconds,
        "inference_seconds": inference_seconds,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export and compare Kokoro-sized decomposed/native LSTM HMONNX presets"
    )
    parser.add_argument(
        "--acoustic-onnx",
        default=("work_dirs/kokoro_merak/precision_split_t32_f120/onnx/kokoro_acoustic_b1_t32_f120.onnx"),
    )
    parser.add_argument(
        "--output-dir",
        default="work_dirs/kokoro_merak/lstm_export_matrix_t32_f120",
    )
    parser.add_argument("--quant-type", default="w16a16_sefp")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--preset",
        action="append",
        choices=[value.name for value in PRESETS],
        help="Run only the selected preset; repeat to select more than one",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    acoustic_path = Path(args.acoustic_onnx).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = onnx.load(acoustic_path)
    initializers = _initializer_map(source)
    selected = [preset for preset in PRESETS if not args.preset or preset.name in args.preset]
    device = torch.device(args.device)
    report_path = output_dir / "lstm_precision_report.json"
    report: dict[str, Any]
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {"presets": {}}
    report.update(
        {
            "source_acoustic_onnx": str(acoustic_path),
            "quant_type": args.quant_type,
            "device": str(device),
            "seed": args.seed,
        }
    )
    report.setdefault("presets", {})

    for index, preset in enumerate(selected):
        print(f"[{index + 1}/{len(selected)}] preparing {preset.name}", flush=True)
        weights = _preset_weights(source, initializers, preset)
        feed = _make_feed(preset, args.seed + PRESETS.index(preset))
        source_model = _make_source_model(preset, weights)
        source_path = output_dir / "source" / f"{preset.name}.onnx"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(source_model, source_path)
        input_path = output_dir / "inputs" / f"{preset.name}.npz"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(input_path, **feed)

        reference_fp32 = _reference_outputs(preset, weights, feed, dtype=torch.float32)
        reference_fp16 = _reference_outputs(preset, weights, feed, dtype=torch.float16)
        variants: dict[str, Any] = {}
        runtime_outputs: dict[str, tuple[np.ndarray, ...]] = {}
        for variant, decompose_lstm in (("native", False), ("decomposed", True)):
            print(f"  exporting {variant}", flush=True)
            hmonnx_path = output_dir / variant / f"{preset.name}_{args.quant_type}.onnx"
            outputs, graph_info = _convert_and_run(
                source_path=source_path,
                output_path=hmonnx_path,
                feed=feed,
                decompose_lstm=decompose_lstm,
                quant_type=args.quant_type,
                device=device,
            )
            runtime_outputs[variant] = outputs
            variants[variant] = {
                "hmonnx": str(hmonnx_path),
                **graph_info,
                "vs_fp32": _output_metrics(reference_fp32, outputs),
                "vs_fp16_reference": _output_metrics(reference_fp16, outputs),
            }

        variants["decomposed"]["vs_native"] = _output_metrics(runtime_outputs["native"], runtime_outputs["decomposed"])
        report["presets"][preset.name] = {
            "shape": {
                "X": [preset.sequence_length, 1, preset.input_size],
                "hidden_size": preset.hidden_size,
                "Y": [
                    preset.sequence_length,
                    2 if preset.direction == "bidirectional" else 1,
                    1,
                    preset.hidden_size,
                ],
            },
            "direction": preset.direction,
            "source_onnx": str(source_path),
            "inputs": str(input_path),
            "variants": variants,
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  report updated: {report_path}", flush=True)

    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
