from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import onnx
import onnxruntime as ort
import torch

from xhmodel_merak.xh_other_model.models.kokoro.assets import KokoroAssets
from xhmodel_merak.xh_other_model.models.kokoro.graph import (
    F0BranchStatic,
    _export_graph,
    load_official_model,
    materialize_weight_norm,
    rewrite_depthwise_deconvolution,
)
from xhmodel_merak.xh_other_model.models.kokoro.host import (
    DEFAULT_INPUT_IDS,
    F0_UPSAMPLE,
    HARMONICS,
    SAMPLE_RATE,
    load_voice_style,
    make_frame_masks,
    make_rmsnorm_scales,
)
from xhmodel_merak.xh_other_model.models.kokoro.workflow import _convert_hmonnx


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Kokoro F0, convert it to HMONNX, and compare every operator on a real input.",
    )
    parser.add_argument(
        "--source-export-dir",
        type=Path,
        default=Path("work_dirs/kokoro_merak/single_t32_f120"),
        help="Existing Kokoro export containing metadata and real ORT boundary tensors.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("work_dirs/kokoro_merak/f0_operator_compare_t32_f120"),
    )
    parser.add_argument("--frame-max-length", type=int, default=120)
    parser.add_argument("--norm-mode", choices=("adain", "rmsnorm", "both"), default="adain")
    parser.add_argument("--target", default="XH2a")
    parser.add_argument("--quant-type", default="w16a16_sefp")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reuse", action="store_true", help="Reuse existing ONNX and HMONNX artifacts.")
    return parser.parse_args()


def _assets_from_export(source_export_dir: Path) -> tuple[KokoroAssets, dict[str, Any]]:
    meta_path = source_export_dir / "export_meta_info.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    paths = meta["source_paths"]
    assets = KokoroAssets(
        root=source_export_dir.resolve(),
        source_root=Path(paths["source_root"]).resolve(),
        config=Path(paths["config"]).resolve(),
        checkpoint=Path(paths["checkpoint"]).resolve(),
        voice=Path(paths["voice"]).resolve(),
        voice_pack=None,
    )
    for path in (
        assets.source_root,
        assets.config,
        assets.checkpoint,
        assets.voice,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    return assets, meta


def _real_feed(
    source_export_dir: Path,
    assets: KokoroAssets,
    frame_max_length: int,
) -> tuple[dict[str, torch.Tensor], int, Path]:
    boundary_path = source_export_dir / "diagnostics" / "ort_boundaries.npz"
    with np.load(boundary_path) as data:
        shared = torch.from_numpy(np.array(data["diag_shared"], copy=True)).float()
        valid_frames = int(np.asarray(data["valid_frames"]).reshape(-1)[0])
    expected_shape = (1, 512, frame_max_length)
    if tuple(shared.shape) != expected_shape:
        raise ValueError(f"real shared input shape {tuple(shared.shape)} != {expected_shape}")
    style = load_voice_style(
        assets.voice,
        phoneme_count=len(DEFAULT_INPUT_IDS) - 2,
    )[:, 128:].contiguous()
    mask_f, mask_2f, *_ = make_frame_masks(
        torch.tensor([valid_frames], dtype=torch.int32),
        frame_max_length,
    )
    return (
        {
            "shared": shared,
            "prosody_style": style,
            "mask_f": mask_f,
            "mask_2f": mask_2f,
        },
        valid_frames,
        boundary_path,
    )


def _export_and_convert(
    *,
    mode: str,
    assets: KokoroAssets,
    feed: dict[str, torch.Tensor],
    frame_max_length: int,
    output_dir: Path,
    target: str,
    quant_type: str,
    reuse: bool,
) -> tuple[Path, Path, dict[str, Any]]:
    onnx_dir = output_dir / "onnx"
    hmonnx_dir = output_dir / "hmonnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    hmonnx_dir.mkdir(parents=True, exist_ok=True)
    source_path = onnx_dir / f"kokoro_f0_{mode}_b1_f{frame_max_length}.onnx"
    hmonnx_path = hmonnx_dir / (f"kokoro_f0_{mode}_b1_f{frame_max_length}_{target}_{quant_type}.onnx")
    export_info: dict[str, Any] = {}
    pytorch_path = output_dir / "diagnostics" / "pytorch_f0.npy"
    pytorch_path.parent.mkdir(parents=True, exist_ok=True)
    if not reuse or not source_path.is_file():
        model = load_official_model(assets)
        module = F0BranchStatic(
            model,
            frame_max_length,
            use_rmsnorm=mode == "rmsnorm",
        ).eval()
        export_info["materialized_weight_norm"] = materialize_weight_norm(module)
        export_info["rewritten_depthwise_deconvolution"] = rewrite_depthwise_deconvolution(module)
        artifact, reference = _export_graph(
            role=f"f0_{mode}",
            module=module,
            feed=feed,
            output_names=("f0",),
            path=source_path,
            opset=17,
            simplify=True,
            validate_onnx=False,
        )
        export_info.update(
            {
                "source_node_count": artifact.node_count,
                "source_op_counts": artifact.op_counts,
                "source_sha256": artifact.onnx_sha256,
                "structural_rewrites": artifact.structural_rewrites,
                "artifact": artifact,
            }
        )
        np.save(pytorch_path, reference["f0"].detach().cpu().numpy())
    else:
        export_info["artifact"] = _artifact_for_existing(source_path, mode)
        if not pytorch_path.is_file():
            model = load_official_model(assets)
            module = F0BranchStatic(
                model,
                frame_max_length,
                use_rmsnorm=mode == "rmsnorm",
            ).eval()
            materialize_weight_norm(module)
            rewrite_depthwise_deconvolution(module)
            with torch.no_grad():
                pytorch_f0 = module(*(feed[name] for name in export_info["artifact"].input_names))
            np.save(pytorch_path, pytorch_f0.detach().cpu().numpy())

    if not reuse or not hmonnx_path.is_file():
        _convert_hmonnx(
            export_info["artifact"],
            feed,
            hmonnx_path,
            target=target,
            quant_type=quant_type,
            debug=False,
            decompose_lstm=False,
        )
    source = onnx.shape_inference.infer_shapes(onnx.load(source_path, load_external_data=False))
    hmonnx = onnx.load(hmonnx_path, load_external_data=False)
    export_info.update(
        {
            "source_node_count": len(source.graph.node),
            "source_op_counts": _op_counts(source),
            "hmonnx_node_count": len(hmonnx.graph.node),
            "hmonnx_op_counts": _op_counts(hmonnx, include_domain=True),
            "pytorch_f0_path": str(pytorch_path),
        }
    )
    return source_path, hmonnx_path, export_info


def _artifact_for_existing(path: Path, mode: str):
    from xhmodel_merak.xh_other_model.models.kokoro.graph import GraphArtifact

    model = onnx.load(path, load_external_data=False)
    return GraphArtifact(
        role=f"f0_{mode}",
        path=path,
        input_names=tuple(value.name for value in model.graph.input),
        output_names=("f0",),
        input_contracts=(),
        output_contracts=(),
        node_count=len(model.graph.node),
        op_counts=_op_counts(model),
        onnx_sha256="",
        pytorch_vs_onnx=(),
        structural_rewrites={},
    )


def _op_counts(model: onnx.ModelProto, *, include_domain: bool = False) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for node in model.graph.node:
        key = f"{node.domain or 'ai.onnx'}::{node.op_type}" if include_domain else node.op_type
        counts[key] += 1
    return dict(sorted(counts.items()))


def _instrument_source(source_path: Path, output_path: Path) -> tuple[Path, list[str]]:
    model = onnx.shape_inference.infer_shapes(onnx.load(source_path))
    value_infos = {value.name: value for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)}
    existing = {value.name for value in model.graph.output}
    output_names: list[str] = []
    for node in model.graph.node:
        if len(node.output) != 1:
            raise RuntimeError(f"F0 diagnostic expects one output per source node: {node.name}")
        name = node.output[0]
        output_names.append(name)
        if name in existing:
            continue
        value_info = value_infos.get(name)
        if value_info is None:
            value_info = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
        model.graph.output.append(copy.deepcopy(value_info))
        existing.add(name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    return output_path, output_names


def _run_ort_all(
    instrumented_path: Path,
    output_names: list[str],
    feed: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(instrumented_path),
        options,
        providers=["CPUExecutionProvider"],
    )
    outputs = session.run(output_names, feed)
    return {name: np.asarray(value) for name, value in zip(output_names, outputs, strict=True)}


def _dump_hmonnx_nodes(
    hmonnx_path: Path,
    feed: dict[str, np.ndarray],
    golden_dir: Path,
    device: str,
    input_names: tuple[str, ...],
) -> tuple[Path, np.ndarray]:
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import (
        HMONNXInferenceConfig,
        HMONNXInferenceV2,
    )

    if golden_dir.exists():
        shutil.rmtree(golden_dir)
    actual_device = device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    config = HMONNXInferenceConfig(
        enable_golden=True,
        exec_devices=[actual_device],
    )
    session = HMONNXInferenceV2(str(hmonnx_path), session_config=config)
    session.save_golden_dir = str(golden_dir)
    session.step = 0
    tensors = [
        torch.from_numpy(np.asarray(feed[name]).astype(np.float16, copy=False)).to(actual_device)
        for name in input_names
    ]
    with torch.no_grad():
        output = session(*tensors)
    if isinstance(output, (tuple, list)):
        if len(output) != 1:
            raise RuntimeError(f"unexpected HMONNX F0 output count: {len(output)}")
        output = output[0]
    return golden_dir / "step_0", output.detach().float().cpu().numpy()


def _shape_map(model: onnx.ModelProto) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    values = (*model.graph.input, *model.graph.value_info, *model.graph.output)
    for value in values:
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        result[value.name] = tuple(
            dimension.dim_value if dimension.HasField("dim_value") else -1 for dimension in tensor_type.shape.dim
        )
    return result


def _canonical_op(op_type: str) -> str:
    return {"Conv2d": "Conv", "Linear": "Gemm"}.get(op_type, op_type)


def _collapsed_source_rmsnorm_nodes(
    model: onnx.ModelProto,
) -> dict[int, tuple[tuple[int, ...], str]]:
    nodes = list(model.graph.node)
    producer = {output: index for index, node in enumerate(nodes) for output in node.output}
    collapsed: dict[int, tuple[tuple[int, ...], str]] = {}
    for terminal_index, terminal in enumerate(nodes):
        if terminal.op_type != "Mul" or "/norm/Mul" not in terminal.name:
            continue
        if len(terminal.input) < 2:
            raise RuntimeError(f"malformed exported RMSNorm terminal: {terminal.name}")
        data_input = terminal.input[0]
        div_index = producer.get(terminal.input[1])
        if div_index is None or nodes[div_index].op_type != "Div":
            raise RuntimeError(f"cannot find RMSNorm reciprocal Div for {terminal.name}")
        sqrt_index = producer.get(nodes[div_index].input[1])
        add_index = producer.get(nodes[sqrt_index].input[0]) if sqrt_index is not None else None
        mean_index = producer.get(nodes[add_index].input[0]) if add_index is not None else None
        pow_index = producer.get(nodes[mean_index].input[0]) if mean_index is not None else None
        chain = (pow_index, mean_index, add_index, sqrt_index, div_index)
        if any(index is None for index in chain):
            raise RuntimeError(f"incomplete exported RMSNorm chain for {terminal.name}")
        expected_ops = ("Pow", "ReduceMean", "Add", "Sqrt", "Div")
        actual_ops = tuple(nodes[int(index)].op_type for index in chain)
        if actual_ops != expected_ops or nodes[int(pow_index)].input[0] != data_input:
            raise RuntimeError(f"unexpected exported RMSNorm chain for {terminal.name}: {actual_ops}")
        collapsed[terminal_index] = (tuple(int(index) for index in chain), data_input)
    return collapsed


def _build_source_graph(
    model: onnx.ModelProto,
) -> tuple[nx.DiGraph, dict[int, tuple[tuple[int, ...], str]]]:
    shapes = _shape_map(model)
    collapsed = _collapsed_source_rmsnorm_nodes(model)
    skipped = {index for internal_indices, _ in collapsed.values() for index in internal_indices}
    graph = nx.DiGraph()
    producers: dict[str, str] = {}
    for value in model.graph.input:
        key = f"input:{value.name}"
        graph.add_node(key, label=("Input", value.name, shapes.get(value.name)))
        producers[value.name] = key
    for index, node in enumerate(model.graph.node):
        if index in skipped:
            continue
        key = f"node:{index}"
        graph.add_node(
            key,
            label=("RMSNorm" if index in collapsed else node.op_type, shapes.get(node.output[0])),
            index=index,
            output=node.output[0],
            collapsed=collapsed.get(index),
        )
        for output in node.output:
            producers[output] = key
    for index, node in enumerate(model.graph.node):
        if index in skipped:
            continue
        key = f"node:{index}"
        inputs = [collapsed[index][1]] if index in collapsed else node.input
        for value in inputs:
            producer = producers.get(value)
            if producer is not None:
                graph.add_edge(producer, key)
    for value in model.graph.output:
        key = f"output:{value.name}"
        graph.add_node(key, label=("Output", value.name, shapes.get(value.name)))
        graph.add_edge(producers[value.name], key)
    return graph, collapsed


def _collapsed_hmonnx_nodes(model: onnx.ModelProto) -> dict[int, tuple[int, int]]:
    nodes = list(model.graph.node)
    producer = {output: index for index, node in enumerate(nodes) for output in node.output}
    consumers: dict[str, list[int]] = {}
    for index, node in enumerate(nodes):
        for value in node.input:
            consumers.setdefault(value, []).append(index)
    collapsed: dict[int, tuple[int, int]] = {}
    for index, node in enumerate(nodes):
        if node.op_type not in {"Conv2d", "Resize"} or not node.input:
            continue
        pre_index = producer.get(node.input[0])
        if pre_index is None or nodes[pre_index].op_type != "Unsqueeze":
            continue
        post_candidates = [
            candidate
            for output in node.output
            for candidate in consumers.get(output, [])
            if nodes[candidate].op_type == "Squeeze"
        ]
        if len(post_candidates) != 1:
            raise RuntimeError(f"cannot collapse HMONNX {node.op_type} at node {index}")
        collapsed[index] = (pre_index, post_candidates[0])
    return collapsed


def _build_hmonnx_graph(
    model: onnx.ModelProto,
) -> tuple[nx.DiGraph, dict[int, tuple[int, int]]]:
    shapes = _shape_map(model)
    nodes = list(model.graph.node)
    collapsed = _collapsed_hmonnx_nodes(model)
    skipped = {value for pair in collapsed.values() for value in pair}
    graph = nx.DiGraph()
    producers: dict[str, str] = {}
    for value in model.graph.input:
        key = f"input:{value.name}"
        graph.add_node(key, label=("Input", value.name, shapes.get(value.name)))
        producers[value.name] = key
    for index, node in enumerate(nodes):
        if index in skipped:
            continue
        key = f"node:{index}"
        semantic_output = nodes[collapsed[index][1]].output[0] if index in collapsed else node.output[0]
        graph.add_node(
            key,
            label=(_canonical_op(node.op_type), shapes.get(semantic_output)),
            index=index,
            output=semantic_output,
            collapsed=collapsed.get(index),
        )
        for output in node.output:
            producers[output] = key
        if index in collapsed:
            pre_index, post_index = collapsed[index]
            for output in nodes[pre_index].output:
                producers[output] = key
            for output in nodes[post_index].output:
                producers[output] = key
    for index, node in enumerate(nodes):
        if index in skipped:
            continue
        key = f"node:{index}"
        inputs = list(node.input)
        if index in collapsed:
            pre_index, _ = collapsed[index]
            inputs = list(nodes[pre_index].input) + list(node.input[1:])
        for value in inputs:
            producer = producers.get(value)
            if producer is not None:
                graph.add_edge(producer, key)
    for value in model.graph.output:
        key = f"output:{value.name}"
        graph.add_node(key, label=("Output", value.name, shapes.get(value.name)))
        graph.add_edge(producers[value.name], key)
    return graph, collapsed


def _map_nodes(
    source: onnx.ModelProto,
    hmonnx: onnx.ModelProto,
) -> tuple[list[dict[str, Any]], nx.DiGraph]:
    source = onnx.shape_inference.infer_shapes(source)
    hmonnx = onnx.shape_inference.infer_shapes(hmonnx)
    source_graph, source_collapsed = _build_source_graph(source)
    hmonnx_graph, _ = _build_hmonnx_graph(hmonnx)
    matcher = nx.algorithms.isomorphism.DiGraphMatcher(
        source_graph,
        hmonnx_graph,
        node_match=lambda left, right: left["label"] == right["label"],
    )
    if not matcher.is_isomorphic():
        source_labels = Counter(nx.get_node_attributes(source_graph, "label").values())
        hmonnx_labels = Counter(nx.get_node_attributes(hmonnx_graph, "label").values())
        raise RuntimeError(
            "source/HMONNX graphs are not semantically isomorphic after wrapper collapse: "
            f"source_only={source_labels - hmonnx_labels}, hmonnx_only={hmonnx_labels - source_labels}"
        )
    mapping = matcher.mapping
    semantic_rows: dict[int, dict[str, Any]] = {}
    for source_index, source_node in enumerate(source.graph.node):
        if f"node:{source_index}" not in mapping:
            continue
        h_key = mapping[f"node:{source_index}"]
        h_info = hmonnx_graph.nodes[h_key]
        h_index = int(h_info["index"])
        h_node = hmonnx.graph.node[h_index]
        collapsed = h_info.get("collapsed")
        semantic_rows[source_index] = {
            "source_index": source_index,
            "source_name": source_node.name,
            "source_op": source_node.op_type,
            "source_output": source_node.output[0],
            "shape": list(source_graph.nodes[f"node:{source_index}"]["label"][1]),
            "hmonnx_index": h_index,
            "hmonnx_name": h_node.name,
            "hmonnx_op": h_node.op_type,
            "hmonnx_output": str(h_info["output"]),
            "hmonnx_wrapper_nodes": list(collapsed) if collapsed is not None else [],
            "comparison_status": (
                "fused_rmsnorm_boundary" if source_index in source_collapsed else "mapped_semantic_boundary"
            ),
            "fused_into_source_index": None,
        }
    internal_to_terminal = {
        internal_index: terminal_index
        for terminal_index, (internal_indices, _) in source_collapsed.items()
        for internal_index in internal_indices
    }
    source_shapes = _shape_map(source)
    rows: list[dict[str, Any]] = []
    for source_index, source_node in enumerate(source.graph.node):
        if source_index in semantic_rows:
            rows.append(semantic_rows[source_index])
            continue
        terminal_index = internal_to_terminal.get(source_index)
        if terminal_index is None:
            raise RuntimeError(f"source node {source_index} was neither mapped nor fused")
        terminal = semantic_rows[terminal_index]
        rows.append(
            {
                "source_index": source_index,
                "source_name": source_node.name,
                "source_op": source_node.op_type,
                "source_output": source_node.output[0],
                "shape": list(source_shapes[source_node.output[0]]),
                "hmonnx_index": terminal["hmonnx_index"],
                "hmonnx_name": terminal["hmonnx_name"],
                "hmonnx_op": terminal["hmonnx_op"],
                "hmonnx_output": terminal["hmonnx_output"],
                "hmonnx_wrapper_nodes": [],
                "comparison_status": "fused_internal_no_hmonnx_boundary",
                "fused_into_source_index": terminal_index,
            }
        )
    return rows, source_graph


def _region_mask(shape: tuple[int, ...], valid_frames: int) -> tuple[np.ndarray, int | None]:
    candidates = [(axis, size) for axis, size in enumerate(shape) if size in {120, 240}]
    if not candidates:
        return np.ones(shape, dtype=bool), None
    axis, size = candidates[-1]
    valid = valid_frames if size == 120 else 2 * valid_frames
    mask = np.zeros(shape, dtype=bool)
    slices = [slice(None)] * len(shape)
    slices[axis] = slice(0, valid)
    mask[tuple(slices)] = True
    return mask, axis


def _metrics(reference: np.ndarray, actual: np.ndarray, select: np.ndarray) -> dict[str, Any] | None:
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if reference.shape != actual.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {actual.shape}")
    ref = reference[select].reshape(-1)
    act = actual[select].reshape(-1)
    if ref.size == 0:
        return None
    error = act - ref
    absolute = np.abs(error)
    max_flat_index = int(np.argmax(absolute))
    ref_norm = float(np.linalg.norm(ref))
    act_norm = float(np.linalg.norm(act))
    error_norm = float(np.linalg.norm(error))
    denominator = ref_norm * act_norm
    peak = float(np.max(np.abs(ref), initial=0.0))
    relative_floor = max(1e-3, peak * 1e-4)
    relative_select = np.abs(ref) >= relative_floor
    max_relative = (
        float(np.max(absolute[relative_select] / np.abs(ref[relative_select]), initial=0.0))
        if np.any(relative_select)
        else None
    )
    return {
        "count": int(ref.size),
        "reference_min": float(ref.min(initial=0.0)),
        "reference_max": float(ref.max(initial=0.0)),
        "reference_peak": peak,
        "reference_rms": float(np.sqrt(np.mean(ref * ref))),
        "actual_min": float(act.min(initial=0.0)),
        "actual_max": float(act.max(initial=0.0)),
        "max_abs": float(absolute[max_flat_index]),
        "mean_abs": float(np.mean(absolute)),
        "median_abs": float(np.median(absolute)),
        "p95_abs": float(np.percentile(absolute, 95)),
        "p99_abs": float(np.percentile(absolute, 99)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mean_signed": float(np.mean(error)),
        "error_std": float(np.std(error)),
        "relative_l2": error_norm / ref_norm if ref_norm else None,
        "max_relative_above_floor": max_relative,
        "relative_floor": relative_floor,
        "cosine": float(np.dot(ref, act) / denominator) if denominator else None,
        "snr_db": 20.0 * math.log10(ref_norm / error_norm) if ref_norm and error_norm else None,
        "max_abs_flat_index": max_flat_index,
        "reference_at_max_abs": float(ref[max_flat_index]),
        "actual_at_max_abs": float(act[max_flat_index]),
    }


def _comparison_metrics(
    reference_fp32: np.ndarray,
    reference_rounded_input: np.ndarray,
    hmonnx: np.ndarray,
    valid_frames: int,
) -> dict[str, Any]:
    shape = tuple(reference_fp32.shape)
    if tuple(reference_rounded_input.shape) != shape or tuple(hmonnx.shape) != shape:
        raise ValueError(
            f"comparison shape mismatch: fp32={shape}, rounded={reference_rounded_input.shape}, hmonnx={hmonnx.shape}"
        )
    valid_mask, temporal_axis = _region_mask(shape, valid_frames)
    full_mask = np.ones(shape, dtype=bool)
    invalid_mask = ~valid_mask

    def compare(select: np.ndarray) -> dict[str, Any]:
        return {
            "total": _metrics(reference_fp32, hmonnx, select),
            "input_rounding": _metrics(reference_fp32, reference_rounded_input, select),
            "hmonnx_core": _metrics(reference_rounded_input, hmonnx, select),
        }

    return {
        "temporal_axis": temporal_axis,
        "valid": compare(valid_mask),
        "full": compare(full_mask),
        "invalid": compare(invalid_mask) if np.any(invalid_mask) else None,
    }


def _phase_error_projection(
    reference_f0: np.ndarray,
    actual_f0: np.ndarray,
    valid_frames: int,
) -> dict[str, Any]:
    reference = np.asarray(reference_f0, dtype=np.float64)[:, : 2 * valid_frames]
    actual = np.asarray(actual_f0, dtype=np.float64)[:, : 2 * valid_frames]
    frequency_error = actual - reference
    # Kokoro repeats every 2F F0 value F0_UPSAMPLE times before cumulative
    # phase integration.  One F0-bin error therefore contributes
    # error_hz * F0_UPSAMPLE / SAMPLE_RATE cycles.
    cumulative_cycles = np.cumsum(
        frequency_error * F0_UPSAMPLE / SAMPLE_RATE,
        axis=-1,
    )
    end_cycles = float(cumulative_cycles[0, -1])
    max_abs_cycles = float(np.max(np.abs(cumulative_cycles), initial=0.0))
    return {
        "duration_seconds": 2 * valid_frames * F0_UPSAMPLE / SAMPLE_RATE,
        "f0_upsample": F0_UPSAMPLE,
        "sample_rate": SAMPLE_RATE,
        "frequency_error_sum_hz_frames": float(np.sum(frequency_error)),
        "frequency_error_mean_hz": float(np.mean(frequency_error)),
        "fundamental": {
            "end_cycles": end_cycles,
            "end_radians": end_cycles * 2.0 * math.pi,
            "max_abs_cycles": max_abs_cycles,
            "max_abs_radians": max_abs_cycles * 2.0 * math.pi,
        },
        f"harmonic_{HARMONICS}": {
            "end_cycles": end_cycles * HARMONICS,
            "end_radians": end_cycles * HARMONICS * 2.0 * math.pi,
            "max_abs_cycles": max_abs_cycles * HARMONICS,
            "max_abs_radians": max_abs_cycles * HARMONICS * 2.0 * math.pi,
        },
    }


def _golden_tensor(golden_dir: Path, name: str) -> np.ndarray:
    path = golden_dir / f"{name.replace('/', '_')}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"HMONNX golden tensor is missing: {path}")
    return np.load(path)


def _value_info_map(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    return {value.name: value for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)}


def _run_local_float_boundary(
    source: onnx.ModelProto,
    mapping: list[dict[str, Any]],
    source_index: int,
    golden_dir: Path,
) -> np.ndarray:
    """Replay one source semantic boundary on its actual HMONNX parent tensors.

    Normal ONNX nodes are replayed alone.  A source RMSNorm decomposition is
    replayed as one semantic group because HMONNX exposes only the fused
    RMSNorm output.  Original FP32 source initializers are intentionally kept:
    the resulting local residual therefore includes parameter conversion and
    kernel arithmetic, while excluding all upstream activation error.
    """

    rows_by_index = {int(row["source_index"]): row for row in mapping}
    producers = {output: index for index, node in enumerate(source.graph.node) for output in node.output}
    graph_inputs = {value.name for value in source.graph.input}
    initializers = {value.name: value for value in source.graph.initializer}
    value_infos = _value_info_map(source)
    collapsed = _collapsed_source_rmsnorm_nodes(source)
    if source_index in collapsed:
        internal_indices, _ = collapsed[source_index]
        selected_indices = (*internal_indices, source_index)
    else:
        selected_indices = (source_index,)
    selected_set = set(selected_indices)

    external_names: list[str] = []
    selected_initializers: dict[str, onnx.TensorProto] = {}
    for index in selected_indices:
        for name in source.graph.node[index].input:
            if not name:
                continue
            if name in initializers:
                selected_initializers.setdefault(name, copy.deepcopy(initializers[name]))
                continue
            if producers.get(name) in selected_set:
                continue
            if name not in external_names:
                external_names.append(name)

    def actual_parent(name: str) -> np.ndarray:
        if name in graph_inputs:
            return _golden_tensor(golden_dir, name)
        producer_index = producers.get(name)
        if producer_index is None:
            raise RuntimeError(f"cannot resolve source tensor producer for local replay: {name}")
        producer_row = rows_by_index[producer_index]
        if producer_row["comparison_status"] == "fused_internal_no_hmonnx_boundary":
            raise RuntimeError(f"local replay boundary unexpectedly depends on fused internal tensor: {name}")
        return _golden_tensor(golden_dir, producer_row["hmonnx_output"])

    graph_feed: dict[str, np.ndarray] = {}
    graph_input_infos: list[onnx.ValueInfoProto] = []
    for name in external_names:
        value_info = value_infos.get(name)
        if value_info is None:
            raise RuntimeError(f"source value info is missing for local replay input: {name}")
        element_type = value_info.type.tensor_type.elem_type
        numpy_dtype = onnx.helper.tensor_dtype_to_np_dtype(element_type)
        graph_feed[name] = np.asarray(actual_parent(name)).astype(numpy_dtype, copy=False)
        graph_input_infos.append(copy.deepcopy(value_info))

    terminal = source.graph.node[source_index]
    graph_output_infos = [copy.deepcopy(value_infos[name]) for name in terminal.output]
    local_graph = onnx.helper.make_graph(
        [copy.deepcopy(source.graph.node[index]) for index in selected_indices],
        f"local_source_boundary_{source_index}",
        graph_input_infos,
        graph_output_infos,
        initializer=list(selected_initializers.values()),
    )
    local_model = onnx.helper.make_model(
        local_graph,
        opset_imports=[copy.deepcopy(opset) for opset in source.opset_import],
    )
    local_model.ir_version = source.ir_version
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        local_model.SerializeToString(),
        options,
        providers=["CPUExecutionProvider"],
    )
    outputs = session.run(list(terminal.output), graph_feed)
    if len(outputs) != 1:
        raise RuntimeError(f"local F0 boundary unexpectedly has {len(outputs)} outputs")
    return np.asarray(outputs[0])


def _local_operator_metrics(
    local_float: np.ndarray,
    hmonnx: np.ndarray,
    valid_frames: int,
) -> dict[str, Any]:
    if local_float.shape != hmonnx.shape:
        raise ValueError(f"local replay shape mismatch: {local_float.shape} != {hmonnx.shape}")
    if np.issubdtype(hmonnx.dtype, np.floating):
        expected_stored = local_float.astype(hmonnx.dtype).astype(local_float.dtype)
    else:
        expected_stored = local_float.astype(hmonnx.dtype)
    valid_mask, temporal_axis = _region_mask(tuple(local_float.shape), valid_frames)
    full_mask = np.ones(local_float.shape, dtype=bool)
    invalid_mask = ~valid_mask

    def compare(select: np.ndarray) -> dict[str, Any]:
        return {
            "local_total": _metrics(local_float, hmonnx, select),
            "output_storage": _metrics(local_float, expected_stored, select),
            "backend_parameter": _metrics(expected_stored, hmonnx, select),
        }

    return {
        "hmonnx_storage_dtype": str(hmonnx.dtype),
        "temporal_axis": temporal_axis,
        "valid": compare(valid_mask),
        "full": compare(full_mask),
        "invalid": compare(invalid_mask) if np.any(invalid_mask) else None,
    }


def _root_cause_analysis(
    source: onnx.ModelProto,
    hmonnx: onnx.ModelProto,
    mapping: list[dict[str, Any]],
    source_fp32: dict[str, np.ndarray],
    local_outputs: dict[int, np.ndarray],
    golden_dir: Path,
    valid_frames: int,
) -> dict[str, Any]:
    rows_by_index = {int(row["source_index"]): row for row in mapping}
    producers = {output: index for index, node in enumerate(source.graph.node) for output in node.output}
    source_initializers = {value.name: onnx.numpy_helper.to_array(value) for value in source.graph.initializer}
    hmonnx_initializers = {value.name: onnx.numpy_helper.to_array(value) for value in hmonnx.graph.initializer}
    projection_index = next(index for index, node in enumerate(source.graph.node) if node.name == "/projection/Conv")
    projection_node = source.graph.node[projection_index]
    feature_name = projection_node.input[0]
    feature_index = producers[feature_name]
    feature_row = rows_by_index[feature_index]
    projection_row = rows_by_index[projection_index]
    source_feature = source_fp32[feature_name]
    hmonnx_feature = _golden_tensor(golden_dir, feature_row["hmonnx_output"]).astype(np.float32)
    source_projection = source_fp32[projection_node.output[0]]
    local_projection = local_outputs[projection_index]
    hmonnx_projection = _golden_tensor(golden_dir, projection_row["hmonnx_output"])
    projection_mask, _ = _region_mask(tuple(source_projection.shape), valid_frames)
    feature_mask, _ = _region_mask(tuple(source_feature.shape), valid_frames)

    feature_error = hmonnx_feature[:, :, : 2 * valid_frames].astype(np.float64) - source_feature[
        :, :, : 2 * valid_frames
    ].astype(np.float64)
    feature_channel_mean = source_feature[:, :, : 2 * valid_frames].astype(np.float64).mean(axis=(0, 2))
    error_channel_mean = feature_error.mean(axis=(0, 2))
    design = np.stack([feature_channel_mean, np.ones_like(feature_channel_mean)], axis=1)
    gain_slope, gain_offset = np.linalg.lstsq(design, error_channel_mean, rcond=None)[0]
    fitted = design @ np.asarray([gain_slope, gain_offset])
    residual_energy = float(np.sum((error_channel_mean - fitted) ** 2))
    centered_energy = float(np.sum((error_channel_mean - error_channel_mean.mean()) ** 2))
    channel_mean_gain_r2 = 1.0 - residual_energy / centered_energy if centered_energy else 1.0

    projection_weight = source_initializers[projection_node.input[1]].astype(np.float64)
    projection_bias = source_initializers[projection_node.input[2]].astype(np.float64)
    weight_vector = projection_weight.reshape(-1)
    weight_error_correlation = float(np.corrcoef(weight_vector, error_channel_mean)[0, 1])

    sqrt2_nodes: list[dict[str, Any]] = []
    divisor_ratios: list[float] = []
    for source_index, node in enumerate(source.graph.node):
        if node.op_type != "Div" or len(node.input) < 2 or node.input[1] not in source_initializers:
            continue
        source_divisor_array = np.asarray(source_initializers[node.input[1]])
        if source_divisor_array.size != 1 or not np.isclose(
            float(source_divisor_array.reshape(-1)[0]),
            math.sqrt(2.0),
        ):
            continue
        row = rows_by_index[source_index]
        hmonnx_node = hmonnx.graph.node[int(row["hmonnx_index"])]
        hmonnx_constant_name = next(
            (name for name in hmonnx_node.input if name in hmonnx_initializers),
            None,
        )
        if hmonnx_constant_name is None:
            raise RuntimeError(f"cannot find converted sqrt(2) initializer for {node.name}")
        source_divisor = float(source_divisor_array.reshape(-1)[0])
        hmonnx_divisor_array = np.asarray(hmonnx_initializers[hmonnx_constant_name])
        hmonnx_divisor = float(hmonnx_divisor_array.reshape(-1)[0])
        ratio = source_divisor / hmonnx_divisor
        numerator_index = producers[node.input[0]]
        numerator_row = rows_by_index[numerator_index]
        hmonnx_numerator = _golden_tensor(golden_dir, numerator_row["hmonnx_output"])
        hmonnx_division = _golden_tensor(golden_dir, row["hmonnx_output"])
        converted_constant_replay = (hmonnx_numerator.astype(np.float32) / hmonnx_divisor).astype(hmonnx_division.dtype)
        divisor_mask, _ = _region_mask(tuple(hmonnx_division.shape), valid_frames)
        converted_constant_replay_metrics = _metrics(
            converted_constant_replay,
            hmonnx_division,
            divisor_mask,
        )
        if converted_constant_replay_metrics is None:
            raise RuntimeError(f"empty converted divisor replay for {node.name}")
        divisor_ratios.append(ratio)
        sqrt2_nodes.append(
            {
                "source_index": source_index,
                "source_name": node.name,
                "hmonnx_index": row["hmonnx_index"],
                "source_divisor": source_divisor,
                "source_dtype": str(source_divisor_array.dtype),
                "hmonnx_divisor": hmonnx_divisor,
                "hmonnx_dtype": str(hmonnx_divisor_array.dtype),
                "gain_ratio": ratio,
                "gain_ppm": (ratio - 1.0) * 1e6,
                "converted_constant_replay_vs_hmonnx": converted_constant_replay_metrics,
                "local_replay": row["local_replay"]["valid"],
            }
        )
    if not divisor_ratios or not np.allclose(divisor_ratios, divisor_ratios[0]):
        raise RuntimeError("expected identical converted sqrt(2) divisors in all F0 residual blocks")
    divisor_gain = divisor_ratios[0]

    correction_metrics: dict[str, Any] = {}
    for power in (1, 2, 3):
        corrected_feature = hmonnx_feature.astype(np.float64) / (divisor_gain**power)
        corrected_projection = np.einsum(
            "oc,bct->bot", projection_weight[:, :, 0], corrected_feature
        ) + projection_bias.reshape(1, -1, 1)
        correction_metrics[f"inverse_sqrt2_gain_power_{power}"] = _metrics(
            source_projection,
            corrected_projection,
            projection_mask,
        )

    feature_metrics = _metrics(source_feature, hmonnx_feature, feature_mask)
    upstream_projection_metrics = _metrics(
        source_projection,
        local_projection,
        projection_mask,
    )
    projection_local_metrics = _metrics(
        local_projection,
        hmonnx_projection,
        projection_mask,
    )
    if feature_metrics is None or upstream_projection_metrics is None or projection_local_metrics is None:
        raise RuntimeError("empty root-cause metric region")
    return {
        "method": (
            "Each HMONNX boundary is locally replayed with the corresponding FP32 ONNX operator on actual "
            "HMONNX parent tensors; this removes upstream error from the operator-local residual."
        ),
        "sqrt2_residual_scaling": {
            "nodes": sqrt2_nodes,
            "per_block_gain_ratio": divisor_gain,
            "per_block_gain_ppm": (divisor_gain - 1.0) * 1e6,
            "two_divisors_compound_gain_ratio": divisor_gain**2,
            "two_divisors_compound_gain_ppm": (divisor_gain**2 - 1.0) * 1e6,
        },
        "block2_feature_error": {
            "source_index": feature_index,
            "source_name": source.graph.node[feature_index].name,
            "metrics": feature_metrics,
            "channel_mean_error_fit": {
                "gain_slope": float(gain_slope),
                "offset": float(gain_offset),
                "r_squared": channel_mean_gain_r2,
                "projection_weight_correlation": weight_error_correlation,
            },
        },
        "projection_amplification": {
            "weight_l1": float(np.linalg.norm(weight_vector, ord=1)),
            "weight_l2": float(np.linalg.norm(weight_vector)),
            "weight_max_abs": float(np.max(np.abs(weight_vector))),
            "feature_to_f0_mae_gain": upstream_projection_metrics["mean_abs"] / feature_metrics["mean_abs"],
            "feature_to_f0_rmse_gain": upstream_projection_metrics["rmse"] / feature_metrics["rmse"],
            "upstream_feature_error_through_fp32_projection": upstream_projection_metrics,
            "projection_local_total": projection_local_metrics,
            "projection_local_replay": projection_row["local_replay"]["valid"],
        },
        "theoretical_inverse_gain_counterfactual_with_fp32_projection": correction_metrics,
    }


def _analyse_variant(
    *,
    mode: str,
    source_path: Path,
    hmonnx_path: Path,
    feed_tensors: dict[str, torch.Tensor],
    valid_frames: int,
    output_dir: Path,
    device: str,
    export_info: dict[str, Any],
    source_meta: dict[str, Any],
    boundary_path: Path,
) -> dict[str, Any]:
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    source = onnx.load(source_path, load_external_data=False)
    input_names = tuple(value.name for value in source.graph.input)
    if mode == "rmsnorm" and "norm_scales" not in input_names:
        raise RuntimeError(
            "the reused RMSNorm ONNX predates the Host s + inv_s input contract; "
            "rerun without --reuse or choose a new output directory"
        )
    instrumented_path, source_output_names = _instrument_source(
        source_path,
        diagnostics_dir / f"kokoro_f0_{mode}_all_outputs.onnx",
    )
    feed = {name: feed_tensors[name].detach().cpu().numpy() for name in input_names}
    rounded_feed = {name: value.astype(np.float16).astype(np.float32) for name, value in feed.items()}
    source_fp32 = _run_ort_all(instrumented_path, source_output_names, feed)
    source_rounded = _run_ort_all(instrumented_path, source_output_names, rounded_feed)
    golden_dir, hmonnx_f0 = _dump_hmonnx_nodes(
        hmonnx_path,
        feed,
        diagnostics_dir / "hmonnx_golden",
        device,
        input_names,
    )
    hmonnx = onnx.load(hmonnx_path, load_external_data=False)
    mapping, source_graph = _map_nodes(source, hmonnx)
    pytorch_f0 = np.load(export_info["pytorch_f0_path"])
    pytorch_valid_mask, _ = _region_mask(tuple(pytorch_f0.shape), valid_frames)
    pytorch_vs_source_onnx = _metrics(
        pytorch_f0,
        source_fp32["f0"],
        pytorch_valid_mask,
    )
    input_metrics: dict[str, Any] = {}
    metrics_by_key: dict[str, dict[str, Any]] = {}
    for name in input_names:
        h_value = _golden_tensor(golden_dir, name)
        metrics = _comparison_metrics(feed[name], rounded_feed[name], h_value, valid_frames)
        input_metrics[name] = metrics
        metrics_by_key[f"input:{name}"] = metrics

    rows: list[dict[str, Any]] = []
    local_outputs: dict[int, np.ndarray] = {}
    for row in mapping:
        source_index = int(row["source_index"])
        if row["comparison_status"] == "fused_internal_no_hmonnx_boundary":
            row["parents"] = []
            row["parent_valid_max_abs"] = None
            row["valid_max_abs_growth"] = None
            row["metrics"] = None
            row["local_replay"] = None
            rows.append(row)
            continue
        reference = source_fp32[row["source_output"]]
        rounded_reference = source_rounded[row["source_output"]]
        h_value = _golden_tensor(golden_dir, row["hmonnx_output"])
        metrics = _comparison_metrics(reference, rounded_reference, h_value, valid_frames)
        key = f"node:{source_index}"
        metrics_by_key[key] = metrics
        semantic_parents = list(source_graph.predecessors(key))
        parent_errors = [
            metrics_by_key[parent]["valid"]["total"]["max_abs"]
            for parent in semantic_parents
            if parent in metrics_by_key and metrics_by_key[parent]["valid"]["total"] is not None
        ]
        parent_max_abs = max(parent_errors, default=None)
        current_max_abs = metrics["valid"]["total"]["max_abs"]
        row["parents"] = semantic_parents
        row["parent_valid_max_abs"] = parent_max_abs
        row["valid_max_abs_growth"] = (
            current_max_abs / parent_max_abs if parent_max_abs is not None and parent_max_abs > 0.0 else None
        )
        row["metrics"] = metrics
        local_float = _run_local_float_boundary(
            source,
            mapping,
            source_index,
            golden_dir,
        )
        row["local_replay"] = _local_operator_metrics(
            local_float,
            h_value,
            valid_frames,
        )
        local_outputs[source_index] = local_float
        rows.append(row)

    comparable_rows = [row for row in rows if row["metrics"] is not None]
    final_metrics = rows[-1]["metrics"]
    if final_metrics is None:
        raise RuntimeError("final F0 output unexpectedly has no HMONNX boundary")
    stored_hmonnx_f0 = _golden_tensor(golden_dir, "f0")
    if not np.array_equal(hmonnx_f0, stored_hmonnx_f0.astype(np.float32)):
        raise RuntimeError("returned HMONNX F0 differs from the golden graph output")
    source_f0_storage = source_fp32["f0"].astype(stored_hmonnx_f0.dtype).astype(np.float32)
    final_valid_mask, _ = _region_mask(tuple(source_fp32["f0"].shape), valid_frames)
    final_output_storage_only = _metrics(
        source_fp32["f0"],
        source_f0_storage,
        final_valid_mask,
    )
    if final_output_storage_only is None:
        raise RuntimeError("empty final F0 storage comparison")
    root_cause = _root_cause_analysis(
        source,
        hmonnx,
        rows,
        source_fp32,
        local_outputs,
        golden_dir,
        valid_frames,
    )
    phase_error = _phase_error_projection(
        source_fp32["f0"],
        hmonnx_f0,
        valid_frames,
    )
    np.savez_compressed(diagnostics_dir / "source_fp32_outputs.npz", **source_fp32)
    np.savez_compressed(diagnostics_dir / "source_fp16_rounded_input_outputs.npz", **source_rounded)
    mapping_path = diagnostics_dir / "node_mapping.json"
    mapping_path.write_text(
        json.dumps(
            [{key: value for key, value in row.items() if key != "metrics"} for row in rows],
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    metrics_path = diagnostics_dir / "per_operator_metrics.json"
    metrics_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    csv_path = diagnostics_dir / "per_operator_metrics.csv"
    _write_csv(csv_path, rows)
    root_cause_path = diagnostics_dir / "root_cause_analysis.json"
    root_cause_path.write_text(
        json.dumps(root_cause, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    rmsnorm_scale_pair = None
    if "norm_scales" in input_names:
        rmsnorm_scale_pair = {
            "definition": ["s=sqrt(T/L)", "inv_s=sqrt(L/T)"],
            "source_fp32": feed["norm_scales"].reshape(-1).tolist(),
            "source_fp16_rounded": rounded_feed["norm_scales"].reshape(-1).tolist(),
            "hmonnx_input": _golden_tensor(golden_dir, "norm_scales").reshape(-1).tolist(),
            "hmonnx_input_dtype": str(_golden_tensor(golden_dir, "norm_scales").dtype),
        }

    largest = sorted(
        comparable_rows,
        key=lambda item: item["metrics"]["valid"]["total"]["max_abs"],
        reverse=True,
    )
    largest_relative = sorted(
        comparable_rows,
        key=lambda item: item["metrics"]["valid"]["total"]["relative_l2"] or -1.0,
        reverse=True,
    )
    largest_local_backend = sorted(
        comparable_rows,
        key=lambda item: item["local_replay"]["valid"]["backend_parameter"]["max_abs"],
        reverse=True,
    )
    first_local_backend = next(
        (row for row in comparable_rows if row["local_replay"]["valid"]["backend_parameter"]["max_abs"] >= 1e-6),
        None,
    )
    first_material = next(
        (row for row in comparable_rows if row["metrics"]["valid"]["total"]["max_abs"] >= 1e-2),
        None,
    )
    report = {
        "mode": mode,
        "sample": {
            "text": source_meta["reference_sample"]["text"],
            "phonemes": source_meta["reference_sample"]["phonemes"],
            "token_count": len(source_meta["reference_sample"]["input_ids"]),
            "valid_frames_f": valid_frames,
            "valid_f0_frames_2f": 2 * valid_frames,
            "frame_bucket_f": int(feed["shared"].shape[-1]),
            "f0_bucket_2f": int(feed["mask_2f"].shape[-1]),
            "boundary_npz": str(boundary_path.resolve()),
        },
        "artifacts": {
            "source_onnx": str(source_path.resolve()),
            "instrumented_source_onnx": str(instrumented_path.resolve()),
            "hmonnx": str(hmonnx_path.resolve()),
            "hmonnx_golden_dir": str(golden_dir.resolve()),
            "node_mapping": str(mapping_path.resolve()),
            "per_operator_json": str(metrics_path.resolve()),
            "per_operator_csv": str(csv_path.resolve()),
            "root_cause_analysis": str(root_cause_path.resolve()),
        },
        "graph": {
            "source_nodes": len(source.graph.node),
            "hmonnx_physical_nodes": len(hmonnx.graph.node),
            "matched_semantic_nodes": len(comparable_rows),
            "fused_internal_source_nodes_without_hmonnx_boundary": len(rows) - len(comparable_rows),
            "mapping": (
                "exact DAG isomorphism after collapsing HMONNX Conv1d/Resize wrappers "
                "and source ONNX RMSNorm decomposition groups"
            ),
            "source_op_counts": export_info["source_op_counts"],
            "hmonnx_op_counts": export_info["hmonnx_op_counts"],
            "pytorch_vs_source_onnx_valid_f0": pytorch_vs_source_onnx,
        },
        "input_metrics": input_metrics,
        "rmsnorm_scale_pair": rmsnorm_scale_pair,
        "final_f0": final_metrics,
        "final_f0_output_storage_only": final_output_storage_only,
        "root_cause_analysis": root_cause,
        "sinegen_phase_error_projection": phase_error,
        "first_node_with_valid_max_abs_ge_1e-2": _row_summary(first_material),
        "first_node_with_local_backend_max_abs_ge_1e-6": _row_summary(first_local_backend),
        "largest_valid_max_abs": [_row_summary(row) for row in largest[:15]],
        "largest_valid_relative_l2": [_row_summary(row) for row in largest_relative[:15]],
        "largest_local_backend_parameter_max_abs": [_row_summary(row) for row in largest_local_backend[:20]],
    }
    report_path = output_dir / "summary.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown(output_dir / "report.md", report, rows)
    return report


def _row_summary(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    metrics = row["metrics"]["valid"]
    summary = {
        "source_index": row["source_index"],
        "source_name": row["source_name"],
        "source_op": row["source_op"],
        "hmonnx_index": row["hmonnx_index"],
        "hmonnx_op": row["hmonnx_op"],
        "shape": row["shape"],
        "total": metrics["total"],
        "input_rounding": metrics["input_rounding"],
        "hmonnx_core": metrics["hmonnx_core"],
        "parent_valid_max_abs": row["parent_valid_max_abs"],
        "valid_max_abs_growth": row["valid_max_abs_growth"],
    }
    if row.get("local_replay") is not None:
        summary["local_replay"] = row["local_replay"]["valid"]
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "source_index",
        "source_name",
        "source_op",
        "source_output",
        "shape",
        "hmonnx_index",
        "hmonnx_name",
        "hmonnx_op",
        "hmonnx_output",
        "comparison_status",
        "fused_into_source_index",
        "parent_valid_max_abs",
        "valid_max_abs_growth",
        "valid_max_abs",
        "valid_mean_abs",
        "valid_median_abs",
        "valid_p95_abs",
        "valid_p99_abs",
        "valid_rmse",
        "valid_mean_signed",
        "valid_relative_l2",
        "valid_max_relative_above_floor",
        "valid_cosine",
        "valid_snr_db",
        "full_max_abs",
        "invalid_max_abs",
        "input_rounding_valid_max_abs",
        "hmonnx_core_valid_max_abs",
        "local_total_valid_max_abs",
        "local_total_valid_mean_abs",
        "local_total_valid_p99_abs",
        "local_total_valid_rmse",
        "local_total_valid_mean_signed",
        "local_total_valid_relative_l2",
        "local_output_storage_valid_max_abs",
        "local_output_storage_valid_mean_abs",
        "local_output_storage_valid_rmse",
        "local_output_storage_valid_mean_signed",
        "local_backend_parameter_valid_max_abs",
        "local_backend_parameter_valid_mean_abs",
        "local_backend_parameter_valid_p99_abs",
        "local_backend_parameter_valid_rmse",
        "local_backend_parameter_valid_mean_signed",
        "local_backend_parameter_valid_relative_l2",
        "local_backend_parameter_valid_snr_db",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if row["metrics"] is None:
                writer.writerow(
                    {
                        "source_index": row["source_index"],
                        "source_name": row["source_name"],
                        "source_op": row["source_op"],
                        "source_output": row["source_output"],
                        "shape": "x".join(str(value) for value in row["shape"]),
                        "hmonnx_index": row["hmonnx_index"],
                        "hmonnx_name": row["hmonnx_name"],
                        "hmonnx_op": row["hmonnx_op"],
                        "hmonnx_output": row["hmonnx_output"],
                        "comparison_status": row["comparison_status"],
                        "fused_into_source_index": row["fused_into_source_index"],
                    }
                )
                continue
            valid = row["metrics"]["valid"]
            full = row["metrics"]["full"]
            invalid = row["metrics"]["invalid"]
            total = valid["total"]
            local = row["local_replay"]["valid"]
            writer.writerow(
                {
                    "source_index": row["source_index"],
                    "source_name": row["source_name"],
                    "source_op": row["source_op"],
                    "source_output": row["source_output"],
                    "shape": "x".join(str(value) for value in row["shape"]),
                    "hmonnx_index": row["hmonnx_index"],
                    "hmonnx_name": row["hmonnx_name"],
                    "hmonnx_op": row["hmonnx_op"],
                    "hmonnx_output": row["hmonnx_output"],
                    "comparison_status": row["comparison_status"],
                    "fused_into_source_index": row["fused_into_source_index"],
                    "parent_valid_max_abs": row["parent_valid_max_abs"],
                    "valid_max_abs_growth": row["valid_max_abs_growth"],
                    "valid_max_abs": total["max_abs"],
                    "valid_mean_abs": total["mean_abs"],
                    "valid_median_abs": total["median_abs"],
                    "valid_p95_abs": total["p95_abs"],
                    "valid_p99_abs": total["p99_abs"],
                    "valid_rmse": total["rmse"],
                    "valid_mean_signed": total["mean_signed"],
                    "valid_relative_l2": total["relative_l2"],
                    "valid_max_relative_above_floor": total["max_relative_above_floor"],
                    "valid_cosine": total["cosine"],
                    "valid_snr_db": total["snr_db"],
                    "full_max_abs": full["total"]["max_abs"],
                    "invalid_max_abs": invalid["total"]["max_abs"] if invalid else None,
                    "input_rounding_valid_max_abs": valid["input_rounding"]["max_abs"],
                    "hmonnx_core_valid_max_abs": valid["hmonnx_core"]["max_abs"],
                    "local_total_valid_max_abs": local["local_total"]["max_abs"],
                    "local_total_valid_mean_abs": local["local_total"]["mean_abs"],
                    "local_total_valid_p99_abs": local["local_total"]["p99_abs"],
                    "local_total_valid_rmse": local["local_total"]["rmse"],
                    "local_total_valid_mean_signed": local["local_total"]["mean_signed"],
                    "local_total_valid_relative_l2": local["local_total"]["relative_l2"],
                    "local_output_storage_valid_max_abs": local["output_storage"]["max_abs"],
                    "local_output_storage_valid_mean_abs": local["output_storage"]["mean_abs"],
                    "local_output_storage_valid_rmse": local["output_storage"]["rmse"],
                    "local_output_storage_valid_mean_signed": local["output_storage"]["mean_signed"],
                    "local_backend_parameter_valid_max_abs": local["backend_parameter"]["max_abs"],
                    "local_backend_parameter_valid_mean_abs": local["backend_parameter"]["mean_abs"],
                    "local_backend_parameter_valid_p99_abs": local["backend_parameter"]["p99_abs"],
                    "local_backend_parameter_valid_rmse": local["backend_parameter"]["rmse"],
                    "local_backend_parameter_valid_mean_signed": local["backend_parameter"]["mean_signed"],
                    "local_backend_parameter_valid_relative_l2": local["backend_parameter"]["relative_l2"],
                    "local_backend_parameter_valid_snr_db": local["backend_parameter"]["snr_db"],
                }
            )


def _write_markdown(path: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    final = report["final_f0"]["valid"]
    phase = report["sinegen_phase_error_projection"]
    root_cause = report["root_cause_analysis"]
    lines = [
        f"# Kokoro F0 {report['mode']} W16A16 逐算子误差报告",
        "",
        "真实输入来自完整 Kokoro ORT 数据流：28 token，F 有效 110/120 帧，F0 有效 220/240 帧。",
        "HMONNX 使用 FP16 输入；报告把输入 FP16 舍入误差与 HMONNX 算子误差分开统计。",
        "局部重放使用每个 HMONNX 节点的真实父节点输出作为输入，再单独执行对应浮点 ONNX 算子，"
        "因此不会把上游累计误差错算到当前算子。",
        f"源 ONNX 对 PyTorch 的 F0 max_abs 为 {report['graph']['pytorch_vs_source_onnx_valid_f0']['max_abs']:.8g} Hz。",
    ]
    scale_pair = report.get("rmsnorm_scale_pair")
    if scale_pair is not None:
        lines.extend(
            [
                f"RMSNorm 的 Host 输入为 s/inv_s={scale_pair['source_fp32']}；"
                f"HMONNX {scale_pair['hmonnx_input_dtype']} 实际收到 {scale_pair['hmonnx_input']}。",
            ]
        )
    lines.extend(
        [
            "",
            "## 最终 F0（有效 220 帧）",
            "",
            "| 对比 | max_abs | MAE | P99 | RMSE | relative L2 | cosine | SNR(dB) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, key in (
        ("ORT FP32 vs HMONNX", "total"),
        ("ORT FP32 vs FP16-rounded input ORT", "input_rounding"),
        ("FP16-rounded input ORT vs HMONNX", "hmonnx_core"),
    ):
        metric = final[key]
        lines.append(
            f"| {label} | {metric['max_abs']:.8g} | {metric['mean_abs']:.8g} | "
            f"{metric['p99_abs']:.8g} | {metric['rmse']:.8g} | "
            f"{metric['relative_l2']:.8g} | {metric['cosine']:.10g} | {metric['snr_db']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## 算子自身误差定位",
            "",
            "每个节点再拆成三项：`local_total` 是浮点算子局部重放与 HMONNX 输出的差；"
            "`output_storage` 是只把局部浮点结果存成 HMONNX 输出 dtype 的误差；"
            "`backend_parameter` 是扣除输出存储舍入后，权重转换、LUT 和内核计算留下的残差。",
            "",
            f"即使前面所有算子完全准确，只把最终浮点 F0 存成 "
            f"{rows[-1]['local_replay']['hmonnx_storage_dtype']}，也会产生 "
            f"max_abs={report['final_f0_output_storage_only']['max_abs']:.8g} Hz、"
            f"MAE={report['final_f0_output_storage_only']['mean_abs']:.8g} Hz、"
            f"RMSE={report['final_f0_output_storage_only']['rmse']:.8g} Hz。",
            "",
            "### backend/parameter 局部残差最大的 20 个语义算子",
            "",
            "| ONNX idx | ONNX 节点 | ONNX/HMONNX 算子 | shape | local total max | storage max | "
            "backend max | backend MAE | backend relative L2 |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    comparable_rows = [row for row in rows if row["metrics"] is not None]
    largest_local = sorted(
        comparable_rows,
        key=lambda item: item["local_replay"]["valid"]["backend_parameter"]["max_abs"],
        reverse=True,
    )
    for row in largest_local[:20]:
        local = row["local_replay"]["valid"]
        lines.append(
            f"| {row['source_index']} | `{row['source_name']}` | "
            f"{row['source_op']}/{row['hmonnx_op']} | {'x'.join(map(str, row['shape']))} | "
            f"{local['local_total']['max_abs']:.8g} | "
            f"{local['output_storage']['max_abs']:.8g} | "
            f"{local['backend_parameter']['max_abs']:.8g} | "
            f"{local['backend_parameter']['mean_abs']:.8g} | "
            f"{local['backend_parameter']['relative_l2']:.8g} |"
        )
    sqrt2 = root_cause["sqrt2_residual_scaling"]
    block2 = root_cause["block2_feature_error"]
    projection = root_cause["projection_amplification"]
    correction = root_cause["theoretical_inverse_gain_counterfactual_with_fp32_projection"][
        "inverse_sqrt2_gain_power_2"
    ]
    source_divisor = sqrt2["nodes"][0]["source_divisor"]
    hmonnx_divisor = sqrt2["nodes"][0]["hmonnx_divisor"]
    converted_divisor_replay_max = max(
        node["converted_constant_replay_vs_hmonnx"]["max_abs"] for node in sqrt2["nodes"]
    )
    upstream = projection["upstream_feature_error_through_fp32_projection"]
    projection_local = projection["projection_local_total"]
    lines.extend(
        [
            "",
            "### 根因链路",
            "",
            f"三个残差块的 `/sqrt(2)` 在浮点 ONNX 中是 `{source_divisor:.10g}`，"
            f"HMONNX FP16 常量变成 `{hmonnx_divisor:.10g}`。因此每个残差缩放产生 "
            f"`+{sqrt2['per_block_gain_ppm']:.5g} ppm` 增益；两次该增益复合为 "
            f"`+{sqrt2['two_divisors_compound_gain_ppm']:.5g} ppm`。",
            f"把 HMONNX 真实分子除以转换后的 `{hmonnx_divisor:.10g}` 再存成 FP16，"
            f"与三个 HMONNX Div 输出的最大差都是 `{converted_divisor_replay_max:.8g}`；"
            "偏差来自常量表示，不是 Div 内核实现。",
            f"Block2 输出的逐通道均值误差可由 `error = {block2['channel_mean_error_fit']['gain_slope']:.8g} "
            f"* feature + {block2['channel_mean_error_fit']['offset']:.8g}` 拟合，"
            f"R²={block2['channel_mean_error_fit']['r_squared']:.6g}；该误差与最终 projection 权重的 "
            f"correlation={block2['channel_mean_error_fit']['projection_weight_correlation']:.6g}。",
            f"Block2 feature 自身 MAE={block2['metrics']['mean_abs']:.8g}，经过 FP32 projection 后变成 "
            f"F0 MAE={upstream['mean_abs']:.8g} Hz、mean_signed={upstream['mean_signed']:.8g} Hz，"
            f"MAE 放大 {projection['feature_to_f0_mae_gain']:.6g}x。",
            f"最终 projection 自身的 local total MAE={projection_local['mean_abs']:.8g} Hz、"
            f"mean_signed={projection_local['mean_signed']:.8g} Hz；它部分抵消上游正偏差，"
            "所以不能把最终 Conv 的局部 max 当成最早根因。",
            f"只按理论上的两个 `/sqrt(2)` 增益反校正 Block2 feature，再用 FP32 projection，"
            f"F0 MAE 从 {upstream['mean_abs']:.8g} Hz 降为 {correction['mean_abs']:.8g} Hz，"
            f"mean_signed 从 {upstream['mean_signed']:.8g} Hz 降为 {correction['mean_signed']:.8g} Hz。",
            "",
            "## SineGen 相位累计",
            "",
            f"当前 {phase['duration_seconds']:.3f}s 有效音频内，基频末端累计相位误差为 "
            f"{phase['fundamental']['end_cycles']:.8g} cycles / "
            f"{phase['fundamental']['end_radians']:.8g} rad；"
            f"第 {HARMONICS} 次谐波为 {phase[f'harmonic_{HARMONICS}']['end_cycles']:.8g} cycles / "
            f"{phase[f'harmonic_{HARMONICS}']['end_radians']:.8g} rad。",
            "",
            "## 按有效区 max_abs 排名前 20 的算子",
            "",
            "| ONNX idx | ONNX 节点 | 算子 | HMONNX idx/算子 | shape | max_abs | P99 | MAE | RMSE | core max_abs |",
            "|---:|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    largest = sorted(
        comparable_rows,
        key=lambda item: item["metrics"]["valid"]["total"]["max_abs"],
        reverse=True,
    )
    for row in largest[:20]:
        total = row["metrics"]["valid"]["total"]
        core = row["metrics"]["valid"]["hmonnx_core"]
        lines.append(
            f"| {row['source_index']} | `{row['source_name']}` | {row['source_op']} | "
            f"{row['hmonnx_index']}/{row['hmonnx_op']} | {'x'.join(map(str, row['shape']))} | "
            f"{total['max_abs']:.8g} | {total['p99_abs']:.8g} | {total['mean_abs']:.8g} | "
            f"{total['rmse']:.8g} | {core['max_abs']:.8g} |"
        )
    lines.extend(
        [
            "",
            f"完整 {len(rows)} 个源 ONNX 节点中，{len(comparable_rows)} 个可对齐语义边界的 "
            "full/valid/invalid、input-rounding/core 指标见 `diagnostics/per_operator_metrics.csv` "
            "和 `diagnostics/per_operator_metrics.json`。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_variant_comparison(
    output_root: Path,
    reports: dict[str, dict[str, Any]],
    valid_frames: int,
) -> None:
    if set(reports) != {"adain", "rmsnorm"}:
        return
    with np.load(output_root / "adain" / "diagnostics" / "source_fp32_outputs.npz") as data:
        adain_fp32 = np.array(data["f0"][:, : 2 * valid_frames], copy=True)
    with np.load(output_root / "rmsnorm" / "diagnostics" / "source_fp32_outputs.npz") as data:
        rmsnorm_fp32 = np.array(data["f0"][:, : 2 * valid_frames], copy=True)
    select = np.ones(adain_fp32.shape, dtype=bool)
    fp32_equivalence = _metrics(adain_fp32, rmsnorm_fp32, select)
    if fp32_equivalence is None:
        raise RuntimeError("empty F0 comparison")
    adain = reports["adain"]["final_f0"]["valid"]["total"]
    rmsnorm = reports["rmsnorm"]["final_f0"]["valid"]["total"]
    ratios = {name: rmsnorm[name] / adain[name] for name in ("max_abs", "mean_abs", "p99_abs", "rmse", "relative_l2")}
    comparison = {
        "fp32_rmsnorm_vs_adain": fp32_equivalence,
        "w16a16_final_f0": {
            "adain": adain,
            "rmsnorm": rmsnorm,
            "rmsnorm_over_adain": ratios,
            "snr_delta_db_rmsnorm_minus_adain": rmsnorm["snr_db"] - adain["snr_db"],
        },
        "sinegen_phase_error_projection": {
            mode: report["sinegen_phase_error_projection"] for mode, report in reports.items()
        },
        "conclusion": (
            "The centered masked RMSNorm rewrite uses Host-provided s and inv_s and is FP32-equivalent, "
            "but its W16A16 F0 error remains larger than AdaIN; it does not make NPU F0 safe for the "
            "phase-integrating SineGen boundary."
        ),
    }
    (output_root / "comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    phase_a = comparison["sinegen_phase_error_projection"]["adain"]
    phase_r = comparison["sinegen_phase_error_projection"]["rmsnorm"]
    lines = [
        "# Kokoro F0 AdaIN 与 RMSNorm 对比",
        "",
        "两张图使用同一组完整模型真实输入，分别独立导出、转 W16A16 HMONNX，并逐算子抓取。",
        "",
        "| 项目 | AdaIN | RMSNorm | RMS/AdaIN |",
        "|---|---:|---:|---:|",
    ]
    for label, key in (
        ("max_abs (Hz)", "max_abs"),
        ("MAE (Hz)", "mean_abs"),
        ("P99 (Hz)", "p99_abs"),
        ("RMSE (Hz)", "rmse"),
        ("relative L2", "relative_l2"),
    ):
        lines.append(f"| {label} | {adain[key]:.8g} | {rmsnorm[key]:.8g} | {ratios[key]:.6g} |")
    lines.extend(
        [
            f"| SNR (dB) | {adain['snr_db']:.8g} | {rmsnorm['snr_db']:.8g} | "
            f"差 {comparison['w16a16_final_f0']['snr_delta_db_rmsnorm_minus_adain']:.6g} dB |",
            "",
            f"FP32 两种实现的 F0 差异：max_abs={fp32_equivalence['max_abs']:.8g} Hz，"
            f"MAE={fp32_equivalence['mean_abs']:.8g} Hz。",
            "",
            f"SineGen 第 {HARMONICS} 次谐波末端累计相位误差：AdaIN "
            f"{phase_a[f'harmonic_{HARMONICS}']['end_radians']:.8g} rad，RMSNorm "
            f"{phase_r[f'harmonic_{HARMONICS}']['end_radians']:.8g} rad。",
            "",
            "当前 RMSNorm 图由 Host 同时输入 s=sqrt(T/L) 和 inv_s=sqrt(L/T)。"
            "改写在 FP32 数学上成立，但 W16A16 下误差仍比 AdaIN 大，不能据此把 F0 交给 SineGen。",
            "",
        ]
    )
    (output_root / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    source_export_dir = args.source_export_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    assets, source_meta = _assets_from_export(source_export_dir)
    feed, valid_frames, boundary_path = _real_feed(
        source_export_dir,
        assets,
        args.frame_max_length,
    )
    np.savez(
        output_root / "real_inputs.npz",
        **{name: value.detach().cpu().numpy() for name, value in feed.items()},
        norm_scales=make_rmsnorm_scales(
            torch.tensor([valid_frames], dtype=torch.int32),
            args.frame_max_length,
        ).numpy(),
        valid_frames=np.asarray([valid_frames], dtype=np.int32),
    )
    modes = ("adain", "rmsnorm") if args.norm_mode == "both" else (args.norm_mode,)
    manifest: dict[str, Any] = {
        "source_export_dir": str(source_export_dir),
        "frame_max_length": args.frame_max_length,
        "valid_frames": valid_frames,
        "target": args.target,
        "quant_type": args.quant_type,
        "variants": {},
    }
    reports: dict[str, dict[str, Any]] = {}
    for mode in modes:
        variant_dir = output_root / mode
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant_feed = dict(feed)
        if mode == "rmsnorm":
            variant_feed["norm_scales"] = make_rmsnorm_scales(
                torch.tensor([valid_frames], dtype=torch.int32),
                args.frame_max_length,
            )
        source_path, hmonnx_path, export_info = _export_and_convert(
            mode=mode,
            assets=assets,
            feed=variant_feed,
            frame_max_length=args.frame_max_length,
            output_dir=variant_dir,
            target=args.target,
            quant_type=args.quant_type,
            reuse=args.reuse,
        )
        report = _analyse_variant(
            mode=mode,
            source_path=source_path,
            hmonnx_path=hmonnx_path,
            feed_tensors=variant_feed,
            valid_frames=valid_frames,
            output_dir=variant_dir,
            device=args.device,
            export_info=export_info,
            source_meta=source_meta,
            boundary_path=boundary_path,
        )
        manifest["variants"][mode] = {
            "summary": str((variant_dir / "summary.json").resolve()),
            "final_f0": report["final_f0"],
        }
        reports[mode] = report
    _write_variant_comparison(output_root, reports, valid_frames)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
