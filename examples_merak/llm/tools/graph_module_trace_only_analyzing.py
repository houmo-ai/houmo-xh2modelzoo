#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


GRAPH_NODE_PREFIX = "graph_module_node::"
CONVERSION_CPU_OPS = {"aten::to", "aten::_to_copy", "aten::copy_"}
SCALAR_CPU_OPS = {"aten::item", "aten::_local_scalar_dense"}
SYNC_RUNTIME_OPS = {"cudaDeviceSynchronize", "cudaStreamSynchronize"}
FAMILY_TOKENS = [
    "mlp_gate_proj",
    "mlp_up_proj",
    "mlp_down_proj",
    "mlp_act_fn",
    "self_attn_q_proj",
    "self_attn_k_proj",
    "self_attn_v_proj",
    "self_attn_o_proj",
    "self_attn_matmul_0",
    "self_attn_matmul",
    "self_attn_masked_softmax",
    "input_layernorm",
    "post_attention_layernorm",
    "rotary_emb",
    "lm_head",
]
SIGNAL_WEIGHTS = {
    "compute": 1.0,
    "sync": 1.25,
    "conversion": 1.35,
    "scalar": 1.35,
}
BOTTLENECK_RECOMMENDATIONS = {
    "compute_hotspot": "Focus on the dominant GPU kernel or fuse adjacent math around it.",
    "fragmented_compute": "Reduce small kernels and intermediate materialization in the GPU path.",
    "sync_wait": "Remove or relocate blocking device or stream synchronizations on the CPU path.",
    "conversion_copy": "Cache constants and eliminate repeated to/_to_copy/copy_ conversion chains.",
    "scalarization": "Remove item/_local_scalar_dense style scalar round-trips and vectorize the path.",
    "secondary": "Lower priority from this trace; optimize after the dominant families.",
}


@dataclass
class BaseMetrics:
    base: str
    user_annotation_count: int = 0
    gpu_annotation_count: int = 0
    user_annotation_ms: float = 0.0
    gpu_annotation_ms: float = 0.0
    cpu_op_count: int = 0
    runtime_count: int = 0
    launch_count: int = 0
    gpu_memcpy_count: int = 0
    nested_kernel_count: int = 0
    launch_runtime_ms: float = 0.0
    gpu_memcpy_total_ms: float = 0.0
    nested_kernel_total_ms: float = 0.0
    cpu_op_intervals_us: List[Tuple[int, int]] = field(default_factory=list)
    conversion_intervals_us: List[Tuple[int, int]] = field(default_factory=list)
    scalar_intervals_us: List[Tuple[int, int]] = field(default_factory=list)
    runtime_intervals_us: List[Tuple[int, int]] = field(default_factory=list)
    sync_runtime_intervals_us: List[Tuple[int, int]] = field(default_factory=list)
    nested_kernel_intervals_us: List[Tuple[int, int]] = field(default_factory=list)
    cpu_op_name_totals: Counter = field(default_factory=Counter)
    runtime_name_totals: Counter = field(default_factory=Counter)
    kernel_name_totals: Counter = field(default_factory=Counter)


@dataclass
class FamilyMetrics:
    family: str
    node_count: int = 0
    user_annotation_node_count: int = 0
    gpu_annotation_node_count: int = 0
    user_annotation_ms: float = 0.0
    gpu_annotation_ms: float = 0.0
    cpu_op_span_ms: float = 0.0
    conversion_cpu_span_ms: float = 0.0
    scalar_cpu_span_ms: float = 0.0
    runtime_span_ms: float = 0.0
    sync_runtime_span_ms: float = 0.0
    nested_kernel_span_ms: float = 0.0
    nested_kernel_total_ms: float = 0.0
    nested_kernel_count: int = 0
    runtime_count: int = 0
    launch_count: int = 0
    launch_runtime_ms: float = 0.0
    gpu_memcpy_total_ms: float = 0.0
    gpu_memcpy_count: int = 0
    cpu_op_names: Counter = field(default_factory=Counter)
    runtime_names: Counter = field(default_factory=Counter)
    kernel_names: Counter = field(default_factory=Counter)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--trace", type=str, required=True, help="Path to a torch.profiler .pt.trace.json file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for generated trace-only analysis files. Defaults to a trace_only_analysis sibling directory.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional prefix for generated filenames. Defaults to the trace basename.",
    )
    parser.add_argument("--topk", type=int, default=12, help="Number of ranked families to print in the console")
    parser.add_argument(
        "--focus-family",
        type=str,
        default="",
        help="Optional family name for a specialized top-node report, for example self_attn_matmul.",
    )
    parser.add_argument(
        "--focus-topk",
        type=int,
        default=20,
        help="Number of top nodes to include in the specialized family report.",
    )
    return parser.parse_args()


def _round_metric(value: float) -> float:
    return round(value, 6)


def _trace_stem(trace_path: Path) -> str:
    name = trace_path.name
    suffix = ".pt.trace.json"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    if name.endswith(".json"):
        return name[:-5]
    return trace_path.stem


def _default_output_dir(trace_path: Path) -> Path:
    trace_parent = trace_path.parent
    if trace_parent.name == "tensorboard":
        return trace_parent.parent / "trace_only_analysis"
    return trace_parent / "trace_only_analysis"


def _family_of(base: str) -> str:
    for token in FAMILY_TOKENS:
        if token in base:
            return token
    return "other"


def _merge_intervals(intervals: Iterable[Tuple[int, int]]) -> int:
    sorted_intervals = sorted((start, end) for start, end in intervals if end > start)
    if not sorted_intervals:
        return 0

    merged_total = 0
    current_start, current_end = sorted_intervals[0]
    for start, end in sorted_intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        merged_total += current_end - current_start
        current_start, current_end = start, end
    merged_total += current_end - current_start
    return merged_total


def _duration_ms(event: Dict[str, object]) -> float:
    return float(event.get("dur", 0) or 0) / 1000.0


def _end_ts_us(event: Dict[str, object]) -> int:
    start = int(event.get("ts", 0) or 0)
    return start + int(event.get("dur", 0) or 0)


def _base_from_event_name(name: str) -> str:
    return name.split("##run=")[0]


def _get_metric(dct: Dict[str, BaseMetrics], base: str) -> BaseMetrics:
    metric = dct.get(base)
    if metric is None:
        metric = BaseMetrics(base=base)
        dct[base] = metric
    return metric


def _load_x_events(trace_path: Path) -> List[Dict[str, object]]:
    with trace_path.open() as handle:
        data = json.load(handle)
    return [event for event in data.get("traceEvents", []) if event.get("ph") == "X"]


def _group_events_by_tid(events: Sequence[Dict[str, object]]) -> Dict[Tuple[object, object], List[Dict[str, object]]]:
    grouped: Dict[Tuple[object, object], List[Dict[str, object]]] = defaultdict(list)
    for event_index, event in enumerate(events):
        item = dict(event)
        item["_index"] = event_index
        item["_end_us"] = _end_ts_us(item)
        grouped[(item.get("pid"), item.get("tid"))].append(item)
    for sequence in grouped.values():
        sequence.sort(key=lambda item: (int(item.get("ts", 0) or 0), int(item["_end_us"]), int(item["_index"])))
    return grouped


def _build_base_metrics(events: Sequence[Dict[str, object]]) -> Dict[str, BaseMetrics]:
    grouped = _group_events_by_tid(events)
    base_metrics: Dict[str, BaseMetrics] = {}
    correlation_to_base: Dict[object, str] = {}

    for sequence in grouped.values():
        stack: List[Dict[str, object]] = []
        for event in sequence:
            ts_us = int(event.get("ts", 0) or 0)
            while stack and int(stack[-1]["_end_us"]) <= ts_us:
                stack.pop()

            event_name = str(event.get("name", ""))
            event_cat = str(event.get("cat", ""))
            event_end_us = int(event["_end_us"])

            if event_name.startswith(GRAPH_NODE_PREFIX):
                base = _base_from_event_name(event_name)
                metric = _get_metric(base_metrics, base)
                if event_cat == "user_annotation":
                    metric.user_annotation_count += 1
                    metric.user_annotation_ms += _duration_ms(event)
                elif event_cat == "gpu_user_annotation":
                    metric.gpu_annotation_count += 1
                    metric.gpu_annotation_ms += _duration_ms(event)

            current_base = ""
            current_gpu_base = ""
            for parent in reversed(stack):
                parent_name = str(parent.get("name", ""))
                if not parent_name.startswith(GRAPH_NODE_PREFIX):
                    continue
                if not current_base:
                    current_base = _base_from_event_name(parent_name)
                if not current_gpu_base and str(parent.get("cat", "")) == "gpu_user_annotation":
                    current_gpu_base = _base_from_event_name(parent_name)
                if current_base and current_gpu_base:
                    break

            if current_base and event_cat == "cpu_op":
                metric = _get_metric(base_metrics, current_base)
                metric.cpu_op_count += 1
                metric.cpu_op_intervals_us.append((ts_us, event_end_us))
                metric.cpu_op_name_totals[event_name] += _duration_ms(event)
                if event_name in CONVERSION_CPU_OPS:
                    metric.conversion_intervals_us.append((ts_us, event_end_us))
                if event_name in SCALAR_CPU_OPS:
                    metric.scalar_intervals_us.append((ts_us, event_end_us))

            if current_base and event_cat == "cuda_runtime":
                metric = _get_metric(base_metrics, current_base)
                metric.runtime_count += 1
                metric.runtime_intervals_us.append((ts_us, event_end_us))
                metric.runtime_name_totals[event_name] += _duration_ms(event)
                if event_name in SYNC_RUNTIME_OPS:
                    metric.sync_runtime_intervals_us.append((ts_us, event_end_us))
                if event_name == "cudaLaunchKernel":
                    metric.launch_count += 1
                    metric.launch_runtime_ms += _duration_ms(event)
                correlation = (event.get("args") or {}).get("correlation")
                if correlation is not None:
                    correlation_to_base.setdefault(correlation, current_base)

            if current_gpu_base and event_cat == "kernel":
                metric = _get_metric(base_metrics, current_gpu_base)
                metric.nested_kernel_count += 1
                metric.nested_kernel_total_ms += _duration_ms(event)
                metric.nested_kernel_intervals_us.append((ts_us, event_end_us))
                metric.kernel_name_totals[event_name] += _duration_ms(event)

            stack.append(event)

    for event in events:
        if str(event.get("cat", "")) != "gpu_memcpy":
            continue
        correlation = (event.get("args") or {}).get("correlation")
        if correlation is None:
            continue
        base = correlation_to_base.get(correlation)
        if not base:
            continue
        metric = _get_metric(base_metrics, base)
        metric.gpu_memcpy_count += 1
        metric.gpu_memcpy_total_ms += _duration_ms(event)

    return base_metrics


def _aggregate_families(base_metrics: Dict[str, BaseMetrics]) -> Dict[str, FamilyMetrics]:
    family_metrics: Dict[str, FamilyMetrics] = {}
    for base, metric in base_metrics.items():
        family = _family_of(base)
        family_metric = family_metrics.get(family)
        if family_metric is None:
            family_metric = FamilyMetrics(family=family)
            family_metrics[family] = family_metric

        family_metric.node_count += 1
        if metric.user_annotation_count > 0:
            family_metric.user_annotation_node_count += 1
        if metric.gpu_annotation_count > 0:
            family_metric.gpu_annotation_node_count += 1
        family_metric.user_annotation_ms += metric.user_annotation_ms
        family_metric.gpu_annotation_ms += metric.gpu_annotation_ms
        family_metric.cpu_op_span_ms += _merge_intervals(metric.cpu_op_intervals_us) / 1000.0
        family_metric.conversion_cpu_span_ms += _merge_intervals(metric.conversion_intervals_us) / 1000.0
        family_metric.scalar_cpu_span_ms += _merge_intervals(metric.scalar_intervals_us) / 1000.0
        family_metric.runtime_span_ms += _merge_intervals(metric.runtime_intervals_us) / 1000.0
        family_metric.sync_runtime_span_ms += _merge_intervals(metric.sync_runtime_intervals_us) / 1000.0
        family_metric.nested_kernel_span_ms += _merge_intervals(metric.nested_kernel_intervals_us) / 1000.0
        family_metric.nested_kernel_total_ms += metric.nested_kernel_total_ms
        family_metric.nested_kernel_count += metric.nested_kernel_count
        family_metric.runtime_count += metric.runtime_count
        family_metric.launch_count += metric.launch_count
        family_metric.launch_runtime_ms += metric.launch_runtime_ms
        family_metric.gpu_memcpy_total_ms += metric.gpu_memcpy_total_ms
        family_metric.gpu_memcpy_count += metric.gpu_memcpy_count
        family_metric.cpu_op_names.update(metric.cpu_op_name_totals)
        family_metric.runtime_names.update(metric.runtime_name_totals)
        family_metric.kernel_names.update(metric.kernel_name_totals)
    return family_metrics


def _top_name(counter: Counter) -> Tuple[str, float]:
    if not counter:
        return "", 0.0
    name, total_ms = counter.most_common(1)[0]
    return str(name), float(total_ms)


def _primary_annotation_category(user_annotation_ms: float, gpu_annotation_ms: float) -> str:
    if gpu_annotation_ms > 0.0:
        return "gpu_user_annotation"
    if user_annotation_ms > 0.0:
        return "user_annotation"
    return ""


def _classify_signals(
    work_unit_count: int,
    nested_kernel_count: int,
    kernel_avg_ms: float,
    compute_signal_ms: float,
    sync_signal_ms: float,
    conversion_signal_ms: float,
    scalar_signal_ms: float,
    kernel_names: Counter,
    runtime_names: Counter,
    cpu_op_names: Counter,
) -> Tuple[str, float, str, float, float, float, float]:
    raw_signals = {
        "compute": compute_signal_ms,
        "sync": sync_signal_ms,
        "conversion": conversion_signal_ms,
        "scalar": scalar_signal_ms,
    }
    weighted_signals = {name: value * SIGNAL_WEIGHTS[name] for name, value in raw_signals.items()}
    sorted_signals = sorted(weighted_signals.items(), key=lambda item: item[1], reverse=True)
    primary_signal_name, primary_signal_score = sorted_signals[0]
    secondary_signal_score = sorted_signals[1][1] if len(sorted_signals) > 1 else 0.0
    tertiary_signal_score = sorted_signals[2][1] if len(sorted_signals) > 2 else 0.0
    priority_score = primary_signal_score + 0.35 * secondary_signal_score + 0.1 * tertiary_signal_score

    if primary_signal_name == "compute":
        if nested_kernel_count > max(work_unit_count, 1) * 4 and kernel_avg_ms < 0.3:
            bottleneck = "fragmented_compute"
        else:
            bottleneck = "compute_hotspot"
    elif primary_signal_name == "sync":
        bottleneck = "sync_wait"
    elif primary_signal_name == "conversion":
        bottleneck = "conversion_copy"
    elif primary_signal_name == "scalar":
        bottleneck = "scalarization"
    else:
        bottleneck = "secondary"

    if bottleneck in {"compute_hotspot", "fragmented_compute"}:
        dominant_event_name, dominant_event_ms = _top_name(kernel_names)
    elif bottleneck == "sync_wait":
        dominant_event_name, dominant_event_ms = _top_name(
            Counter({name: total for name, total in runtime_names.items() if name in SYNC_RUNTIME_OPS})
        )
    elif bottleneck in {"conversion_copy", "scalarization"}:
        tracked_names = CONVERSION_CPU_OPS if bottleneck == "conversion_copy" else SCALAR_CPU_OPS
        dominant_event_name, dominant_event_ms = _top_name(
            Counter({name: total for name, total in cpu_op_names.items() if name in tracked_names})
        )
    else:
        dominant_event_name, dominant_event_ms = _top_name(cpu_op_names)

    return (
        bottleneck,
        priority_score,
        dominant_event_name,
        dominant_event_ms,
        compute_signal_ms,
        sync_signal_ms,
        max(conversion_signal_ms, scalar_signal_ms),
    )


def _classify_family(metric: FamilyMetrics) -> Tuple[str, float, str, float, float, float, float]:
    kernel_avg_ms = metric.nested_kernel_total_ms / metric.nested_kernel_count if metric.nested_kernel_count else 0.0
    return _classify_signals(
        work_unit_count=metric.node_count,
        nested_kernel_count=metric.nested_kernel_count,
        kernel_avg_ms=kernel_avg_ms,
        compute_signal_ms=metric.nested_kernel_span_ms,
        sync_signal_ms=metric.sync_runtime_span_ms,
        conversion_signal_ms=metric.conversion_cpu_span_ms,
        scalar_signal_ms=metric.scalar_cpu_span_ms,
        kernel_names=metric.kernel_names,
        runtime_names=metric.runtime_names,
        cpu_op_names=metric.cpu_op_names,
    )


def _build_rows(family_metrics: Dict[str, FamilyMetrics]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    breakdown_rows: List[Dict[str, object]] = []
    priority_rows: List[Dict[str, object]] = []

    for family, metric in family_metrics.items():
        top_kernel_name, top_kernel_ms = _top_name(metric.kernel_names)
        top_runtime_name, top_runtime_ms = _top_name(metric.runtime_names)
        top_cpu_name, top_cpu_ms = _top_name(metric.cpu_op_names)
        avg_kernel_ms = (
            metric.nested_kernel_total_ms / metric.nested_kernel_count if metric.nested_kernel_count else 0.0
        )
        (
            bottleneck,
            priority_score,
            dominant_event_name,
            dominant_event_ms,
            compute_signal_ms,
            sync_signal_ms,
            removable_signal_ms,
        ) = _classify_family(metric)

        breakdown_row = {
            "family": family,
            "node_count": metric.node_count,
            "user_annotation_node_count": metric.user_annotation_node_count,
            "gpu_annotation_node_count": metric.gpu_annotation_node_count,
            "user_annotation_ms": _round_metric(metric.user_annotation_ms),
            "gpu_annotation_ms": _round_metric(metric.gpu_annotation_ms),
            "cpu_op_span_ms": _round_metric(metric.cpu_op_span_ms),
            "conversion_cpu_span_ms": _round_metric(metric.conversion_cpu_span_ms),
            "scalar_cpu_span_ms": _round_metric(metric.scalar_cpu_span_ms),
            "runtime_span_ms": _round_metric(metric.runtime_span_ms),
            "sync_runtime_span_ms": _round_metric(metric.sync_runtime_span_ms),
            "nested_kernel_span_ms": _round_metric(metric.nested_kernel_span_ms),
            "nested_kernel_total_ms": _round_metric(metric.nested_kernel_total_ms),
            "nested_kernel_count": metric.nested_kernel_count,
            "avg_nested_kernel_ms": _round_metric(avg_kernel_ms),
            "runtime_count": metric.runtime_count,
            "launch_count": metric.launch_count,
            "launch_runtime_ms": _round_metric(metric.launch_runtime_ms),
            "gpu_memcpy_count": metric.gpu_memcpy_count,
            "gpu_memcpy_total_ms": _round_metric(metric.gpu_memcpy_total_ms),
            "top_kernel_name": top_kernel_name,
            "top_kernel_ms": _round_metric(top_kernel_ms),
            "top_runtime_name": top_runtime_name,
            "top_runtime_ms": _round_metric(top_runtime_ms),
            "top_cpu_op_name": top_cpu_name,
            "top_cpu_op_ms": _round_metric(top_cpu_ms),
        }
        breakdown_rows.append(breakdown_row)

        priority_rows.append(
            {
                "family": family,
                "bottleneck_type": bottleneck,
                "priority_score": _round_metric(priority_score),
                "compute_signal_ms": _round_metric(compute_signal_ms),
                "sync_signal_ms": _round_metric(sync_signal_ms),
                "removable_signal_ms": _round_metric(removable_signal_ms),
                "nested_kernel_span_ms": _round_metric(metric.nested_kernel_span_ms),
                "nested_kernel_total_ms": _round_metric(metric.nested_kernel_total_ms),
                "nested_kernel_count": metric.nested_kernel_count,
                "avg_nested_kernel_ms": _round_metric(avg_kernel_ms),
                "sync_runtime_span_ms": _round_metric(metric.sync_runtime_span_ms),
                "conversion_cpu_span_ms": _round_metric(metric.conversion_cpu_span_ms),
                "scalar_cpu_span_ms": _round_metric(metric.scalar_cpu_span_ms),
                "launch_count": metric.launch_count,
                "runtime_count": metric.runtime_count,
                "gpu_memcpy_total_ms": _round_metric(metric.gpu_memcpy_total_ms),
                "dominant_event_name": dominant_event_name,
                "dominant_event_ms": _round_metric(dominant_event_ms),
                "recommendation": BOTTLENECK_RECOMMENDATIONS[bottleneck],
            }
        )

    breakdown_rows.sort(key=lambda row: (-float(row["nested_kernel_total_ms"]), row["family"]))
    priority_rows.sort(key=lambda row: (-float(row["priority_score"]), row["family"]))
    for priority_rank, row in enumerate(priority_rows, start=1):
        row["priority_rank"] = priority_rank
    return breakdown_rows, priority_rows


def _build_node_rows(base_metrics: Dict[str, BaseMetrics]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    breakdown_rows: List[Dict[str, object]] = []
    priority_rows: List[Dict[str, object]] = []

    for base, metric in base_metrics.items():
        family = _family_of(base)
        annotation_category = _primary_annotation_category(metric.user_annotation_ms, metric.gpu_annotation_ms)
        cpu_op_span_ms = _merge_intervals(metric.cpu_op_intervals_us) / 1000.0
        conversion_cpu_span_ms = _merge_intervals(metric.conversion_intervals_us) / 1000.0
        scalar_cpu_span_ms = _merge_intervals(metric.scalar_intervals_us) / 1000.0
        runtime_span_ms = _merge_intervals(metric.runtime_intervals_us) / 1000.0
        sync_runtime_span_ms = _merge_intervals(metric.sync_runtime_intervals_us) / 1000.0
        nested_kernel_span_ms = _merge_intervals(metric.nested_kernel_intervals_us) / 1000.0
        avg_kernel_ms = metric.nested_kernel_total_ms / metric.nested_kernel_count if metric.nested_kernel_count else 0.0
        top_kernel_name, top_kernel_ms = _top_name(metric.kernel_name_totals)
        top_runtime_name, top_runtime_ms = _top_name(metric.runtime_name_totals)
        top_cpu_name, top_cpu_ms = _top_name(metric.cpu_op_name_totals)
        (
            bottleneck,
            priority_score,
            dominant_event_name,
            dominant_event_ms,
            compute_signal_ms,
            sync_signal_ms,
            removable_signal_ms,
        ) = _classify_signals(
            work_unit_count=1,
            nested_kernel_count=metric.nested_kernel_count,
            kernel_avg_ms=avg_kernel_ms,
            compute_signal_ms=nested_kernel_span_ms,
            sync_signal_ms=sync_runtime_span_ms,
            conversion_signal_ms=conversion_cpu_span_ms,
            scalar_signal_ms=scalar_cpu_span_ms,
            kernel_names=metric.kernel_name_totals,
            runtime_names=metric.runtime_name_totals,
            cpu_op_names=metric.cpu_op_name_totals,
        )

        short_node_name = base[len(GRAPH_NODE_PREFIX) :] if base.startswith(GRAPH_NODE_PREFIX) else base
        breakdown_rows.append(
            {
                "node_base": base,
                "node_name": short_node_name,
                "family": family,
                "annotation_category": annotation_category,
                "user_annotation_count": metric.user_annotation_count,
                "gpu_annotation_count": metric.gpu_annotation_count,
                "user_annotation_ms": _round_metric(metric.user_annotation_ms),
                "gpu_annotation_ms": _round_metric(metric.gpu_annotation_ms),
                "cpu_op_count": metric.cpu_op_count,
                "cpu_op_span_ms": _round_metric(cpu_op_span_ms),
                "conversion_cpu_span_ms": _round_metric(conversion_cpu_span_ms),
                "scalar_cpu_span_ms": _round_metric(scalar_cpu_span_ms),
                "runtime_count": metric.runtime_count,
                "runtime_span_ms": _round_metric(runtime_span_ms),
                "sync_runtime_span_ms": _round_metric(sync_runtime_span_ms),
                "launch_count": metric.launch_count,
                "launch_runtime_ms": _round_metric(metric.launch_runtime_ms),
                "nested_kernel_count": metric.nested_kernel_count,
                "nested_kernel_span_ms": _round_metric(nested_kernel_span_ms),
                "nested_kernel_total_ms": _round_metric(metric.nested_kernel_total_ms),
                "avg_nested_kernel_ms": _round_metric(avg_kernel_ms),
                "gpu_memcpy_count": metric.gpu_memcpy_count,
                "gpu_memcpy_total_ms": _round_metric(metric.gpu_memcpy_total_ms),
                "top_kernel_name": top_kernel_name,
                "top_kernel_ms": _round_metric(top_kernel_ms),
                "top_runtime_name": top_runtime_name,
                "top_runtime_ms": _round_metric(top_runtime_ms),
                "top_cpu_op_name": top_cpu_name,
                "top_cpu_op_ms": _round_metric(top_cpu_ms),
            }
        )
        priority_rows.append(
            {
                "node_base": base,
                "node_name": short_node_name,
                "family": family,
                "annotation_category": annotation_category,
                "bottleneck_type": bottleneck,
                "priority_score": _round_metric(priority_score),
                "compute_signal_ms": _round_metric(compute_signal_ms),
                "sync_signal_ms": _round_metric(sync_signal_ms),
                "removable_signal_ms": _round_metric(removable_signal_ms),
                "nested_kernel_span_ms": _round_metric(nested_kernel_span_ms),
                "nested_kernel_total_ms": _round_metric(metric.nested_kernel_total_ms),
                "nested_kernel_count": metric.nested_kernel_count,
                "avg_nested_kernel_ms": _round_metric(avg_kernel_ms),
                "sync_runtime_span_ms": _round_metric(sync_runtime_span_ms),
                "conversion_cpu_span_ms": _round_metric(conversion_cpu_span_ms),
                "scalar_cpu_span_ms": _round_metric(scalar_cpu_span_ms),
                "launch_count": metric.launch_count,
                "runtime_count": metric.runtime_count,
                "gpu_memcpy_total_ms": _round_metric(metric.gpu_memcpy_total_ms),
                "dominant_event_name": dominant_event_name,
                "dominant_event_ms": _round_metric(dominant_event_ms),
                "recommendation": BOTTLENECK_RECOMMENDATIONS[bottleneck],
            }
        )

    breakdown_rows.sort(key=lambda row: (-float(row["nested_kernel_total_ms"]), row["node_base"]))
    priority_rows.sort(key=lambda row: (-float(row["priority_score"]), row["node_base"]))
    for priority_rank, row in enumerate(priority_rows, start=1):
        row["priority_rank"] = priority_rank
    return breakdown_rows, priority_rows


def _write_csv(output_file: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(output_file: Path, rows: Sequence[Dict[str, object]], headers: Sequence[str]) -> None:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    output_file.write_text("\n".join(lines) + "\n")


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _print_console_summary(priority_rows: Sequence[Dict[str, object]], topk: int) -> None:
    limit = max(min(topk, len(priority_rows)), 0)
    print(f"Top {limit} trace-only family priorities:")
    for row in priority_rows[:limit]:
        print(
            f"[{row['priority_rank']:02d}] {row['family']}: bottleneck={row['bottleneck_type']} "
            f"score={float(row['priority_score']):.3f} kernel_span={float(row['nested_kernel_span_ms']):.3f} ms "
            f"sync_span={float(row['sync_runtime_span_ms']):.3f} ms conversion_span={float(row['conversion_cpu_span_ms']):.3f} ms "
            f"scalar_span={float(row['scalar_cpu_span_ms']):.3f} ms dominant={row['dominant_event_name']}"
        )


def _print_node_console_summary(priority_rows: Sequence[Dict[str, object]], topk: int) -> None:
    limit = max(min(topk, len(priority_rows)), 0)
    print(f"Top {limit} trace-only node priorities:")
    for row in priority_rows[:limit]:
        print(
            f"[{row['priority_rank']:02d}] {row['node_name']}: family={row['family']} bottleneck={row['bottleneck_type']} "
            f"score={float(row['priority_score']):.3f} kernel_span={float(row['nested_kernel_span_ms']):.3f} ms "
            f"sync_span={float(row['sync_runtime_span_ms']):.3f} ms conversion_span={float(row['conversion_cpu_span_ms']):.3f} ms "
            f"scalar_span={float(row['scalar_cpu_span_ms']):.3f} ms dominant={row['dominant_event_name']}"
        )


def _build_focus_family_rows(
    focus_family: str,
    node_breakdown_rows: Sequence[Dict[str, object]],
    node_priority_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    breakdown_by_base = {str(row["node_base"]): row for row in node_breakdown_rows}
    rows: List[Dict[str, object]] = []
    for priority_row in node_priority_rows:
        if str(priority_row.get("family", "")) != focus_family:
            continue
        node_base = str(priority_row["node_base"])
        breakdown_row = breakdown_by_base[node_base]
        sync_span_ms = float(priority_row["sync_runtime_span_ms"])
        kernel_span_ms = float(priority_row["nested_kernel_span_ms"])
        conversion_span_ms = float(priority_row["conversion_cpu_span_ms"])
        scalar_span_ms = float(priority_row["scalar_cpu_span_ms"])
        runtime_count = int(priority_row["runtime_count"])
        launch_count = int(priority_row["launch_count"])
        kernel_count = int(priority_row["nested_kernel_count"])
        row = dict(priority_row)
        row["sync_to_kernel_ratio"] = _round_metric(_safe_divide(sync_span_ms, kernel_span_ms))
        row["removable_to_kernel_ratio"] = _round_metric(_safe_divide(max(conversion_span_ms, scalar_span_ms), kernel_span_ms))
        row["launches_per_kernel"] = _round_metric(_safe_divide(float(launch_count), float(kernel_count)))
        row["runtime_per_launch"] = _round_metric(_safe_divide(float(runtime_count), float(max(launch_count, 1))))
        row["top_runtime_name"] = breakdown_row["top_runtime_name"]
        row["top_runtime_ms"] = breakdown_row["top_runtime_ms"]
        row["top_kernel_name"] = breakdown_row["top_kernel_name"]
        row["top_kernel_ms"] = breakdown_row["top_kernel_ms"]
        row["top_cpu_op_name"] = breakdown_row["top_cpu_op_name"]
        row["top_cpu_op_ms"] = breakdown_row["top_cpu_op_ms"]
        rows.append(row)
    for family_rank, row in enumerate(rows, start=1):
        row["focus_rank"] = family_rank
    return rows


def _focus_family_summary_lines(
    focus_family: str,
    focus_rows: Sequence[Dict[str, object]],
    base_metrics: Dict[str, BaseMetrics],
) -> List[str]:
    if not focus_rows:
        return [f"# Trace-Only Focus Report: {focus_family}", "", "No nodes matched this family."]

    filtered_metrics = [metric for base, metric in base_metrics.items() if _family_of(base) == focus_family]
    runtime_counter: Counter = Counter()
    kernel_counter: Counter = Counter()
    cpu_counter: Counter = Counter()
    for metric in filtered_metrics:
        runtime_counter.update(metric.runtime_name_totals)
        kernel_counter.update(metric.kernel_name_totals)
        cpu_counter.update(metric.cpu_op_name_totals)

    node_count = len(focus_rows)
    avg_priority_score = _safe_divide(sum(float(row["priority_score"]) for row in focus_rows), float(node_count))
    avg_sync_span_ms = _safe_divide(sum(float(row["sync_runtime_span_ms"]) for row in focus_rows), float(node_count))
    avg_kernel_span_ms = _safe_divide(sum(float(row["nested_kernel_span_ms"]) for row in focus_rows), float(node_count))
    avg_conversion_span_ms = _safe_divide(sum(float(row["conversion_cpu_span_ms"]) for row in focus_rows), float(node_count))
    avg_scalar_span_ms = _safe_divide(sum(float(row["scalar_cpu_span_ms"]) for row in focus_rows), float(node_count))
    avg_launch_count = _safe_divide(sum(float(row["launch_count"]) for row in focus_rows), float(node_count))
    avg_sync_to_kernel_ratio = _safe_divide(sum(float(row["sync_to_kernel_ratio"]) for row in focus_rows), float(node_count))
    dominant_runtime_name, dominant_runtime_ms = _top_name(runtime_counter)
    dominant_kernel_name, dominant_kernel_ms = _top_name(kernel_counter)
    dominant_cpu_name, dominant_cpu_ms = _top_name(cpu_counter)

    diagnosis = (
        f"The family is sync-bound in this trace: average sync span is {avg_sync_span_ms:.3f} ms per node, "
        f"while average nested kernel span is only {avg_kernel_span_ms:.3f} ms. "
        f"The dominant runtime event is {dominant_runtime_name or '<none>'} at {dominant_runtime_ms:.3f} ms total, "
        f"which indicates the CPU is mostly blocked waiting for device completion rather than spending time launching kernels."
    )
    if avg_launch_count > 0:
        diagnosis += (
            f" Each node still issues about {avg_launch_count:.2f} launches on average, so the path is not a single monolithic kernel; "
            f"it mixes real GPU work with explicit synchronization."
        )
    if avg_conversion_span_ms > 0.0 or avg_scalar_span_ms > 0.0:
        diagnosis += (
            f" Conversion or scalarization is secondary here: average conversion span is {avg_conversion_span_ms:.3f} ms and "
            f"average scalar span is {avg_scalar_span_ms:.3f} ms per node."
        )

    lines = [
        f"# Trace-Only Focus Report: {focus_family}",
        "",
        "## Summary",
        "",
        f"- Node count: {node_count}",
        f"- Average priority score: {avg_priority_score:.3f}",
        f"- Average sync span per node: {avg_sync_span_ms:.3f} ms",
        f"- Average nested kernel span per node: {avg_kernel_span_ms:.3f} ms",
        f"- Average sync to kernel ratio: {avg_sync_to_kernel_ratio:.3f}",
        f"- Average launches per node: {avg_launch_count:.3f}",
        f"- Dominant runtime event: {dominant_runtime_name} ({dominant_runtime_ms:.3f} ms total)",
        f"- Dominant kernel: {dominant_kernel_name} ({dominant_kernel_ms:.3f} ms total)",
        f"- Dominant CPU op: {dominant_cpu_name} ({dominant_cpu_ms:.3f} ms total)",
        "",
        "## Diagnosis",
        "",
        diagnosis,
        "",
        "## Top Nodes",
        "",
    ]
    return lines


def _write_focus_family_markdown(output_file: Path, summary_lines: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    headers = [
        "focus_rank",
        "node_name",
        "bottleneck_type",
        "priority_score",
        "sync_runtime_span_ms",
        "nested_kernel_span_ms",
        "sync_to_kernel_ratio",
        "launch_count",
        "top_runtime_name",
        "top_kernel_name",
        "top_cpu_op_name",
        "dominant_event_name",
    ]
    lines = list(summary_lines)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    output_file.write_text("\n".join(lines) + "\n")


def _print_focus_family_console_summary(focus_family: str, rows: Sequence[Dict[str, object]], topk: int) -> None:
    limit = max(min(topk, len(rows)), 0)
    if limit == 0:
        print(f"No nodes found for focus family: {focus_family}")
        return
    print(f"Top {limit} trace-only nodes in focus family {focus_family}:")
    for row in rows[:limit]:
        print(
            f"[{int(row['focus_rank']):02d}] {row['node_name']}: sync={float(row['sync_runtime_span_ms']):.3f} ms "
            f"kernel={float(row['nested_kernel_span_ms']):.3f} ms ratio={float(row['sync_to_kernel_ratio']):.3f} "
            f"launches={int(row['launch_count'])} top_runtime={row['top_runtime_name']}"
        )


def main() -> None:
    args = parse_args()
    trace_path = Path(args.trace).expanduser().resolve()
    if not trace_path.is_file():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else _default_output_dir(trace_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or _trace_stem(trace_path)

    events = _load_x_events(trace_path)
    base_metrics = _build_base_metrics(events)
    family_metrics = _aggregate_families(base_metrics)
    breakdown_rows, priority_rows = _build_rows(family_metrics)
    node_breakdown_rows, node_priority_rows = _build_node_rows(base_metrics)
    focus_family_rows: List[Dict[str, object]] = []

    breakdown_csv = output_dir / f"{prefix}_trace_only_family_breakdown.csv"
    priority_csv = output_dir / f"{prefix}_trace_only_family_priority.csv"
    priority_md = output_dir / f"{prefix}_trace_only_family_priority.md"
    node_breakdown_csv = output_dir / f"{prefix}_trace_only_node_breakdown.csv"
    node_priority_csv = output_dir / f"{prefix}_trace_only_node_priority.csv"
    node_priority_md = output_dir / f"{prefix}_trace_only_node_priority.md"
    focus_csv = None
    focus_md = None

    _write_csv(
        breakdown_csv,
        [
            "family",
            "node_count",
            "user_annotation_node_count",
            "gpu_annotation_node_count",
            "user_annotation_ms",
            "gpu_annotation_ms",
            "cpu_op_span_ms",
            "conversion_cpu_span_ms",
            "scalar_cpu_span_ms",
            "runtime_span_ms",
            "sync_runtime_span_ms",
            "nested_kernel_span_ms",
            "nested_kernel_total_ms",
            "nested_kernel_count",
            "avg_nested_kernel_ms",
            "runtime_count",
            "launch_count",
            "launch_runtime_ms",
            "gpu_memcpy_count",
            "gpu_memcpy_total_ms",
            "top_kernel_name",
            "top_kernel_ms",
            "top_runtime_name",
            "top_runtime_ms",
            "top_cpu_op_name",
            "top_cpu_op_ms",
        ],
        breakdown_rows,
    )
    _write_csv(
        priority_csv,
        [
            "priority_rank",
            "family",
            "bottleneck_type",
            "priority_score",
            "compute_signal_ms",
            "sync_signal_ms",
            "removable_signal_ms",
            "nested_kernel_span_ms",
            "nested_kernel_total_ms",
            "nested_kernel_count",
            "avg_nested_kernel_ms",
            "sync_runtime_span_ms",
            "conversion_cpu_span_ms",
            "scalar_cpu_span_ms",
            "launch_count",
            "runtime_count",
            "gpu_memcpy_total_ms",
            "dominant_event_name",
            "dominant_event_ms",
            "recommendation",
        ],
        priority_rows,
    )
    _write_csv(
        node_breakdown_csv,
        [
            "node_base",
            "node_name",
            "family",
            "annotation_category",
            "user_annotation_count",
            "gpu_annotation_count",
            "user_annotation_ms",
            "gpu_annotation_ms",
            "cpu_op_count",
            "cpu_op_span_ms",
            "conversion_cpu_span_ms",
            "scalar_cpu_span_ms",
            "runtime_count",
            "runtime_span_ms",
            "sync_runtime_span_ms",
            "launch_count",
            "launch_runtime_ms",
            "nested_kernel_count",
            "nested_kernel_span_ms",
            "nested_kernel_total_ms",
            "avg_nested_kernel_ms",
            "gpu_memcpy_count",
            "gpu_memcpy_total_ms",
            "top_kernel_name",
            "top_kernel_ms",
            "top_runtime_name",
            "top_runtime_ms",
            "top_cpu_op_name",
            "top_cpu_op_ms",
        ],
        node_breakdown_rows,
    )
    _write_csv(
        node_priority_csv,
        [
            "priority_rank",
            "node_base",
            "node_name",
            "family",
            "annotation_category",
            "bottleneck_type",
            "priority_score",
            "compute_signal_ms",
            "sync_signal_ms",
            "removable_signal_ms",
            "nested_kernel_span_ms",
            "nested_kernel_total_ms",
            "nested_kernel_count",
            "avg_nested_kernel_ms",
            "sync_runtime_span_ms",
            "conversion_cpu_span_ms",
            "scalar_cpu_span_ms",
            "launch_count",
            "runtime_count",
            "gpu_memcpy_total_ms",
            "dominant_event_name",
            "dominant_event_ms",
            "recommendation",
        ],
        node_priority_rows,
    )
    _write_markdown(
        priority_md,
        priority_rows,
        [
            "priority_rank",
            "family",
            "bottleneck_type",
            "priority_score",
            "nested_kernel_span_ms",
            "sync_runtime_span_ms",
            "conversion_cpu_span_ms",
            "scalar_cpu_span_ms",
            "avg_nested_kernel_ms",
            "launch_count",
            "dominant_event_name",
            "recommendation",
        ],
    )
    _write_markdown(
        node_priority_md,
        node_priority_rows,
        [
            "priority_rank",
            "node_name",
            "family",
            "annotation_category",
            "bottleneck_type",
            "priority_score",
            "nested_kernel_span_ms",
            "sync_runtime_span_ms",
            "conversion_cpu_span_ms",
            "scalar_cpu_span_ms",
            "avg_nested_kernel_ms",
            "launch_count",
            "dominant_event_name",
            "recommendation",
        ],
    )

    if args.focus_family:
        focus_family_rows = _build_focus_family_rows(args.focus_family, node_breakdown_rows, node_priority_rows)
        focus_family_rows = focus_family_rows[: max(args.focus_topk, 0)]
        focus_family_slug = args.focus_family.replace("/", "_")
        focus_csv = output_dir / f"{prefix}_trace_only_focus_{focus_family_slug}_top{max(args.focus_topk, 0)}.csv"
        focus_md = output_dir / f"{prefix}_trace_only_focus_{focus_family_slug}_top{max(args.focus_topk, 0)}.md"
        _write_csv(
            focus_csv,
            [
                "focus_rank",
                "priority_rank",
                "node_base",
                "node_name",
                "family",
                "annotation_category",
                "bottleneck_type",
                "priority_score",
                "compute_signal_ms",
                "sync_signal_ms",
                "removable_signal_ms",
                "nested_kernel_span_ms",
                "nested_kernel_total_ms",
                "nested_kernel_count",
                "avg_nested_kernel_ms",
                "sync_runtime_span_ms",
                "conversion_cpu_span_ms",
                "scalar_cpu_span_ms",
                "sync_to_kernel_ratio",
                "removable_to_kernel_ratio",
                "launch_count",
                "launches_per_kernel",
                "runtime_count",
                "runtime_per_launch",
                "gpu_memcpy_total_ms",
                "top_runtime_name",
                "top_runtime_ms",
                "top_kernel_name",
                "top_kernel_ms",
                "top_cpu_op_name",
                "top_cpu_op_ms",
                "dominant_event_name",
                "dominant_event_ms",
                "recommendation",
            ],
            focus_family_rows,
        )
        _write_focus_family_markdown(
            focus_md,
            _focus_family_summary_lines(args.focus_family, focus_family_rows, base_metrics),
            focus_family_rows,
        )

    _print_console_summary(priority_rows, args.topk)
    _print_node_console_summary(node_priority_rows, args.topk)
    if args.focus_family:
        _print_focus_family_console_summary(args.focus_family, focus_family_rows, min(args.topk, args.focus_topk))
    print(f"Saved family breakdown CSV: {breakdown_csv}")
    print(f"Saved family priority CSV: {priority_csv}")
    print(f"Saved family priority Markdown: {priority_md}")
    print(f"Saved node breakdown CSV: {node_breakdown_csv}")
    print(f"Saved node priority CSV: {node_priority_csv}")
    print(f"Saved node priority Markdown: {node_priority_md}")
    if focus_csv is not None and focus_md is not None:
        print(f"Saved focus family CSV: {focus_csv}")
        print(f"Saved focus family Markdown: {focus_md}")


if __name__ == "__main__":
    main()
