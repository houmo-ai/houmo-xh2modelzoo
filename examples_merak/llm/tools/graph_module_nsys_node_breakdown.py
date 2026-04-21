#!/usr/bin/env python3

import argparse
import csv
import re
import shutil
import sqlite3
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


NODE_RANGE_PREFIX = "graph_module_node::"
NVTX_PUSHPOP_EVENT_TYPE = 59
NVTX_STARTEND_EVENT_TYPE = 60
CUDA_API_VERSION_SUFFIX_RE = re.compile(r"(_v\d+|_(ptsz|ptds))$")


@dataclass
class TimeEvent:
    start_ns: int
    end_ns: int
    name: str

    @property
    def duration_ns(self) -> int:
        return max(self.end_ns - self.start_ns, 0)


@dataclass
class RuntimeEvent(TimeEvent):
    correlation_id: Optional[int] = None


@dataclass
class NodeInstance:
    instance_index: int
    nvtx_range: str
    start_ns: int
    end_ns: int
    global_tid: int
    pid: int
    tid: int
    node_index: Optional[int] = None
    node_name: str = ""
    node_op: str = ""
    node_target: str = ""
    module_type: str = ""
    runtime_events: List[RuntimeEvent] = field(default_factory=list)
    osrt_events: List[TimeEvent] = field(default_factory=list)
    gpu_total_ns: int = 0
    gpu_kernel_total_ns: int = 0
    gpu_memcpy_total_ns: int = 0
    gpu_memset_total_ns: int = 0
    gpu_op_count: int = 0
    kernel_count: int = 0
    memcpy_count: int = 0
    memset_count: int = 0
    queue_total_ns: int = 0
    gpu_projected_start_ns: Optional[int] = None
    gpu_projected_end_ns: Optional[int] = None
    gpu_correlation_ids: set = field(default_factory=set)
    gpu_name_counter: Counter = field(default_factory=Counter)
    cpu_sample_counter: Counter = field(default_factory=Counter)
    cpu_sample_total: int = 0

    @property
    def wall_ns(self) -> int:
        return max(self.end_ns - self.start_ns, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--report", type=str, required=True, help="Path to a .nsys-rep or .sqlite report")
    parser.add_argument(
        "--sqlite",
        type=str,
        default="",
        help="Optional explicit SQLite export path. If missing, <report>.sqlite is used or exported on demand.",
    )
    parser.add_argument(
        "--node-manifest",
        type=str,
        default="",
        help="Optional GraphModule node manifest CSV produced by graph_module_nsight_profiler.py",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        help="Prefix for output CSV files. Defaults to the SQLite/report basename.",
    )
    parser.add_argument(
        "--nsys-bin",
        type=str,
        default="",
        help="Optional nsys executable used when exporting SQLite from a .nsys-rep file",
    )
    parser.add_argument(
        "--force-export",
        action="store_true",
        help="Force re-exporting the SQLite file even if it already exists",
    )
    return parser.parse_args()


def decode_pid(global_tid: int) -> int:
    return (global_tid // 0x1000000) % 0x1000000


def decode_tid(global_tid: int) -> int:
    return global_tid % 0x1000000


def ns_to_ms(value_ns: int) -> float:
    return float(value_ns) / 1_000_000.0


def ns_to_us(value_ns: int) -> float:
    return float(value_ns) / 1_000.0


def resolve_nsys_bin(explicit_nsys_bin: str) -> str:
    if explicit_nsys_bin:
        nsys_bin = Path(explicit_nsys_bin)
        if nsys_bin.is_file() and nsys_bin.exists():
            return str(nsys_bin)
        raise FileNotFoundError(f"nsys executable not found: {explicit_nsys_bin}")

    candidate = shutil.which("nsys")
    if candidate:
        return candidate

    raise FileNotFoundError(
        "Could not locate nsys. Pass --nsys-bin explicitly or generate the SQLite export ahead of time."
    )


def ensure_sqlite_export(
    report_path: Path,
    sqlite_path: Path,
    explicit_nsys_bin: str,
    force_export: bool,
) -> Path:
    if report_path.suffix == ".sqlite":
        return report_path

    if sqlite_path.exists() and not force_export:
        return sqlite_path

    nsys_bin = resolve_nsys_bin(explicit_nsys_bin)
    export_cmd = [
        nsys_bin,
        "export",
        "--type",
        "sqlite",
        "--force-overwrite=true",
        "--output",
        str(sqlite_path),
        str(report_path),
    ]
    subprocess.run(export_cmd, check=True)
    if not sqlite_path.exists():
        raise FileNotFoundError(f"Expected SQLite export was not created: {sqlite_path}")
    return sqlite_path


def connect_sqlite(sqlite_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    return conn


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1", (table_name,)
    ).fetchone()
    return row is not None


def build_in_clause(values: Sequence[int]) -> Tuple[str, List[int]]:
    placeholders = ",".join(["?"] * len(values))
    return placeholders, list(values)


def merge_intervals(intervals: Iterable[Tuple[int, int]]) -> int:
    merged_duration = 0
    sorted_intervals = sorted((start, end) for start, end in intervals if end > start)
    if not sorted_intervals:
        return 0

    current_start, current_end = sorted_intervals[0]
    for start, end in sorted_intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        merged_duration += current_end - current_start
        current_start, current_end = start, end
    merged_duration += current_end - current_start
    return merged_duration


def parse_nvtx_range(range_name: str) -> Tuple[str, str]:
    payload = range_name[len(NODE_RANGE_PREFIX) :] if range_name.startswith(NODE_RANGE_PREFIX) else range_name
    parts = payload.split("::", 1)
    node_name = parts[0]
    node_target = parts[1] if len(parts) > 1 else ""
    return node_name, node_target


def normalize_cuda_api_name(name: str) -> str:
    return CUDA_API_VERSION_SUFFIX_RE.sub("", name)


def load_manifest(node_manifest_path: str) -> Dict[str, Dict[str, str]]:
    if not node_manifest_path:
        return {}

    manifest_path = Path(node_manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Node manifest not found: {manifest_path}")

    manifest_by_range: Dict[str, Dict[str, str]] = {}
    with manifest_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            manifest_by_range[row["nvtx_range"]] = row
    return manifest_by_range


def load_node_ranges(conn: sqlite3.Connection, manifest_by_range: Dict[str, Dict[str, str]]) -> List[NodeInstance]:
    if not table_exists(conn, "NVTX_EVENTS"):
        raise RuntimeError("Nsight report does not contain NVTX_EVENTS. Rerun with --trace=nvtx.")

    query = """
    SELECT
        NVTX_EVENTS.start AS start_ns,
        NVTX_EVENTS.end AS end_ns,
        NVTX_EVENTS.globalTid AS global_tid,
        COALESCE(NVTX_EVENTS.text, nvtx_strings.value) AS range_name
    FROM NVTX_EVENTS
    LEFT JOIN StringIds AS nvtx_strings
        ON NVTX_EVENTS.textId = nvtx_strings.id
    WHERE NVTX_EVENTS.eventType IN (?, ?)
        AND NVTX_EVENTS.end IS NOT NULL
        AND COALESCE(NVTX_EVENTS.text, nvtx_strings.value) LIKE ?
    ORDER BY NVTX_EVENTS.globalTid, NVTX_EVENTS.start
    """
    rows = conn.execute(query, (NVTX_PUSHPOP_EVENT_TYPE, NVTX_STARTEND_EVENT_TYPE, f"{NODE_RANGE_PREFIX}%")).fetchall()
    if not rows:
        raise RuntimeError(
            "No graph_module_node NVTX ranges were found in the report. Use profile-mode full or node_profile_only."
        )

    node_instances: List[NodeInstance] = []
    for instance_index, row in enumerate(rows):
        range_name = row["range_name"]
        node_name, node_target = parse_nvtx_range(range_name)
        global_tid = int(row["global_tid"])
        manifest_row = manifest_by_range.get(range_name, {})
        node_index_value = manifest_row.get("node_index")
        node_instances.append(
            NodeInstance(
                instance_index=instance_index,
                nvtx_range=range_name,
                start_ns=int(row["start_ns"]),
                end_ns=int(row["end_ns"]),
                global_tid=global_tid,
                pid=decode_pid(global_tid),
                tid=decode_tid(global_tid),
                node_index=int(node_index_value) if node_index_value not in {None, ""} else None,
                node_name=manifest_row.get("node_name", node_name),
                node_op=manifest_row.get("node_op", ""),
                node_target=manifest_row.get("node_target", node_target),
                module_type=manifest_row.get("module_type", ""),
            )
        )

    return node_instances


def build_thread_index(node_instances: Sequence[NodeInstance]) -> Dict[int, List[NodeInstance]]:
    thread_index: Dict[int, List[NodeInstance]] = defaultdict(list)
    for node in node_instances:
        thread_index[node.global_tid].append(node)

    for node_list in thread_index.values():
        node_list.sort(key=lambda item: item.start_ns)
    return thread_index


def find_node_for_event(
    start_ns: int,
    end_ns: int,
    global_tid: int,
    thread_index: Dict[int, List[NodeInstance]],
) -> Optional[NodeInstance]:
    node_list = thread_index.get(global_tid)
    if not node_list:
        return None

    for node in node_list:
        if node.start_ns <= start_ns and end_ns <= node.end_ns:
            return node
        if start_ns < node.start_ns:
            break
    return None


def top_level_events(events: Sequence[TimeEvent]) -> List[TimeEvent]:
    ordered = sorted(events, key=lambda item: (item.start_ns, -item.end_ns))
    selected: List[TimeEvent] = []
    for event in ordered:
        if not selected:
            selected.append(event)
            continue
        previous = selected[-1]
        if event.start_ns >= previous.start_ns and event.end_ns <= previous.end_ns:
            continue
        selected.append(event)
    return selected


def load_runtime_events(
    conn: sqlite3.Connection,
    thread_index: Dict[int, List[NodeInstance]],
) -> Dict[int, Tuple[NodeInstance, RuntimeEvent]]:
    if not table_exists(conn, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        raise RuntimeError(
            "Nsight report does not contain CUPTI_ACTIVITY_KIND_RUNTIME. Rerun with --trace=cuda and "
            "--cuda-trace-all-apis=true."
        )

    tids = sorted(thread_index.keys())
    placeholders, bind_values = build_in_clause(tids)
    time_min = min(node.start_ns for nodes in thread_index.values() for node in nodes)
    time_max = max(node.end_ns for nodes in thread_index.values() for node in nodes)
    query = f"""
    SELECT
        CUPTI_ACTIVITY_KIND_RUNTIME.start AS start_ns,
        CUPTI_ACTIVITY_KIND_RUNTIME.end AS end_ns,
        CUPTI_ACTIVITY_KIND_RUNTIME.globalTid AS global_tid,
        CUPTI_ACTIVITY_KIND_RUNTIME.correlationId AS correlation_id,
        COALESCE(runtime_strings.value, '') AS api_name
    FROM CUPTI_ACTIVITY_KIND_RUNTIME
    LEFT JOIN StringIds AS runtime_strings
        ON CUPTI_ACTIVITY_KIND_RUNTIME.nameId = runtime_strings.id
    WHERE CUPTI_ACTIVITY_KIND_RUNTIME.start <= ?
        AND CUPTI_ACTIVITY_KIND_RUNTIME.end >= ?
        AND CUPTI_ACTIVITY_KIND_RUNTIME.globalTid IN ({placeholders})
    ORDER BY CUPTI_ACTIVITY_KIND_RUNTIME.globalTid, CUPTI_ACTIVITY_KIND_RUNTIME.start, CUPTI_ACTIVITY_KIND_RUNTIME.end
    """
    rows = conn.execute(query, [time_max, time_min, *bind_values]).fetchall()

    runtime_by_correlation: Dict[int, Tuple[NodeInstance, RuntimeEvent]] = {}
    for row in rows:
        start_ns = int(row["start_ns"])
        end_ns = int(row["end_ns"])
        global_tid = int(row["global_tid"])
        node = find_node_for_event(start_ns, end_ns, global_tid, thread_index)
        if node is None:
            continue
        runtime_event = RuntimeEvent(
            start_ns=start_ns,
            end_ns=end_ns,
            name=normalize_cuda_api_name(row["api_name"] or "[unknown_cuda_api]"),
            correlation_id=int(row["correlation_id"]) if row["correlation_id"] is not None else None,
        )
        node.runtime_events.append(runtime_event)
        if runtime_event.correlation_id and runtime_event.correlation_id not in runtime_by_correlation:
            runtime_by_correlation[runtime_event.correlation_id] = (node, runtime_event)

    return runtime_by_correlation


def load_gpu_ops(
    conn: sqlite3.Connection,
    runtime_by_correlation: Dict[int, Tuple[NodeInstance, RuntimeEvent]],
) -> None:
    if not runtime_by_correlation:
        return

    correlation_ids = sorted(runtime_by_correlation.keys())
    placeholders, bind_values = build_in_clause(correlation_ids)

    if table_exists(conn, "CUPTI_ACTIVITY_KIND_KERNEL"):
        kernel_query = f"""
        SELECT
            CUPTI_ACTIVITY_KIND_KERNEL.start AS start_ns,
            CUPTI_ACTIVITY_KIND_KERNEL.end AS end_ns,
            CUPTI_ACTIVITY_KIND_KERNEL.correlationId AS correlation_id,
            COALESCE(kernel_short.value, kernel_demangled.value, kernel_mangled.value, '[kernel]') AS op_name
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        LEFT JOIN StringIds AS kernel_short
            ON CUPTI_ACTIVITY_KIND_KERNEL.shortName = kernel_short.id
        LEFT JOIN StringIds AS kernel_demangled
            ON CUPTI_ACTIVITY_KIND_KERNEL.demangledName = kernel_demangled.id
        LEFT JOIN StringIds AS kernel_mangled
            ON CUPTI_ACTIVITY_KIND_KERNEL.mangledName = kernel_mangled.id
        WHERE CUPTI_ACTIVITY_KIND_KERNEL.correlationId IN ({placeholders})
        ORDER BY CUPTI_ACTIVITY_KIND_KERNEL.start
        """
        for row in conn.execute(kernel_query, bind_values):
            record_gpu_op(runtime_by_correlation, row, kind="kernel")

    if table_exists(conn, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        memcpy_query = f"""
        SELECT
            CUPTI_ACTIVITY_KIND_MEMCPY.start AS start_ns,
            CUPTI_ACTIVITY_KIND_MEMCPY.end AS end_ns,
            CUPTI_ACTIVITY_KIND_MEMCPY.correlationId AS correlation_id,
            COALESCE(memcpy_kind.label, memcpy_kind.name, 'Memcpy') AS op_name
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        LEFT JOIN ENUM_CUDA_MEMCPY_OPER AS memcpy_kind
            ON CUPTI_ACTIVITY_KIND_MEMCPY.copyKind = memcpy_kind.id
        WHERE CUPTI_ACTIVITY_KIND_MEMCPY.correlationId IN ({placeholders})
        ORDER BY CUPTI_ACTIVITY_KIND_MEMCPY.start
        """
        for row in conn.execute(memcpy_query, bind_values):
            record_gpu_op(runtime_by_correlation, row, kind="memcpy")

    if table_exists(conn, "CUPTI_ACTIVITY_KIND_MEMSET"):
        memset_query = f"""
        SELECT
            CUPTI_ACTIVITY_KIND_MEMSET.start AS start_ns,
            CUPTI_ACTIVITY_KIND_MEMSET.end AS end_ns,
            CUPTI_ACTIVITY_KIND_MEMSET.correlationId AS correlation_id,
            'Memset' AS op_name
        FROM CUPTI_ACTIVITY_KIND_MEMSET
        WHERE CUPTI_ACTIVITY_KIND_MEMSET.correlationId IN ({placeholders})
        ORDER BY CUPTI_ACTIVITY_KIND_MEMSET.start
        """
        for row in conn.execute(memset_query, bind_values):
            record_gpu_op(runtime_by_correlation, row, kind="memset")


def record_gpu_op(
    runtime_by_correlation: Dict[int, Tuple[NodeInstance, RuntimeEvent]],
    row: sqlite3.Row,
    kind: str,
) -> None:
    correlation_id = row["correlation_id"]
    if correlation_id is None:
        return
    mapping = runtime_by_correlation.get(int(correlation_id))
    if mapping is None:
        return
    owning_node, runtime_event = mapping

    start_ns = int(row["start_ns"])
    end_ns = int(row["end_ns"])
    duration_ns = max(end_ns - start_ns, 0)
    if duration_ns <= 0:
        return

    op_name = row["op_name"] or f"[{kind}]"
    owning_node.gpu_total_ns += duration_ns
    owning_node.gpu_op_count += 1
    owning_node.gpu_correlation_ids.add(int(correlation_id))
    owning_node.queue_total_ns += max(start_ns - runtime_event.end_ns, 0)
    owning_node.gpu_name_counter[op_name] += duration_ns
    if owning_node.gpu_projected_start_ns is None or start_ns < owning_node.gpu_projected_start_ns:
        owning_node.gpu_projected_start_ns = start_ns
    if owning_node.gpu_projected_end_ns is None or end_ns > owning_node.gpu_projected_end_ns:
        owning_node.gpu_projected_end_ns = end_ns

    if kind == "kernel":
        owning_node.gpu_kernel_total_ns += duration_ns
        owning_node.kernel_count += 1
    elif kind == "memcpy":
        owning_node.gpu_memcpy_total_ns += duration_ns
        owning_node.memcpy_count += 1
    elif kind == "memset":
        owning_node.gpu_memset_total_ns += duration_ns
        owning_node.memset_count += 1


def load_osrt_events(conn: sqlite3.Connection, thread_index: Dict[int, List[NodeInstance]]) -> None:
    if not table_exists(conn, "OSRT_API"):
        return

    tids = sorted(thread_index.keys())
    placeholders, bind_values = build_in_clause(tids)
    time_min = min(node.start_ns for nodes in thread_index.values() for node in nodes)
    time_max = max(node.end_ns for nodes in thread_index.values() for node in nodes)
    query = f"""
    SELECT
        OSRT_API.start AS start_ns,
        OSRT_API.end AS end_ns,
        OSRT_API.globalTid AS global_tid,
        COALESCE(osrt_strings.value, '[unknown_osrt]') AS api_name
    FROM OSRT_API
    LEFT JOIN StringIds AS osrt_strings
        ON OSRT_API.nameId = osrt_strings.id
    WHERE OSRT_API.start <= ?
        AND OSRT_API.end >= ?
        AND OSRT_API.globalTid IN ({placeholders})
    ORDER BY OSRT_API.globalTid, OSRT_API.start, OSRT_API.end
    """
    rows = conn.execute(query, [time_max, time_min, *bind_values]).fetchall()
    for row in rows:
        start_ns = int(row["start_ns"])
        end_ns = int(row["end_ns"])
        global_tid = int(row["global_tid"])
        node = find_node_for_event(start_ns, end_ns, global_tid, thread_index)
        if node is None:
            continue
        node.osrt_events.append(TimeEvent(start_ns=start_ns, end_ns=end_ns, name=row["api_name"]))


def load_cpu_samples(conn: sqlite3.Connection, thread_index: Dict[int, List[NodeInstance]]) -> None:
    if not table_exists(conn, "COMPOSITE_EVENTS") or not table_exists(conn, "SAMPLING_CALLCHAINS"):
        return

    tids = sorted(thread_index.keys())
    placeholders, bind_values = build_in_clause(tids)
    time_min = min(node.start_ns for nodes in thread_index.values() for node in nodes)
    time_max = max(node.end_ns for nodes in thread_index.values() for node in nodes)
    query = f"""
    SELECT
        COMPOSITE_EVENTS.start AS start_ns,
        COMPOSITE_EVENTS.globalTid AS global_tid,
        CASE WHEN COMPOSITE_EVENTS.cpuCycles IS NOT NULL AND COMPOSITE_EVENTS.cpuCycles > 0
            THEN COMPOSITE_EVENTS.cpuCycles
            ELSE 1
        END AS sample_weight,
        COALESCE(sample_strings.value, printf('0x%x', SAMPLING_CALLCHAINS.originalIP), '[unknown_symbol]') AS symbol_name
    FROM COMPOSITE_EVENTS
    JOIN SAMPLING_CALLCHAINS
        ON COMPOSITE_EVENTS.id = SAMPLING_CALLCHAINS.id
        AND SAMPLING_CALLCHAINS.stackDepth = 0
    LEFT JOIN StringIds AS sample_strings
        ON SAMPLING_CALLCHAINS.symbol = sample_strings.id
    WHERE COMPOSITE_EVENTS.start BETWEEN ? AND ?
        AND COMPOSITE_EVENTS.globalTid IN ({placeholders})
    ORDER BY COMPOSITE_EVENTS.globalTid, COMPOSITE_EVENTS.start
    """
    rows = conn.execute(query, [time_min, time_max, *bind_values]).fetchall()
    for row in rows:
        global_tid = int(row["global_tid"])
        timestamp_ns = int(row["start_ns"])
        node = find_node_for_event(timestamp_ns, timestamp_ns, global_tid, thread_index)
        if node is None:
            continue
        sample_weight = int(row["sample_weight"])
        node.cpu_sample_counter[row["symbol_name"]] += sample_weight
        node.cpu_sample_total += sample_weight


def gpu_projected_ns(node: NodeInstance) -> int:
    if node.gpu_projected_start_ns is None or node.gpu_projected_end_ns is None:
        return 0
    return max(node.gpu_projected_end_ns - node.gpu_projected_start_ns, 0)


def runtime_event_is_sync(event: RuntimeEvent) -> bool:
    api_name = event.name.lower()
    return any(token in api_name for token in ("synchronize", "waitevent", "waitvalue", "streamwaitevent"))


def runtime_event_is_launch_like(event: RuntimeEvent, corrid_has_gpu_op: bool) -> bool:
    api_name = event.name.lower()
    if corrid_has_gpu_op:
        return True
    return api_name.startswith(("cudalaunch", "culaunch", "cudamemcpy", "cumemcpy", "cudamemset", "cumemset")) or (
        "graphlaunch" in api_name or "launchhostfunc" in api_name
    )


def counter_top_name(counter: Counter) -> str:
    if not counter:
        return ""
    return counter.most_common(1)[0][0]


def weighted_counter_from_events(events: Sequence[TimeEvent]) -> Counter:
    counter: Counter = Counter()
    for event in events:
        counter[event.name] += event.duration_ns
    return counter


def finalize_node(node: NodeInstance) -> Dict[str, object]:
    runtime_events = top_level_events(node.runtime_events)
    osrt_events = top_level_events(node.osrt_events)
    gpu_span_ns = gpu_projected_ns(node)

    launch_api_total_ns = 0
    sync_api_total_ns = 0
    other_cuda_api_total_ns = 0
    launch_api_count = 0
    sync_api_count = 0
    other_cuda_api_count = 0
    api_name_counter: Counter = Counter()
    osrt_name_counter = weighted_counter_from_events(osrt_events)

    for event in runtime_events:
        event_duration_ns = event.duration_ns
        api_name_counter[event.name] += event_duration_ns
        if runtime_event_is_sync(event):
            sync_api_total_ns += event_duration_ns
            sync_api_count += 1
        elif runtime_event_is_launch_like(event, event.correlation_id in node.gpu_correlation_ids):
            launch_api_total_ns += event_duration_ns
            launch_api_count += 1
        else:
            other_cuda_api_total_ns += event_duration_ns
            other_cuda_api_count += 1

    cuda_api_total_ns = merge_intervals((event.start_ns, event.end_ns) for event in runtime_events)
    osrt_total_ns = merge_intervals((event.start_ns, event.end_ns) for event in osrt_events)
    cpu_compute_ns = max(node.wall_ns - gpu_span_ns - cuda_api_total_ns - osrt_total_ns, 0)

    avg_gpu_op_ns = int(node.gpu_total_ns / node.gpu_op_count) if node.gpu_op_count else 0
    avg_launch_api_ns = int(launch_api_total_ns / launch_api_count) if launch_api_count else 0

    gpu_ratio = gpu_span_ns / node.wall_ns if node.wall_ns else 0.0
    launch_ratio = launch_api_total_ns / node.wall_ns if node.wall_ns else 0.0
    sync_ratio = sync_api_total_ns / node.wall_ns if node.wall_ns else 0.0
    cpu_ratio = cpu_compute_ns / node.wall_ns if node.wall_ns else 0.0
    osrt_ratio = osrt_total_ns / node.wall_ns if node.wall_ns else 0.0

    dominant_reason = diagnose_node(
        gpu_ratio=gpu_ratio,
        launch_ratio=launch_ratio,
        cpu_ratio=cpu_ratio,
        osrt_ratio=osrt_ratio,
        gpu_op_count=node.gpu_op_count,
        avg_gpu_op_ns=avg_gpu_op_ns,
        gpu_span_ns=gpu_span_ns,
        launch_api_total_ns=launch_api_total_ns,
    )
    diagnosis = build_diagnosis(
        dominant_reason=dominant_reason,
        gpu_span_ns=gpu_span_ns,
        launch_api_total_ns=launch_api_total_ns,
        sync_api_total_ns=sync_api_total_ns,
        cpu_compute_ns=cpu_compute_ns,
        osrt_total_ns=osrt_total_ns,
        queue_total_ns=node.queue_total_ns,
        gpu_op_count=node.gpu_op_count,
        launch_api_count=launch_api_count,
        avg_gpu_op_ns=avg_gpu_op_ns,
        avg_launch_api_ns=avg_launch_api_ns,
        top_cpu_symbol=counter_top_name(node.cpu_sample_counter),
        top_osrt_name=counter_top_name(osrt_name_counter),
    )

    return {
        "instance_index": node.instance_index,
        "node_index": node.node_index if node.node_index is not None else "",
        "node_name": node.node_name,
        "node_op": node.node_op,
        "module_type": node.module_type,
        "node_target": node.node_target,
        "nvtx_range": node.nvtx_range,
        "pid": node.pid,
        "tid": node.tid,
        "start_ns": node.start_ns,
        "end_ns": node.end_ns,
        "wall_ms": ns_to_ms(node.wall_ns),
        "cuda_api_total_ms": ns_to_ms(cuda_api_total_ns),
        "launch_api_total_ms": ns_to_ms(launch_api_total_ns),
        "sync_api_total_ms": ns_to_ms(sync_api_total_ns),
        "other_cuda_api_ms": ns_to_ms(other_cuda_api_total_ns),
        "osrt_total_ms": ns_to_ms(osrt_total_ns),
        "cpu_compute_ms": ns_to_ms(cpu_compute_ns),
        "gpu_projected_ms": ns_to_ms(gpu_span_ns),
        "gpu_kernel_ms": ns_to_ms(node.gpu_kernel_total_ns),
        "gpu_memcpy_ms": ns_to_ms(node.gpu_memcpy_total_ns),
        "gpu_memset_ms": ns_to_ms(node.gpu_memset_total_ns),
        "queue_ms": ns_to_ms(node.queue_total_ns),
        "gpu_op_count": node.gpu_op_count,
        "kernel_count": node.kernel_count,
        "memcpy_count": node.memcpy_count,
        "memset_count": node.memset_count,
        "launch_api_count": launch_api_count,
        "sync_api_count": sync_api_count,
        "other_cuda_api_count": other_cuda_api_count,
        "avg_gpu_op_us": ns_to_us(avg_gpu_op_ns),
        "avg_launch_api_us": ns_to_us(avg_launch_api_ns),
        "gpu_projected_ratio": gpu_ratio,
        "launch_ratio": launch_ratio,
        "sync_ratio": sync_ratio,
        "cpu_compute_ratio": cpu_ratio,
        "osrt_ratio": osrt_ratio,
        "dominant_cuda_api": counter_top_name(api_name_counter),
        "dominant_gpu_op": counter_top_name(node.gpu_name_counter),
        "top_osrt_api": counter_top_name(osrt_name_counter),
        "top_cpu_symbol": counter_top_name(node.cpu_sample_counter),
        "cpu_sample_total": node.cpu_sample_total,
        "dominant_reason": dominant_reason,
        "diagnosis": diagnosis,
    }


def diagnose_node(
    gpu_ratio: float,
    launch_ratio: float,
    cpu_ratio: float,
    osrt_ratio: float,
    gpu_op_count: int,
    avg_gpu_op_ns: int,
    gpu_span_ns: int,
    launch_api_total_ns: int,
) -> str:
    if gpu_op_count == 0:
        return "cpu_host_dominant"

    if osrt_ratio >= max(gpu_ratio, launch_ratio, cpu_ratio) and osrt_ratio >= 0.20:
        return "cpu_host_dominant"

    if (
        launch_ratio >= 0.12
        and gpu_op_count >= 6
        and avg_gpu_op_ns <= 200_000
        and launch_api_total_ns >= max(int(gpu_span_ns * 0.25), 1)
        and launch_ratio >= cpu_ratio
    ):
        return "launch_api_dominant"

    if cpu_ratio >= max(gpu_ratio, launch_ratio) and cpu_ratio >= 0.30:
        return "cpu_host_dominant"

    if gpu_ratio >= max(launch_ratio, cpu_ratio) and gpu_ratio >= 0.25:
        return "gpu_kernel_dominant"

    if launch_ratio >= max(gpu_ratio, cpu_ratio) and launch_ratio >= 0.15:
        return "launch_api_dominant"

    if cpu_ratio >= 0.20 and cpu_ratio >= launch_ratio:
        return "cpu_host_dominant"

    return "mixed"


def build_diagnosis(
    dominant_reason: str,
    gpu_span_ns: int,
    launch_api_total_ns: int,
    sync_api_total_ns: int,
    cpu_compute_ns: int,
    osrt_total_ns: int,
    queue_total_ns: int,
    gpu_op_count: int,
    launch_api_count: int,
    avg_gpu_op_ns: int,
    avg_launch_api_ns: int,
    top_cpu_symbol: str,
    top_osrt_name: str,
) -> str:
    if dominant_reason == "gpu_kernel_dominant":
        return (
            f"GPU span {ns_to_ms(gpu_span_ns):.3f} ms dominates; launch {ns_to_ms(launch_api_total_ns):.3f} ms, "
            f"sync {ns_to_ms(sync_api_total_ns):.3f} ms, queue {ns_to_ms(queue_total_ns):.3f} ms, "
            f"GPU ops {gpu_op_count}."
        )

    if dominant_reason == "launch_api_dominant":
        return (
            f"Launch APIs took {ns_to_ms(launch_api_total_ns):.3f} ms across {launch_api_count} submit calls; "
            f"avg launch {ns_to_us(avg_launch_api_ns):.1f} us, avg GPU op {ns_to_us(avg_gpu_op_ns):.1f} us."
        )

    if dominant_reason == "cpu_host_dominant":
        parts = [f"Host residual outside GPU span/CUDA APIs is {ns_to_ms(cpu_compute_ns):.3f} ms"]
        if osrt_total_ns > 0:
            parts.append(f"OSRT {ns_to_ms(osrt_total_ns):.3f} ms")
        if top_osrt_name:
            parts.append(f"top OSRT {top_osrt_name}")
        if top_cpu_symbol:
            parts.append(f"top CPU sample {top_cpu_symbol}")
        return "; ".join(parts) + "."

    return (
        f"Mixed node: GPU span {ns_to_ms(gpu_span_ns):.3f} ms, launch {ns_to_ms(launch_api_total_ns):.3f} ms, "
        f"sync {ns_to_ms(sync_api_total_ns):.3f} ms, host {ns_to_ms(cpu_compute_ns):.3f} ms."
    )


def build_summary_rows(trace_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, object]] = {}
    for row in trace_rows:
        key = str(row["nvtx_range"])
        if key not in grouped:
            grouped[key] = {
                "node_index": row["node_index"],
                "node_name": row["node_name"],
                "node_op": row["node_op"],
                "module_type": row["module_type"],
                "node_target": row["node_target"],
                "nvtx_range": key,
                "instance_count": 0,
                "total_wall_ms": 0.0,
                "total_cuda_api_total_ms": 0.0,
                "total_launch_api_total_ms": 0.0,
                "total_sync_api_total_ms": 0.0,
                "total_other_cuda_api_ms": 0.0,
                "total_osrt_total_ms": 0.0,
                "total_cpu_compute_ms": 0.0,
                "total_gpu_projected_ms": 0.0,
                "total_gpu_kernel_ms": 0.0,
                "total_gpu_memcpy_ms": 0.0,
                "total_gpu_memset_ms": 0.0,
                "total_queue_ms": 0.0,
                "total_gpu_op_count": 0,
                "total_kernel_count": 0,
                "total_memcpy_count": 0,
                "total_memset_count": 0,
                "total_launch_api_count": 0,
                "total_sync_api_count": 0,
                "total_other_cuda_api_count": 0,
                "cuda_api_counter": Counter(),
                "gpu_op_counter": Counter(),
                "osrt_counter": Counter(),
                "cpu_counter": Counter(),
                "diagnosis": row["diagnosis"],
            }

        summary = grouped[key]
        summary["instance_count"] += 1
        for numeric_key in (
            "wall_ms",
            "cuda_api_total_ms",
            "launch_api_total_ms",
            "sync_api_total_ms",
            "other_cuda_api_ms",
            "osrt_total_ms",
            "cpu_compute_ms",
            "gpu_projected_ms",
            "gpu_kernel_ms",
            "gpu_memcpy_ms",
            "gpu_memset_ms",
            "queue_ms",
        ):
            summary[f"total_{numeric_key}"] += float(row[numeric_key])
        for count_key in (
            "gpu_op_count",
            "kernel_count",
            "memcpy_count",
            "memset_count",
            "launch_api_count",
            "sync_api_count",
            "other_cuda_api_count",
        ):
            summary[f"total_{count_key}"] += int(row[count_key])
        if row["dominant_cuda_api"]:
            summary["cuda_api_counter"][row["dominant_cuda_api"]] += float(row["launch_api_total_ms"]) + float(
                row["sync_api_total_ms"]
            )
        if row["dominant_gpu_op"]:
            summary["gpu_op_counter"][row["dominant_gpu_op"]] += float(row["gpu_projected_ms"])
        if row["top_osrt_api"]:
            summary["osrt_counter"][row["top_osrt_api"]] += float(row["osrt_total_ms"])
        if row["top_cpu_symbol"]:
            summary["cpu_counter"][row["top_cpu_symbol"]] += int(row["cpu_sample_total"])

    summary_rows: List[Dict[str, object]] = []
    for summary in grouped.values():
        instance_count = int(summary["instance_count"])
        avg_gpu_op_us = 0.0
        if summary["total_gpu_op_count"]:
            avg_gpu_op_us = (
                float(summary["total_gpu_kernel_ms"])
                + float(summary["total_gpu_memcpy_ms"])
                + float(summary["total_gpu_memset_ms"])
            )
            avg_gpu_op_us = (avg_gpu_op_us * 1000.0) / int(summary["total_gpu_op_count"])
        avg_launch_api_us = 0.0
        if summary["total_launch_api_count"]:
            avg_launch_api_us = (float(summary["total_launch_api_total_ms"]) * 1000.0) / int(
                summary["total_launch_api_count"]
            )

        wall_ms = float(summary["total_wall_ms"])
        gpu_projected_ms = float(summary["total_gpu_projected_ms"])
        launch_ms = float(summary["total_launch_api_total_ms"])
        cpu_ms = float(summary["total_cpu_compute_ms"])
        osrt_ms = float(summary["total_osrt_total_ms"])
        dominant_reason = diagnose_node(
            gpu_ratio=(gpu_projected_ms / wall_ms) if wall_ms else 0.0,
            launch_ratio=(launch_ms / wall_ms) if wall_ms else 0.0,
            cpu_ratio=(cpu_ms / wall_ms) if wall_ms else 0.0,
            osrt_ratio=(osrt_ms / wall_ms) if wall_ms else 0.0,
            gpu_op_count=int(summary["total_gpu_op_count"]),
            avg_gpu_op_ns=int(avg_gpu_op_us * 1000.0),
            gpu_span_ns=int(gpu_projected_ms * 1_000_000.0),
            launch_api_total_ns=int(launch_ms * 1_000_000.0),
        )
        summary_rows.append(
            {
                "node_index": summary["node_index"],
                "node_name": summary["node_name"],
                "node_op": summary["node_op"],
                "module_type": summary["module_type"],
                "node_target": summary["node_target"],
                "nvtx_range": summary["nvtx_range"],
                "instance_count": instance_count,
                "total_wall_ms": wall_ms,
                "avg_wall_ms": wall_ms / instance_count if instance_count else 0.0,
                "avg_cuda_api_total_ms": float(summary["total_cuda_api_total_ms"]) / instance_count
                if instance_count
                else 0.0,
                "avg_launch_api_total_ms": launch_ms / instance_count if instance_count else 0.0,
                "avg_sync_api_total_ms": float(summary["total_sync_api_total_ms"]) / instance_count
                if instance_count
                else 0.0,
                "avg_other_cuda_api_ms": float(summary["total_other_cuda_api_ms"]) / instance_count
                if instance_count
                else 0.0,
                "avg_osrt_total_ms": osrt_ms / instance_count if instance_count else 0.0,
                "avg_cpu_compute_ms": cpu_ms / instance_count if instance_count else 0.0,
                "avg_gpu_projected_ms": gpu_projected_ms / instance_count if instance_count else 0.0,
                "avg_gpu_kernel_ms": float(summary["total_gpu_kernel_ms"]) / instance_count if instance_count else 0.0,
                "avg_gpu_memcpy_ms": float(summary["total_gpu_memcpy_ms"]) / instance_count if instance_count else 0.0,
                "avg_gpu_memset_ms": float(summary["total_gpu_memset_ms"]) / instance_count if instance_count else 0.0,
                "avg_queue_ms": float(summary["total_queue_ms"]) / instance_count if instance_count else 0.0,
                "avg_gpu_op_count": float(summary["total_gpu_op_count"]) / instance_count if instance_count else 0.0,
                "avg_launch_api_count": float(summary["total_launch_api_count"]) / instance_count
                if instance_count
                else 0.0,
                "avg_sync_api_count": float(summary["total_sync_api_count"]) / instance_count
                if instance_count
                else 0.0,
                "avg_gpu_op_us": avg_gpu_op_us,
                "avg_launch_api_us": avg_launch_api_us,
                "gpu_projected_ratio": (gpu_projected_ms / wall_ms) if wall_ms else 0.0,
                "launch_ratio": (launch_ms / wall_ms) if wall_ms else 0.0,
                "cpu_compute_ratio": (cpu_ms / wall_ms) if wall_ms else 0.0,
                "osrt_ratio": (osrt_ms / wall_ms) if wall_ms else 0.0,
                "dominant_cuda_api": counter_top_name(summary["cuda_api_counter"]),
                "dominant_gpu_op": counter_top_name(summary["gpu_op_counter"]),
                "top_osrt_api": counter_top_name(summary["osrt_counter"]),
                "top_cpu_symbol": counter_top_name(summary["cpu_counter"]),
                "dominant_reason": dominant_reason,
            }
        )

    summary_rows.sort(key=lambda row: sortable_node_index(row["node_index"], row["total_wall_ms"]))
    return summary_rows


def sortable_node_index(node_index: object, fallback_value: float) -> Tuple[int, int, float]:
    if node_index in {None, ""}:
        return (1, 0, -fallback_value)
    return (0, int(node_index), -fallback_value)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows were generated for output file: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    report_path = Path(args.report).resolve()
    sqlite_path = Path(args.sqlite).resolve() if args.sqlite else report_path.with_suffix(".sqlite")
    sqlite_path = ensure_sqlite_export(report_path, sqlite_path, args.nsys_bin, args.force_export)

    output_prefix = Path(args.output_prefix).resolve() if args.output_prefix else sqlite_path.with_suffix("")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    manifest_by_range = load_manifest(args.node_manifest)

    with connect_sqlite(sqlite_path) as conn:
        node_instances = load_node_ranges(conn, manifest_by_range)
        thread_index = build_thread_index(node_instances)
        runtime_by_correlation = load_runtime_events(conn, thread_index)
        load_gpu_ops(conn, runtime_by_correlation)
        load_osrt_events(conn, thread_index)
        load_cpu_samples(conn, thread_index)

    trace_rows = [finalize_node(node) for node in node_instances]
    trace_rows.sort(key=lambda row: sortable_node_index(row["node_index"], row["wall_ms"]))
    summary_rows = build_summary_rows(trace_rows)

    trace_csv = output_prefix.parent / f"{output_prefix.name}_node_breakdown_trace.csv"
    summary_csv = output_prefix.parent / f"{output_prefix.name}_node_breakdown_summary.csv"
    write_csv(trace_csv, trace_rows)
    write_csv(summary_csv, summary_rows)

    print(f"SQLite input: {sqlite_path}")
    print(f"Trace CSV: {trace_csv}")
    print(f"Summary CSV: {summary_csv}")
    print("Top node breakdown summary:")
    for row in sorted(summary_rows, key=lambda item: float(item["avg_wall_ms"]), reverse=True)[:10]:
        print(
            f"  node={row['node_name']} avg_wall={float(row['avg_wall_ms']):.3f} ms "
            f"gpu={float(row['avg_gpu_projected_ms']):.3f} ms launch={float(row['avg_launch_api_total_ms']):.3f} ms "
            f"cpu={float(row['avg_cpu_compute_ms']):.3f} ms reason={row['dominant_reason']}"
        )


if __name__ == "__main__":
    main()
