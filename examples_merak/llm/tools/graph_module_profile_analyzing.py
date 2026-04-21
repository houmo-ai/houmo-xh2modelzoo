import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _parse_float(value: str) -> float:
    return float(value) if value not in (None, "") else 0.0


def _parse_int(value: str) -> int:
    return int(value) if value not in (None, "") else 0


def _round_metric(value: float) -> float:
    return round(value, 6)


def _normalize_module_type(value: str) -> str:
    normalized = value.strip() if value else ""
    return normalized or "<unknown>"


def _discover_mode_summary_files(data_dir: Path) -> Tuple[Dict[str, Path], List[Tuple[str, Path]]]:
    mode_to_summary: Dict[str, Path] = {}
    missing_files: List[Tuple[str, Path]] = []
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue

        mode = child.name
        summary_file = child / f"node_profile_summary_{mode}.csv"
        if summary_file.is_file():
            mode_to_summary[mode] = summary_file
        else:
            missing_files.append((mode, summary_file))

    return mode_to_summary, missing_files


def _load_mode_summary(summary_file: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with summary_file.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                {
                    "node_name": row.get("node_name", ""),
                    "node_op": row.get("node_op", ""),
                    "module_type": row.get("module_type", ""),
                    "node_target": row.get("node_target", ""),
                    "call_count": _parse_int(row.get("call_count", "0")),
                    "avg_ms": _parse_float(row.get("avg_ms", "0")),
                    "total_ms": _parse_float(row.get("total_ms", "0")),
                }
            )
    return rows


def _build_metric_stats(entry: Dict[str, Any], modes: List[str], metric_name: str) -> Dict[str, Any]:
    pairs: List[Tuple[str, float]] = []
    for mode in modes:
        value = entry.get(f"{mode}_{metric_name}", "")
        if value != "":
            pairs.append((mode, float(value)))

    prefix = metric_name.replace("_ms", "")
    if not pairs:
        return {
            f"fastest_mode_by_{prefix}": "",
            f"slowest_mode_by_{prefix}": "",
            f"{metric_name}_min": "",
            f"{metric_name}_max": "",
            f"{metric_name}_spread": "",
            f"{metric_name}_ratio": "",
        }

    pairs.sort(key=lambda item: item[1])
    min_mode, min_value = pairs[0]
    max_mode, max_value = pairs[-1]
    ratio = ""
    if len(pairs) >= 2 and min_value > 0:
        ratio = _round_metric(max_value / min_value)

    return {
        f"fastest_mode_by_{prefix}": min_mode,
        f"slowest_mode_by_{prefix}": max_mode,
        f"{metric_name}_min": _round_metric(min_value),
        f"{metric_name}_max": _round_metric(max_value),
        f"{metric_name}_spread": _round_metric(max_value - min_value),
        f"{metric_name}_ratio": ratio,
    }


def _build_mode_overview(mode_to_rows: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    overview_rows: List[Dict[str, Any]] = []
    for mode in sorted(mode_to_rows):
        rows = mode_to_rows[mode]
        overview_rows.append(
            {
                "mode": mode,
                "node_count": len(rows),
                "call_count_sum": sum(row["call_count"] for row in rows),
                "avg_ms_sum": _round_metric(sum(row["avg_ms"] for row in rows)),
                "total_ms_sum": _round_metric(sum(row["total_ms"] for row in rows)),
            }
        )
    return overview_rows


def _build_module_type_breakdown_rows(mode_to_rows: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    breakdown_rows: List[Dict[str, Any]] = []
    for mode in sorted(mode_to_rows):
        source_rows = mode_to_rows[mode]
        mode_total_ms = sum(row["total_ms"] for row in source_rows)
        rows_by_module_type: Dict[str, Dict[str, Any]] = {}
        for source_row in source_rows:
            module_type = _normalize_module_type(source_row.get("module_type", ""))
            row = rows_by_module_type.setdefault(
                module_type,
                {
                    "mode": mode,
                    "module_type": module_type,
                    "node_count": 0,
                    "call_count_sum": 0,
                    "total_ms_sum": 0.0,
                },
            )
            row["node_count"] += 1
            row["call_count_sum"] += source_row["call_count"]
            row["total_ms_sum"] += source_row["total_ms"]

        mode_rows: List[Dict[str, Any]] = []
        for row in rows_by_module_type.values():
            total_ms_sum = row["total_ms_sum"]
            row["total_ms_sum"] = _round_metric(total_ms_sum)
            row["mode_total_ms_ratio"] = _round_metric(total_ms_sum / mode_total_ms) if mode_total_ms > 0 else 0.0
            mode_rows.append(row)

        mode_rows.sort(key=lambda row: (-float(row["total_ms_sum"]), row["module_type"]))
        breakdown_rows.extend(mode_rows)

    return breakdown_rows


def _build_node_comparison_rows(
    mode_to_rows: Dict[str, List[Dict[str, Any]]], sort_by: str
) -> Tuple[List[str], List[Dict[str, Any]]]:
    modes = sorted(mode_to_rows)
    rows_by_node: Dict[str, Dict[str, Any]] = {}
    for mode, mode_rows in mode_to_rows.items():
        for source_row in mode_rows:
            node_name = source_row["node_name"]
            row = rows_by_node.setdefault(
                node_name,
                {
                    "node_name": node_name,
                    "node_op": source_row["node_op"],
                    "module_type": source_row["module_type"],
                    "node_target": source_row["node_target"],
                },
            )

            if not row["node_op"] and source_row["node_op"]:
                row["node_op"] = source_row["node_op"]
            if not row["module_type"] and source_row["module_type"]:
                row["module_type"] = source_row["module_type"]
            if not row["node_target"] and source_row["node_target"]:
                row["node_target"] = source_row["node_target"]

            row[f"{mode}_call_count"] = source_row["call_count"]
            row[f"{mode}_avg_ms"] = _round_metric(source_row["avg_ms"])
            row[f"{mode}_total_ms"] = _round_metric(source_row["total_ms"])

    comparison_rows: List[Dict[str, Any]] = []
    for row in rows_by_node.values():
        present_modes: List[str] = []
        missing_modes: List[str] = []
        for mode in modes:
            avg_key = f"{mode}_avg_ms"
            total_key = f"{mode}_total_ms"
            call_key = f"{mode}_call_count"
            if avg_key not in row:
                row[avg_key] = ""
                row[total_key] = ""
                row[call_key] = ""
                missing_modes.append(mode)
            else:
                present_modes.append(mode)

        row["present_modes"] = len(present_modes)
        row["missing_modes"] = ",".join(missing_modes)
        row.update(_build_metric_stats(row, modes, "avg_ms"))
        row.update(_build_metric_stats(row, modes, "total_ms"))
        comparison_rows.append(row)

    def _sort_metric(row: Dict[str, Any], key: str) -> float:
        value = row.get(key, "")
        return float(value) if value != "" else -1.0

    comparison_rows.sort(
        key=lambda row: (
            -_sort_metric(row, sort_by),
            -_sort_metric(row, "total_ms_spread"),
            row["node_name"],
        )
    )
    return modes, comparison_rows


def _write_csv(output_file: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _build_comparison_fieldnames(modes: List[str]) -> List[str]:
    fieldnames = ["node_name", "node_op", "module_type", "node_target"]
    for mode in modes:
        fieldnames.extend([f"{mode}_avg_ms", f"{mode}_total_ms"])
    fieldnames.extend(
        [
            "total_ms_ratio",
        ]
    )
    return fieldnames


def _build_module_type_breakdown_fieldnames() -> List[str]:
    return ["mode", "module_type", "node_count", "call_count_sum", "total_ms_sum", "mode_total_ms_ratio"]


def _print_console_summary(
    modes: List[str],
    missing_files: List[Tuple[str, Path]],
    comparison_rows: List[Dict[str, Any]],
    overview_rows: List[Dict[str, Any]],
    module_type_breakdown_rows: List[Dict[str, Any]],
    sort_by: str,
    topk: int,
) -> None:
    print(f"Loaded {len(modes)} mode(s): {', '.join(modes)}")
    if missing_files:
        for mode, summary_file in missing_files:
            print(f"Skipped mode '{mode}': missing summary file {summary_file}")

    print("Mode overview:")
    for row in overview_rows:
        print(
            f"  {row['mode']}: node_count={row['node_count']} total_ms_sum={row['total_ms_sum']:.6f} "
            f"avg_ms_sum={row['avg_ms_sum']:.6f} call_count_sum={row['call_count_sum']}"
        )

    module_type_topk = min(topk, 5)
    print(f"Top {module_type_topk} module_type(s) by total_ms_sum per mode:")
    for mode in modes:
        mode_rows = [row for row in module_type_breakdown_rows if row["mode"] == mode][:module_type_topk]
        if not mode_rows:
            print(f"  {mode}: <no module_type rows>")
            continue

        print(f"  {mode}:")
        for row in mode_rows:
            print(
                f"    {row['module_type']}: total_ms_sum={row['total_ms_sum']:.6f} "
                f"ratio={row['mode_total_ms_ratio']:.2%} node_count={row['node_count']}"
            )

    print(f"Top {min(topk, len(comparison_rows))} node(s) sorted by {sort_by}:")
    for index, row in enumerate(comparison_rows[:topk], start=1):
        print(
            f"  [{index:03d}] {row['node_name']} avg_spread={row['avg_ms_spread']} "
            f"total_spread={row['total_ms_spread']} "
            f"fastest_avg={row['fastest_mode_by_avg']} slowest_avg={row['slowest_mode_by_avg']}"
        )


def main(args):
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data_dir does not exist or is not a directory: {data_dir}")

    mode_to_summary_file, missing_files = _discover_mode_summary_files(data_dir)
    if not mode_to_summary_file:
        raise FileNotFoundError(
            f"No node_profile_summary_<mode>.csv files were found under mode subdirectories of {data_dir}"
        )

    mode_to_rows = {mode: _load_mode_summary(summary_file) for mode, summary_file in mode_to_summary_file.items()}
    overview_rows = _build_mode_overview(mode_to_rows)
    module_type_breakdown_rows = _build_module_type_breakdown_rows(mode_to_rows)
    modes, comparison_rows = _build_node_comparison_rows(mode_to_rows, args.sort_by)

    comparison_output = Path(args.output) if args.output else data_dir / "node_profile_mode_comparison.csv"
    overview_output = (
        Path(args.overview_output) if args.overview_output else data_dir / "node_profile_mode_overview.csv"
    )
    module_type_output = (
        Path(args.module_type_output)
        if args.module_type_output
        else data_dir / "node_profile_mode_module_type_breakdown.csv"
    )
    topk_output = (
        Path(args.topk_output) if args.topk_output else data_dir / f"node_profile_mode_comparison_topk_{args.topk}.csv"
    )

    comparison_fieldnames = _build_comparison_fieldnames(modes)
    _write_csv(comparison_output, comparison_fieldnames, comparison_rows)
    _write_csv(overview_output, ["mode", "node_count", "call_count_sum", "avg_ms_sum", "total_ms_sum"], overview_rows)
    _write_csv(module_type_output, _build_module_type_breakdown_fieldnames(), module_type_breakdown_rows)
    _write_csv(topk_output, comparison_fieldnames, comparison_rows[: args.topk])

    _print_console_summary(
        modes,
        missing_files,
        comparison_rows,
        overview_rows,
        module_type_breakdown_rows,
        args.sort_by,
        args.topk,
    )
    print(f"Wrote mode comparison CSV to: {comparison_output}")
    print(f"Wrote mode overview CSV to: {overview_output}")
    print(f"Wrote module_type breakdown CSV to: {module_type_output}")
    print(f"Wrote top-k comparison CSV to: {topk_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=str, default="work_dirs/qwen3_8b_legacy_xh2a_2k_graph_profiler")
    parser.add_argument(
        "--sort-by",
        type=str,
        default="avg_ms_spread",
        choices=["avg_ms_spread", "total_ms_spread", "avg_ms_ratio", "total_ms_ratio"],
        help="metric used to sort compared nodes",
    )
    parser.add_argument("--topk", type=int, default=50, help="number of hottest comparison rows to export and print")
    parser.add_argument("--output", type=str, default=None, help="output CSV path for the full comparison table")
    parser.add_argument("--overview-output", type=str, default=None, help="output CSV path for the per-mode overview")
    parser.add_argument(
        "--module-type-output",
        type=str,
        default=None,
        help="output CSV path for the per-mode module_type total-ms breakdown",
    )
    parser.add_argument("--topk-output", type=str, default=None, help="output CSV path for the top-k comparison rows")
    args = parser.parse_args()
    main(args)
