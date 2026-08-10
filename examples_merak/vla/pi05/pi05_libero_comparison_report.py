#!/usr/bin/env python3
"""Build an auditable FP-versus-HMONNX LIBERO closed-loop report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_eval_files(
    paths: list[Path],
    *,
    label: str,
    expected_episodes: int,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]], float]:
    tasks: dict[int, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    total_eval_s = 0.0

    for path in paths:
        path = path.resolve()
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        aggregated = payload.get("aggregated", payload)
        per_task = aggregated.get("per_task")
        if not isinstance(per_task, list) or not per_task:
            raise ValueError(f"{path}: missing aggregated.per_task")

        source_task_ids: list[int] = []
        for record in per_task:
            task_id = int(record["task_id"])
            if task_id in tasks:
                raise ValueError(f"{label}: duplicate task_id {task_id}")
            if record.get("task_group") != "libero_object":
                raise ValueError(f"{path}: expected task_group libero_object")

            metrics = record.get("metrics", {})
            successes = metrics.get("successes")
            if not isinstance(successes, list) or len(successes) != expected_episodes:
                raise ValueError(
                    f"{path}: task {task_id} expected {expected_episodes} episodes, "
                    f"got {len(successes) if isinstance(successes, list) else 'invalid'}"
                )
            if any(not isinstance(value, bool) for value in successes):
                raise ValueError(f"{path}: task {task_id} has non-boolean successes")

            tasks[task_id] = {
                "successes": successes,
                "video_paths": metrics.get("video_paths", []),
                "source": str(path),
            }
            source_task_ids.append(task_id)

        overall = aggregated.get("overall", {})
        total_eval_s += float(overall.get("eval_s", 0.0))
        sources.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "task_ids": sorted(source_task_ids),
                "n_episodes": sum(len(tasks[task_id]["successes"]) for task_id in source_task_ids),
            }
        )

    return tasks, sources, total_eval_s


def _wilson_interval(successes: int, episodes: int, confidence: float = 0.95) -> list[float]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / episodes
    denominator = 1.0 + z * z / episodes
    center = (proportion + z * z / (2.0 * episodes)) / denominator
    radius = (
        z * math.sqrt(proportion * (1.0 - proportion) / episodes + z * z / (4.0 * episodes * episodes)) / denominator
    )
    return [100.0 * (center - radius), 100.0 * (center + radius)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "task_id",
        "episodes",
        "fp_successes",
        "fp_success_rate_pct",
        "hmonnx_successes",
        "hmonnx_success_rate_pct",
        "delta_pp",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    overall = report["overall"]
    lines = [
        "# LIBERO Object FP vs HMONNX Closed-Loop Report",
        "",
        "## Result",
        "",
        f"- FP reference: **{overall['fp_successes']}/{overall['episodes']} ({overall['fp_success_rate_pct']:.1f}%)**",
        f"- HMONNX W8A8: **{overall['hmonnx_successes']}/{overall['episodes']} "
        f"({overall['hmonnx_success_rate_pct']:.1f}%)**",
        f"- Delta: **{overall['delta_pp']:+.1f} percentage points** "
        f"({overall['hmonnx_successes'] - overall['fp_successes']:+d} episodes)",
        f"- Relative success retention: **{overall['relative_retention_pct']:.2f}%**",
        "",
        "## Per Task",
        "",
        "| Task | Episodes | FP | HMONNX | Delta |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["per_task"]:
        lines.append(
            f"| {row['task_id']} | {row['episodes']} | "
            f"{row['fp_successes']} ({row['fp_success_rate_pct']:.1f}%) | "
            f"{row['hmonnx_successes']} ({row['hmonnx_success_rate_pct']:.1f}%) | "
            f"{row['delta_pp']:+.1f} pp |"
        )
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            f"- Suite: `{report['protocol']['suite']}`",
            f"- Tasks: `{report['protocol']['task_ids']}`",
            f"- Episodes per task: `{report['protocol']['episodes_per_task']}`",
            f"- Start seed: `{report['protocol']['start_seed']}`",
            f"- Matching process partitions: `{report['protocol']['process_partitions']}`",
            "- Closed-loop policy noise follows the evaluator RNG stream; this is an end-to-end "
            "success-rate comparison, not a fixed-noise per-step numerical comparison.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot(path: Path, report: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    background = "#08111f"
    panel = "#101d2f"
    text = "#edf4ff"
    muted = "#9eb0c7"
    grid = "#2a3a50"
    fp_color = "#42d3b2"
    hmonnx_color = "#ffad66"

    rows = report["per_task"]
    overall = report["overall"]
    task_ids = [row["task_id"] for row in rows]
    fp_rates = [row["fp_success_rate_pct"] for row in rows]
    hmonnx_rates = [row["hmonnx_success_rate_pct"] for row in rows]

    figure = plt.figure(figsize=(15, 8.6), facecolor=background)
    layout = GridSpec(3, 4, figure=figure, height_ratios=[0.78, 2.5, 0.8], hspace=0.42, wspace=0.28)

    cards = [
        ("FP REFERENCE", f"{overall['fp_success_rate_pct']:.1f}%", f"{overall['fp_successes']}/200", fp_color),
        (
            "HMONNX W8A8",
            f"{overall['hmonnx_success_rate_pct']:.1f}%",
            f"{overall['hmonnx_successes']}/200",
            hmonnx_color,
        ),
        ("DELTA", f"{overall['delta_pp']:+.1f} pp", "HMONNX - FP", text),
        ("RETENTION", f"{overall['relative_retention_pct']:.2f}%", "relative success", text),
    ]
    for index, (title, value, detail, color) in enumerate(cards):
        axis = figure.add_subplot(layout[0, index])
        axis.set_facecolor(panel)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color(grid)
        axis.text(0.06, 0.76, title, color=muted, fontsize=10, fontweight="bold", transform=axis.transAxes)
        axis.text(0.06, 0.35, value, color=color, fontsize=24, fontweight="bold", transform=axis.transAxes)
        axis.text(0.06, 0.10, detail, color=muted, fontsize=9, transform=axis.transAxes)

    axis = figure.add_subplot(layout[1, :])
    axis.set_facecolor(panel)
    positions = list(range(len(task_ids)))
    width = 0.36
    fp_bars = axis.bar(
        [value - width / 2 for value in positions],
        fp_rates,
        width,
        label="FP reference",
        color=fp_color,
    )
    hmonnx_bars = axis.bar(
        [value + width / 2 for value in positions],
        hmonnx_rates,
        width,
        label="HMONNX W8A8",
        color=hmonnx_color,
    )
    axis.set_ylim(0, 108)
    axis.set_xticks(positions, [f"Task {task_id}" for task_id in task_ids], color=text)
    axis.set_ylabel("Success rate (%)", color=muted)
    axis.tick_params(axis="y", colors=muted)
    axis.grid(axis="y", color=grid, linewidth=0.8, alpha=0.75)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color(grid)
    axis.legend(loc="lower left", frameon=False, labelcolor=text, ncols=2)
    axis.bar_label(fp_bars, fmt="%.0f%%", padding=3, color=fp_color, fontsize=9, fontweight="bold")
    axis.bar_label(
        hmonnx_bars,
        fmt="%.0f%%",
        padding=3,
        color=hmonnx_color,
        fontsize=9,
        fontweight="bold",
    )

    summary_axis = figure.add_subplot(layout[2, :])
    summary_axis.set_facecolor(panel)
    summary_axis.set_xticks([])
    summary_axis.set_yticks([])
    for spine in summary_axis.spines.values():
        spine.set_color(grid)
    deltas = [row["hmonnx_successes"] - row["fp_successes"] for row in rows]
    same = sum(delta == 0 for delta in deltas)
    higher = sum(delta > 0 for delta in deltas)
    lower = sum(delta < 0 for delta in deltas)
    fp_interval = overall["fp_wilson_95_pct"]
    hmonnx_interval = overall["hmonnx_wilson_95_pct"]
    summary_axis.text(
        0.025,
        0.62,
        f"TASK CONSISTENCY   {same} same  |  {higher} HMONNX higher  |  {lower} HMONNX lower",
        color=text,
        fontsize=12,
        fontweight="bold",
        transform=summary_axis.transAxes,
    )
    summary_axis.text(
        0.025,
        0.20,
        f"Wilson 95% CI   FP {fp_interval[0]:.1f}-{fp_interval[1]:.1f}%   |   "
        f"HMONNX {hmonnx_interval[0]:.1f}-{hmonnx_interval[1]:.1f}%",
        color=muted,
        fontsize=10,
        transform=summary_axis.transAxes,
    )
    summary_axis.text(
        0.975,
        0.20,
        "LIBERO Object | tasks 0-9 | 20 episodes/task | start seed 1000 | matching partitions",
        color=muted,
        fontsize=9,
        ha="right",
        transform=summary_axis.transAxes,
    )

    figure.suptitle(
        "PI0.5 LIBERO Closed-Loop Accuracy",
        x=0.06,
        y=0.985,
        ha="left",
        color=text,
        fontsize=20,
        fontweight="bold",
    )
    figure.text(
        0.06,
        0.947,
        "FP reference vs native-RMSNorm HMONNX CUDA Graph runtime",
        color=muted,
        fontsize=11,
    )
    figure.savefig(path, dpi=220, facecolor=background, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp-eval-info", action="append", type=Path, required=True)
    parser.add_argument("--hmonnx-eval-info", action="append", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--start-seed", type=int, default=1000)
    args = parser.parse_args()

    fp_tasks, fp_sources, fp_eval_s = _load_eval_files(
        args.fp_eval_info,
        label="fp",
        expected_episodes=args.episodes_per_task,
    )
    hmonnx_tasks, hmonnx_sources, hmonnx_eval_s = _load_eval_files(
        args.hmonnx_eval_info,
        label="hmonnx",
        expected_episodes=args.episodes_per_task,
    )
    if set(fp_tasks) != set(hmonnx_tasks):
        raise ValueError(f"task IDs differ: FP={sorted(fp_tasks)}, HMONNX={sorted(hmonnx_tasks)}")
    expected_task_ids = list(range(10))
    if sorted(fp_tasks) != expected_task_ids:
        raise ValueError(f"expected tasks {expected_task_ids}, got {sorted(fp_tasks)}")

    fp_partitions = sorted(tuple(source["task_ids"]) for source in fp_sources)
    hmonnx_partitions = sorted(tuple(source["task_ids"]) for source in hmonnx_sources)
    if fp_partitions != hmonnx_partitions:
        raise ValueError(f"process partitions differ: FP={fp_partitions}, HMONNX={hmonnx_partitions}")

    per_task: list[dict[str, Any]] = []
    fp_successes = 0
    hmonnx_successes = 0
    for task_id in expected_task_ids:
        fp_count = sum(fp_tasks[task_id]["successes"])
        hmonnx_count = sum(hmonnx_tasks[task_id]["successes"])
        episodes = args.episodes_per_task
        fp_successes += fp_count
        hmonnx_successes += hmonnx_count
        per_task.append(
            {
                "task_id": task_id,
                "episodes": episodes,
                "fp_successes": fp_count,
                "fp_success_rate_pct": 100.0 * fp_count / episodes,
                "hmonnx_successes": hmonnx_count,
                "hmonnx_success_rate_pct": 100.0 * hmonnx_count / episodes,
                "delta_pp": 100.0 * (hmonnx_count - fp_count) / episodes,
            }
        )

    total_episodes = args.episodes_per_task * len(expected_task_ids)
    fp_rate = 100.0 * fp_successes / total_episodes
    hmonnx_rate = 100.0 * hmonnx_successes / total_episodes
    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "report_version": 2,
        "protocol": {
            "suite": "libero_object",
            "task_ids": expected_task_ids,
            "episodes_per_task": args.episodes_per_task,
            "total_episodes_per_runtime": total_episodes,
            "start_seed": args.start_seed,
            "process_partitions": [list(partition) for partition in fp_partitions],
            "policy_noise": "evaluator process RNG stream",
        },
        "overall": {
            "episodes": total_episodes,
            "fp_successes": fp_successes,
            "fp_success_rate_pct": fp_rate,
            "fp_wilson_95_pct": _wilson_interval(fp_successes, total_episodes),
            "hmonnx_successes": hmonnx_successes,
            "hmonnx_success_rate_pct": hmonnx_rate,
            "hmonnx_wilson_95_pct": _wilson_interval(hmonnx_successes, total_episodes),
            "delta_pp": hmonnx_rate - fp_rate,
            "relative_retention_pct": 100.0 * hmonnx_rate / fp_rate,
            "fp_aggregate_eval_s": fp_eval_s,
            "hmonnx_aggregate_eval_s": hmonnx_eval_s,
        },
        "per_task": per_task,
        "sources": {"fp": fp_sources, "hmonnx": hmonnx_sources},
        "notes": [
            "FP and HMONNX use identical task partitions and start seed.",
            "The legacy sequential FP 177/200 run is intentionally excluded because its process RNG schedule differs.",
            (
                "Closed-loop success is an end-to-end metric; fixed-noise numerical parity is reported "
                "separately on DROID."
            ),
            (
                "Complete persisted-rollout video coverage and file-level provenance are emitted by "
                "pi05_libero_video_report.py."
            ),
        ],
    }

    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    png_path = output_prefix.with_suffix(".png")
    markdown_path = output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(csv_path, per_task)
    _write_markdown(markdown_path, report)
    _plot(png_path, report)

    print(json.dumps(report["overall"], indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
