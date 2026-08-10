#!/usr/bin/env python3
"""Generate a customer-facing action comparison report from two JSONL files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


PROVENANCE_KEYS = ("sample_id", "dataset_index", "input_sha256", "noise_key", "noise_sha256")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_records(path: Path) -> dict[int, dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    return {int(record["sample_id"]): record for record in records}


def _actions(record: dict[str, Any]) -> np.ndarray:
    actions = np.asarray(record["actions"], dtype=np.float32)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2:
        raise ValueError(f"Expected [horizon, action_dim], got {actions.shape}")
    return actions


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Action shape mismatch: {reference.shape} != {candidate.shape}")
    difference = candidate - reference
    flat_reference = reference.reshape(-1)
    flat_candidate = candidate.reshape(-1)
    denominator = max(float(np.linalg.norm(flat_reference) * np.linalg.norm(flat_candidate)), 1e-12)
    return {
        "cosine": float(np.dot(flat_reference, flat_candidate) / denominator),
        "mae": float(np.abs(difference).mean()),
        "rmse": float(np.sqrt(np.square(difference).mean())),
        "max_abs": float(np.abs(difference).max()),
    }


def build_report(
    reference_path: Path,
    candidate_path: Path,
    cosine_threshold: float,
    mae_threshold: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    references = _load_records(reference_path)
    candidates = _load_records(candidate_path)
    sample_ids = sorted(references.keys() & candidates.keys())
    if not sample_ids:
        raise ValueError("The reference and candidate files have no matched sample IDs")

    sample_metrics = []
    reference_actions = []
    candidate_actions = []
    for sample_id in sample_ids:
        reference_record = references[sample_id]
        candidate_record = candidates[sample_id]
        for key in PROVENANCE_KEYS:
            if reference_record.get(key) != candidate_record.get(key):
                raise ValueError(
                    f"Sample {sample_id} provenance mismatch for {key}: "
                    f"{reference_record.get(key)!r} != {candidate_record.get(key)!r}"
                )
        reference = _actions(reference_record)
        candidate = _actions(candidate_record)
        metrics = _metrics(reference, candidate)
        sample_metrics.append(
            {
                "sample_id": sample_id,
                "dataset_index": int(reference_record["dataset_index"]),
                **metrics,
                "cosine_pass": metrics["cosine"] >= cosine_threshold,
                "mae_pass": metrics["mae"] <= mae_threshold,
            }
        )
        reference_actions.append(reference)
        candidate_actions.append(candidate)

    reference_array = np.stack(reference_actions)
    candidate_array = np.stack(candidate_actions)
    aggregate = _metrics(reference_array, candidate_array)
    aggregate.update(
        {
            "num_samples": len(sample_ids),
            "action_shape": list(reference_array.shape[1:]),
            "mean_sample_cosine": float(np.mean([item["cosine"] for item in sample_metrics])),
            "min_sample_cosine": float(np.min([item["cosine"] for item in sample_metrics])),
            "mean_sample_mae": float(np.mean([item["mae"] for item in sample_metrics])),
            "max_sample_mae": float(np.max([item["mae"] for item in sample_metrics])),
            "cosine_pass_count": sum(item["cosine_pass"] for item in sample_metrics),
            "mae_pass_count": sum(item["mae_pass"] for item in sample_metrics),
            "joint_pass_count": sum(item["cosine_pass"] and item["mae_pass"] for item in sample_metrics),
        }
    )

    per_dimension = []
    for dimension in range(reference_array.shape[-1]):
        metrics = _metrics(reference_array[..., dimension], candidate_array[..., dimension])
        per_dimension.append({"dimension": dimension, **metrics})

    latencies = np.asarray(
        [float(candidates[sample_id]["latency_ms_wall"]) for sample_id in sample_ids],
        dtype=np.float64,
    )
    first_runtime = candidates[sample_ids[0]].get("runtime")
    sessions = first_runtime.get("sessions", {}) if isinstance(first_runtime, dict) else {}
    captured_sessions = [name for name, status in sessions.items() if status.get("cuda_graph_captured")]
    report = {
        "title": "PI0.5 DROID HMONNX vs GT",
        "reference": {"path": str(reference_path), "sha256": _sha256(reference_path)},
        "candidate": {"path": str(candidate_path), "sha256": _sha256(candidate_path)},
        "thresholds": {"cosine": cosine_threshold, "mae": mae_threshold},
        "aggregate": aggregate,
        "latency_ms": {
            "first": float(latencies[0]),
            "steady_mean": float(latencies[1:].mean()) if len(latencies) > 1 else float(latencies[0]),
            "steady_min": float(latencies[1:].min()) if len(latencies) > 1 else float(latencies[0]),
            "steady_max": float(latencies[1:].max()) if len(latencies) > 1 else float(latencies[0]),
        },
        "cuda_graph": {
            "requested": first_runtime.get("requested") if isinstance(first_runtime, dict) else None,
            "captured_sessions": captured_sessions,
            "captured_count": len(captured_sessions),
            "session_count": len(sessions),
        },
        "per_dimension": per_dimension,
        "per_sample": sample_metrics,
    }
    return report, reference_array, candidate_array


def _write_csv(path: Path, sample_metrics: list[dict[str, Any]]) -> None:
    columns = ("sample_id", "dataset_index", "cosine", "mae", "rmse", "max_abs", "cosine_pass", "mae_pass")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(sample_metrics)


def _write_chart(
    path: Path,
    report: dict[str, Any],
    reference: np.ndarray,
    candidate: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregate = report["aggregate"]
    thresholds = report["thresholds"]
    sample_metrics = report["per_sample"]
    sample_ids = np.asarray([item["sample_id"] for item in sample_metrics])
    cosines = np.asarray([item["cosine"] for item in sample_metrics])
    maes = np.asarray([item["mae"] for item in sample_metrics])
    dimension_cosines = np.asarray([item["cosine"] for item in report["per_dimension"]])
    dimension_maes = np.asarray([item["mae"] for item in report["per_dimension"]])

    figure, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor="#f3f6fa")
    figure.subplots_adjust(top=0.77, hspace=0.33, wspace=0.25)
    figure.suptitle(report["title"], fontsize=22, fontweight="bold", y=0.965, color="#172033")
    figure.text(
        0.5,
        0.925,
        (
            f"{aggregate['num_samples']} fixed samples | action {aggregate['action_shape']} | "
            f"CUDA Graph {report['cuda_graph']['captured_count']}/{report['cuda_graph']['session_count']} sessions"
        ),
        ha="center",
        fontsize=11,
        color="#526077",
    )

    cards = (
        ("MEAN COSINE", f"{aggregate['mean_sample_cosine']:.6f}", "#176b87"),
        ("MIN COSINE", f"{aggregate['min_sample_cosine']:.6f}", "#7b4eaa"),
        ("MEAN MAE", f"{aggregate['mean_sample_mae']:.6f}", "#b65d32"),
        (
            "JOINT PASS",
            f"{aggregate['joint_pass_count']}/{aggregate['num_samples']}",
            "#237a57",
        ),
    )
    for index, (label, value, color) in enumerate(cards):
        x = 0.13 + index * 0.245
        figure.text(
            x,
            0.845,
            f"{label}\n{value}",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=color,
            bbox={"boxstyle": "round,pad=0.75", "facecolor": "white", "edgecolor": "#d8e0eb"},
        )

    cosine_axis, mae_axis, dimension_axis, scatter_axis = axes.reshape(-1)
    cosine_axis.plot(sample_ids, cosines, color="#176b87", linewidth=1.6, marker="o", markersize=3)
    cosine_axis.axhline(thresholds["cosine"], color="#d97843", linestyle="--", label="threshold")
    cosine_axis.axhline(aggregate["mean_sample_cosine"], color="#176b87", linestyle=":", label="mean")
    cosine_axis.set_title("Per-sample cosine similarity")
    cosine_axis.set_xlabel("Sample ID")
    cosine_axis.set_ylabel("Cosine")
    cosine_axis.set_ylim(min(0.9, cosines.min() - 0.01), 1.002)
    cosine_axis.legend(loc="lower left")

    mae_axis.plot(sample_ids, maes, color="#b65d32", linewidth=1.6, marker="o", markersize=3)
    mae_axis.axhline(thresholds["mae"], color="#d97843", linestyle="--", label="threshold")
    mae_axis.axhline(aggregate["mean_sample_mae"], color="#b65d32", linestyle=":", label="mean")
    mae_axis.set_title("Per-sample mean absolute error")
    mae_axis.set_xlabel("Sample ID")
    mae_axis.set_ylabel("MAE")
    mae_axis.set_ylim(0, max(maes.max() * 1.12, thresholds["mae"] * 1.5))
    mae_axis.legend(loc="upper left")

    dimensions = np.arange(len(dimension_cosines))
    bars = dimension_axis.bar(dimensions, dimension_cosines, color="#4d8eaa", alpha=0.9)
    dimension_axis.axhline(thresholds["cosine"], color="#d97843", linestyle="--")
    dimension_axis.set_title("Per-action-dimension cosine and MAE")
    dimension_axis.set_xlabel("Action dimension")
    dimension_axis.set_ylabel("Cosine")
    dimension_axis.set_ylim(min(0.9, dimension_cosines.min() - 0.01), 1.002)
    twin_axis = dimension_axis.twinx()
    twin_axis.plot(dimensions, dimension_maes, color="#b65d32", marker="D", linewidth=1.5, label="MAE")
    twin_axis.set_ylabel("MAE", color="#b65d32")
    for bar, value in zip(bars, dimension_cosines, strict=True):
        dimension_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value - 0.003,
            f"{value:.3f}",
            ha="center",
            va="top",
            fontsize=8,
            color="white",
        )

    flat_reference = reference.reshape(-1)
    flat_candidate = candidate.reshape(-1)
    scatter_axis.scatter(flat_reference, flat_candidate, s=5, alpha=0.18, color="#3d7f72", rasterized=True)
    lower = float(min(flat_reference.min(), flat_candidate.min()))
    upper = float(max(flat_reference.max(), flat_candidate.max()))
    scatter_axis.plot([lower, upper], [lower, upper], color="#d97843", linestyle="--", linewidth=1.4)
    scatter_axis.set_title("All action values: GT vs HMONNX")
    scatter_axis.set_xlabel("GT")
    scatter_axis.set_ylabel("HMONNX")
    scatter_axis.set_aspect("equal", adjustable="box")

    for axis in axes.reshape(-1):
        axis.set_facecolor("white")
        axis.grid(True, color="#e4e9f0", linewidth=0.8, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    figure.text(
        0.01,
        0.012,
        (
            f"Thresholds: cosine >= {thresholds['cosine']:.2f}, MAE <= {thresholds['mae']:.2f} | "
            f"Steady HMONNX latency: {report['latency_ms']['steady_mean']:.1f} ms/chunk"
        ),
        fontsize=9,
        color="#526077",
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--cosine-threshold", type=float, default=0.99)
    parser.add_argument("--mae-threshold", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_path = args.reference.resolve()
    candidate_path = args.candidate.resolve()
    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    report, reference, candidate = build_report(
        reference_path,
        candidate_path,
        args.cosine_threshold,
        args.mae_threshold,
    )
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    chart_path = output_prefix.with_suffix(".png")
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_csv(csv_path, report["per_sample"])
    _write_chart(chart_path, report, reference, candidate)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "chart": str(chart_path), **report["aggregate"]}))


if __name__ == "__main__":
    main()
