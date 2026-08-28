from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from examples_merak.audio.kokoro.bucketed_inference_demo import _resolve_voice_path
from xhmodel_merak.xh_other_model.models.kokoro.bucketed_runtime import (
    BUCKETED_PRECISION_SPLIT_GRAPH_MODE,
    KokoroBucketedRuntime,
)
from xhmodel_merak.xh_other_model.models.kokoro.buckets import audio_seconds_to_frames
from xhmodel_merak.xh_other_model.models.kokoro.host import DEFAULT_INPUT_IDS, load_voice_style
from xhmodel_merak.xh_other_model.models.kokoro.runtime import OrtRunner, Runner


def tensor_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference64 = np.asarray(reference, dtype=np.float64).reshape(-1)
    candidate64 = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if reference64.shape != candidate64.shape:
        return {
            "reference_elements": int(reference64.size),
            "candidate_elements": int(candidate64.size),
            "shape_equal": False,
        }
    difference = np.abs(reference64 - candidate64)
    denominator = float(np.linalg.norm(reference64) * np.linalg.norm(candidate64))
    return {
        "elements": int(reference64.size),
        "shape_equal": True,
        "finite": bool(np.isfinite(candidate64).all()),
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()) if difference.size else 0.0,
        "p99_abs": float(np.percentile(difference, 99.0)) if difference.size else 0.0,
        "rmse": float(np.sqrt(np.mean(difference**2))) if difference.size else 0.0,
        "cosine": (float(np.dot(reference64, candidate64) / denominator) if denominator else None),
    }


class ComparingRunner:
    """Measure a candidate graph using the exact feed consumed by ORT."""

    def __init__(self, reference: Runner, candidate: Runner) -> None:
        self.reference = reference
        self.candidate = candidate
        self.calls: list[dict[str, dict[str, Any]]] = []

    def run(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        reference = self.reference.run(feed)
        candidate = self.candidate.run(feed)
        if reference.keys() != candidate.keys():
            raise ValueError(
                f"output mismatch: reference={list(reference)}, candidate={list(candidate)}"
            )
        self.calls.append(
            {name: tensor_metrics(reference[name], candidate[name]) for name in reference}
        )
        # Downstream graphs consume FP32 reference outputs. This isolates each
        # graph's local error instead of compounding errors from earlier graphs.
        return reference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare bucketed Kokoro ORT and HMONNX with identical per-graph inputs"
    )
    parser.add_argument("--export-dir", required=True)
    parser.add_argument(
        "--model-dir",
        help="legacy fallback for exports without an embedded NumPy voice pack",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lstm-variant", choices=("native", "decomposed"), default="native")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--token-bucket", type=int)
    parser.add_argument("--audio-seconds", type=int)
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_dir = Path(args.export_dir).expanduser().resolve()
    metadata = json.loads((export_dir / "export_meta_info.json").read_text(encoding="utf-8"))
    if metadata.get("graph_mode") != BUCKETED_PRECISION_SPLIT_GRAPH_MODE:
        raise ValueError(
            "compare_backends.py supports the production bucketed_precision_split export"
        )

    tokens = np.asarray(DEFAULT_INPUT_IDS, dtype=np.int32)
    voice_path = _resolve_voice_path(export_dir, metadata, model_dir=args.model_dir)
    style = load_voice_style(voice_path, phoneme_count=tokens.size - 2).numpy()
    forced_frame = None if args.audio_seconds is None else audio_seconds_to_frames(args.audio_seconds)

    candidate_runtime = KokoroBucketedRuntime(
        export_dir,
        metadata,
        backend="hmonnx",
        lstm_variant=args.lstm_variant,
        device=args.device,
    )
    comparison_runtime = KokoroBucketedRuntime(
        export_dir,
        metadata,
        backend="ort",
        lstm_variant=args.lstm_variant,
        device=args.device,
    )

    comparisons: dict[tuple[str, str], ComparingRunner] = {}
    candidate_artifacts: dict[str, str] = {}
    for role in metadata["component_order"]:
        for key, bucket in metadata["components"][role]["buckets"].items():
            reference_path = export_dir / bucket["onnx_file"]
            candidate = candidate_runtime._runner(str(role), str(key))
            comparing = ComparingRunner(OrtRunner(reference_path), candidate)
            comparison_runtime._runners[(str(role), str(key))] = comparing
            comparisons[(str(role), str(key))] = comparing
            if "hmonnx_variants" in bucket:
                candidate_entry = bucket["hmonnx_variants"][args.lstm_variant]
                candidate_file = candidate_entry.get("hmonnx_file", candidate_entry.get("status"))
            else:
                candidate_file = bucket.get("hmonnx_file", bucket["onnx_file"])
            candidate_artifacts[f"{role}/{key}"] = str(candidate_file)

    synthesize_kwargs = {
        "speed": args.speed,
        "seed": args.seed,
        "token_bucket": args.token_bucket,
        "frame_bucket": forced_frame,
    }
    reference_waveform, reference_synthesis = comparison_runtime.synthesize(
        tokens,
        style,
        **synthesize_kwargs,
    )
    candidate_waveform, candidate_synthesis = candidate_runtime.synthesize(
        tokens,
        style,
        **synthesize_kwargs,
    )

    local_graph_deltas = {
        f"{role}/{key}": {"calls": len(runner.calls), "steps": runner.calls}
        for (role, key), runner in comparisons.items()
        if runner.calls
    }
    report = {
        "reference_backend": "onnxruntime-cpu-fp32",
        "candidate_backend": "hmonnx",
        "lstm_variant": args.lstm_variant,
        "candidate_artifacts": candidate_artifacts,
        "reference_synthesis": reference_synthesis,
        "candidate_synthesis": candidate_synthesis,
        "route_equal": reference_synthesis.get("route") == candidate_synthesis.get("route"),
        "duration_exact": reference_synthesis.get("duration") == candidate_synthesis.get("duration"),
        "waveform": tensor_metrics(reference_waveform, candidate_waveform),
        "local_graph_deltas": local_graph_deltas,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output_json:
        output = Path(args.output_json).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(output)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
