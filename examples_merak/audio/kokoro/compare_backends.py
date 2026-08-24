from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from xhmodel_merak.xh_other_model.models.kokoro.assets import resolve_model_assets
from xhmodel_merak.xh_other_model.models.kokoro.graph import GRAPH_ROLES
from xhmodel_merak.xh_other_model.models.kokoro.host import (
    DEFAULT_INPUT_IDS,
    load_voice_style,
)
from xhmodel_merak.xh_other_model.models.kokoro.runtime import (
    HmonnxRunner,
    KokoroStaticRuntime,
    OrtRunner,
    Runner,
)


def tensor_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference64 = np.asarray(reference, dtype=np.float64).reshape(-1)
    candidate64 = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if reference64.shape != candidate64.shape:
        raise ValueError(f"shape mismatch: reference={reference64.shape}, candidate={candidate64.shape}")
    difference = reference64 - candidate64
    denominator = float(np.linalg.norm(reference64) * np.linalg.norm(candidate64))
    return {
        "elements": int(reference64.size),
        "finite": bool(np.isfinite(candidate64).all()),
        "max_abs": float(np.max(np.abs(difference), initial=0.0)),
        "mean_abs": float(np.mean(np.abs(difference))) if difference.size else 0.0,
        "rmse": float(np.sqrt(np.mean(difference**2))) if difference.size else 0.0,
        "cosine": (float(np.dot(reference64, candidate64) / denominator) if denominator else None),
    }


def parse_hmonnx_overrides(
    values: list[str],
    export_dir: Path,
) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in values:
        role, separator, raw_path = value.partition("=")
        if not separator or role not in GRAPH_ROLES or not raw_path:
            raise ValueError(
                f"--hmonnx-override must be ROLE=PATH with ROLE in {', '.join(GRAPH_ROLES)}; got {value!r}"
            )
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = export_dir / path
        overrides[role] = path.resolve()
    return overrides


class ComparingRunner:
    """Measure an HMONNX graph with the exact feed consumed by the ORT path."""

    def __init__(self, reference: Runner, candidate: Runner) -> None:
        self.reference = reference
        self.candidate = candidate
        self.calls: list[dict[str, dict[str, Any]]] = []

    def run(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        reference = self.reference.run(feed)
        candidate = self.candidate.run(feed)
        if reference.keys() != candidate.keys():
            raise ValueError(f"output mismatch: reference={list(reference)}, candidate={list(candidate)}")
        self.calls.append({name: tensor_metrics(reference[name], candidate[name]) for name in reference})
        return reference


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Kokoro ORT and HMONNX graphs with identical inputs")
    parser.add_argument("--model-dir", required=True, help="source asset directory")
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--hmonnx-override",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help="replace one component artifact; repeat to override multiple roles",
    )
    parser.add_argument(
        "--ort-fallback",
        action="append",
        default=[],
        choices=GRAPH_ROLES,
        help="run this candidate role with static ONNX/ORT; repeat for multiple roles",
    )
    parser.add_argument("--output-json")
    args = parser.parse_args()

    export_dir = Path(args.export_dir).expanduser().resolve()
    overrides = parse_hmonnx_overrides(args.hmonnx_override, export_dir)
    ort_fallback = set(args.ort_fallback)
    conflict = sorted(ort_fallback.intersection(overrides))
    if conflict:
        raise ValueError(f"roles cannot use both HMONNX override and ORT fallback: {conflict}")
    metadata = json.loads((export_dir / "export_meta_info.json").read_text(encoding="utf-8"))
    assets = resolve_model_assets(args.model_dir)
    tokens = np.asarray(DEFAULT_INPUT_IDS, dtype=np.int32)
    style = load_voice_style(assets.voice, phoneme_count=tokens.size - 2)

    ort_runners: dict[str, Runner] = {}
    hmonnx_runners: dict[str, Runner] = {}
    comparing_runners: dict[str, ComparingRunner] = {}
    candidate_artifacts: dict[str, str] = {}
    for role in GRAPH_ROLES:
        component = metadata["components"][role]
        reference = OrtRunner(export_dir / component["onnx_file"])
        if role in ort_fallback:
            candidate_path = export_dir / component["onnx_file"]
            candidate: Runner = OrtRunner(candidate_path)
            candidate_backend = "onnxruntime-cpu"
        else:
            candidate_path = overrides.get(role, export_dir / component["hmonnx_file"])
            candidate = HmonnxRunner(
                candidate_path,
                device=args.device,
            )
            candidate_backend = "hmonnx"
        ort_runners[role] = reference
        hmonnx_runners[role] = candidate
        comparing_runners[role] = ComparingRunner(reference, candidate)
        candidate_artifacts[role] = f"{candidate_backend}:{candidate_path}"

    runtime_options = {
        "text_max_length": int(metadata["text_max_length"]),
        "frame_max_length": int(metadata["frame_max_length"]),
        "lstm_chunk_length": int(metadata["lstm_chunk_length"]),
        "seed": int(metadata["seed"]),
    }
    reference_runtime = KokoroStaticRuntime(comparing_runners, **runtime_options)
    reference_waveform, reference_synthesis = reference_runtime.synthesize(
        tokens,
        style,
        speed=args.speed,
        seed=args.seed,
    )
    candidate_runtime = KokoroStaticRuntime(hmonnx_runners, **runtime_options)
    candidate_waveform, candidate_synthesis = candidate_runtime.synthesize(
        tokens,
        style,
        speed=args.speed,
        seed=args.seed,
    )

    report = {
        "reference_backend": "onnxruntime-cpu",
        "candidate_backend": "hybrid" if ort_fallback else "hmonnx",
        "ort_fallback": sorted(ort_fallback),
        "candidate_artifacts": candidate_artifacts,
        "reference_synthesis": reference_synthesis,
        "candidate_synthesis": candidate_synthesis,
        "duration_exact": (reference_synthesis["duration"] == candidate_synthesis["duration"]),
        "waveform": tensor_metrics(reference_waveform, candidate_waveform),
        "local_graph_deltas": {
            role: {
                "calls": len(comparing_runners[role].calls),
                "steps": comparing_runners[role].calls,
            }
            for role in GRAPH_ROLES
        },
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
