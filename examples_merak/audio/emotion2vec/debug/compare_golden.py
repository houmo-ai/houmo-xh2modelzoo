from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from xhmodel_merak.xh_llm.models.emotion2vec.configuration_emotion2vec import Emotion2vecModelMeta
from xhmodel_merak.xh_llm.models.emotion2vec.emotion2vec_hmonnx_inference import Emotion2vecHMONNXModel


def compare_golden(golden_dir: str, meta_path: str, audio: str) -> dict[str, float]:
    meta = Emotion2vecModelMeta.from_json_file(meta_path)
    model = Emotion2vecHMONNXModel(meta)
    result = model.extract_file(audio)
    frame_golden = np.load(Path(golden_dir) / "frame_features.npy")
    utterance_golden = np.load(Path(golden_dir) / "utterance_feature.npy")
    frame = result["frame_features"].cpu().numpy().astype(np.float64)
    utterance = result["utterance_feature"].cpu().numpy().astype(np.float64)
    frame_golden = frame_golden.astype(np.float64)
    utterance_golden = utterance_golden.astype(np.float64)
    frame_cos = float(
        np.dot(frame.flatten(), frame_golden.flatten()) / (np.linalg.norm(frame) * np.linalg.norm(frame_golden) + 1e-8)
    )
    utterance_cos = float(
        np.dot(utterance, utterance_golden) / (np.linalg.norm(utterance) * np.linalg.norm(utterance_golden) + 1e-8)
    )
    mae = float(np.mean(np.abs(frame - frame_golden)))
    max_abs = float(np.max(np.abs(frame - frame_golden)))
    return {"frame_cos": frame_cos, "utterance_cos": utterance_cos, "mae": mae, "max_abs": max_abs}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden-dir", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--min-frame-cos", type=float, default=0.9)
    parser.add_argument("--min-utterance-cos", type=float, default=0.9)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = compare_golden(args.golden_dir, args.meta, args.audio)
    print(metrics)
    if metrics["frame_cos"] < args.min_frame_cos or metrics["utterance_cos"] < args.min_utterance_cos:
        raise SystemExit(
            "Golden validation failed: "
            f"frame_cos={metrics['frame_cos']:.6f} (required {args.min_frame_cos:.6f}), "
            f"utterance_cos={metrics['utterance_cos']:.6f} (required {args.min_utterance_cos:.6f})"
        )


if __name__ == "__main__":
    main()
