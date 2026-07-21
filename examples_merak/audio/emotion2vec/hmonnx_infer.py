from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from xhmodel_merak.xh_llm.models.emotion2vec.configuration_emotion2vec import Emotion2vecModelMeta
from xhmodel_merak.xh_llm.models.emotion2vec.emotion2vec_hmonnx_inference import Emotion2vecHMONNXModel


def main() -> None:
    args = build_parser().parse_args()
    meta = Emotion2vecModelMeta.from_json_file(args.meta)
    model = Emotion2vecHMONNXModel(meta)
    result = model.extract_file(args.audio)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    np.save(Path(args.output) / "frame_features.npy", result["frame_features"].cpu().numpy())
    np.save(Path(args.output) / "utterance_feature.npy", result["utterance_feature"].cpu().numpy())
    if result["logits"] is not None:
        np.save(Path(args.output) / "logits.npy", result["logits"].cpu().numpy())
    np.save(Path(args.output) / "probabilities.npy", result["probabilities"].cpu().numpy())
    print(
        {
            "predicted_label": result["predicted_label"],
            "labels": result["labels"],
            "scores": result["scores"],
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    return parser


if __name__ == "__main__":
    main()
