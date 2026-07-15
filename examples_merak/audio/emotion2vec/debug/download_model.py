from __future__ import annotations

import argparse
from pathlib import Path


def download_model(output_dir: str, model_id: str = "iic/emotion2vec_plus_large") -> str:
    try:
        from modelscope import snapshot_download
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError("modelscope is required to download emotion2vec checkpoints") from exc

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    return snapshot_download(model_id, local_dir=str(target_dir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/models/emotion2vec_plus_large")
    parser.add_argument("--model-id", default="iic/emotion2vec_plus_large")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(download_model(args.output_dir, args.model_id))


if __name__ == "__main__":
    main()
