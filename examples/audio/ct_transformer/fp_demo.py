import argparse
from pathlib import Path

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

DEFAULT_MODEL_DIR = Path("/data02/datasets/funasr/CT-Transformer")
DEFAULT_INPUT = Path(__file__).resolve().parent / "examples.txt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-revision", type=str, default="v2.0.4")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--split-size", type=int, default=20)
    args = parser.parse_args()

    inference_pipeline = pipeline(
        task=Tasks.punctuation,
        model=str(args.model_dir),
        model_revision=args.model_revision,
    )

    rec_result = inference_pipeline(args.input, split_size=args.split_size)
    print(rec_result)


if __name__ == "__main__":
    main()