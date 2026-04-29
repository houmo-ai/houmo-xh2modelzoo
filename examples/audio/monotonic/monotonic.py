import argparse
import json
import os
from pathlib import Path

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

DEFAULT_MODEL_DIR = Path("/data02/datasets/funasr/Monotonic")
DEFAULT_AUDIO_PATH = DEFAULT_MODEL_DIR / "example" / "asr_example.wav"
DEFAULT_TEXT_PATH = DEFAULT_MODEL_DIR / "example" / "text.txt"
DEFAULT_OUTPUT_DIR = Path("./tmp")
DEFAULT_DEVICE = "cuda:0"


def read_text_input(text: str | None, text_path: Path) -> str:
    if text is not None:
        return text.strip()
    return text_path.read_text(encoding="utf-8").strip()


def normalize_token_text(text: str) -> list[str]:
    if any(ch.isspace() for ch in text):
        tokens = [token for token in text.split() if token]
        return tokens
    compact = text.replace(" ", "").strip()
    return list(compact)


def collapse_spaces(text: str) -> str:
    return "".join(normalize_token_text(text))


def validate_timestamp_result(result, expected_text: str) -> dict:
    if not result:
        raise ValueError("empty result from speech_timestamp pipeline")

    item = result[0]
    predicted_text = item.get("text", "")
    timestamps = item.get("timestamp", [])
    expected_tokens = normalize_token_text(expected_text)

    if collapse_spaces(predicted_text) != collapse_spaces(expected_text):
        raise ValueError(
            f"text mismatch:\nexpected: {expected_text}\npredicted: {predicted_text}"
        )

    if len(timestamps) != len(expected_tokens):
        raise ValueError(
            f"timestamp/token count mismatch: {len(timestamps)} vs {len(expected_tokens)}"
        )

    previous_end = -1
    for index, span in enumerate(timestamps):
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            raise ValueError(f"invalid timestamp span at index {index}: {span}")
        start, end = int(span[0]), int(span[1])
        if start < 0 or end < start:
            raise ValueError(f"invalid timestamp ordering at index {index}: {span}")
        if start < previous_end:
            raise ValueError(
                f"timestamp overlap/non-monotonic at index {index}: prev_end={previous_end}, span={span}"
            )
        previous_end = end

    return {
        "text_match": True,
        "token_count": len(expected_tokens),
        "timestamp_count": len(timestamps),
        "first_span": timestamps[0] if timestamps else None,
        "last_span": timestamps[-1] if timestamps else None,
    }


def build_pipeline(model_dir: Path, model_revision: str, output_dir: Path):
    return pipeline(
        task=Tasks.speech_timestamp,
        model=str(model_dir),
        model_revision=model_revision,
        output_dir=str(output_dir),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-revision", type=str, default="v2.0.4")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO_PATH)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--text-path", type=Path, default=DEFAULT_TEXT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Respect explicit device choice while still allowing users to bind GPU 5 via CUDA_VISIBLE_DEVICES=5.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.device.split(":")[-1] if args.device.startswith("cuda:") else os.environ.get("CUDA_VISIBLE_DEVICES", ""))

    text = read_text_input(args.text, args.text_path)
    inference_pipeline = build_pipeline(args.model_dir, args.model_revision, args.output_dir)
    rec_result = inference_pipeline(input=(str(args.audio), text), data_type=("sound", "text"))

    print(json.dumps(rec_result, ensure_ascii=False, indent=2))

    if args.validate:
        summary = validate_timestamp_result(rec_result, text)
        print(json.dumps({"validation": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
