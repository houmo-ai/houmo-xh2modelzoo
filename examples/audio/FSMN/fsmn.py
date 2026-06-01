import argparse
from pathlib import Path

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

DEFAULT_MODEL_DIR = Path("/data02/datasets/funasr/FSMN")
DEFAULT_AUDIO_PATH = DEFAULT_MODEL_DIR / "example" / "vad_example.wav"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-revision", type=str, default="v2.0.4")
    parser.add_argument("--input", type=str, default=str(DEFAULT_AUDIO_PATH))
    parser.add_argument("--chunk-size", type=int, default=120000)
    parser.add_argument("--max-end-silence-time", type=int, default=None)
    args = parser.parse_args()

    inference_pipeline = pipeline(
        task=Tasks.voice_activity_detection,
        model=str(args.model_dir),
        model_revision=args.model_revision,
    )

    call_kwargs = {"input": args.input, "chunk_size": args.chunk_size}
    if args.max_end_silence_time is not None:
        call_kwargs["max_end_silence_time"] = args.max_end_silence_time

    segments_result = inference_pipeline(**call_kwargs)
    print(segments_result)


if __name__ == "__main__":
    main()