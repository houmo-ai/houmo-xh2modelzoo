"""Build mixed WikiText/MMLU calibration JSONL for MiniCPM5 AutoRound."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer


def _is_unsupported_verification_mode_error(error: TypeError) -> bool:
    message = str(error).lower()
    return "unexpected keyword argument" in message and "verification_mode" in message


def _load_calibration_dataset(
    *,
    dataset_name: str,
    dataset_config: str,
    split: str,
    cache_dir: str | None,
    required_fields: set[str],
):
    """Load a calibration dataset and report actionable cache/schema errors."""
    try:
        load_kwargs = {"split": split, "cache_dir": cache_dir}
        if dataset_name == "wikitext":
            load_kwargs["verification_mode"] = "no_checks"
        try:
            dataset = load_dataset(dataset_name, dataset_config, **load_kwargs)
        except TypeError as exc:
            if dataset_name != "wikitext" or not _is_unsupported_verification_mode_error(exc):
                raise
            # Older datasets versions do not expose verification_mode.
            load_kwargs.pop("verification_mode")
            dataset = load_dataset(dataset_name, dataset_config, **load_kwargs)
    except Exception as exc:
        cache_hint = f" (cache_dir={cache_dir!r})" if cache_dir else ""
        raise RuntimeError(
            f"Failed to load {dataset_name}/{dataset_config} split {split!r}{cache_hint}. "
            "Check network access or provide a cache_dir containing the dataset."
        ) from exc

    if dataset is None:
        raise ValueError(f"{dataset_name}/{dataset_config} split {split!r} returned no dataset")
    try:
        dataset_size = len(dataset)
    except TypeError:
        dataset_size = None
    if dataset_size == 0:
        raise ValueError(f"{dataset_name}/{dataset_config} split {split!r} is empty")

    columns = set(getattr(dataset, "column_names", ()) or ())
    missing = sorted(required_fields - columns)
    if missing:
        available = ", ".join(sorted(columns)) or "<none>"
        raise ValueError(
            f"{dataset_name}/{dataset_config} split {split!r} is missing required "
            f"fields: {', '.join(missing)}; available fields: {available}"
        )
    return dataset


def _render_mmlu_prompt(document: dict[str, Any], include_answer: bool = True) -> str:
    subject = str(document.get("subject", "general knowledge")).replace("_", " ")
    question = str(document["question"]).strip()
    choices = document["choices"]
    lines = [
        f"The following is a multiple choice question about {subject}.",
        "",
        question,
    ]
    lines.extend(f"{letter}. {str(choice).strip()}" for letter, choice in zip("ABCD", choices, strict=True))
    if include_answer and document.get("answer") is not None:
        answer = int(document["answer"])
        if answer not in range(4):
            raise ValueError(f"MMLU answer must be in [0, 3], got {answer}")
        lines.append(f"Answer: {'ABCD'[answer]}")
    else:
        lines.append("Answer:")
    return "\n".join(lines)


def _load_text_streams(
    *,
    cache_dir: str | None,
    mmlu_split: str,
    mmlu_samples: int,
    seed: int,
    include_answers: bool,
) -> tuple[str, str]:
    wikitext = _load_calibration_dataset(
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        split="train",
        cache_dir=cache_dir,
        required_fields={"text"},
    )
    wiki_rows = [str(row).strip() for row in wikitext["text"] if str(row).strip()]
    if not wiki_rows:
        raise ValueError("WikiText-2 train split contains no non-empty text")

    mmlu = _load_calibration_dataset(
        dataset_name="cais/mmlu",
        dataset_config="all",
        split=mmlu_split,
        cache_dir=cache_dir,
        required_fields={"subject", "question", "choices", "answer"},
    )
    mmlu_rows = [dict(row) for row in mmlu]
    if not mmlu_rows:
        raise ValueError(f"MMLU split {mmlu_split!r} contains no examples")
    rng = random.Random(seed)
    rng.shuffle(mmlu_rows)
    if mmlu_samples > 0:
        mmlu_rows = mmlu_rows[:mmlu_samples]

    mmlu_text = "\n\n".join(_render_mmlu_prompt(row, include_answers) for row in mmlu_rows)
    wiki_text = "\n\n".join(wiki_rows)
    return wiki_text, mmlu_text


def _build_interleaved_segments(
    *,
    tokenizer: Any,
    wiki_text: str,
    mmlu_text: str,
    segment_count: int,
    segment_length: int,
) -> list[str]:
    if segment_count < 1 or segment_length < 1:
        raise ValueError("segment_count and segment_length must be positive")

    streams = {
        "wikitext": list(tokenizer(wiki_text, add_special_tokens=False)["input_ids"]),
        "mmlu": list(tokenizer(mmlu_text, add_special_tokens=False)["input_ids"]),
    }
    offsets = {name: 0 for name in streams}
    segments = []
    for index in range(segment_count):
        source_name = "mmlu" if index % 2 == 0 else "wikitext"
        source_ids = streams[source_name]
        start = offsets[source_name]
        end = start + segment_length
        if end > len(source_ids):
            raise ValueError(
                f"{source_name} calibration stream has only {len(source_ids)} tokens; "
                f"need {end} tokens for segment {index + 1}"
            )
        segment_ids = source_ids[start:end]
        offsets[source_name] = end
        segments.append(
            tokenizer.decode(
                segment_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
    return segments


def build_calibration_jsonl(
    *,
    model_dir: str,
    output: Path,
    cache_dir: str | None,
    mmlu_split: str,
    mmlu_samples: int,
    segment_count: int,
    segment_length: int,
    seed: int,
    include_answers: bool,
) -> Path:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=True)
    wiki_text, mmlu_text = _load_text_streams(
        cache_dir=cache_dir,
        mmlu_split=mmlu_split,
        mmlu_samples=mmlu_samples,
        seed=seed,
        include_answers=include_answers,
    )
    segments = _build_interleaved_segments(
        tokenizer=tokenizer,
        wiki_text=wiki_text,
        mmlu_text=mmlu_text,
        segment_count=segment_count,
        segment_length=segment_length,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for segment in segments:
            stream.write(json.dumps({"text": segment}, ensure_ascii=False) + "\n")
    return output.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MiniCPM5 mixed AutoRound calibration JSONL.")
    parser.add_argument("--model-dir", required=True, help="MiniCPM5 HF model directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--mmlu-split", default="test", choices=("test", "validation", "auxiliary_train"))
    parser.add_argument("--mmlu-samples", type=int, default=128)
    parser.add_argument("--segment-count", type=int, default=8)
    parser.add_argument("--segment-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--without-answers", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_calibration_jsonl(
        model_dir=args.model_dir,
        output=args.output,
        cache_dir=args.cache_dir,
        mmlu_split=args.mmlu_split,
        mmlu_samples=args.mmlu_samples,
        segment_count=args.segment_count,
        segment_length=args.segment_length,
        seed=args.seed,
        include_answers=not args.without_answers,
    )
    print(f"Wrote {args.segment_count} calibration segments to {output}")


if __name__ == "__main__":
    main()
