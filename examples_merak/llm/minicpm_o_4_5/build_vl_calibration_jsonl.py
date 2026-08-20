from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


try:
    import numpy as np
except ModuleNotFoundError:  # Keep the dataset builder usable outside the quantization env.
    np = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a balanced text calibration set from VL result files")
    parser.add_argument("--source", action="append", required=True, metavar="LABEL=JSON")
    parser.add_argument("--text-source", action="append", default=[], metavar="LABEL=JSONL")
    parser.add_argument("--samples-per-source", type=int, default=16)
    parser.add_argument(
        "--text-samples-per-source",
        type=int,
        default=None,
        help="Override the per-source count for --text-source entries",
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=None,
        help="Sample this many records globally, matching Qwen-VL's data_files/calib_samples flow",
    )
    parser.add_argument("--seed", type=int, default=1024, help="Deterministic sampling seed")
    parser.add_argument(
        "--sampling-mode",
        choices=("first", "random"),
        default="first",
        help="Per-source sampling mode; --total-samples always uses global random sampling",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_source(raw: str) -> tuple[str, Path]:
    label, separator, path = raw.partition("=")
    if not separator or not label or not path:
        raise ValueError(f"expected LABEL=JSON for --source, got {raw!r}")
    return label, Path(path)


def build_text(item: dict) -> str:
    prompt_parts: list[str] = []
    for content in item.get("struct") or []:
        content_type = content.get("type")
        if content_type == "image":
            prompt_parts.append("[Image]")
        elif content_type == "text":
            value = str(content.get("value") or "").strip()
            if value:
                prompt_parts.append(value)
    response = str(item.get("response") or "").strip()
    if not prompt_parts or not response:
        return ""
    return "User:\n" + "\n".join(prompt_parts) + "\nAssistant:\n" + response


def load_vl_records(path: Path) -> list[tuple[str, object, int]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    usable: list[tuple[str, object, int]] = []
    for item in records:
        text = build_text(item)
        if text:
            usable.append((text, item.get("index"), len(str(item.get("response") or ""))))
    return sorted(usable, key=lambda row: row[2], reverse=True)


def load_text_records(path: Path) -> list[tuple[str, object, int]]:
    records: list[tuple[str, object, int]] = []
    with path.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            payload = json.loads(line)
            text = str(payload.get("text") or "").strip()
            if text:
                records.append((text, payload.get("index", line_index), len(text)))
    return sorted(records, key=lambda row: row[2], reverse=True)


def main() -> None:
    args = parse_args()
    if args.samples_per_source <= 0:
        raise ValueError("--samples-per-source must be positive")
    if args.text_samples_per_source is not None and args.text_samples_per_source <= 0:
        raise ValueError("--text-samples-per-source must be positive")
    if args.total_samples is not None and args.total_samples <= 0:
        raise ValueError("--total-samples must be positive")

    sources: list[tuple[str, list[tuple[str, object, int]]]] = []
    for raw_source in args.source:
        label, path = parse_source(raw_source)
        usable = load_vl_records(path)
        if args.total_samples is None and len(usable) < args.samples_per_source:
            raise ValueError(f"source {label} has {len(usable)} usable records; need {args.samples_per_source}")
        sources.append((label, usable))
    for raw_source in args.text_source:
        label, path = parse_source(raw_source)
        usable = load_text_records(path)
        requested = args.text_samples_per_source or args.samples_per_source
        if args.total_samples is None and len(usable) < requested:
            raise ValueError(f"source {label} has {len(usable)} usable records; need {requested}")
        sources.append((label, usable))

    if not sources:
        raise ValueError("at least one --source or --text-source is required")

    rng = np.random.default_rng(args.seed) if np is not None else random.Random(args.seed)

    def choose_indices(length: int, count: int) -> list[int]:
        if np is not None:
            return [int(index) for index in rng.choice(length, size=count, replace=False)]
        return rng.sample(range(length), count)

    selected_sources: list[tuple[str, list[tuple[str, object, int]]]]
    if args.total_samples is not None:
        # Qwen-VL passes several data_files to VLLMCustomDataset and samples
        # calib_samples records from the concatenated dataset.  Keep that
        # behavior here rather than forcing an artificial per-source quota.
        combined = [
            (label, text, source_index, response_length)
            for label, records in sources
            for text, source_index, response_length in records
        ]
        # VLLMCustomDataset sorts the concatenated Qwen-VL records by assistant
        # response length before get_vllm_custom_data samples them.
        combined.sort(key=lambda row: row[3], reverse=True)
        if len(combined) < args.total_samples:
            raise ValueError(f"combined sources have {len(combined)} usable records; need {args.total_samples}")
        chosen = [combined[index] for index in choose_indices(len(combined), args.total_samples)]
        selected_sources = [
            (label, [(text, source_index, response_length)]) for label, text, source_index, response_length in chosen
        ]
        # Each selected item is already in the Qwen-VL global sampling order.
        output_rows = [
            {
                "text": text,
                "source": label,
                "source_index": source_index,
                "sample_index": sample_index,
                "sampling_seed": args.seed,
                "sampling_mode": "global_random",
            }
            for sample_index, (label, records) in enumerate(selected_sources)
            for text, source_index, _ in records
        ]
    else:
        selected_sources = []
        for source_index, (label, records) in enumerate(sources):
            requested = (
                args.text_samples_per_source
                if source_index >= len(args.source) and args.text_samples_per_source is not None
                else args.samples_per_source
            )
            if args.sampling_mode == "random":
                selected = [records[index] for index in choose_indices(len(records), requested)]
            else:
                selected = records[:requested]
            selected_sources.append((label, selected))
        output_rows = []
        for round_robin_index in range(max(len(records) for _, records in selected_sources)):
            for label, records in selected_sources:
                if round_robin_index >= len(records):
                    continue
                text, source_index, _ = records[round_robin_index]
                output_rows.append(
                    {
                        "text": text,
                        "source": label,
                        "source_index": source_index,
                        "round_robin_index": round_robin_index,
                        "sampling_seed": args.seed,
                        "sampling_mode": f"per_source_{args.sampling_mode}",
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for payload in output_rows:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    metadata = {
        "format": "minicpm_o_4_5_text_calibration_v2",
        "sampling_mode": output_rows[0]["sampling_mode"],
        "seed": args.seed,
        "total_samples": len(output_rows),
        "source_counts": {label: sum(1 for row in output_rows if row["source"] == label) for label, _ in sources},
        "input_sources": [{"label": label, "records": len(records)} for label, records in sources],
        "jsonl_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
