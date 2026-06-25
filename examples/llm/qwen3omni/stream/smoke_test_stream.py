# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test Qwen3-Omni HMONNX stream artifacts end-to-end.

This script runs the streaming CLI with a small token budget and validates the
contract expected from vLLM-style realtime speech streaming:

* ``stream_events.jsonl`` exists.
* At least one ``thinker_token`` event is emitted.
* At least one ``audio_chunk`` event is emitted before ``complete``.
* The generated wav file exists when audio chunks were produced.

It is intentionally lightweight and delegates model/artifact loading to
``generate_stream.py``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def _run_generate(args) -> Path:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "generate_stream.py"),
        "--model",
        args.model,
        "--work-dir",
        args.work_dir,
        "--case",
        args.case,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--codec-chunk-frames",
        str(args.codec_chunk_frames),
        "--codec-left-context-frames",
        str(args.codec_left_context_frames),
        "--speaker",
        args.speaker,
    ]
    if args.device_map:
        command.extend(["--device-map", args.device_map])
    if args.debug:
        command.append("--debug")

    subprocess.run(command, check=True, cwd=str(SCRIPT_DIR.parent.parent.parent.parent))
    return Path(args.work_dir) / f"stream_output_{args.case}"


def _load_events(output_dir: Path) -> list[dict]:
    events_path = output_dir / "stream_events.jsonl"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing stream event log: {events_path}")
    events = []
    with open(events_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


def _validate(output_dir: Path, events: list[dict]):
    if not events:
        raise AssertionError("No stream events were produced")
    event_types = [event.get("type") for event in events]
    if "complete" not in event_types:
        raise AssertionError(f"Missing complete event. Event types: {event_types}")
    if "thinker_token" not in event_types:
        raise AssertionError(f"Missing thinker_token event. Event types: {event_types}")
    if "audio_chunk" not in event_types:
        raise AssertionError(f"Missing audio_chunk event. Event types: {event_types}")
    first_audio_idx = event_types.index("audio_chunk")
    complete_idx = event_types.index("complete")
    if first_audio_idx > complete_idx:
        raise AssertionError("audio_chunk was emitted after complete; stream is not realtime")
    wav_file = output_dir / "generated_audio.wav"
    if not wav_file.exists():
        raise FileNotFoundError(f"Missing generated audio: {wav_file}")


def main():
    parser = argparse.ArgumentParser(description="Smoke-test Qwen3-Omni HMONNX stream artifacts")
    parser.add_argument("--model", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--case", default="text", choices=["text", "vision", "audio", "multimodal"])
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--codec-chunk-frames", type=int, default=5)
    parser.add_argument("--codec-left-context-frames", type=int, default=2)
    parser.add_argument("--speaker", default="Ethan")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    output_dir = _run_generate(args)
    events = _load_events(output_dir)
    _validate(output_dir, events)
    print(f"Qwen3-Omni stream smoke test passed: {output_dir}")


if __name__ == "__main__":
    main()
