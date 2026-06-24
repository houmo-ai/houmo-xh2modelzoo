# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Top-level Cosmos3-Nano quantization orchestration.

This script is intentionally thin: component scripts own graph construction,
quantization options, and smoke comparisons.  The top-level entrypoint only
orders components, passes common flags, and writes one combined manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


COSMOS3_ROOT = Path(__file__).resolve().parent
if str(COSMOS3_ROOT) not in sys.path:
    sys.path.insert(0, str(COSMOS3_ROOT))

from common.paths import default_model_root  # noqa: E402

DEFAULT_MODEL_ROOT = default_model_root()
COMPONENTS = ("reasoner", "generator", "forward_dynamics", "policy")


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    script: str
    summary_name: str


SPECS = {
    "reasoner": ComponentSpec("reasoner", "quantize_reasoner.py", "quantize_reasoner_summary.json"),
    "generator": ComponentSpec("generator", "quantize_generator.py", "quantize_generator_summary.json"),
    "forward_dynamics": ComponentSpec(
        "forward_dynamics",
        "quantize_forward_dynamics.py",
        "quantize_forward_dynamics_summary.json",
    ),
    "policy": ComponentSpec("policy", "quantize_policy.py", "quantize_policy_summary.json"),
}


def parse_csv(value: str, valid: tuple[str, ...]) -> list[str]:
    if value.strip().lower() == "all":
        return list(valid)
    items = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in items if item not in valid]
    if unknown:
        raise ValueError(f"unknown component(s): {unknown}; valid={valid}")
    return list(dict.fromkeys(items))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=COSMOS3_ROOT / "data" / "quantized_all")
    parser.add_argument("--components", default="all", help="Comma-separated subset or 'all'.")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--export-device", default="cpu")
    parser.add_argument("--runtime-device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--input-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--quant-type", default="w8a16_sefp")
    parser.add_argument("--vae-quant-type", default="w16a16_sefp")
    parser.add_argument("--sound-quant-type", default="w16a16_sefp")
    parser.add_argument("--policy-segments", default="", help="Optional policy segmented plan, e.g. 0:6,6:1,7:2.")
    parser.add_argument("--policy-fp-segments", default="", help="Policy segment ranges to keep FP ONNX.")
    parser.add_argument("--fd-segments", default="", help="Optional forward-dynamics segmented plan.")
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--latent-height", type=int, default=4)
    parser.add_argument("--latent-width", type=int, default=4)
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--skip-sound-tokenizer", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def command_to_string(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def component_out_dir(args: argparse.Namespace, component: str) -> Path:
    return args.out_dir / component


def build_command(args: argparse.Namespace, component: str) -> list[str]:
    spec = SPECS[component]
    command = [
        args.python,
        str(COSMOS3_ROOT / "quantize" / spec.script),
        "--model-root",
        str(args.model_root),
        "--out-dir",
        str(component_out_dir(args, component)),
        "--profile",
        args.profile,
        "--python",
        args.python,
        "--export-device",
        args.export_device,
        "--runtime-device",
        args.runtime_device,
        "--dtype",
        args.dtype,
        "--input-dtype",
        args.input_dtype,
        "--quant-type",
        args.quant_type,
        "--seed",
        str(args.seed),
    ]
    if component in {"generator", "forward_dynamics"}:
        command.extend(
            [
                "--vae-quant-type",
                args.vae_quant_type,
                "--frames",
                str(args.frames),
                "--latent-height",
                str(args.latent_height),
                "--latent-width",
                str(args.latent_width),
                "--num-inference-steps",
                str(args.num_inference_steps),
            ]
        )
    if component == "generator":
        command.extend(["--sound-quant-type", args.sound_quant_type])
        if args.skip_sound_tokenizer:
            command.append("--skip-sound-tokenizer")
    if component == "forward_dynamics" and args.fd_segments:
        command.extend(["--segments", args.fd_segments])
    if component == "policy":
        if args.policy_segments:
            command.extend(["--segments", args.policy_segments])
        if args.policy_fp_segments:
            command.extend(["--fp-segments", args.policy_fp_segments])
    if args.force:
        command.append("--force")
    if args.dry_run:
        command.append("--dry-run")
    if args.continue_on_error:
        command.append("--continue-on-error")
    if args.debug:
        command.append("--debug")
    return command


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def status_from_summary(summary: dict[str, Any] | None) -> str:
    if summary is None:
        return "missing_summary"
    stages = summary.get("stages")
    if not isinstance(stages, list):
        return "bad_summary"
    statuses = [stage.get("status") for stage in stages if isinstance(stage, dict)]
    if any(status == "failed" for status in statuses):
        return "failed"
    if statuses and all(status in {"done", "skipped", "dry_run"} for status in statuses):
        return "ok"
    return "unknown"


def main() -> None:
    args = parse_args()
    components = parse_csv(args.components, COMPONENTS)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    env = os.environ.copy()

    for component in components:
        spec = SPECS[component]
        command = build_command(args, component)
        log_path = log_dir / f"{component}.log"
        started = time.monotonic()
        print(f"[run] {component}")
        print(command_to_string(command))
        returncode = 0
        if not args.dry_run:
            with log_path.open("w", encoding="utf-8") as log_file:
                process = subprocess.run(
                    command,
                    cwd=str(COSMOS3_ROOT.parents[3]),
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            returncode = int(process.returncode)
        elapsed = time.monotonic() - started
        summary_path = component_out_dir(args, component) / spec.summary_name
        summary = load_json(summary_path)
        component_status = "dry_run" if args.dry_run else status_from_summary(summary)
        if returncode != 0:
            component_status = "failed"
        record = {
            "component": component,
            "status": component_status,
            "returncode": returncode,
            "elapsed_sec": elapsed,
            "command": command,
            "log": str(log_path),
            "summary": str(summary_path),
            "out_dir": str(component_out_dir(args, component)),
        }
        results.append(record)
        if returncode != 0 and not args.continue_on_error:
            break

    payload = {
        "model_root": str(args.model_root.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "profile": args.profile,
        "components": components,
        "quant_type": args.quant_type,
        "vae_quant_type": args.vae_quant_type,
        "sound_quant_type": args.sound_quant_type,
        "normalize_force_fp32": True,
        "results": results,
    }
    summary_path = args.out_dir / "quantize_all_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Cosmos3-Nano Full Quantization", "", "| Component | Status | Log | Summary |", "| --- | --- | --- | --- |"]
    for item in results:
        lines.append(f"| {item['component']} | {item['status']} | `{item['log']}` | `{item['summary']}` |")
    (args.out_dir / "quantize_all_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[summary] {summary_path}")

    if any(item["status"] == "failed" for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
