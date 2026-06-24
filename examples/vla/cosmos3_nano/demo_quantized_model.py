# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Run one fixed-sample demo against quantized Cosmos3-Nano artifacts.

Cosmos3-Nano is exported as multiple fixed-shape HMONNX/ONNX graphs, so there
is no single monolithic model file to call. This demo chooses one recorded
compare/runtime stage from the quantization summary, reruns it on its built-in
sample input, and prints the resulting report metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


COSMOS3_ROOT = Path(__file__).resolve().parent
REPO_ROOT = COSMOS3_ROOT.parents[3]
DEFAULT_QUANT_DIR = COSMOS3_ROOT / "data" / "quantized_all"
COMPONENT_SUMMARIES = {
    "reasoner": "quantize_reasoner_summary.json",
    "generator": "quantize_generator_summary.json",
    "forward_dynamics": "quantize_forward_dynamics_summary.json",
    "policy": "quantize_policy_summary.json",
}
DEMO_STAGE_KEYWORDS = {
    "reasoner": (
        "reasoner_e2e_prefill_kv_decode",
        "reasoner_e2e",
        "prefill_kv_compare",
        "decode_stack_compare",
    ),
    "generator": (
        "generator_vae_pipeline",
        "cfg_scheduler_loop",
        "generator_realpack",
        "vae_decoder_full_split",
    ),
    "forward_dynamics": (
        "fd_vae_pipeline",
        "fd_segmented_scheduler",
        "fd_scheduler_loop",
        "fd_realpack",
    ),
    "policy": (
        "policy_segmented_chain",
        "policy_backbone_head_chain",
        "policy_action_backbone",
        "policy_action_head",
    ),
}
METRIC_KEYS = {
    "cosine",
    "raw_action_cosine",
    "mean_abs_diff",
    "max_abs_diff",
    "mse",
    "raw_action_mse",
    "raw_action_endpoint_error",
    "pc_success",
    "avg_sum_reward",
    "avg_max_reward",
    "top1_match",
    "prefill_logits_cosine",
    "decode_logits_cosine",
    "final_latents_cosine",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--quant-dir", type=Path, default=DEFAULT_QUANT_DIR)
    parser.add_argument("--component", choices=tuple(COMPONENT_SUMMARIES) + ("auto",), default="auto")
    parser.add_argument("--stage", default="auto", help="Exact summary stage name, or 'auto'.")
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--keep-existing-report", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def command_to_string(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def resolve_quant_path(base: Path, value: Any) -> Path | None:
    if not value:
        return None
    try:
        path = Path(str(value))
    except Exception:
        return None
    if not path.is_absolute():
        path = base / path
    return path


def discover_component_summaries(quant_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    all_summary = quant_dir / "quantize_all_summary.json"
    if all_summary.exists():
        data = load_json(all_summary)
        if not isinstance(data, dict):
            print(f"[warn] quantize_all_summary.json is not a dict: {all_summary}")
        else:
            for item in data.get("results", []):
                if not isinstance(item, dict) or item.get("component") not in COMPONENT_SUMMARIES:
                    continue
                component = str(item["component"])
                resolved = resolve_quant_path(quant_dir, item.get("summary"))
                if resolved is None:
                    print(f"[warn] invalid summary path in quantize_all_summary for {component!r}: {item.get('summary')!r}")
                    continue
                if resolved.exists():
                    found[component] = resolved
                else:
                    print(f"[warn] summary path not found in quantize_all_summary for {component!r}: {resolved}")
    for component, filename in COMPONENT_SUMMARIES.items():
        for path in (quant_dir / component / filename, quant_dir / filename):
            if path.exists():
                found.setdefault(component, path)
    return found


def report_stage(stage: dict[str, Any]) -> bool:
    return bool(stage.get("reports")) and isinstance(stage.get("command"), list)


def stage_report_paths(stage: dict[str, Any], quant_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for path in stage.get("reports", []):
        resolved = resolve_quant_path(quant_dir, path)
        if resolved is not None:
            paths.append(resolved)
    return paths


def command_outputs_exist(stage: dict[str, Any], quant_dir: Path) -> bool:
    expected = [
        resolved
        for resolved in (
            resolve_quant_path(quant_dir, path) for path in stage.get("expected", [])
        )
        if resolved is not None
    ]
    reports = stage_report_paths(stage, quant_dir)
    return all(path.exists() for path in expected + reports)


def choose_stage(component: str, summary: dict[str, Any], wanted: str) -> dict[str, Any]:
    stages = [stage for stage in summary.get("stages", []) if isinstance(stage, dict) and report_stage(stage)]
    if wanted != "auto":
        for stage in stages:
            if stage.get("name") == wanted:
                return stage
        raise ValueError(f"stage {wanted!r} not found for component {component}")

    keywords = DEMO_STAGE_KEYWORDS.get(component, ())
    for keyword in keywords:
        for stage in stages:
            if keyword in str(stage.get("name", "")):
                return stage
    if stages:
        return stages[0]
    raise ValueError(f"no runnable report stage found for component {component}")


def choose_component(summaries: dict[str, Path], requested: str) -> str:
    if requested != "auto":
        if requested not in summaries:
            raise FileNotFoundError(f"summary for component {requested!r} not found")
        return requested
    for component in ("policy", "forward_dynamics", "generator", "reasoner"):
        if component in summaries:
            return component
    raise FileNotFoundError("no quantization summary found")


def collect_metrics(path: Path, quant_dir: Path) -> dict[str, Any]:
    resolved = resolve_quant_path(quant_dir, path)
    if resolved is None or not resolved.exists() or resolved.suffix != ".json":
        return {}
    try:
        data = load_json(resolved)
    except Exception:
        return {}
    metrics: dict[str, Any] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                name = f"{prefix}.{key}" if prefix else str(key)
                if key in METRIC_KEYS and isinstance(item, (int, float, bool)):
                    metrics[name] = item
                else:
                    visit(name, item)
        elif isinstance(value, list):
            for index, item in enumerate(value[:4]):
                visit(f"{prefix}[{index}]", item)

    visit("", data)
    return metrics


def run_demo_stage(stage: dict[str, Any], log_path: Path) -> int:
    command = [str(item) for item in stage["command"]]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=os.environ.copy(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.monotonic() - started
    print(f"[demo] returncode={process.returncode} elapsed_sec={elapsed:.2f}")
    print(f"[demo] log={log_path}")
    return int(process.returncode)


def main() -> None:
    args = parse_args()
    summaries = discover_component_summaries(args.quant_dir)
    component = choose_component(summaries, args.component)
    summary_path = summaries[component]
    summary = load_json(summary_path)
    stage = choose_stage(component, summary, args.stage)
    reports = stage_report_paths(stage, args.quant_dir)
    log_dir = args.log_dir or args.quant_dir / "demo_logs"
    log_path = log_dir / component / f"{stage.get('name', 'demo')}.log"

    print(f"[demo] component={component}")
    print(f"[demo] summary={summary_path}")
    print(f"[demo] stage={stage.get('name')}")
    if args.print_command:
        print(f"[demo] command={command_to_string([str(item) for item in stage['command']])}")

    if args.keep_existing_report and command_outputs_exist(stage, args.quant_dir):
        print("[demo] existing report/artifacts found; skip rerun")
        returncode = 0
    else:
        returncode = run_demo_stage(stage, log_path)

    payload = {
        "component": component,
        "stage": stage.get("name"),
        "returncode": returncode,
        "reports": [],
    }
    for report in reports:
        metrics = collect_metrics(report, args.quant_dir)
        payload["reports"].append({"path": str(report), "exists": report.exists(), "metrics": metrics})
        print(f"[report] {report} exists={report.exists()}")
        if metrics:
            for key, value in sorted(metrics.items()):
                print(f"  {key}: {value}")

    output = args.quant_dir / "demo_quantized_model_summary.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[summary] {output}")
    if returncode != 0:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
