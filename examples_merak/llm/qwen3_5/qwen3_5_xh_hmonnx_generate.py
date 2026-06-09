"""Qwen3.5/Qwen3.6 HMONNX generate CLI backed by workflow_runtime."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

from xhquant.utils import TimeProfiler
from xhquant.utils.memory_tracker import MemoryTracker


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from xhmodel_merak.xh_llm.models.qwen3_5.workflow_runtime import (  # noqa: E402
    hmonnx_generate,
    print_quick_test_result,
)


try:
    from validate_hm_release_layout import ensure_step_artifact_links
except ImportError:  # pragma: no cover - optional local helper
    ensure_step_artifact_links = None


def _resolve_export_dir(config: str) -> Path:
    path = Path(config).resolve()
    if path.is_dir():
        return path
    return path.parent


def _parse_device_arg(device_arg: str) -> str | list[int]:
    if device_arg is None:
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        else:
            return ["cpu"]

    tokens = [token.strip().lower() for token in (device_arg or "").split(",") if token.strip()]
    if not tokens:
        raise ValueError("--device must be 'cpu' or a comma-separated list of GPU ids, e.g. '0' or '0,1'.")

    if len(tokens) == 1 and tokens[0] == "cpu":
        return [
            "cpu",
        ]

    if "cpu" in tokens:
        raise ValueError("--device cannot mix 'cpu' with GPU ids.")

    gpu_ids: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        if token.startswith("cuda:"):
            token = token.split(":", 1)[1].strip()
        if not token.isdigit():
            raise ValueError(f"Unsupported device token: {token}")
        gpu_id = int(token)
        if gpu_id in seen:
            continue
        seen.add(gpu_id)
        gpu_ids.append(gpu_id)
    return gpu_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to exported golden_meta_info.json or meta file")
    parser.add_argument("--fast", action="store_true", help="run in fast mode")
    parser.add_argument("--debug", action="store_true", help="run in debug mode")
    parser.add_argument("--image-path", type=str, default="./data/images/demo_qwen3_vl.jpeg")
    parser.add_argument("--prompt", type=str, default="Describe this image.")
    parser.add_argument("--think", action="store_true", help="enable think mode")
    parser.add_argument("--golden", action="store_true", help="save golden outputs")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--do-sample", dest="do_sample", action="store_true", default=True)
    parser.add_argument("--no-sample", dest="do_sample", action="store_false")
    parser.add_argument(
        "--min-output-tokens",
        type=int,
        default=0,
        help="fail if the generated continuation is shorter than this many tokens",
    )
    parser.add_argument(
        "--auto-offload",
        action="store_true",
        help="Whether to enable auto offload, only for debug and development",
    )
    parser.add_argument("--cuda-graph", action="store_true", help="Enable CUDA Graph execution when supported.")
    parser.add_argument("--use-v2", action="store_true", help="Use HMONNXInferenceV2 instead of the legacy runtime.")

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device: 'cpu', one GPU id such as '0', or multiple GPU ids such as '0,1'.",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    devices = _parse_device_arg(args.device)
    gpu_ids = [device for device in devices if device != "cpu"]

    from loguru import logger

    with (
        MemoryTracker(device=gpu_ids, name="hmonnx generate", logger=logger),
        TimeProfiler("hmonnx generate", logger=logger),
    ):
        result = hmonnx_generate(
            meta_file=args.config,
            prompt=args.prompt,
            image_path=args.image_path,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            think=args.think,
            fast=args.fast,
            debug=args.debug,
            golden=args.golden,
            min_output_tokens=args.min_output_tokens,
            auto_offload=args.auto_offload,
            cuda_graph=args.cuda_graph,
            device_map=devices,
        )
    print_quick_test_result(result)

    if args.golden and ensure_step_artifact_links is not None:
        export_dir = _resolve_export_dir(args.config)
        if (export_dir / "golden_meta_info.json").is_file():
            ensure_step_artifact_links(export_dir)
            print(f"Ensured HM golden step artifact links under {export_dir}")
        else:
            print(f"Skip HM step artifact link repair; golden_meta_info.json not found under {export_dir}")


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.use_v2:
        os.environ["ENABLE_HMINFERENCE_V2"] = "1"

    main(args)
