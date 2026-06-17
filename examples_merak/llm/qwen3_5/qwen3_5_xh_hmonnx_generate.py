"""Qwen3.5/Qwen3.6 HMONNX generate CLI backed by workflow_runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


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
    return parser


def main(args: argparse.Namespace) -> None:
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
    main(build_parser().parse_args())
