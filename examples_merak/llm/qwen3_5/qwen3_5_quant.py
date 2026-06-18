"""Quantize Qwen3.5/Qwen3.6 models through the Merak workflow API.

This is the small quant-only entrypoint for integrations that only need to
produce a GPTQModel-compatible HF artifact first.  Topology and AutoRound
settings stay in the workflow YAML; this CLI only receives paths, device, and
an explicit optional source override.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _is_unsafe_delete_target(path: Path) -> bool:
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    return resolved in {Path("/").resolve(), Path.home().resolve(), cwd}


def _remove_output_dir_if_needed(output_dir: str, force: bool) -> None:
    path = Path(output_dir)
    if not path.exists():
        return
    if not force:
        raise FileExistsError(f"Quant output_dir already exists: {path}; pass --force to overwrite")
    if path.is_symlink() or _is_unsafe_delete_target(path):
        raise ValueError(f"Refusing to delete unsafe quant output_dir: {path.resolve()}")
    shutil.rmtree(path)


def _build_quant_overrides(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.base and args.existing_hf_model_dir:
        raise ValueError("--base and --existing-hf-model-dir are mutually exclusive")
    if args.base:
        return {"quant": None}
    if args.existing_hf_model_dir:
        return {
            "quant": {
                "algorithm": "existing_hf",
                "artifact_format": "gptqmodel_hf",
                "source_algorithm": args.source_algorithm,
                "existing_hf_model_dir": args.existing_hf_model_dir,
            }
        }
    return None


def main(args: argparse.Namespace) -> None:
    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

    _remove_output_dir_if_needed(args.output_dir, args.force)
    workflow = AutoLLMWorkflow.from_config(
        hf_model_dir=args.hf_model_dir,
        config_path=args.config,
        seed=args.seed,
        debug=args.debug,
    )
    quant_result = workflow.quant(
        output_dir=args.output_dir,
        device=args.device,
        config_overrides=_build_quant_overrides(args),
    )
    print(json.dumps(quant_result.__dict__, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantize Qwen3.5/Qwen3.6 Dense or MoE using workflow YAML defaults.")
    parser.add_argument("--hf-model-dir", required=True, help="Original HuggingFace model directory")
    parser.add_argument("--config", required=True, help="Workflow YAML containing the quant section")
    parser.add_argument("--output-dir", required=True, help="Quant work/output directory")
    parser.add_argument("--device", default="cuda", help="AutoRound device, e.g. cuda, cuda:0, or cpu")
    parser.add_argument("--seed", type=int, default=1024, help="Workflow seed; AutoRound seed comes from YAML")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode on the workflow")
    parser.add_argument("--force", action="store_true", help="Remove output-dir before quantization")
    parser.add_argument(
        "--base",
        action="store_true",
        help="Skip quantization explicitly and return a base-model QuantResult",
    )
    parser.add_argument(
        "--existing-hf-model-dir",
        default="",
        help="Reuse an already quantized HF/GPTQModel directory instead of running AutoRound",
    )
    parser.add_argument(
        "--source-algorithm",
        default="autoround",
        help="Algorithm label for --existing-hf-model-dir QuantResult metadata",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
