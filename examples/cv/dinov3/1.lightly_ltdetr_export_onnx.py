# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export the LightlyTrain DINOv3 LT-DETR detector to ONNX.

Run without arguments to export the default model:

    python examples/cv/dinov3/1.lightly_ltdetr_export_onnx.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from dinov3_common import (
    DEFAULT_DATA_CACHE_DIR,
    DEFAULT_MODEL_CACHE_DIR,
    DEFAULT_MODEL_NAME,
    DEFAULT_ONNX,
    load_lightly_model,
)


def export_onnx(args: argparse.Namespace) -> Path:
    """Load the LightlyTrain model and export it to ONNX."""
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    model = load_lightly_model(args)
    simplify = not args.no_simplify
    if simplify and importlib.util.find_spec("onnxslim") is None:
        print("onnxslim is not installed; exporting without simplification.")
        simplify = False
    model.export_onnx(
        out,
        precision=args.precision,
        batch_size=args.batch_size,
        dynamic_batch_size=args.dynamic_batch,
        opset_version=args.opset,
        simplify=simplify,
        verify=not args.no_verify,
        num_channels=args.num_channels,
    )
    print(f"ONNX exported: {out}")
    return out


def build_parser() -> argparse.ArgumentParser:
    """Build the ONNX export CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="LightlyTrain model name or checkpoint path.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Model/runtime device.")
    parser.add_argument("--model-cache-dir", default=str(DEFAULT_MODEL_CACHE_DIR), help="LightlyTrain model cache directory.")
    parser.add_argument("--data-cache-dir", default=str(DEFAULT_DATA_CACHE_DIR), help="LightlyTrain data cache directory.")
    parser.add_argument("--out", default=str(DEFAULT_ONNX), help="Output ONNX path.")
    parser.add_argument("--precision", default="fp32", choices=["auto", "fp32", "fp16"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dynamic-batch", action="store_true", help="Export a dynamic batch dimension.")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--num-channels", type=int, default=None)
    parser.add_argument("--no-simplify", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> Any:
    """Parse export arguments and run ONNX export."""
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "export-onnx":
        argv = argv[1:]
    args = build_parser().parse_args(argv)
    return export_onnx(args)


if __name__ == "__main__":
    main()
