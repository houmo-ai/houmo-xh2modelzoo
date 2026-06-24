# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""One-command Cosmos3-Nano reasoner quantization orchestration.

This script intentionally reuses the export/compare entrypoints instead of
duplicating graph construction.  The default profile is a small smoke run; use
``--profile full`` for the 36-layer/full-image reasoner path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _runner import QuantRunConfig, QuantStage, run_stages, write_markdown_summary, write_summary  # noqa: E402


COSMOS3_ROOT = Path(__file__).resolve().parents[1]
if str(COSMOS3_ROOT) not in sys.path:
    sys.path.insert(0, str(COSMOS3_ROOT))

from common.paths import default_model_root  # noqa: E402

DEFAULT_MODEL_ROOT = default_model_root()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=COSMOS3_ROOT / "data" / "quantized_reasoner")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--quant-type", default="w8a16_sefp")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--export-device", default="cpu")
    parser.add_argument("--runtime-device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--input-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def py(args: argparse.Namespace, relative_script: str) -> list[str]:
    return [args.python, str(COSMOS3_ROOT / relative_script)]


def add_debug(args: argparse.Namespace, command: list[str]) -> list[str]:
    return [*command, "--debug"] if args.debug else command


def build_stages(args: argparse.Namespace) -> list[QuantStage]:
    out = args.out_dir.resolve()
    model_root = args.model_root.resolve()
    num_layers = 2 if args.profile == "smoke" else 36
    seq = 8 if args.profile == "smoke" else 53
    image_tokens = 4 if args.profile == "smoke" else 49
    past_seq = seq
    layer_suffix = f"layers0_{num_layers - 1}"

    vision_patch_dir = out / "vision_patch_embed"
    vision_core_dir = out / "vision_encoder_core"
    text_seq_dir = out / f"text_embedding_seq{seq}"
    text_seq1_dir = out / "text_embedding_seq1"
    prefill_dir = out / f"prefill_kv_{num_layers}layers_seq{seq}"
    decode_dir = out / f"decode_{num_layers}layers_past{past_seq}"
    logits_seq_dir = out / f"logits_head_seq{seq}"
    logits_seq1_dir = out / "logits_head_seq1"
    e2e_dir = out / f"e2e_prefill_kv_decode_{args.profile}"

    vision_patch_hmonnx = vision_patch_dir / "cosmos3_nano_vision_patch_embed.hmonnx.onnx"
    vision_core_hmonnx = vision_core_dir / "cosmos3_nano_vision_encoder_core.hmonnx.onnx"
    text_seq_hmonnx = text_seq_dir / f"cosmos3_nano_text_embedding_seq{seq}.hmonnx.onnx"
    text_seq1_hmonnx = text_seq1_dir / "cosmos3_nano_text_embedding_seq1.hmonnx.onnx"
    prefill_hmonnx = prefill_dir / f"cosmos3_nano_transformer_{layer_suffix}_prefill_kv_stack.hmonnx.onnx"
    decode_hmonnx = decode_dir / f"cosmos3_nano_transformer_{layer_suffix}_decode_stack.hmonnx.onnx"
    logits_seq_hmonnx = logits_seq_dir / f"cosmos3_nano_transformer_logits_head_seq{seq}.hmonnx.onnx"
    logits_seq1_hmonnx = logits_seq1_dir / "cosmos3_nano_transformer_logits_head_seq1.hmonnx.onnx"
    e2e_report = e2e_dir / "reasoner_e2e_prefill_kv_decode_hmonnx_report.json"

    stages = [
        QuantStage(
            name="vision_patch_embed",
            command=[
                *py(args, "export/reasoner/vision_encoder/export_vision_patch_embed.py"),
                "--model",
                str(model_root),
                "--out-dir",
                str(vision_patch_dir),
                "--height",
                str(args.height),
                "--width",
                str(args.width),
                "--dtype",
                args.dtype,
                "--device",
                args.export_device,
                "--quant-type",
                args.quant_type,
                "--convert-hmonnx",
            ],
            expected=(vision_patch_hmonnx,),
        ),
        QuantStage(
            name="vision_encoder_core",
            command=[
                *py(args, "export/reasoner/vision_encoder/export_vision_encoder_core.py"),
                "--model",
                str(model_root),
                "--out-dir",
                str(vision_core_dir),
                "--height",
                str(args.height),
                "--width",
                str(args.width),
                "--dtype",
                args.dtype,
                "--device",
                args.export_device,
                "--quant-type",
                args.quant_type,
                "--attention-layout",
                "4d",
                "--convert-hmonnx",
            ],
            expected=(vision_core_hmonnx,),
        ),
        QuantStage(
            name=f"text_embedding_seq{seq}",
            command=[
                *py(args, "export/reasoner/text_embedding/export_text_embedding.py"),
                "--model",
                str(model_root),
                "--out-dir",
                str(text_seq_dir),
                "--hmonnx-name",
                text_seq_hmonnx.name,
                "--seq",
                str(seq),
                "--quant-type",
                args.quant_type,
                "--convert-hmonnx",
            ],
            expected=(text_seq_hmonnx,),
        ),
        QuantStage(
            name="text_embedding_seq1",
            command=[
                *py(args, "export/reasoner/text_embedding/export_text_embedding.py"),
                "--model",
                str(model_root),
                "--out-dir",
                str(text_seq1_dir),
                "--hmonnx-name",
                text_seq1_hmonnx.name,
                "--seq",
                "1",
                "--quant-type",
                args.quant_type,
                "--convert-hmonnx",
            ],
            expected=(text_seq1_hmonnx,),
        ),
        QuantStage(
            name=f"prefill_kv_{num_layers}layers_seq{seq}",
            command=[
                *py(args, "export/reasoner/ar_prefill_kv/export_transformer_prefill_kv_stack.py"),
                "--model",
                str(model_root),
                "--out-dir",
                str(prefill_dir),
                "--start-layer",
                "0",
                "--num-layers",
                str(num_layers),
                "--seq",
                str(seq),
                "--dtype",
                args.dtype,
                "--device",
                args.export_device,
                "--quant-type",
                args.quant_type,
                "--convert-hmonnx",
            ],
            expected=(prefill_hmonnx,),
        ),
        QuantStage(
            name=f"decode_{num_layers}layers_past{past_seq}",
            command=[
                *py(args, "export/reasoner/ar_decode/export_transformer_decode_stack.py"),
                "--model",
                str(model_root),
                "--out-dir",
                str(decode_dir),
                "--start-layer",
                "0",
                "--num-layers",
                str(num_layers),
                "--past-seq",
                str(past_seq),
                "--dtype",
                args.dtype,
                "--device",
                args.export_device,
                "--quant-type",
                args.quant_type,
                "--convert-hmonnx",
            ],
            expected=(decode_hmonnx,),
        ),
        QuantStage(
            name=f"logits_head_seq{seq}",
            command=[
                *py(args, "export/reasoner/logits_head/export_transformer_logits_head.py"),
                "--model",
                str(model_root),
                "--out-dir",
                str(logits_seq_dir),
                "--hmonnx-name",
                logits_seq_hmonnx.name,
                "--seq",
                str(seq),
                "--dtype",
                args.dtype,
                "--device",
                args.export_device,
                "--quant-type",
                args.quant_type,
                "--convert-hmonnx",
            ],
            expected=(logits_seq_hmonnx,),
        ),
        QuantStage(
            name="logits_head_seq1",
            command=[
                *py(args, "export/reasoner/logits_head/export_transformer_logits_head.py"),
                "--model",
                str(model_root),
                "--out-dir",
                str(logits_seq1_dir),
                "--hmonnx-name",
                logits_seq1_hmonnx.name,
                "--seq",
                "1",
                "--dtype",
                args.dtype,
                "--device",
                args.export_device,
                "--quant-type",
                args.quant_type,
                "--convert-hmonnx",
            ],
            expected=(logits_seq1_hmonnx,),
        ),
    ]

    e2e_command = [
        *py(args, "compare/reasoner/e2e/compare_reasoner_e2e_prefill_kv_decode_hmonnx.py"),
        "--model",
        str(model_root),
        "--text-hmonnx",
        str(text_seq_hmonnx),
        "--text-seq1-hmonnx",
        str(text_seq1_hmonnx),
        "--vision-patch-hmonnx",
        str(vision_patch_hmonnx),
        "--vision-core-hmonnx",
        str(vision_core_hmonnx),
        "--prefill-kv-hmonnx",
        str(prefill_hmonnx),
        "--decode-hmonnx",
        str(decode_hmonnx),
        "--logits-seq8-hmonnx",
        str(logits_seq_hmonnx),
        "--logits-seq1-hmonnx",
        str(logits_seq1_hmonnx),
        "--report",
        str(e2e_report),
        "--seq",
        str(seq),
        "--image-tokens",
        str(image_tokens),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--num-layers",
        str(num_layers),
        "--device",
        args.runtime_device,
        "--input-dtype",
        args.input_dtype,
    ]
    stages.append(
        QuantStage(
            name=f"reasoner_e2e_{args.profile}",
            command=add_debug(args, e2e_command),
            expected=(e2e_report,),
            reports=(e2e_report,),
            note="Runs text+vision fusion, prefill-KV, decode and logits through runtime.",
        )
    )
    return stages


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.out_dir / "logs"
    stages = build_stages(args)
    results = run_stages(
        stages,
        QuantRunConfig(
            dry_run=args.dry_run,
            force=args.force,
            continue_on_error=args.continue_on_error,
            log_dir=log_dir,
        ),
    )
    payload = {
        "model_root": str(args.model_root.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "profile": args.profile,
        "quant_type": args.quant_type,
        "normalize_force_fp32": True,
    }
    write_summary(args.out_dir / "quantize_reasoner_summary.json", payload, results)
    write_markdown_summary(args.out_dir / "quantize_reasoner_summary.md", "Cosmos3-Nano Reasoner Quantization", results)


if __name__ == "__main__":
    main()
