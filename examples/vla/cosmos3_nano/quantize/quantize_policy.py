# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""One-command Cosmos3-Nano policy quantization orchestration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _runner import (  # noqa: E402
    QuantRunConfig,
    QuantStage,
    run_stages,
    write_markdown_summary,
    write_summary,
)


COSMOS3_ROOT = Path(__file__).resolve().parents[1]
if str(COSMOS3_ROOT) not in sys.path:
    sys.path.insert(0, str(COSMOS3_ROOT))

from common.paths import default_model_root  # noqa: E402

DEFAULT_MODEL_ROOT = default_model_root()
DEFAULT_UND_SEQ = 13


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=COSMOS3_ROOT / "data" / "quantized_policy")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--num-layers-override", type=int, default=None)
    parser.add_argument(
        "--scaling-layers",
        default="",
        help="Comma-separated layer counts for error localization, for example 1,2,4,8,16,36.",
    )
    parser.add_argument(
        "--segments",
        default="",
        help="Comma-separated contiguous segmented plan, for example 0:8,8:8,16:8,24:12.",
    )
    parser.add_argument(
        "--fp-segments",
        default="",
        help="Comma-separated segment ranges to keep as FP ONNX, for example 35 or 34-35 or 35:1.",
    )
    parser.add_argument("--quant-type", default="w8a16_sefp")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--export-device", default="cpu")
    parser.add_argument("--runtime-device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--input-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--action-head-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--action-seq", type=int, default=4)
    parser.add_argument("--raw-action-dim", type=int, default=7)
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def py(args: argparse.Namespace, relative_script: str) -> list[str]:
    return [args.python, str(COSMOS3_ROOT / relative_script)]


def add_debug(args: argparse.Namespace, command: list[str]) -> list[str]:
    return [*command, "--debug"] if args.debug else command


def layer_counts(args: argparse.Namespace) -> list[int]:
    if args.scaling_layers.strip():
        values = [int(item) for item in args.scaling_layers.split(",") if item.strip()]
        return list(dict.fromkeys(values))
    if args.num_layers_override is not None:
        return [int(args.num_layers_override)]
    return [2 if args.profile == "smoke" else 36]


def segment_plan(args: argparse.Namespace) -> list[tuple[int, int]]:
    if not args.segments.strip():
        return []
    segments: list[tuple[int, int]] = []
    for item in args.segments.split(","):
        item = item.strip()
        if not item:
            continue
        start_text, length_text = item.split(":", 1)
        segments.append((int(start_text), int(length_text)))
    if not segments:
        return []
    expected_start = segments[0][0]
    for start, length in segments:
        if start != expected_start:
            raise ValueError(f"segments must be contiguous; expected start {expected_start}, got {start}")
        if length <= 0:
            raise ValueError(f"segment length must be positive: {start}:{length}")
        expected_start += length
    return segments


def parse_fp_segment_set(value: str | None) -> set[tuple[int, int]]:
    if not value or not value.strip():
        return set()
    segments: set[tuple[int, int]] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            start_text, length_text = item.split(":", 1)
            segments.add((int(start_text), int(length_text)))
        elif "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"invalid fp segment range: {item}")
            segments.add((start, end - start + 1))
        else:
            segments.add((int(item), 1))
    return segments


def policy_suffix(num_layers: int, action_seq: int, raw_action_dim: int, domain_id: int) -> str:
    end_layer = num_layers - 1
    layer_part = "layer0" if num_layers == 1 else f"layers0_{end_layer}"
    # The default prompt tokenizes to 13 text-side tokens after Cosmos packing.
    return f"{layer_part}_realpack_und{DEFAULT_UND_SEQ}_act{action_seq}_raw{raw_action_dim}_domain{domain_id}"


def segment_suffix(
    segment_kind: str,
    start_layer: int,
    num_layers: int,
    action_seq: int,
    raw_action_dim: int,
    domain_id: int,
) -> str:
    end_layer = start_layer + num_layers - 1
    layer_part = f"layer{start_layer}" if num_layers == 1 else f"layers{start_layer}_{end_layer}"
    return (
        f"{segment_kind}_{layer_part}_realpack_und{DEFAULT_UND_SEQ}_act{action_seq}"
        f"_raw{raw_action_dim}_domain{domain_id}"
    )


def add_backbone_stages(
    args: argparse.Namespace,
    stages: list[QuantStage],
    num_layers: int,
    action_head_onnx: Path,
) -> None:
    out = args.out_dir.resolve()
    model_root = args.model_root.resolve()
    transformer_dir = model_root / "transformer"
    suffix = policy_suffix(num_layers, args.action_seq, args.raw_action_dim, args.domain_id)
    backbone_dir = out / f"action_denoiser_backbone_{num_layers}layers"
    backbone_hmonnx = backbone_dir / f"cosmos3_nano_policy_action_{suffix}.hmonnx.onnx"
    backbone_report = backbone_dir / f"{suffix}_hmonnx_compare_report.json"
    chain_report = out / "action_chain" / f"{suffix}_action_chain_compare_report.json"

    stages.extend(
        [
            QuantStage(
                name=f"policy_action_backbone_{num_layers}layers",
                command=[
                    *py(args, "export/policy/action_denoiser/export_official_action_denoiser_boundary.py"),
                    "--model",
                    str(transformer_dir),
                    "--model-root",
                    str(model_root),
                    "--out-dir",
                    str(backbone_dir),
                    "--start-layer",
                    "0",
                    "--num-layers",
                    str(num_layers),
                    "--action-seq",
                    str(args.action_seq),
                    "--raw-action-dim",
                    str(args.raw_action_dim),
                    "--domain-id",
                    str(args.domain_id),
                    "--seed",
                    str(args.seed),
                    "--dtype",
                    args.dtype,
                    "--device",
                    args.export_device,
                    "--quant-type",
                    args.quant_type,
                    "--convert-hmonnx",
                ],
                expected=(backbone_hmonnx,),
                note="Exports backbone hidden only; action_proj_out/action head stays FP ONNX.",
            ),
            QuantStage(
                name=f"policy_action_backbone_{num_layers}layers_compare",
                command=add_debug(
                    args,
                    [
                        *py(args, "compare/policy/action_denoiser/compare_official_action_denoiser_boundary_hmonnx.py"),
                        "--model",
                        str(transformer_dir),
                        "--model-root",
                        str(model_root),
                        "--hmonnx",
                        str(backbone_hmonnx),
                        "--report",
                        str(backbone_report),
                        "--start-layer",
                        "0",
                        "--num-layers",
                        str(num_layers),
                        "--action-seq",
                        str(args.action_seq),
                        "--raw-action-dim",
                        str(args.raw_action_dim),
                        "--domain-id",
                        str(args.domain_id),
                        "--seed",
                        str(args.seed),
                        "--device",
                        args.runtime_device,
                        "--input-dtype",
                        args.input_dtype,
                    ],
                ),
                expected=(backbone_report,),
                reports=(backbone_report,),
            ),
            QuantStage(
                name=f"policy_action_chain_{num_layers}layers",
                command=add_debug(
                    args,
                    [
                        *py(args, "compare/policy/e2e/compare_policy_action_backbone_head_chain.py"),
                        "--model",
                        str(transformer_dir),
                        "--model-root",
                        str(model_root),
                        "--backbone-hmonnx",
                        str(backbone_hmonnx),
                        "--action-head-onnx",
                        str(action_head_onnx),
                        "--report",
                        str(chain_report),
                        "--start-layer",
                        "0",
                        "--num-layers",
                        str(num_layers),
                        "--action-seq",
                        str(args.action_seq),
                        "--raw-action-dim",
                        str(args.raw_action_dim),
                        "--domain-id",
                        str(args.domain_id),
                        "--seed",
                        str(args.seed),
                        "--device",
                        args.runtime_device,
                        "--input-dtype",
                        args.input_dtype,
                    ],
                ),
                expected=(chain_report,),
                reports=(chain_report,),
            ),
        ]
    )


def add_segment_stages(args: argparse.Namespace, stages: list[QuantStage], action_head_onnx: Path) -> None:
    segments = segment_plan(args)
    if not segments:
        return
    fp_segments = parse_fp_segment_set(args.fp_segments)
    unknown_fp_segments = fp_segments.difference(set(segments))
    if unknown_fp_segments:
        raise ValueError(f"--fp-segments must match entries in --segments; unmatched: {sorted(unknown_fp_segments)}")

    out = args.out_dir.resolve()
    model_root = args.model_root.resolve()
    transformer_dir = model_root / "transformer"
    segment_paths: list[Path] = []
    segment_names: list[str] = []

    for index, (start_layer, num_layers) in enumerate(segments):
        kind = "init" if index == 0 else "stack"
        suffix = segment_suffix(kind, start_layer, num_layers, args.action_seq, args.raw_action_dim, args.domain_id)
        segment_dir = out / "action_denoiser_segments" / suffix
        onnx_path = segment_dir / f"cosmos3_nano_policy_action_segment_{suffix}.onnx"
        hmonnx_path = segment_dir / f"cosmos3_nano_policy_action_segment_{suffix}.hmonnx.onnx"
        patched_onnx_path = onnx_path.with_name(onnx_path.stem + ".scatter_i64.onnx")
        keep_fp = (start_layer, num_layers) in fp_segments
        segment_paths.append(patched_onnx_path if keep_fp else hmonnx_path)
        segment_names.append(f"{start_layer}_{start_layer + num_layers - 1}" + ("fp" if keep_fp else ""))

        command = [
            *py(args, "export/policy/action_denoiser/export_official_action_denoiser_segment.py"),
            "--model",
            str(transformer_dir),
            "--model-root",
            str(model_root),
            "--out-dir",
            str(segment_dir),
            "--segment-kind",
            kind,
            "--start-layer",
            str(start_layer),
            "--num-layers",
            str(num_layers),
            "--action-seq",
            str(args.action_seq),
            "--raw-action-dim",
            str(args.raw_action_dim),
            "--domain-id",
            str(args.domain_id),
            "--seed",
            str(args.seed),
            "--dtype",
            args.dtype,
            "--device",
            args.export_device,
            "--quant-type",
            args.quant_type,
        ]
        if keep_fp:
            command.append("--patch-scatternd-indices-int64")
        else:
            command.append("--convert-hmonnx")

        stages.append(
            QuantStage(
                name=(
                    f"policy_action_segment_{kind}_{start_layer}_{start_layer + num_layers - 1}"
                    + ("_fp" if keep_fp else "")
                ),
                command=command,
                expected=(patched_onnx_path if keep_fp else hmonnx_path,),
                note="FP ONNX segment uses ScatterND int64 patch."
                if keep_fp
                else "Segmented backbone output is full packed hidden states for host chaining.",
            )
        )

    plan_name = "_".join(segment_names)
    report = out / "action_segmented_chain" / f"segments_{plan_name}_action_chain_compare_report.json"
    command = [
        *py(args, "compare/policy/e2e/compare_policy_action_segmented_chain.py"),
        "--model",
        str(transformer_dir),
        "--model-root",
        str(model_root),
        "--segments",
        args.segments,
        "--segment-hmonnx",
        *[str(path) for path in segment_paths],
        "--action-head-onnx",
        str(action_head_onnx),
        "--report",
        str(report),
        "--action-seq",
        str(args.action_seq),
        "--raw-action-dim",
        str(args.raw_action_dim),
        "--domain-id",
        str(args.domain_id),
        "--seed",
        str(args.seed),
        "--device",
        args.runtime_device,
        "--input-dtype",
        args.input_dtype,
    ]
    stages.append(
        QuantStage(
            name=f"policy_action_segmented_chain_{plan_name}",
            command=add_debug(args, command),
            expected=(report,),
            reports=(report,),
        )
    )

def build_stages(args: argparse.Namespace) -> list[QuantStage]:
    out = args.out_dir.resolve()
    model_root = args.model_root.resolve()
    transformer_dir = model_root / "transformer"
    action_head_dir = out / "action_head_fp16"
    action_head_onnx = action_head_dir / "cosmos3_nano_policy_action_head_fp16.onnx"
    action_head_report = action_head_dir / "action_head_onnx_compare_report.json"

    stages: list[QuantStage] = [
        QuantStage(
            name="policy_action_head_fp16",
            command=[
                *py(args, "export/policy/action_head/export_official_action_head.py"),
                "--model",
                str(transformer_dir),
                "--out-dir",
                str(action_head_dir),
                "--dtype",
                args.action_head_dtype,
                "--device",
                args.export_device,
                "--action-seq",
                str(args.action_seq),
                "--domain-id",
                str(args.domain_id),
            ],
            expected=(action_head_onnx,),
            note="FP action head; not W8A16-quantized in the first policy split.",
        ),
        QuantStage(
            name="policy_action_head_fp16_compare",
            command=[
                *py(args, "compare/policy/action_head/compare_official_action_head_onnx.py"),
                "--model",
                str(transformer_dir),
                "--onnx",
                str(action_head_onnx),
                "--report",
                str(action_head_report),
                "--dtype",
                args.action_head_dtype,
                "--action-seq",
                str(args.action_seq),
                "--raw-action-dim",
                str(args.raw_action_dim),
                "--domain-id",
                str(args.domain_id),
            ],
            expected=(action_head_report,),
            reports=(action_head_report,),
        ),
    ]
    if segment_plan(args):
        add_segment_stages(args, stages, action_head_onnx)
    else:
        for count in layer_counts(args):
            add_backbone_stages(args, stages, count, action_head_onnx)
    return stages


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stages = build_stages(args)
    results = run_stages(
        stages,
        QuantRunConfig(
            dry_run=args.dry_run,
            force=args.force,
            continue_on_error=args.continue_on_error,
            log_dir=args.out_dir / "logs",
        ),
    )
    segments = segment_plan(args)
    counts = [] if segments else layer_counts(args)
    payload = {
        "model_root": str(args.model_root.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "profile": args.profile,
        "layer_counts": counts,
        "segments": [{"start_layer": start, "num_layers": length} for start, length in segments],
        "quant_type": args.quant_type,
        "action_head_dtype": args.action_head_dtype,
        "normalize_force_fp32": True,
        "split": {
            "backbone": (
                "action_proj_in + time/action embeddings + MoT stack, "
                "HMONNX W8A16 with optional FP ONNX segments"
            ),
            "segment_boundary": "full packed hidden states stay on host between HMONNX/ONNX segments",
            "action_head": "action_proj_out, FP ONNX",
            "host_runtime": "scheduler loop, raw_action_dim unpadding, de-normalization, clamps",
        },
    }
    write_summary(args.out_dir / "quantize_policy_summary.json", payload, results)
    write_markdown_summary(args.out_dir / "quantize_policy_summary.md", "Cosmos3-Nano Policy Quantization", results)


if __name__ == "__main__":
    main()
