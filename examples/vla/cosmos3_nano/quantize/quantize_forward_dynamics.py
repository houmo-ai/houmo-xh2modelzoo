# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""One-command Cosmos3-Nano forward dynamics quantization orchestration."""

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
    parser.add_argument("--out-dir", type=Path, default=COSMOS3_ROOT / "data" / "quantized_forward_dynamics")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--quant-type", default="w8a16_sefp")
    parser.add_argument("--vae-dir", type=Path, default=COSMOS3_ROOT / "data" / "vae_decoder_split_stage")
    parser.add_argument("--vae-quant-type", default="w16a16_sefp")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--export-device", default="cpu")
    parser.add_argument("--runtime-device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--input-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--latent-height", type=int, default=4)
    parser.add_argument("--latent-width", type=int, default=4)
    parser.add_argument("--action-seq", type=int, default=16)
    parser.add_argument("--raw-action-dim", type=int, default=29)
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--num-layers-override", type=int, default=None, help="Override profile layer count for staged scaling tests.")
    parser.add_argument(
        "--segments",
        default="",
        help="Comma-separated contiguous segmented plan, for example 0:8,8:8,16:8,24:12.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def py(args: argparse.Namespace, relative_script: str) -> list[str]:
    return [args.python, str(COSMOS3_ROOT / relative_script)]


def add_debug(args: argparse.Namespace, command: list[str]) -> list[str]:
    return [*command, "--debug"] if args.debug else command


def fd_suffix(
    num_layers: int,
    und_seq: int,
    vision_seq: int,
    action_seq: int,
    frames: int,
    latent_height: int,
    latent_width: int,
) -> str:
    return (
        f"layers0_{num_layers - 1}_realpack_und{und_seq}_vision{vision_seq}_action{action_seq}"
        f"_t{frames}_lh{latent_height}_lw{latent_width}"
    )


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


def fd_segment_suffix(
    segment_kind: str,
    start_layer: int,
    num_layers: int,
    und_seq: int,
    vision_seq: int,
    action_seq: int,
    frames: int,
    latent_height: int,
    latent_width: int,
) -> str:
    end_layer = start_layer + num_layers - 1
    layer_part = f"layer{start_layer}" if num_layers == 1 else f"layers{start_layer}_{end_layer}"
    return (
        f"{segment_kind}_{layer_part}_realpack_und{und_seq}_vision{vision_seq}_action{action_seq}"
        f"_t{frames}_lh{latent_height}_lw{latent_width}"
    )


def add_segment_stages(args: argparse.Namespace, stages: list[QuantStage]) -> None:
    segments = segment_plan(args)
    if not segments:
        return
    out = args.out_dir.resolve()
    model_root = args.model_root.resolve()
    transformer_dir = model_root / "transformer"
    vision_seq = args.frames * (args.latent_height // 2) * (args.latent_width // 2)
    und_seq = 8
    segment_hmonnx_paths: list[Path] = []
    segment_names: list[str] = []

    for index, (start_layer, num_layers) in enumerate(segments):
        kind = "init" if index == 0 else "stack"
        suffix = fd_segment_suffix(
            kind,
            start_layer,
            num_layers,
            und_seq,
            vision_seq,
            args.action_seq,
            args.frames,
            args.latent_height,
            args.latent_width,
        )
        segment_dir = out / "fd_realpack_segments" / suffix
        hmonnx_path = segment_dir / f"cosmos3_nano_forward_dynamics_segment_{suffix}.hmonnx.onnx"
        segment_hmonnx_paths.append(hmonnx_path)
        segment_names.append(f"{start_layer}_{start_layer + num_layers - 1}")
        stages.append(
            QuantStage(
                name=f"fd_segment_{kind}_{start_layer}_{start_layer + num_layers - 1}",
                command=[
                    *py(args, "export/forward_dynamics/action_denoiser/export_fd_real_packed_segment.py"),
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
                    "--frames",
                    str(args.frames),
                    "--latent-height",
                    str(args.latent_height),
                    "--latent-width",
                    str(args.latent_width),
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
                expected=(hmonnx_path,),
                note="Segmented FD stack output is full packed hidden states for host chaining.",
            )
        )

    plan_name = "_".join(segment_names)
    report = out / "fd_segmented_chain" / f"segments_{plan_name}_compare_report.json"
    stages.append(
        QuantStage(
            name=f"fd_segmented_chain_{plan_name}",
            command=add_debug(
                args,
                [
                    *py(args, "compare/forward_dynamics/action_denoiser/compare_fd_segmented_chain_hmonnx.py"),
                    "--model",
                    str(transformer_dir),
                    "--model-root",
                    str(model_root),
                    "--segments",
                    args.segments,
                    "--segment-hmonnx",
                    *[str(path) for path in segment_hmonnx_paths],
                    "--report",
                    str(report),
                    "--frames",
                    str(args.frames),
                    "--latent-height",
                    str(args.latent_height),
                    "--latent-width",
                    str(args.latent_width),
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
            expected=(report,),
            reports=(report,),
        )
    )

    scheduler_report = out / "fd_segmented_scheduler_loop" / f"segments_{plan_name}_scheduler_loop_report.json"
    stages.append(
        QuantStage(
            name=f"fd_segmented_scheduler_{plan_name}",
            command=add_debug(
                args,
                [
                    *py(args, "compare/forward_dynamics/action_denoiser/compare_fd_segmented_scheduler_loop_hmonnx.py"),
                    "--model",
                    str(transformer_dir),
                    "--model-root",
                    str(model_root),
                    "--segments",
                    args.segments,
                    "--segment-hmonnx",
                    *[str(path) for path in segment_hmonnx_paths],
                    "--report",
                    str(scheduler_report),
                    "--frames",
                    str(args.frames),
                    "--latent-height",
                    str(args.latent_height),
                    "--latent-width",
                    str(args.latent_width),
                    "--action-seq",
                    str(args.action_seq),
                    "--raw-action-dim",
                    str(args.raw_action_dim),
                    "--domain-id",
                    str(args.domain_id),
                    "--seed",
                    str(args.seed),
                    "--num-inference-steps",
                    str(args.num_inference_steps),
                    "--torch-device",
                    "cpu",
                    "--hmonnx-device",
                    args.runtime_device,
                    "--input-dtype",
                    args.input_dtype,
                    "--torch-dtype",
                    "float32",
                ],
            ),
            expected=(scheduler_report,),
            reports=(scheduler_report,),
        )
    )


def build_stages(args: argparse.Namespace) -> list[QuantStage]:
    out = args.out_dir.resolve()
    model_root = args.model_root.resolve()
    transformer_dir = model_root / "transformer"
    num_layers = int(args.num_layers_override) if args.num_layers_override is not None else (2 if args.profile == "smoke" else 36)

    if segment_plan(args):
        stages: list[QuantStage] = []
        add_segment_stages(args, stages)
        return stages

    fd_dir = out / f"fd_realpack_{num_layers}layers"
    # The default prompt tokenizes to 8 text-side tokens after Cosmos packing.
    # If users override the prompt in the export script, the graph name will differ.
    vision_seq = args.frames * (args.latent_height // 2) * (args.latent_width // 2)
    und_seq = 8
    suffix = fd_suffix(
        num_layers,
        und_seq,
        vision_seq,
        args.action_seq,
        args.frames,
        args.latent_height,
        args.latent_width,
    )
    fd_hmonnx = fd_dir / f"cosmos3_nano_forward_dynamics_{suffix}.hmonnx.onnx"
    fd_report = fd_dir / f"{suffix}_hmonnx_compare_report.json"
    scheduler_report = out / "scheduler_loop" / f"{suffix}_scheduler_loop_report.json"
    pipeline_report = out / "fd_vae_pipeline" / f"{suffix}_fd_vae_pipeline_report.json"

    return [
        QuantStage(
            name=f"fd_realpack_{num_layers}layers",
            command=[
                *py(args, "export/forward_dynamics/action_denoiser/export_fd_real_packed_boundary.py"),
                "--model",
                str(transformer_dir),
                "--model-root",
                str(model_root),
                "--out-dir",
                str(fd_dir),
                "--start-layer",
                "0",
                "--num-layers",
                str(num_layers),
                "--frames",
                str(args.frames),
                "--latent-height",
                str(args.latent_height),
                "--latent-width",
                str(args.latent_width),
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
            expected=(fd_hmonnx,),
            note="Forward dynamics denoiser: text + future vision latent + action tokens.",
        ),
        QuantStage(
            name=f"fd_realpack_{num_layers}layers_compare",
            command=add_debug(
                args,
                [
                    *py(args, "compare/forward_dynamics/action_denoiser/compare_fd_real_packed_boundary_hmonnx.py"),
                    "--model",
                    str(transformer_dir),
                    "--model-root",
                    str(model_root),
                    "--hmonnx",
                    str(fd_hmonnx),
                    "--report",
                    str(fd_report),
                    "--start-layer",
                    "0",
                    "--num-layers",
                    str(num_layers),
                    "--frames",
                    str(args.frames),
                    "--latent-height",
                    str(args.latent_height),
                    "--latent-width",
                    str(args.latent_width),
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
            expected=(fd_report,),
            reports=(fd_report,),
        ),
        QuantStage(
            name=f"fd_scheduler_loop_{num_layers}layers",
            command=add_debug(
                args,
                [
                    *py(args, "compare/forward_dynamics/action_denoiser/compare_fd_scheduler_loop_hmonnx.py"),
                    "--model",
                    str(transformer_dir),
                    "--model-root",
                    str(model_root),
                    "--hmonnx",
                    str(fd_hmonnx),
                    "--report",
                    str(scheduler_report),
                    "--num-layers",
                    str(num_layers),
                    "--frames",
                    str(args.frames),
                    "--latent-height",
                    str(args.latent_height),
                    "--latent-width",
                    str(args.latent_width),
                    "--action-seq",
                    str(args.action_seq),
                    "--raw-action-dim",
                    str(args.raw_action_dim),
                    "--domain-id",
                    str(args.domain_id),
                    "--seed",
                    str(args.seed),
                    "--num-inference-steps",
                    str(args.num_inference_steps),
                    "--torch-device",
                    "cpu",
                    "--hmonnx-device",
                    args.runtime_device,
                    "--input-dtype",
                    args.input_dtype,
                ],
            ),
            expected=(scheduler_report,),
            reports=(scheduler_report,),
        ),
        QuantStage(
            name=f"fd_vae_pipeline_{num_layers}layers",
            command=add_debug(
                args,
                [
                    *py(args, "compare/forward_dynamics/pipeline/compare_fd_vae_pipeline_hmonnx.py"),
                    "--model",
                    str(transformer_dir),
                    "--model-root",
                    str(model_root),
                    "--denoiser-hmonnx",
                    str(fd_hmonnx),
                    "--vae-dir",
                    str(args.vae_dir.resolve()),
                    "--report",
                    str(pipeline_report),
                    "--num-layers",
                    str(num_layers),
                    "--frames",
                    str(args.frames),
                    "--latent-height",
                    str(args.latent_height),
                    "--latent-width",
                    str(args.latent_width),
                    "--action-seq",
                    str(args.action_seq),
                    "--raw-action-dim",
                    str(args.raw_action_dim),
                    "--domain-id",
                    str(args.domain_id),
                    "--seed",
                    str(args.seed),
                    "--num-inference-steps",
                    str(args.num_inference_steps),
                    "--quant-type",
                    args.vae_quant_type,
                    "--torch-device",
                    "cpu",
                    "--hmonnx-device",
                    args.runtime_device,
                    "--input-dtype",
                    args.input_dtype,
                    "--torch-dtype",
                    "float16",
                ],
            ),
            expected=(pipeline_report,),
            reports=(pipeline_report,),
            note="Reuses generator VAE decoder split-chain HMONNX files.",
        ),
    ]

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
    num_layers = int(args.num_layers_override) if args.num_layers_override is not None else (2 if args.profile == "smoke" else 36)
    payload = {
        "model_root": str(args.model_root.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "profile": args.profile,
        "num_layers": num_layers,
        "segments": args.segments if segment_plan(args) else None,
        "quant_type": args.quant_type,
        "vae_dir": str(args.vae_dir.resolve()),
        "vae_quant_type": args.vae_quant_type,
        "normalize_force_fp32": True,
        "reuse": {
            "observation_encoder": "reuse generator VAE encoder export when full FD pipeline is needed",
            "observation_decoder": "reuse generator VAE decoder split-chain",
            "scheduler_loop": "host/runtime rollout; not exported as one graph",
        },
    }
    write_summary(args.out_dir / "quantize_forward_dynamics_summary.json", payload, results)
    write_markdown_summary(
        args.out_dir / "quantize_forward_dynamics_summary.md",
        "Cosmos3-Nano Forward Dynamics Quantization",
        results,
    )


if __name__ == "__main__":
    main()
