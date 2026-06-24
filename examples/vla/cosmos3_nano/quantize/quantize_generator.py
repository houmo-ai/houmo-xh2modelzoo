# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""One-command Cosmos3-Nano generator quantization orchestration."""

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
VAE_SPLIT_STAGES = ("up0", "up1", "up2", "up3", "head")
SOUND_DECODER_STAGES = ("conv_in", "block0", "block1", "block2", "block3", "block4", "head")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=COSMOS3_ROOT / "data" / "quantized_generator")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--quant-type", default="w8a16_sefp", help="Default generator denoiser quant type.")
    parser.add_argument("--vae-quant-type", default="w16a16_sefp")
    parser.add_argument("--sound-quant-type", default="w16a16_sefp")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--export-device", default="cpu")
    parser.add_argument("--runtime-device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--input-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--latent-height", type=int, default=4)
    parser.add_argument("--latent-width", type=int, default=4)
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-sound-tokenizer", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def py(args: argparse.Namespace, relative_script: str) -> list[str]:
    return [args.python, str(COSMOS3_ROOT / relative_script)]


def add_debug(args: argparse.Namespace, command: list[str]) -> list[str]:
    return [*command, "--debug"] if args.debug else command


def denoiser_suffix(num_layers: int, frames: int, latent_height: int, latent_width: int) -> str:
    return f"layers0_{num_layers - 1}_realpack_und8_gen4_t{frames}_lh{latent_height}_lw{latent_width}"


def vae_pre_mid_name(frames: int, latent_height: int, latent_width: int, quant_type: str) -> str:
    return f"vae_decoder_stage_mid_t{frames}_lh{latent_height}_lw{latent_width}_{quant_type}_normfp32_cacheloop.hmonnx.onnx"


def vae_split_name(stage: str, frames: int, latent_height: int, latent_width: int, quant_type: str) -> str:
    return f"vae_decoder_{stage}_split_t{frames}_lh{latent_height}_lw{latent_width}_{quant_type}_normfp32.hmonnx.onnx"


def sound_split_name(stage: str, latent_frames: int, quant_type: str) -> str:
    return f"sound_tokenizer_decoder_{stage}_t{latent_frames}_{quant_type}.hmonnx.onnx"


def build_stages(args: argparse.Namespace) -> list[QuantStage]:
    out = args.out_dir.resolve()
    model_root = args.model_root.resolve()
    transformer_dir = model_root / "transformer"
    sound_tokenizer_dir = model_root / "sound_tokenizer"
    num_layers = 2 if args.profile == "smoke" else 36

    denoiser_dir = out / f"denoiser_realpack_{num_layers}layers"
    denoiser_graph_suffix = denoiser_suffix(num_layers, args.frames, args.latent_height, args.latent_width)
    denoiser_hmonnx = denoiser_dir / f"cosmos3_nano_generator_{denoiser_graph_suffix}.hmonnx.onnx"
    denoiser_report = denoiser_dir / f"{denoiser_graph_suffix}_hmonnx_compare_report.json"
    scheduler_report = out / "scheduler_loop" / f"cfg_scheduler_loop_{num_layers}layers_report.json"

    vae_encoder_dir = out / "vae_encoder"
    vae_encoder_hmonnx = vae_encoder_dir / "cosmos3_nano_vae_encoder.hmonnx.onnx"
    vae_encoder_report = vae_encoder_dir / "vae_encoder_hmonnx_compare_report.json"

    vae_dir = out / "vae_decoder_split_stage"
    vae_pre_mid = vae_dir / vae_pre_mid_name(args.frames, args.latent_height, args.latent_width, args.vae_quant_type)
    vae_split_paths = tuple(
        vae_dir / vae_split_name(stage, args.frames, args.latent_height, args.latent_width, args.vae_quant_type)
        for stage in VAE_SPLIT_STAGES
    )
    vae_full_report = (
        vae_dir
        / f"vae_decoder_full_split_chain_t{args.frames}_lh{args.latent_height}_lw{args.latent_width}_{args.vae_quant_type}_normfp32_report.json"
    )

    sound_dir = out / "sound_tokenizer_decoder_split"
    sound_latent_frames = 2
    sound_paths = tuple(sound_dir / sound_split_name(stage, sound_latent_frames, args.sound_quant_type) for stage in SOUND_DECODER_STAGES)
    sound_report = sound_dir / f"sound_tokenizer_decoder_split_chain_t{sound_latent_frames}_{args.sound_quant_type}_report.json"
    pipeline_report = out / "generator_vae_pipeline" / f"generator_vae_pipeline_{num_layers}layers_report.json"

    stages: list[QuantStage] = [
        QuantStage(
            name=f"denoiser_realpack_{num_layers}layers",
            command=[
                *py(args, "export/generator/diffusion_denoiser/export_official_real_packed_latent_boundary.py"),
                "--model",
                str(transformer_dir),
                "--model-root",
                str(model_root),
                "--out-dir",
                str(denoiser_dir),
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
            expected=(denoiser_hmonnx,),
            note="Full profile attempts the monolithic 36-layer boundary and can be heavy.",
        ),
        QuantStage(
            name=f"denoiser_realpack_{num_layers}layers_compare",
            command=add_debug(
                args,
                [
                    *py(args, "compare/generator/diffusion_denoiser/compare_official_real_packed_latent_boundary_hmonnx.py"),
                    "--model",
                    str(transformer_dir),
                    "--model-root",
                    str(model_root),
                    "--hmonnx",
                    str(denoiser_hmonnx),
                    "--report",
                    str(denoiser_report),
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
                    "--seed",
                    str(args.seed),
                    "--device",
                    args.runtime_device,
                    "--input-dtype",
                    args.input_dtype,
                ],
            ),
            expected=(denoiser_report,),
            reports=(denoiser_report,),
        ),
        QuantStage(
            name=f"scheduler_loop_{num_layers}layers",
            command=add_debug(
                args,
                [
                    *py(args, "compare/generator/scheduler_loop/compare_cfg_scheduler_loop_hmonnx.py"),
                    "--model",
                    str(transformer_dir),
                    "--model-root",
                    str(model_root),
                    "--hmonnx",
                    str(denoiser_hmonnx),
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
                    "--num-inference-steps",
                    str(args.num_inference_steps),
                    "--guidance-scale",
                    str(args.guidance_scale),
                    "--torch-device",
                    args.runtime_device,
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
            name="vae_encoder",
            command=[
                *py(args, "export/generator/vae_encoder/export_vae_encoder.py"),
                "--model",
                str(model_root),
                "--out-dir",
                str(vae_encoder_dir),
                "--frames",
                str(args.frames),
                "--height",
                str(args.latent_height * 16),
                "--width",
                str(args.latent_width * 16),
                "--dtype",
                args.dtype,
                "--device",
                args.export_device,
                "--quant-type",
                args.vae_quant_type,
                "--lower-conv3d-to-2d",
                "--convert-hmonnx",
            ],
            expected=(vae_encoder_hmonnx,),
        ),
        QuantStage(
            name="vae_encoder_compare",
            command=add_debug(
                args,
                [
                    *py(args, "compare/generator/vae_encoder/compare_vae_encoder_hmonnx.py"),
                    "--model",
                    str(model_root),
                    "--hmonnx",
                    str(vae_encoder_hmonnx),
                    "--report",
                    str(vae_encoder_report),
                    "--frames",
                    str(args.frames),
                    "--height",
                    str(args.latent_height * 16),
                    "--width",
                    str(args.latent_width * 16),
                    "--device",
                    args.runtime_device,
                    "--input-dtype",
                    args.input_dtype,
                ],
            ),
            expected=(vae_encoder_report,),
            reports=(vae_encoder_report,),
        ),
        QuantStage(
            name="vae_decoder_pre_mid",
            command=add_debug(
                args,
                [
                    *py(args, "compare/generator/vae_decoder/locate_vae_decoder_stage_hmonnx.py"),
                    "--model",
                    str(model_root),
                    "--out-dir",
                    str(vae_dir),
                    "--stage",
                    "mid",
                    "--latent-frames",
                    str(args.frames),
                    "--latent-height",
                    str(args.latent_height),
                    "--latent-width",
                    str(args.latent_width),
                    "--quant-type",
                    args.vae_quant_type,
                    "--device",
                    args.runtime_device,
                    "--input-dtype",
                    args.input_dtype,
                    "--normalize-force-fp32",
                    "--official-cache-loop",
                ],
            ),
            expected=(vae_pre_mid,),
        ),
    ]

    for stage in VAE_SPLIT_STAGES:
        stage_hmonnx = vae_dir / vae_split_name(stage, args.frames, args.latent_height, args.latent_width, args.vae_quant_type)
        stages.append(
            QuantStage(
                name=f"vae_decoder_{stage}",
                command=add_debug(
                    args,
                    [
                        *py(args, "compare/generator/vae_decoder/compare_vae_decoder_split_stage_hmonnx.py"),
                        "--model",
                        str(model_root),
                        "--out-dir",
                        str(vae_dir),
                        "--stage",
                        stage,
                        "--latent-frames",
                        str(args.frames),
                        "--latent-height",
                        str(args.latent_height),
                        "--latent-width",
                        str(args.latent_width),
                        "--quant-type",
                        args.vae_quant_type,
                        "--device",
                        args.runtime_device,
                        "--input-dtype",
                        args.input_dtype,
                    ],
                ),
                expected=(stage_hmonnx,),
            )
        )

    stages.append(
        QuantStage(
            name="vae_decoder_full_split_chain_compare",
            command=add_debug(
                args,
                [
                    *py(args, "compare/generator/vae_decoder/compare_vae_decoder_full_split_chain_hmonnx.py"),
                    "--model",
                    str(model_root),
                    "--split-dir",
                    str(vae_dir),
                    "--report",
                    str(vae_full_report),
                    "--latent-frames",
                    str(args.frames),
                    "--latent-height",
                    str(args.latent_height),
                    "--latent-width",
                    str(args.latent_width),
                    "--quant-type",
                    args.vae_quant_type,
                    "--device",
                    args.runtime_device,
                    "--input-dtype",
                    args.input_dtype,
                ],
            ),
            expected=(vae_full_report,),
            reports=(vae_full_report,),
        )
    )

    if not args.skip_sound_tokenizer:
        stages.extend(
            [
                QuantStage(
                    name="sound_tokenizer_decoder_split",
                    command=[
                        *py(args, "export/generator/audio_codec/export_sound_tokenizer_decoder_split.py"),
                        "--sound-tokenizer-dir",
                        str(sound_tokenizer_dir),
                        "--out-dir",
                        str(sound_dir),
                        "--latent-frames",
                        str(sound_latent_frames),
                        "--dtype",
                        args.dtype,
                        "--device",
                        args.export_device,
                        "--quant-type",
                        args.sound_quant_type,
                        "--convert-hmonnx",
                    ],
                    expected=sound_paths,
                    note="Cosmos3-Nano package contains sound tokenizer decoder weights only; encoder is not exported here.",
                ),
                QuantStage(
                    name="sound_tokenizer_decoder_split_compare",
                    command=add_debug(
                        args,
                        [
                            *py(args, "compare/generator/audio_codec/compare_sound_tokenizer_decoder_split_chain_hmonnx.py"),
                            "--sound-tokenizer-dir",
                            str(sound_tokenizer_dir),
                            "--split-dir",
                            str(sound_dir),
                            "--report",
                            str(sound_report),
                            "--latent-frames",
                            str(sound_latent_frames),
                            "--quant-type",
                            args.sound_quant_type,
                            "--hmonnx-device",
                            args.runtime_device,
                            "--input-dtype",
                            args.input_dtype,
                        ],
                    ),
                    expected=(sound_report,),
                    reports=(sound_report,),
                ),
            ]
        )

    stages.append(
        QuantStage(
            name=f"generator_vae_pipeline_{num_layers}layers",
            command=add_debug(
                args,
                [
                    *py(args, "compare/generator/pipeline/compare_generator_vae_pipeline_hmonnx.py"),
                    "--model",
                    str(transformer_dir),
                    "--model-root",
                    str(model_root),
                    "--denoiser-hmonnx",
                    str(denoiser_hmonnx),
                    "--vae-dir",
                    str(vae_dir),
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
                    "--num-inference-steps",
                    str(args.num_inference_steps),
                    "--guidance-scale",
                    str(args.guidance_scale),
                    "--quant-type",
                    args.vae_quant_type,
                    "--torch-device",
                    args.runtime_device,
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
        )
    )
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
    payload = {
        "model_root": str(args.model_root.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "profile": args.profile,
        "quant_type": args.quant_type,
        "vae_quant_type": args.vae_quant_type,
        "sound_quant_type": args.sound_quant_type,
        "normalize_force_fp32": True,
        "sound_tokenizer_encoder": "not present in Cosmos3-Nano local weights; decoder-only package",
    }
    write_summary(args.out_dir / "quantize_generator_summary.json", payload, results)
    write_markdown_summary(args.out_dir / "quantize_generator_summary.md", "Cosmos3-Nano Generator Quantization", results)


if __name__ == "__main__":
    main()
