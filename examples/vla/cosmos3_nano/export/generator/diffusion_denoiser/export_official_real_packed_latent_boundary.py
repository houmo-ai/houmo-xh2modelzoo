# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export Cosmos3-Nano generator latent boundary with official host packing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


_COSMOS3_ROOT = Path(__file__).resolve().parents[3]
if str(_COSMOS3_ROOT) not in sys.path:
    sys.path.insert(0, str(_COSMOS3_ROOT))

from common.paths import default_model_root  # noqa: E402
from common.official_cosmos3 import DEFAULT_OFFICIAL_TRANSFORMER, load_official_transformer  # noqa: E402
from export.common_quant.attention.export_transformer_rope import torch_dtype  # noqa: E402
from export.common_quant.quant_config import enable_normalize_force_fp32  # noqa: E402
from runtime.generator_runtime import (  # noqa: E402
    OfficialCosmos3LatentDenoiserBoundaryWrapper,
    make_generator_real_packed_latent_boundary_inputs,
)


DEFAULT_MODEL_ROOT = default_model_root()
DEFAULT_OUTPUT_ROOT = _COSMOS3_ROOT / "data" / "generator_official_real_packed_latent_boundary_2layers_smoke"
DEFAULT_PROMPT = "a cinematic photo of a red sports car on a mountain road at sunset"


def graph_suffix(num_layers: int, frames: int, latent_height: int, latent_width: int) -> str:
    return f"layers0_{num_layers - 1}_realpack_und8_gen4_t{frames}_lh{latent_height}_lw{latent_width}"


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=Path, default=DEFAULT_OFFICIAL_TRANSFORMER)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--latent-height", type=int, default=4)
    parser.add_argument("--latent-width", type=int, default=4)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--timestep", type=float, default=0.5)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--quant-type", default="w8a16_sefp")
    parser.add_argument("--hmonnx-name", default="")
    parser.add_argument("--convert-hmonnx", action="store_true")
    parser.add_argument("--normalize-force-fp32", action="store_true", default=True)
    return parser.parse_args()


def export_onnx(
    wrapper: torch.nn.Module,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    input_names: list[str],
    output_names: list[str],
    onnx_path: Path,
    opset: int,
) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            inputs,
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            do_constant_folding=True,
        )


def convert_hmonnx(
    onnx_path: Path,
    hmonnx_path: Path,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    input_names: list[str],
    output_names: list[str],
    quant_type: str,
    normalize_force_fp32: bool,
) -> None:
    from xhquant.api import DeviceType, QuantScheme, convert_onnx_to_hmonnx, create_quant_config, xhquant_init

    xhquant_init(None)
    quant_config = create_quant_config(QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type))
    if normalize_force_fp32:
        quant_config = enable_normalize_force_fp32(quant_config)
    hmonnx_path.parent.mkdir(parents=True, exist_ok=True)
    convert_onnx_to_hmonnx(
        str(onnx_path),
        [item.detach().cpu() for item in inputs],
        DeviceType.XH2a,
        str(hmonnx_path),
        quant_config=quant_config,
        input_names=input_names,
        output_names=output_names,
    )


def main() -> None:
    args = parse_args()
    dtype = torch_dtype(args.dtype)
    device = torch.device(args.device)

    transformer = load_official_transformer(args.model, torch_dtype=dtype).to(device)
    inputs, input_names, output_names, pack_meta = make_generator_real_packed_latent_boundary_inputs(
        transformer=transformer,
        model_root=args.model_root,
        prompt=args.prompt,
        frames=args.frames,
        latent_height=args.latent_height,
        latent_width=args.latent_width,
        seed=args.seed,
        dtype=dtype,
        device=device,
        timestep=args.timestep,
    )
    und_seq = int(pack_meta["und_seq"])
    gen_seq = int(pack_meta["gen_seq"])
    wrapper = OfficialCosmos3LatentDenoiserBoundaryWrapper(
        transformer,
        start_layer=args.start_layer,
        num_layers=args.num_layers,
        und_seq=und_seq,
        frames=args.frames,
        latent_height=args.latent_height,
        latent_width=args.latent_width,
    ).to(device=device, dtype=dtype)
    wrapper.eval()

    suffix = graph_suffix(args.num_layers, args.frames, args.latent_height, args.latent_width)
    onnx_path = args.out_dir / f"cosmos3_nano_generator_{suffix}.onnx"
    hmonnx_name = args.hmonnx_name or f"cosmos3_nano_generator_{suffix}.hmonnx.onnx"
    hmonnx_path = args.out_dir / hmonnx_name
    meta_path = args.out_dir / f"{suffix}_meta.json"

    export_onnx(wrapper, inputs, input_names, output_names, onnx_path, args.opset)
    if args.convert_hmonnx:
        convert_hmonnx(
            onnx_path,
            hmonnx_path,
            inputs,
            input_names,
            output_names,
            args.quant_type,
            args.normalize_force_fp32,
        )

    payload = {
        "model": str(args.model),
        "model_root": str(args.model_root),
        "onnx": str(onnx_path),
        "hmonnx": str(hmonnx_path) if args.convert_hmonnx else None,
        "input_names": input_names,
        "output_names": output_names,
        "inputs": {name: tensor_summary(tensor) for name, tensor in zip(input_names, inputs)},
        "pack": pack_meta,
        "start_layer": args.start_layer,
        "num_layers": args.num_layers,
        "frames": args.frames,
        "latent_height": args.latent_height,
        "latent_width": args.latent_width,
        "dtype": args.dtype,
        "quant_type": args.quant_type,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
