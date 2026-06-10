#!/usr/bin/env python3
"""Static FP compute profile for Wan2.2 A14B and TI2V-5B.

This tool intentionally avoids loading weights.  It profiles the official
Wan2.2 module structure from config-level dimensions and reports FLOPs at the
module boundary needed for architecture review.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable


TFLOP = 1e12
PFLOP = 1e15
GIB = 1024**3


@dataclass(frozen=True)
class Shape:
    name: str
    width: int
    height: int
    frames: int = 81
    vae_t_stride: int = 4
    vae_h_stride: int = 8
    vae_w_stride: int = 8
    latent_channels: int = 16
    fps: int = 16

    @property
    def f_lat(self) -> int:
        return (self.frames - 1) // self.vae_t_stride + 1

    @property
    def h_lat(self) -> int:
        return self.height // self.vae_h_stride

    @property
    def w_lat(self) -> int:
        return self.width // self.vae_w_stride

    @property
    def tokens(self) -> int:
        return self.f_lat * (self.h_lat // 2) * (self.w_lat // 2)

    @property
    def duration_sec(self) -> float:
        return self.frames / self.fps


@dataclass(frozen=True)
class Arch:
    name: str
    dim: int
    ffn_dim: int
    num_layers: int
    text_len: int = 512
    patch_volume: int = 4
    out_dim: int = 16
    param_count: int = 14_288_706_624
    experts: int = 2
    vae: str = "wan2.1"


ARCHES = {
    "a14b": Arch(
        name="Wan2.2-A14B",
        dim=5120,
        ffn_dim=13824,
        num_layers=40,
        out_dim=16,
        param_count=14_288_706_624,
        experts=2,
        vae="wan2.1",
    ),
    "ti2v-5b": Arch(
        name="Wan2.2-TI2V-5B",
        dim=3072,
        ffn_dim=14336,
        num_layers=30,
        out_dim=48,
        param_count=5_000_000_000,
        experts=1,
        vae="wan2.2",
    ),
}


def fmt_flops(value: float) -> str:
    if value >= PFLOP:
        return f"{value / PFLOP:.3f} PFLOPs"
    return f"{value / TFLOP:.3f} TFLOPs"


def linear_flops(tokens: int, in_dim: int, out_dim: int) -> int:
    return 2 * tokens * in_dim * out_dim


def conv3d_patch_flops(tokens: int, in_dim: int, out_dim: int, patch_volume: int) -> int:
    return 2 * tokens * in_dim * patch_volume * out_dim


def wan_dit_forward(shape: Shape, arch: Arch, in_dim: int | None = None) -> dict[str, float]:
    s = shape.tokens
    c = arch.dim
    ffn = arch.ffn_dim
    text_len = arch.text_len
    layers = arch.num_layers
    patch_volume = arch.patch_volume
    in_dim = shape.latent_channels if in_dim is None else in_dim

    patch = conv3d_patch_flops(s, in_dim, c, patch_volume)
    text_embed = linear_flops(text_len, 4096, c) + linear_flops(text_len, c, c)
    time_embed = linear_flops(s, 256, c) + linear_flops(s, c, c) + linear_flops(s, c, c * 6)

    self_proj = 4 * linear_flops(s, c, c)
    self_attn = 4 * s * s * c
    cross_proj = linear_flops(s, c, c) + 2 * linear_flops(text_len, c, c) + linear_flops(s, c, c)
    cross_attn = 4 * s * text_len * c
    ffn_flops = 2 * linear_flops(s, c, ffn)
    norms_mod = 16 * s * c
    block = self_proj + self_attn + cross_proj + cross_attn + ffn_flops + norms_mod
    head = linear_flops(s, c, arch.out_dim * patch_volume)

    total = patch + text_embed + time_embed + layers * block + head
    return {
        "patch_embed": patch,
        "text_embed_in_wan": text_embed,
        "time_embed_modulation": time_embed,
        "self_attn_projections_per_layer": self_proj,
        "self_attn_qk_av_per_layer": self_attn,
        "cross_attn_projections_per_layer": cross_proj,
        "cross_attn_qk_av_per_layer": cross_attn,
        "ffn_per_layer": ffn_flops,
        "norms_modulation_per_layer": norms_mod,
        "block_per_layer": block,
        "blocks_all_layers": layers * block,
        "head_unpatchify": head,
        "single_forward_total": total,
    }


def t5_encoder(prompt_count: int = 2, text_len: int = 512) -> dict[str, float]:
    d = 4096
    d_attn = 4096
    ffn = 10240
    layers = 24

    attn_proj = 4 * linear_flops(text_len, d, d_attn)
    attn_score = 4 * text_len * text_len * d_attn
    ffn_flops = 3 * linear_flops(text_len, d, ffn)
    norms = 8 * text_len * d
    layer = attn_proj + attn_score + ffn_flops + norms
    single_prompt = layers * layer
    return {
        "t5_self_attn_proj_per_layer": attn_proj,
        "t5_self_attn_qk_av_per_layer": attn_score,
        "t5_ffn_per_layer": ffn_flops,
        "t5_layer_total": layer,
        "t5_single_prompt_total": single_prompt,
        "t5_cond_uncond_total": single_prompt * prompt_count,
    }


def vae_static_estimate(shape: Shape, mode: str = "decode", vae: str = "wan2.1") -> dict[str, float]:
    """Conservative static estimate for Wan VAE.

    Official VAE code uses causal 3D convs with temporal chunk cache. This
    counts equivalent full-volume convolutions at each resolution stage and is
    meant for relative module sizing, not kernel-level benchmarking.
    """

    if vae == "wan2.2":
        if mode == "decode":
            stages = [
                (48, 1024, shape.f_lat, shape.h_lat, shape.w_lat, 4),
                (1024, 1024, shape.f_lat, shape.h_lat, shape.w_lat, 9),
                (1024, 1024, min(shape.frames, shape.f_lat * 2), shape.h_lat * 2, shape.w_lat * 2, 9),
                (1024, 512, min(shape.frames, shape.f_lat * 4), shape.h_lat * 4, shape.w_lat * 4, 9),
                (512, 256, shape.frames, shape.h_lat * 8, shape.w_lat * 8, 9),
                (256, 12, shape.frames, shape.h_lat * 8, shape.w_lat * 8, 1),
            ]
        else:
            # Wan2.2 VAE patchifies input video spatially by 2 before encode,
            # so the final pre-latent feature grid is H/16 x W/16.
            stages = [
                (12, 160, shape.frames, shape.h_lat * 8, shape.w_lat * 8, 1),
                (160, 160, shape.frames, shape.h_lat * 8, shape.w_lat * 8, 6),
                (160, 320, min(shape.frames, shape.f_lat * 4), shape.h_lat * 4, shape.w_lat * 4, 6),
                (320, 640, min(shape.frames, shape.f_lat * 2), shape.h_lat * 2, shape.w_lat * 2, 6),
                (640, 640, shape.f_lat, shape.h_lat, shape.w_lat, 6),
                (640, 96, shape.f_lat, shape.h_lat, shape.w_lat, 1),
            ]
    elif mode == "decode":
        # channel, temporal, height, width snapshots through decoder stages.
        stages = [
            (16, 384, shape.f_lat, shape.h_lat, shape.w_lat, 3),  # latent -> hidden + middle
            (384, 384, shape.f_lat, shape.h_lat, shape.w_lat, 7),
            (384, 384, min(shape.frames, shape.f_lat * 2), shape.h_lat * 2, shape.w_lat * 2, 7),
            (384, 192, min(shape.frames, shape.f_lat * 4), shape.h_lat * 4, shape.w_lat * 4, 7),
            (192, 96, shape.frames, shape.h_lat * 8, shape.w_lat * 8, 7),
            (96, 3, shape.frames, shape.height, shape.width, 1),
        ]
    else:
        stages = [
            (3, 96, shape.frames, shape.height, shape.width, 1),
            (96, 192, shape.frames, shape.h_lat * 8, shape.w_lat * 8, 5),
            (192, 384, min(shape.frames, shape.f_lat * 4), shape.h_lat * 4, shape.w_lat * 4, 5),
            (384, 384, min(shape.frames, shape.f_lat * 2), shape.h_lat * 2, shape.w_lat * 2, 5),
            (384, 32, shape.f_lat, shape.h_lat, shape.w_lat, 3),
        ]

    total = 0
    rows = {}
    for idx, (cin, cout, t, h, w, convs) in enumerate(stages, start=1):
        # Most blocks use 3x3x3 causal convs.  Some resample convs are 2D/1x1;
        # using 27 here is a conservative upper estimate.
        flops = convs * 2 * t * h * w * cin * cout * 27
        rows[f"vae_{mode}_stage_{idx}"] = flops
        total += flops
    rows[f"vae_{mode}_total_static_estimate"] = total
    return rows


def param_memory(arch: Arch) -> dict[str, float]:
    single_expert_params = arch.param_count
    return {
        "dit_single_expert_fp32_gib": single_expert_params * 4 / GIB,
        "dit_single_expert_bf16_gib": single_expert_params * 2 / GIB,
        "dit_all_experts_fp32_gib": single_expert_params * arch.experts * 4 / GIB,
        "dit_all_experts_bf16_gib": single_expert_params * arch.experts * 2 / GIB,
        "t5_file_gib": 10.58,
        "vae_file_gib": 0.47 if arch.vae == "wan2.1" else 1.0,
    }


def markdown_table(title: str, rows: Iterable[tuple[str, float]]) -> str:
    lines = [f"### {title}", "", "| 模块 | FLOPs |", "| --- | ---: |"]
    for name, value in rows:
        lines.append(f"| {name} | {fmt_flops(value)} |")
    return "\n".join(lines)


def build_markdown(shape: Shape, arch: Arch, steps: int = 40, cfg_passes: int = 2, i2v: bool = False) -> str:
    dit_in_dim = 36 if i2v and arch.name.endswith("A14B") else shape.latent_channels
    dit = wan_dit_forward(shape, arch, in_dim=dit_in_dim)
    t5 = t5_encoder(prompt_count=cfg_passes)
    vae_decode = vae_static_estimate(shape, "decode", arch.vae)
    vae_encode = vae_static_estimate(shape, "encode", arch.vae) if i2v else {}

    single = dit["single_forward_total"]
    per_step = single * cfg_passes
    denoise_total = per_step * steps
    end_to_end = t5["t5_cond_uncond_total"] + denoise_total + vae_decode["vae_decode_total_static_estimate"]
    if i2v:
        end_to_end += vae_encode["vae_encode_total_static_estimate"]

    rows = [
        ("T5 encoder cond+uncond", t5["t5_cond_uncond_total"]),
        ("Wan DiT single forward", single),
        ("Wan DiT CFG per sampling step", per_step),
        (f"Wan DiT denoise loop ({steps} steps)", denoise_total),
    ]
    if i2v:
        rows.append(("VAE encode image condition (static estimate)", vae_encode["vae_encode_total_static_estimate"]))
    rows.extend(
        [
            ("VAE decode final latent (static estimate)", vae_decode["vae_decode_total_static_estimate"]),
            ("End-to-end generation estimate", end_to_end),
        ]
    )

    md = [
        f"## {arch.name} {'I2V' if i2v else 'T2V'} {shape.name} 模块级 FLOPs",
        "",
        f"- 视频 shape: `[3,{shape.frames},{shape.height},{shape.width}]`",
        f"- 输出时长: `{shape.duration_sec:.2f}s` ({shape.frames} frames / {shape.fps} fps)",
        f"- latent shape: `[{shape.latent_channels},{shape.f_lat},{shape.h_lat},{shape.w_lat}]`",
        f"- DiT patch tokens: `{shape.tokens:,}`",
        f"- 采样步数: `{steps}`；CFG 前向次数/步: `{cfg_passes}`",
        "",
        markdown_table("端到端模块拆分", rows),
        "",
        markdown_table(
            "Wan DiT 单次前向内部拆分",
            [
                ("Patch embedding", dit["patch_embed"]),
                ("Wan text embedding", dit["text_embed_in_wan"]),
                ("Time embedding + modulation", dit["time_embed_modulation"]),
                ("Self-attn projections / layer", dit["self_attn_projections_per_layer"]),
                ("Self-attn QK+AV / layer", dit["self_attn_qk_av_per_layer"]),
                ("Cross-attn projections / layer", dit["cross_attn_projections_per_layer"]),
                ("Cross-attn QK+AV / layer", dit["cross_attn_qk_av_per_layer"]),
                ("FFN / layer", dit["ffn_per_layer"]),
                ("Norm + modulation / layer", dit["norms_modulation_per_layer"]),
                (f"{arch.num_layers} transformer blocks", dit["blocks_all_layers"]),
                ("Head + unpatchify linear", dit["head_unpatchify"]),
                ("Single WanModel forward total", dit["single_forward_total"]),
            ],
        ),
        "",
        markdown_table(
            "T5 encoder 内部拆分",
            [
                ("Self-attn projections / layer", t5["t5_self_attn_proj_per_layer"]),
                ("Self-attn QK+AV / layer", t5["t5_self_attn_qk_av_per_layer"]),
                ("Gated FFN / layer", t5["t5_ffn_per_layer"]),
                ("Single layer total", t5["t5_layer_total"]),
                ("Single prompt 24-layer encoder", t5["t5_single_prompt_total"]),
                ("Cond + uncond prompts", t5["t5_cond_uncond_total"]),
            ],
        ),
        "",
        markdown_table("VAE decode 静态近似拆分", vae_decode.items()),
    ]
    if i2v:
        md.extend(["", markdown_table("VAE encode 静态近似拆分", vae_encode.items())])

    mem = param_memory(arch)
    md.extend(
        [
            "",
            "### 显存/权重口径",
            "",
            "| 项 | GiB |",
            "| --- | ---: |",
            f"| DiT 单 expert FP32 | {mem['dit_single_expert_fp32_gib']:.2f} |",
            f"| DiT 单 expert BF16 | {mem['dit_single_expert_bf16_gib']:.2f} |",
            f"| DiT 总 expert FP32 | {mem['dit_all_experts_fp32_gib']:.2f} |",
            f"| DiT 总 expert BF16 | {mem['dit_all_experts_bf16_gib']:.2f} |",
            f"| T5 encoder 文件 | {mem['t5_file_gib']:.2f} |",
            f"| VAE 文件 | {mem['vae_file_gib']:.2f} |",
            "",
            "> VAE 为静态卷积近似：官方实现使用 causal 3D conv + temporal chunk cache，真实 kernel 级 FLOPs/带宽需 profiler 复核。",
        ]
    )
    return "\n".join(md)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=sorted(ARCHES), default="a14b")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--vae-h-stride", type=int, default=None)
    parser.add_argument("--vae-w-stride", type=int, default=None)
    parser.add_argument("--latent-channels", type=int, default=None)
    parser.add_argument("--name", default="720P")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--i2v", action="store_true")
    args = parser.parse_args()

    arch = ARCHES[args.arch]
    default_spatial_stride = 16 if args.arch == "ti2v-5b" else 8
    default_latent_channels = 48 if args.arch == "ti2v-5b" else 16
    shape = Shape(
        args.name,
        args.width,
        args.height,
        args.frames,
        fps=args.fps,
        vae_h_stride=args.vae_h_stride or default_spatial_stride,
        vae_w_stride=args.vae_w_stride or default_spatial_stride,
        latent_channels=args.latent_channels or default_latent_channels,
    )
    print(build_markdown(shape, arch, steps=args.steps, i2v=args.i2v))


if __name__ == "__main__":
    main()
