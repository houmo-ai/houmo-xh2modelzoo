# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export and validate the VLA-JEPA Qwen3-VL visual encoder boundary."""

from __future__ import annotations

import sys
from pathlib import Path

_VLA_JEPA_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _VLA_JEPA_ROOT.parents[2]
for _path in (_VLA_JEPA_ROOT, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torch import nn

from export.export_action_head import tensor_summary
from export.export_context_encoder import DEFAULT_OUT_DIR, DEFAULT_SAMPLE, load_sample
from common.inspect_policy import DEFAULT_MODEL, load_policy
from common.paths import set_default_libero_config_path



class QwenVisualPatchEmbedLinear(nn.Module):
    """Linear equivalent of HF Qwen3-VL Conv3d patch embedding.

    VLA-JEPA already feeds flattened patch pixels as ``[num_patches, 1536]``.
    The HF Conv3d uses a kernel covering the whole ``[3, 2, 16, 16]`` patch,
    so it is exactly a linear projection over the flattened patch.
    """

    def __init__(self, patch_embed: nn.Module) -> None:
        super().__init__()
        self.patch_size = patch_embed.patch_size
        self.temporal_patch_size = patch_embed.temporal_patch_size
        self.in_channels = patch_embed.in_channels
        self.embed_dim = patch_embed.embed_dim
        in_features = self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
        self.proj = nn.Linear(in_features, self.embed_dim, bias=patch_embed.proj.bias is not None)
        self.proj.weight.data.copy_(patch_embed.proj.weight.detach().reshape(self.embed_dim, in_features))
        if patch_embed.proj.bias is not None:
            self.proj.bias.data.copy_(patch_embed.proj.bias.detach())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states.to(dtype=self.proj.weight.dtype))


def replace_visual_patch_embed_for_export(qwen_model: nn.Module) -> str:
    visual = qwen_model.model.visual
    previous = type(visual.patch_embed).__name__
    if not isinstance(visual.patch_embed, QwenVisualPatchEmbedLinear):
        visual.patch_embed = QwenVisualPatchEmbedLinear(visual.patch_embed).to(
            device=visual.pos_embed.weight.device,
            dtype=visual.pos_embed.weight.dtype,
        )
    return previous


def configure_visual_for_export(qwen_model: nn.Module) -> str:
    """Use traceable visual attention for ONNX export.

    HF Qwen3-VL defaults to SDPA here. PyTorch ONNX export currently cannot
    lower scaled_dot_product_attention when GQA is enabled, while the eager path
    is explicit MatMul + Softmax and matches the project wrapper strategy.
    """
    visual = qwen_model.model.visual
    previous = getattr(visual.config, "_attn_implementation", None)
    visual.config._attn_implementation = "eager"
    return str(previous)


class QwenVisualAttentionNoSplit(nn.Module):
    """Traceable visual attention without the fused qkv Split node."""

    def __init__(self, attention: nn.Module, chunk_lengths: list[int]) -> None:
        super().__init__()
        self.dim = attention.dim
        self.num_heads = attention.num_heads
        self.head_dim = attention.head_dim
        self.scaling = attention.scaling
        self.chunk_offsets = self._make_offsets(chunk_lengths)
        self.num_chunks = len(self.chunk_offsets)
        for chunk_idx, (start, end) in enumerate(self.chunk_offsets):
            self.register_buffer(
                f"chunk_indices_{chunk_idx}",
                torch.arange(start, end).long(),
                persistent=False,
            )
        half_dim = self.head_dim // 2
        rotate_indices = torch.cat((torch.arange(half_dim, self.head_dim), torch.arange(0, half_dim))).long()
        rotate_sign = torch.cat((-torch.ones(half_dim), torch.ones(half_dim)))
        self.register_buffer("rotate_indices", rotate_indices, persistent=False)
        self.register_buffer("rotate_sign", rotate_sign, persistent=False)

        self.q_proj = nn.Linear(self.dim, self.dim, bias=True)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=True)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=True)
        self.proj = attention.proj

        qkv_weight = attention.qkv.weight.detach()
        qkv_bias = attention.qkv.bias.detach()
        self.q_proj.weight.data.copy_(qkv_weight[: self.dim])
        self.k_proj.weight.data.copy_(qkv_weight[self.dim : 2 * self.dim])
        self.v_proj.weight.data.copy_(qkv_weight[2 * self.dim :])
        self.q_proj.bias.data.copy_(qkv_bias[: self.dim])
        self.k_proj.bias.data.copy_(qkv_bias[self.dim : 2 * self.dim])
        self.v_proj.bias.data.copy_(qkv_bias[2 * self.dim :])

        self.q_proj.to(device=qkv_weight.device, dtype=qkv_weight.dtype)
        self.k_proj.to(device=qkv_weight.device, dtype=qkv_weight.dtype)
        self.v_proj.to(device=qkv_weight.device, dtype=qkv_weight.dtype)

    @staticmethod
    def _make_offsets(chunk_lengths: list[int]) -> list[tuple[int, int]]:
        offsets = []
        start = 0
        for length in chunk_lengths:
            end = start + int(length)
            offsets.append((start, end))
            start = end
        return offsets

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        indices = self.rotate_indices.to(device=x.device)
        sign = self.rotate_sign.to(device=x.device, dtype=x.dtype)
        return x.index_select(-1, indices) * sign

    def apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_q_dtype = q.dtype
        orig_k_dtype = k.dtype
        q = q.float()
        k = k.float()
        cos = cos.unsqueeze(-2).float()
        sin = sin.unsqueeze(-2).float()
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del cu_seqlens, rotary_pos_emb, kwargs
        seq_length = hidden_states.shape[0]
        query_states = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, self.head_dim)

        if position_embeddings is None:
            raise ValueError("position_embeddings is required for QwenVisualAttentionNoSplit")
        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).reshape(1, self.num_heads, seq_length, self.head_dim)
        key_states = key_states.transpose(0, 1).reshape(1, self.num_heads, seq_length, self.head_dim)
        value_states = value_states.transpose(0, 1).reshape(1, self.num_heads, seq_length, self.head_dim)

        attn_outputs = []
        for chunk_idx in range(self.num_chunks):
            indices = getattr(self, f"chunk_indices_{chunk_idx}").to(device=query_states.device)
            query_chunk = query_states.index_select(2, indices)
            key_chunk = key_states.index_select(2, indices)
            value_chunk = value_states.index_select(2, indices)
            attn_weights = torch.matmul(query_chunk, key_chunk.transpose(2, 3)) * self.scaling
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_chunk.dtype)
            attn_output = torch.matmul(attn_weights, value_chunk)
            attn_outputs.append(attn_output.transpose(1, 2))

        attn_output = torch.cat(attn_outputs, dim=1).reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)


def replace_visual_attention_for_export(qwen_model: nn.Module, image_grid_thw: torch.Tensor) -> dict[str, Any]:
    visual = qwen_model.model.visual
    grid_thw = image_grid_thw.detach().cpu()
    chunk_lengths = (grid_thw[:, 1] * grid_thw[:, 2]).repeat_interleave(grid_thw[:, 0]).tolist()
    chunk_lengths = [int(item) for item in chunk_lengths]
    previous = []
    replaced = 0
    for block in visual.blocks:
        previous.append(type(block.attn).__name__)
        if not isinstance(block.attn, QwenVisualAttentionNoSplit):
            block.attn = QwenVisualAttentionNoSplit(block.attn, chunk_lengths)
            replaced += 1
    return {
        "previous": sorted(set(previous)),
        "export": "QwenVisualAttentionNoSplit",
        "replaced_blocks": replaced,
        "chunk_lengths": chunk_lengths,
    }


class QwenVisualEncoderReference(nn.Module):
    """Original HF visual boundary used by VLA-JEPA."""

    def __init__(self, qwen_model: nn.Module) -> None:
        super().__init__()
        self.qwen_model = qwen_model

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.qwen_model.model.get_image_features(
            pixel_values,
            image_grid_thw,
            return_dict=True,
        )
        image_embeds = torch.cat(outputs.pooler_output, dim=0)
        deepstack_features = tuple(outputs.deepstack_features)
        return (
            image_embeds,
            deepstack_features[0],
            deepstack_features[1],
            deepstack_features[2],
        )


class QwenVisualEncoder(nn.Module):
    """Static-shape export wrapper for the VLA-JEPA Qwen3-VL visual tower."""

    def __init__(self, qwen_model: nn.Module, image_grid_thw: torch.Tensor) -> None:
        super().__init__()
        self.visual = qwen_model.model.visual
        grid_thw = image_grid_thw.detach().to(device=self.visual.pos_embed.weight.device)

        with torch.no_grad():
            pos_embeds = self.visual.fast_pos_embed_interpolate(grid_thw)
            rotary_pos_emb = self.visual.rot_pos_emb(grid_thw)
            seq_len = pos_embeds.shape[0]
            rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
                dim=0, dtype=grid_thw.dtype
            )
            cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

        self.register_buffer("pos_embeds", pos_embeds.detach(), persistent=False)
        self.register_buffer("cos", emb.cos().detach(), persistent=False)
        self.register_buffer("sin", emb.sin().detach(), persistent=False)
        self.register_buffer("cu_seqlens", cu_seqlens.detach(), persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = self.visual.patch_embed(pixel_values)
        hidden_states = hidden_states + self.pos_embeds.to(device=hidden_states.device, dtype=hidden_states.dtype)
        position_embeddings = (
            self.cos.to(device=hidden_states.device),
            self.sin.to(device=hidden_states.device),
        )
        cu_seqlens = self.cu_seqlens.to(device=hidden_states.device)

        deepstack_feature_lists = []
        for layer_num, block in enumerate(self.visual.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_num in self.visual.deepstack_visual_indexes:
                deepstack_feature = self.visual.deepstack_merger_list[
                    self.visual.deepstack_visual_indexes.index(layer_num)
                ](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        image_embeds = self.visual.merger(hidden_states)
        return (
            image_embeds,
            deepstack_feature_lists[0],
            deepstack_feature_lists[1],
            deepstack_feature_lists[2],
        )


def diff_summary(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    ref = reference.detach().cpu().float().numpy()
    cand = candidate.detach().cpu().float().numpy()
    diff = np.abs(ref - cand)
    return {
        "reference": tensor_summary(reference),
        "candidate": tensor_summary(candidate),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "p95_abs_diff": float(np.percentile(diff, 95)),
        "allclose_rtol_1e-4_atol_1e-4": bool(np.allclose(ref, cand, rtol=1e-4, atol=1e-4)),
        "allclose_rtol_1e-3_atol_1e-3": bool(np.allclose(ref, cand, rtol=1e-3, atol=1e-3)),
    }


def validate_onnx(
    onnx_path: Path,
    inputs: tuple[torch.Tensor, torch.Tensor],
    reference_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, Any]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = [item.name for item in session.get_inputs()]
    input_by_name = {"pixel_values": inputs[0]}
    if len(inputs) > 1:
        input_by_name["image_grid_thw"] = inputs[1]
    feed = {name: input_by_name[name].detach().cpu().numpy() for name in input_names}
    ort_outputs = session.run(None, feed)
    output_names = [item.name for item in session.get_outputs()]
    return {
        "input_names": input_names,
        "folded_inputs": [name for name in input_by_name if name not in input_names],
        "output_names": output_names,
        "outputs": {
            name: diff_summary(reference, torch.from_numpy(candidate))
            for name, reference, candidate in zip(output_names, reference_outputs, ort_outputs, strict=True)
        },
    }


def export_onnx(
    model: nn.Module,
    inputs: tuple[torch.Tensor, torch.Tensor],
    onnx_path: Path,
    opset: int,
) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        inputs,
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["pixel_values"] if len(inputs) == 1 else ["pixel_values", "image_grid_thw"],
        output_names=[
            "image_embeds",
            "deepstack_feature_0",
            "deepstack_feature_1",
            "deepstack_feature_2",
        ],
    )


def convert_to_hmonnx(
    onnx_path: Path,
    hmonnx_path: Path,
    inputs: tuple[torch.Tensor, torch.Tensor],
    quant_type: str,
    simplify: bool,
) -> dict[str, Any]:
    from xhquant.api import DeviceType, QuantScheme, convert_onnx_to_hmonnx, create_quant_config

    quant_config = create_quant_config(
        QuantScheme(
            target_device=DeviceType.XH2a,
            quant_type=quant_type,
        )
    )
    input_by_name = {"pixel_values": inputs[0]}
    if len(inputs) > 1:
        input_by_name["image_grid_thw"] = inputs[1]
    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    initializer_names = {item.name for item in onnx_model.graph.initializer}
    graph_input_names = [item.name for item in onnx_model.graph.input if item.name not in initializer_names]
    convert_inputs = [input_by_name[name].detach().cpu() for name in graph_input_names]

    hmonnx_path.parent.mkdir(parents=True, exist_ok=True)
    convert_onnx_to_hmonnx(
        str(onnx_path),
        convert_inputs,
        DeviceType.XH2a,
        str(hmonnx_path),
        quant_config,
        simplify=simplify,
    )
    return {
        "hmonnx_path": str(hmonnx_path),
        "quant_type": quant_type,
        "simplify": simplify,
        "graph_input_names": graph_input_names,
        "folded_inputs": [name for name in input_by_name if name not in graph_input_names],
        "size_mb": hmonnx_path.stat().st_size / 1024 / 1024,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample", default=DEFAULT_SAMPLE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--skip-ort", action="store_true")
    parser.add_argument("--convert-hmonnx", action="store_true")
    parser.add_argument("--quant-type", default="w8a8_sefp")
    parser.add_argument("--hmonnx-file", default=None)
    parser.add_argument("--no-simplify-hmonnx", action="store_true")
    args = parser.parse_args()

    set_default_libero_config_path()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(43)

    out_dir = Path(args.out_dir) / "qwen_visual_encoder"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "export_qwen_visual_encoder_report.json"
    onnx_path = out_dir / "vla_jepa_qwen_visual_encoder.onnx"
    hmonnx_path = (
        Path(args.hmonnx_file)
        if args.hmonnx_file
        else out_dir / f"vla_jepa_qwen_visual_encoder_{args.quant_type}.hmonnx.onnx"
    )

    sample = load_sample(args.sample, args.device)
    qwen_inputs = sample["qwen_inputs"]
    inputs = (
        qwen_inputs["pixel_values"],
        qwen_inputs["image_grid_thw"],
    )

    policy = load_policy(args.model, args.device, disable_world_model=True)
    policy.eval()
    qwen_model = policy.model.qwen.model
    previous_attention_impl = configure_visual_for_export(qwen_model)
    reference_wrapper = QwenVisualEncoderReference(qwen_model).eval().to(args.device)

    device_type = torch.device(args.device).type
    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        reference_outputs = reference_wrapper(*inputs)
        previous_patch_embed = replace_visual_patch_embed_for_export(qwen_model)
        attention_replacement = replace_visual_attention_for_export(qwen_model, inputs[1])
        wrapper = QwenVisualEncoder(qwen_model, inputs[1]).eval().to(args.device)
        candidate_outputs = wrapper(inputs[0])

    report: dict[str, Any] = {
        "model": args.model,
        "sample": args.sample,
        "device": args.device,
        "visual_attention_impl": {
            "previous": previous_attention_impl,
            "export": "eager",
        },
        "visual_patch_embed_impl": {
            "previous": previous_patch_embed,
            "export": type(qwen_model.model.visual.patch_embed).__name__,
        },
        "visual_attention_replacement": attention_replacement,
        "inputs": {
            "pixel_values": tensor_summary(inputs[0]),
            "image_grid_thw": tensor_summary(inputs[1]),
        },
        "outputs": {
            "image_embeds": diff_summary(reference_outputs[0], candidate_outputs[0]),
            "deepstack_feature_0": diff_summary(reference_outputs[1], candidate_outputs[1]),
            "deepstack_feature_1": diff_summary(reference_outputs[2], candidate_outputs[2]),
            "deepstack_feature_2": diff_summary(reference_outputs[3], candidate_outputs[3]),
        },
    }

    if args.export_onnx:
        try:
            export_model = wrapper.float().cpu()
            export_inputs = (inputs[0].float().cpu(),)
            with torch.no_grad():
                export_reference_outputs = export_model(*export_inputs)
            export_onnx(export_model, export_inputs, onnx_path, args.opset)
            onnx_model = onnx.load(str(onnx_path), load_external_data=True)
            onnx.checker.check_model(onnx_model)
        except Exception as exc:  # noqa: BLE001 - export diagnostics are part of route discovery.
            report["onnx_export"] = "failed"
            report["onnx_export_error_type"] = type(exc).__name__
            report["onnx_export_error"] = str(exc)
        else:
            report["onnx_export"] = "ok"
            report["onnx_path"] = str(onnx_path)
            report["onnx_size_mb"] = onnx_path.stat().st_size / 1024 / 1024
            if args.skip_ort:
                report["onnxruntime"] = "skipped by --skip-ort"
            else:
                report["onnxruntime"] = validate_onnx(onnx_path, export_inputs, export_reference_outputs)
    else:
        report["onnx_export"] = "skipped; pass --export-onnx to attempt export"

    if args.convert_hmonnx:
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX file does not exist: {onnx_path}")
        convert_inputs = (inputs[0].float().cpu(),)
        try:
            report["hmonnx"] = convert_to_hmonnx(
                onnx_path,
                hmonnx_path,
                convert_inputs,
                args.quant_type,
                simplify=not args.no_simplify_hmonnx,
            )
        except Exception as exc:  # noqa: BLE001 - conversion diagnostics are part of wrapper bring-up.
            report["hmonnx"] = "failed"
            report["hmonnx_error_type"] = type(exc).__name__
            report["hmonnx_error"] = str(exc)

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
