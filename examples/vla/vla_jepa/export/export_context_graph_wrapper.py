
# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

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
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers.models.qwen3_vl import modeling_qwen3_vl as hf_qwen3_vl

from export.export_action_head import tensor_summary
from common.inspect_policy import DEFAULT_MODEL, load_policy
from common.paths import output_str, set_default_libero_config_path

from xh_model_zoo.xh_llm.models.builder import XHLLM_TRACEABLE_MODULES, wrap_llm_model
from xh_model_zoo.xh_llm.models.qwen3_vl._llm_model_impl import (
    _Qwen3VLForConditionalGeneration,
    _Qwen3VLModel,
    _Qwen3VLTextAttention,
    _Qwen3VLTextDecoderLayer,
    _Qwen3VLTextModel,
    _Qwen3VLTextRMSNorm,
    _Qwen3VLTextRotaryEmbedding,
)

DEFAULT_SAMPLE = output_str("context_encoder", "dummy_sample.pt")
DEFAULT_OUT_DIR = output_str("context_encoder", "fixed_context")
GRAPH_INPUT_NAMES = [
    "inputs_embeds",
    "time_position_ids",
    "height_position_ids",
    "width_position_ids",
    "past_seq_length",
    "current_input_length",
    "embodied_positions",
    "deepstack_visual_embed_0",
    "deepstack_visual_embed_1",
    "deepstack_visual_embed_2",
]
SEQUENCE_INPUT_NAMES = {
    "inputs_embeds",
    "time_position_ids",
    "height_position_ids",
    "width_position_ids",
    "deepstack_visual_embed_0",
    "deepstack_visual_embed_1",
    "deepstack_visual_embed_2",
}


def _pad_tensor_to_sequence_length(tensor: torch.Tensor, target_seq_len: int) -> torch.Tensor:
    if tensor.dim() == 1:
        seq_dim = 0
    elif tensor.dim() == 3:
        seq_dim = 1
    else:
        return tensor

    seq_len = int(tensor.shape[seq_dim])
    if seq_len == target_seq_len:
        return tensor
    if seq_len > target_seq_len:
        raise ValueError(f"Cannot pad context input with seq_len={seq_len} to shorter target_seq_len={target_seq_len}")

    pad_shape = list(tensor.shape)
    pad_shape[seq_dim] = target_seq_len - seq_len
    padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, padding], dim=seq_dim)


def pad_graph_inputs_to_sequence_length(
    inputs: tuple[torch.Tensor, ...],
    target_seq_len: int | None,
) -> tuple[torch.Tensor, ...]:
    if target_seq_len is None:
        return inputs

    padded_inputs: list[torch.Tensor] = []
    for name, tensor in zip(GRAPH_INPUT_NAMES, inputs, strict=True):
        if name in SEQUENCE_INPUT_NAMES:
            padded_inputs.append(_pad_tensor_to_sequence_length(tensor, int(target_seq_len)))
        elif name == "current_input_length":
            padded_inputs.append(torch.tensor([int(target_seq_len)], dtype=tensor.dtype, device=tensor.device))
        else:
            padded_inputs.append(tensor)
    return tuple(padded_inputs)


def register_hf_qwen3_vl_wrap_cls() -> None:
    """Bridge upstream Transformers Qwen3-VL classes to project wrappers."""
    pairs = {
        hf_qwen3_vl.Qwen3VLTextRMSNorm: _Qwen3VLTextRMSNorm,
        hf_qwen3_vl.Qwen3VLTextRotaryEmbedding: _Qwen3VLTextRotaryEmbedding,
        hf_qwen3_vl.Qwen3VLTextAttention: _Qwen3VLTextAttention,
        hf_qwen3_vl.Qwen3VLTextDecoderLayer: _Qwen3VLTextDecoderLayer,
        hf_qwen3_vl.Qwen3VLTextModel: _Qwen3VLTextModel,
        hf_qwen3_vl.Qwen3VLModel: _Qwen3VLModel,
        hf_qwen3_vl.Qwen3VLForConditionalGeneration: _Qwen3VLForConditionalGeneration,
    }
    for hf_cls, wrapper_cls in pairs.items():
        if hf_cls not in XHLLM_TRACEABLE_MODULES:
            XHLLM_TRACEABLE_MODULES.register_module({hf_cls: hf_cls.__name__}, wrapper_cls)


class StaticSequenceSlice(nn.Module):
    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self.seq_len = int(seq_len)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states[:, : self.seq_len, :]


class NativeCausalMaskedSoftmax(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, attn_weights: torch.Tensor, past_seq_length: torch.Tensor | None = None) -> torch.Tensor:
        del past_seq_length
        q_len = attn_weights.shape[-2]
        k_len = attn_weights.shape[-1]
        q_pos = torch.arange(q_len, device=attn_weights.device).view(q_len, 1)
        k_pos = torch.arange(k_len, device=attn_weights.device).view(1, k_len)
        causal_mask = (k_pos <= q_pos).view(1, 1, q_len, k_len)
        min_value = torch.finfo(attn_weights.dtype).min
        attn_weights = attn_weights.masked_fill(~causal_mask, min_value)
        return torch.softmax(attn_weights, dim=self.dim)


class NativeRMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone())
        self.variance_epsilon = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        states = hidden_states.float()
        variance = (states * states).mean(-1, keepdim=True)
        states = states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * states.to(input_dtype)


def native_rotate_half(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    half = self.head_dim // 2
    x1, x2 = torch.split(x, [half, half], dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def native_attention_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    past_seq_length: torch.Tensor | None = None,
    current_input_length: torch.Tensor | None = None,
    past_k_cache: torch.Tensor | None = None,
    past_v_cache: torch.Tensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, None, None]:
    del current_input_length, past_k_cache, past_v_cache
    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = self.q_norm(query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim))
    query_states = query_states.transpose(1, 2)
    key_states = self.k_norm(key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim))
    key_states = key_states.transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

    query_states = query_states * self.kv_scale
    key_states = key_states.transpose(2, 3)
    key_states = key_states.index_select(1, self.kv_repeat_indices)
    attn_weights = torch.matmul(query_states, key_states)
    attn_weights = self.masked_softmax(attn_weights, past_seq_length)

    value_states = value_states.index_select(1, self.kv_repeat_indices)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.config.num_attention_heads * self.head_dim)
    attn_output = self.o_proj(attn_output)
    return attn_output, None, None


def patch_attention_for_onnx_export(wrapped_qwen: nn.Module) -> None:
    for layer in wrapped_qwen.language_model.layers:
        attn = layer.self_attn
        attn.enable_rope = False
        if hasattr(attn, "rope"):
            delattr(attn, "rope")
        kv_indices = torch.arange(attn.config.num_key_value_heads, dtype=torch.long).repeat_interleave(
            attn.num_key_value_groups
        )
        attn.register_buffer("kv_repeat_indices", kv_indices, persistent=False)
        attn.rotate_half = types.MethodType(native_rotate_half, attn)
        attn.forward = types.MethodType(native_attention_forward, attn)
        attn.masked_softmax = NativeCausalMaskedSoftmax(dim=-1)


def patch_rms_norm_for_hmonnx_export(wrapped_qwen: nn.Module) -> None:
    for module in wrapped_qwen.modules():
        norm = getattr(module, "norm", None)
        if norm is not None and type(norm).__name__ == "RMSNorm" and hasattr(norm, "weight"):
            module.norm = NativeRMSNorm(norm.weight, norm.variance_epsilon)


class WrappedQwenContextGraph(nn.Module):
    def __init__(self, wrapped_qwen: nn.Module) -> None:
        super().__init__()
        self.wrapped_qwen = wrapped_qwen

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        time_position_ids: torch.Tensor,
        height_position_ids: torch.Tensor,
        width_position_ids: torch.Tensor,
        past_seq_length: torch.Tensor,
        current_input_length: torch.Tensor,
        embodied_positions: torch.Tensor,
        deepstack_visual_embed_0: torch.Tensor,
        deepstack_visual_embed_1: torch.Tensor,
        deepstack_visual_embed_2: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.wrapped_qwen(
            inputs_embeds=inputs_embeds,
            time_position_ids=time_position_ids,
            hight_position_ids=height_position_ids,
            width_position_ids=width_position_ids,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            deepstack_visual_embed_0=deepstack_visual_embed_0,
            deepstack_visual_embed_1=deepstack_visual_embed_1,
            deepstack_visual_embed_2=deepstack_visual_embed_2,
        )
        hidden_states = outputs.last_hidden_state
        gather_index = embodied_positions.reshape(32)
        return hidden_states.index_select(1, gather_index)


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
        "allclose_rtol_1e-2_atol_1e-2": bool(np.allclose(ref, cand, rtol=1e-2, atol=1e-2)),
        "allclose_rtol_5e-2_atol_5e-2": bool(np.allclose(ref, cand, rtol=5e-2, atol=5e-2)),
        "allclose_rtol_1e-1_atol_1e-1": bool(np.allclose(ref, cand, rtol=1e-1, atol=1e-1)),
    }


def dense_deepstack(inputs_embeds: torch.Tensor, visual_pos_masks: torch.Tensor, sparse_embed: torch.Tensor) -> torch.Tensor:
    dense = torch.zeros_like(inputs_embeds)
    dense[visual_pos_masks] = sparse_embed.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    return dense


def build_graph_inputs(
    sample: dict[str, Any],
    device: str,
    dtype: torch.dtype,
    target_seq_len: int | None = None,
) -> tuple[torch.Tensor, ...]:
    qwen_inputs = sample["qwen_inputs"]
    context_inputs = sample.get("context_graph_inputs")
    if context_inputs is None:
        context_inputs = torch.load(
            output_str("context_encoder", "float16", "context_graph_inputs.pt"),
            map_location="cpu",
        )

    if all(name in context_inputs for name in GRAPH_INPUT_NAMES):
        prepared_inputs = []
        for name in GRAPH_INPUT_NAMES:
            tensor = context_inputs[name].to(device=device)
            if tensor.is_floating_point():
                tensor = tensor.to(dtype=dtype)
            prepared_inputs.append(tensor)
        return pad_graph_inputs_to_sequence_length(tuple(prepared_inputs), target_seq_len)

    inputs_embeds = context_inputs["inputs_embeds"].to(device=device, dtype=dtype)
    position_ids = context_inputs["position_ids"].to(device=device)
    visual_pos_masks = context_inputs["visual_pos_masks"].to(device=device).bool()
    embodied_positions = context_inputs["embodied_positions"].to(device=device).long()

    seq_len = inputs_embeds.shape[1]
    past_seq_length = torch.zeros(1, dtype=torch.int32, device=device)
    current_input_length = torch.tensor([seq_len], dtype=torch.int32, device=device)

    dense_0 = dense_deepstack(inputs_embeds, visual_pos_masks, context_inputs["deepstack_visual_embed_0"].to(device))
    dense_1 = dense_deepstack(inputs_embeds, visual_pos_masks, context_inputs["deepstack_visual_embed_1"].to(device))
    dense_2 = dense_deepstack(inputs_embeds, visual_pos_masks, context_inputs["deepstack_visual_embed_2"].to(device))

    if position_ids.dim() == 3:
        position_ids = position_ids[:, 0, :]

    inputs = (
        inputs_embeds,
        position_ids[0].long(),
        position_ids[1].long(),
        position_ids[2].long(),
        past_seq_length,
        current_input_length,
        embodied_positions,
        dense_0,
        dense_1,
        dense_2,
    )
    return pad_graph_inputs_to_sequence_length(inputs, target_seq_len)


def build_wrapper(model_path: str, device: str, dtype: torch.dtype, seq_len: int) -> nn.Module:
    policy = load_policy(model_path, device, disable_world_model=True)
    policy.eval()
    qwen = policy.model.qwen.model
    qwen_model = qwen.model
    if hasattr(qwen_model, "visual"):
        qwen_model.visual = nn.Identity()
    register_hf_qwen3_vl_wrap_cls()
    wrapped = wrap_llm_model(
        qwen_model,
        {
            "batch_size": 1,
            "max_sequence_length": seq_len,
            "max_pe_length": 32768,
            "input_sequence_length": seq_len,
            "use_cache": False,
            "num_logits_to_keep": 0,
            "enable_rope": False,
            "kv_cache": {"cache_axis": 2},
        },
    )
    wrapped.language_model.slice = nn.Identity()
    patch_attention_for_onnx_export(wrapped)
    patch_rms_norm_for_hmonnx_export(wrapped)
    graph = WrappedQwenContextGraph(wrapped).to(device=device, dtype=dtype).eval()
    graph.wrapped_class_report = {
        "qwen_model": type(wrapped).__name__,
        "language_model": type(wrapped.language_model).__name__,
        "first_layer": type(wrapped.language_model.layers[0]).__name__,
        "first_attention": type(wrapped.language_model.layers[0].self_attn).__name__,
        "first_attention_rope_enabled": bool(wrapped.language_model.layers[0].self_attn.enable_rope),
        "first_attention_masked_softmax": type(wrapped.language_model.layers[0].self_attn.masked_softmax).__name__,
        "first_rms_norm": type(wrapped.language_model.layers[0].input_layernorm.norm).__name__,
        "sequence_slice": type(wrapped.language_model.slice).__name__,
    }
    return graph


def export_onnx(model: nn.Module, inputs: tuple[torch.Tensor, ...], onnx_path: Path, opset: int) -> None:
    torch.onnx.export(
        model,
        inputs,
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=GRAPH_INPUT_NAMES,
        output_names=["conditioning_tokens"],
    )


def select_onnx_example_inputs(onnx_path: Path, inputs: tuple[torch.Tensor, ...]) -> tuple[list[torch.Tensor], list[str]]:
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    onnx_input_names = [value.name for value in model.graph.input]
    input_by_name = dict(zip(GRAPH_INPUT_NAMES, inputs))
    return [input_by_name[name].detach().cpu() for name in onnx_input_names], onnx_input_names


def qwen3vl_matmul_ops() -> dict[str, Any]:
    return {
        "MatMul": {
            "act_scheme": {
                "bits": 8,
                "fp_mode": "sefp",
            },
            "act_schema_2": {
                "bits": 16,
                "fp_mode": "sefp",
            },
        }
    }


def convert_to_hmonnx(
    onnx_path: Path,
    hmonnx_path: Path,
    inputs: tuple[torch.Tensor, ...],
    quant_type: str,
    use_qwen3vl_matmul_ops: bool = False,
    linear_quant_type: str | None = None,
) -> dict[str, Any]:
    from xhquant.api import DeviceType, QuantScheme, convert_onnx_to_hmonnx, create_quant_config

    example_inputs, onnx_input_names = select_onnx_example_inputs(onnx_path, inputs)
    ops: dict[str, Any] = {}
    if use_qwen3vl_matmul_ops:
        ops.update(qwen3vl_matmul_ops())
    if linear_quant_type:
        ops["Linear"] = linear_quant_type

    if ops:
        quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type, ops=ops)
    else:
        quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)
    convert_onnx_to_hmonnx(
        str(onnx_path),
        example_inputs,
        DeviceType.XH2a,
        str(hmonnx_path),
        quant_config,
        simplify=False,
    )
    return {
        "hmonnx_path": str(hmonnx_path),
        "size_mb": hmonnx_path.stat().st_size / 1024 / 1024,
        "onnx_input_names": onnx_input_names,
        "quant_type": quant_type,
        "use_qwen3vl_matmul_ops": use_qwen3vl_matmul_ops,
        "linear_quant_type": linear_quant_type,
        "ops": ops,
    }


def fuse_hmonnx_rmsnorm(input_path: Path, output_path: Path) -> dict[str, Any]:
    import onnx

    from patches.patch_hmonnx_rmsnorm_casts import fuse_rmsnorm

    model = onnx.load(str(input_path), load_external_data=False)
    fused_count = fuse_rmsnorm(model)
    onnx.save(model, str(output_path))
    return {
        "input": str(input_path),
        "output": str(output_path),
        "fused_rmsnorm_patterns": fused_count,
        "size_mb": output_path.stat().st_size / 1024 / 1024,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample", default=DEFAULT_SAMPLE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--convert-hmonnx", action="store_true")
    parser.add_argument("--quant-type", default="w8a16_sefp")
    parser.add_argument("--qwen3vl-matmul-ops", action="store_true")
    parser.add_argument("--fuse-rmsnorm", action="store_true")
    args = parser.parse_args()

    set_default_libero_config_path()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(41)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"vla_jepa_context_graph_wrapper_{args.dtype}.onnx"
    hmonnx_path = out_dir / f"vla_jepa_context_graph_wrapper_{args.dtype}_{args.quant_type}.hmonnx.onnx"
    fused_hmonnx_path = out_dir / f"vla_jepa_context_graph_wrapper_{args.dtype}_{args.quant_type}.fused_rmsnorm.hmonnx.onnx"
    report_path = out_dir / f"export_context_graph_wrapper_{args.dtype}_report.json"

    sample = torch.load(args.sample, map_location="cpu", weights_only=False)
    inputs = build_graph_inputs(sample, args.device, dtype, args.context_length)
    wrapper = build_wrapper(args.model, args.device, dtype, int(inputs[0].shape[1]))

    with torch.no_grad():
        candidate = wrapper(*inputs)
    reference = sample["conditioning_tokens"].to(args.device)

    report: dict[str, Any] = {
        "model": args.model,
        "sample": args.sample,
        "device": args.device,
        "dtype": args.dtype,
        "opset": args.opset,
        "context_length": args.context_length,
        "quant_type": args.quant_type,
        "use_qwen3vl_matmul_ops": bool(args.qwen3vl_matmul_ops),
        "fuse_rmsnorm": bool(args.fuse_rmsnorm),
        "inputs": {str(idx): tensor_summary(item) for idx, item in enumerate(inputs)},
        "wrapped_classes": getattr(wrapper, "wrapped_class_report", {}),
        "pytorch_compare": diff_summary(reference, candidate),
    }

    if args.export_onnx:
        export_onnx(wrapper, inputs, onnx_path, args.opset)
        report["onnx_path"] = str(onnx_path)
        report["onnx_size_mb"] = onnx_path.stat().st_size / 1024 / 1024
    if args.convert_hmonnx:
        if not onnx_path.exists():
            export_onnx(wrapper, inputs, onnx_path, args.opset)
            report["onnx_path"] = str(onnx_path)
            report["onnx_size_mb"] = onnx_path.stat().st_size / 1024 / 1024
        report["hmonnx"] = convert_to_hmonnx(
            onnx_path,
            hmonnx_path,
            inputs,
            args.quant_type,
            use_qwen3vl_matmul_ops=bool(args.qwen3vl_matmul_ops),
        )
        if args.fuse_rmsnorm:
            report["fused_hmonnx"] = fuse_hmonnx_rmsnorm(hmonnx_path, fused_hmonnx_path)

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
