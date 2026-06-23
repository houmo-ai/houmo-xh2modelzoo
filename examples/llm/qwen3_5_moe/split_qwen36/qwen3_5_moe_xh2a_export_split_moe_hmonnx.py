# Copyright 2026 HOUMO AI
#
# File: qwen3_5_moe_xh2a_export_split_moe_hmonnx.py
# Description:
#   Experimental split-MoE export for Qwen3.5-MoE.  It exports every decoder
#   block as four kinds of HMONNX subgraphs:
#     1. PreMoE: attention + router + shared expert
#     2. Expert: one routed expert MLP per layer/expert id
#     3. PostMoE: one shared decode-only NPU aggregation graph
#     4. Head: final norm + lm_head logits graph
#
# Usage (small smoke export):
#   python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_split_moe_hmonnx.py \
#       --model /path/to/Qwen3.5-35B-A3B \
#       --context-length 2048 --decode-sequence-length 1 --prefill-sequence-length 256 \
#       --export-layers 0 --export-experts 0,1 --quant-type w8a8h0_sefp
#
# Usage (full split export):
#   python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_split_moe_hmonnx.py \
#       --model /path/to/Qwen3.5-35B-A3B \
#       --context-length 2048 --decode-sequence-length 1 --prefill-sequence-length 256 \
#       --export-layers all --export-experts all --quant-type w8a8h0_sefp
#
# Usage (GPTQModel weights, split export):
#   python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_split_moe_hmonnx.py \
#       --model /path/to/Qwen3.5-35B-A3B \
#       --quant-weight /data01/datasets/qwen36moe-no-rotate-attn8-shared8-n256-iter400 \
#       --context-length 2048 --decode-sequence-length 1 --prefill-sequence-length 256 \
#       --export-layers all --export-experts all \
#       --premoe-quant-type w8a8h0_sefp --expert-quant-type w4a8h0_ssfp

# Usage (export final norm + lm_head only):
#   python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_split_moe_hmonnx.py \
#       --model /path/to/Qwen3.5-35B-A3B \
#       --context-length 2048 --decode-sequence-length 1 \
#       --export-parts head --premoe-quant-type w8a8h0_sefp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os.path as osp
import re
import shutil
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from xh_model_zoo.utils.memory_tracker import MemoryTracker
from xh_model_zoo.utils.time_profiler import TimeProfiler
from xh_model_zoo.xh_llm.models.base_converter import BaseConverter
from xh_model_zoo.xh_llm.models.qwen3_5_moe import Qwen3_5MoeConvertConfig
from xh_model_zoo.xh_llm.models.qwen3_5_moe.qwen3_5_moe_converter import (
    _LINEAR_CONV_CACHE_BRANCHES,
    Qwen3_5MoeConverterXH2a,
    _extract_quant_method,
    _get_text_config,
    _get_text_model,
    _linear_split_conv_dims,
    _patch_hmonnx_standard_add_ops,
)
from xhquant.api import (
    CacheTensor,
    Config,
    DeviceType,
    QuantScheme,
    convert_fx_model_to_quanted_model,
    convert_quanted_model_to_hmonnx,
    get_root_logger,
    xhquant_init,
)


class _PreMoEMLPSlice(nn.Module):
    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.gate = deepcopy(mlp.gate)
        self.shared_expert = deepcopy(mlp.shared_expert)
        self.shared_expert_gate = deepcopy(mlp.shared_expert_gate)


class _PreMoELayerSlice(nn.Module):
    def __init__(self, layer: nn.Module, layer_type: str):
        super().__init__()
        self.input_layernorm = deepcopy(layer.input_layernorm)
        self.post_attention_layernorm = deepcopy(layer.post_attention_layernorm)
        if layer_type == "full_attention":
            self.self_attn = deepcopy(layer.self_attn)
        elif layer_type == "linear_attention":
            self.linear_attn = deepcopy(layer.linear_attn)
        else:
            raise ValueError(f"Unsupported PreMoE layer_type: {layer_type}")
        self.mlp = _PreMoEMLPSlice(layer.mlp)


class FullAttentionPreMoEExportModule(nn.Module):
    """Per-layer PreMoE graph for full-attention Qwen3.5-MoE blocks.

    Outputs match the Host scheduler ABI:
      moe_input, topk_id, topk_gate, shared_out, residual1

    KV cache updates are kept in the same implicit CacheTensor/LLMCacheV2 path
    used by the standard full-graph export; there are no explicit KV outputs.
    """

    def __init__(self, text_model: nn.Module, layer_idx: int, top_k: int):
        super().__init__()
        self.rotary_emb = deepcopy(text_model.rotary_emb)
        self.layer = _PreMoELayerSlice(text_model.layers[layer_idx], "full_attention")
        self.top_k = int(top_k)

    def _position_embeddings(
        self,
        time_position_ids: torch.Tensor,
        hight_position_ids: torch.Tensor,
        width_position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.rotary_emb.cos_cached
        sin = self.rotary_emb.sin_cached

        time_cos = cos[time_position_ids] * self.rotary_emb.time_mask
        time_sin = sin[time_position_ids] * self.rotary_emb.time_mask
        hight_cos = cos[hight_position_ids] * self.rotary_emb.hight_mask
        hight_sin = sin[hight_position_ids] * self.rotary_emb.hight_mask
        width_cos = cos[width_position_ids] * self.rotary_emb.width_mask
        width_sin = sin[width_position_ids] * self.rotary_emb.width_mask

        combined_cos = (time_cos + hight_cos + width_cos).squeeze(-2).unsqueeze(1)
        combined_sin = (time_sin + hight_sin + width_sin).squeeze(-2).unsqueeze(1)
        return combined_cos, combined_sin

    def _route_and_shared(
        self,
        moe_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = self.layer.mlp.gate(moe_input)
        router_prob = F.softmax(router_logits, dim=-1)
        topk_gate, topk_id = torch.topk(router_prob, k=self.top_k, dim=-1)
        topk_gate = topk_gate / topk_gate.sum(dim=-1, keepdim=True)
        shared_out = self.layer.mlp.shared_expert(moe_input)
        shared_out = torch.sigmoid(self.layer.mlp.shared_expert_gate(moe_input)) * shared_out
        return topk_id, topk_gate, shared_out

    def forward(
        self,
        hidden_in: torch.Tensor,
        time_position_ids: torch.Tensor,
        hight_position_ids: torch.Tensor,
        width_position_ids: torch.Tensor,
        past_seq_length: torch.Tensor,
        current_input_length: torch.Tensor,
        past_key_cache: torch.Tensor,
        past_value_cache: torch.Tensor,
    ):
        residual0 = hidden_in
        x = self.layer.input_layernorm(hidden_in)
        position_embeddings = self._position_embeddings(
            time_position_ids,
            hight_position_ids,
            width_position_ids,
        )
        attn_out, _, _ = self.layer.self_attn(
            hidden_states=x,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            position_embeddings=position_embeddings,
            past_k_cache=past_key_cache,
            past_v_cache=past_value_cache,
        )
        hidden_attn = residual0 + attn_out
        residual1 = hidden_attn
        moe_input = self.layer.post_attention_layernorm(hidden_attn)
        topk_id, topk_gate, shared_out = self._route_and_shared(moe_input)
        return moe_input, topk_id, topk_gate, shared_out, residual1


class LinearAttentionPreMoEExportModule(nn.Module):
    """Per-layer PreMoE graph for linear-attention Qwen3.5-MoE blocks."""

    def __init__(self, text_model: nn.Module, layer_idx: int, top_k: int, split_conv_cache: bool):
        super().__init__()
        self.layer = _PreMoELayerSlice(text_model.layers[layer_idx], "linear_attention")
        self.top_k = int(top_k)
        self.split_conv_cache = bool(split_conv_cache)

    def _route_and_shared(
        self,
        moe_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = self.layer.mlp.gate(moe_input)
        router_prob = F.softmax(router_logits, dim=-1)
        topk_gate, topk_id = torch.topk(router_prob, k=self.top_k, dim=-1)
        topk_gate = topk_gate / topk_gate.sum(dim=-1, keepdim=True)
        shared_out = self.layer.mlp.shared_expert(moe_input)
        shared_out = torch.sigmoid(self.layer.mlp.shared_expert_gate(moe_input)) * shared_out
        return topk_id, topk_gate, shared_out

    def _forward_impl(
        self,
        hidden_in: torch.Tensor,
        current_input_length: torch.Tensor,
        linear_attn_mask: torch.Tensor,
        past_conv_cache,
        past_recurrent_state: torch.Tensor,
    ):
        residual0 = hidden_in
        x = self.layer.input_layernorm(hidden_in)
        attn_out, conv_cache_out, recurrent_state_out = self.layer.linear_attn(
            hidden_states=x,
            conv_cache=past_conv_cache,
            recurrent_state=past_recurrent_state,
            linear_attn_mask=linear_attn_mask,
            current_input_length=current_input_length,
        )
        hidden_attn = residual0 + attn_out
        residual1 = hidden_attn
        moe_input = self.layer.post_attention_layernorm(hidden_attn)
        topk_id, topk_gate, shared_out = self._route_and_shared(moe_input)
        if isinstance(conv_cache_out, (tuple, list)):
            conv_outputs = tuple(conv_cache_out)
        else:
            conv_outputs = (conv_cache_out,)
        return (moe_input, topk_id, topk_gate, shared_out, residual1) + conv_outputs + (recurrent_state_out,)

    def forward(
        self,
        hidden_in: torch.Tensor,
        current_input_length: torch.Tensor,
        linear_attn_mask: torch.Tensor,
        past_conv_cache_q: torch.Tensor,
        past_conv_cache_k: torch.Tensor,
        past_conv_cache_v: torch.Tensor,
        past_recurrent_state: torch.Tensor,
    ):
        past_conv_cache = (
            (past_conv_cache_q, past_conv_cache_k, past_conv_cache_v) if self.split_conv_cache else past_conv_cache_q
        )
        return self._forward_impl(
            hidden_in,
            current_input_length,
            linear_attn_mask,
            past_conv_cache,
            past_recurrent_state,
        )


class SingleExpertExportModule(nn.Module):
    """One routed expert MLP: down_proj(silu(gate_proj(x)) * up_proj(x))."""

    def __init__(self, mlp: nn.Module, expert_id: int):
        super().__init__()
        self.expert_id = int(expert_id)
        self.gate_proj, self.up_proj, self.down_proj = self._build_linears(mlp, self.expert_id)

    @staticmethod
    def _maybe_copy_bias(linear: nn.Linear, bias_storage, expert_id: Optional[int] = None) -> None:
        if bias_storage is None or linear.bias is None:
            return
        bias = bias_storage if expert_id is None else bias_storage[expert_id]
        linear.bias.data.copy_(bias.detach().to(linear.bias.device, linear.bias.dtype))

    @staticmethod
    def _maybe_copy_quant_weight(linear: nn.Linear, quant_weight_storage, expert_id: Optional[int] = None) -> None:
        if quant_weight_storage is None:
            return
        quant_weight = quant_weight_storage if expert_id is None else quant_weight_storage[expert_id]
        quant_weight = quant_weight.detach().clone().to(device=linear.weight.device)
        if "quant_weight" in linear._buffers:
            linear._buffers["quant_weight"] = quant_weight
        else:
            linear.register_buffer("quant_weight", quant_weight)

    @staticmethod
    def _expert_linear_quant_weight(expert: nn.Module, linear_name: str):
        linear = getattr(expert, linear_name, None)
        return None if linear is None else getattr(linear, "quant_weight", None)

    @staticmethod
    def _build_linears(mlp: nn.Module, expert_id: int) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
        gate_b_idx = up_b_idx = down_b_idx = None
        gate_qw = up_qw = down_qw = None
        gate_qw_idx = up_qw_idx = down_qw_idx = None
        if hasattr(mlp, "moeblock"):
            moeblock = mlp.moeblock
            gate_w = moeblock.expert_gate_proj_weight[expert_id].detach()
            up_w = moeblock.expert_up_proj_weight[expert_id].detach()
            down_w = moeblock.expert_down_proj_weight[expert_id].detach()
            gate_b = getattr(moeblock, "expert_gate_proj_bias", None)
            up_b = getattr(moeblock, "expert_up_proj_bias", None)
            down_b = getattr(moeblock, "expert_down_proj_bias", None)
            gate_b_idx = up_b_idx = down_b_idx = expert_id
            gate_qw = getattr(moeblock, "expert_gate_proj_quant_weight", None)
            up_qw = getattr(moeblock, "expert_up_proj_quant_weight", None)
            down_qw = getattr(moeblock, "expert_down_proj_quant_weight", None)
            gate_qw_idx = up_qw_idx = down_qw_idx = expert_id
        elif hasattr(mlp, "experts") and hasattr(mlp.experts, "gate_up_proj"):
            experts = mlp.experts
            intermediate_dim = int(experts.intermediate_dim)
            gate_up = experts.gate_up_proj[expert_id].detach()
            gate_w = gate_up[:intermediate_dim, :]
            up_w = gate_up[intermediate_dim:, :]
            down_w = experts.down_proj[expert_id].detach()
            gate_b = up_b = down_b = None
            gate_up_qw = getattr(experts, "gate_up_proj_quant_weight", None)
            if gate_up_qw is not None:
                gate_up_qw = gate_up_qw[expert_id].detach()
                gate_qw = gate_up_qw[:intermediate_dim, :]
                up_qw = gate_up_qw[intermediate_dim:, :]
            down_proj_qw = getattr(experts, "down_proj_quant_weight", None)
            if down_proj_qw is not None:
                down_qw = down_proj_qw[expert_id].detach()
        elif hasattr(mlp, "experts"):
            expert = mlp.experts[expert_id]
            gate_w = expert.gate_proj.weight.detach()
            up_w = expert.up_proj.weight.detach()
            down_w = expert.down_proj.weight.detach()
            gate_b = expert.gate_proj.bias.detach() if expert.gate_proj.bias is not None else None
            up_b = expert.up_proj.bias.detach() if expert.up_proj.bias is not None else None
            down_b = expert.down_proj.bias.detach() if expert.down_proj.bias is not None else None
            gate_qw = SingleExpertExportModule._expert_linear_quant_weight(expert, "gate_proj")
            up_qw = SingleExpertExportModule._expert_linear_quant_weight(expert, "up_proj")
            down_qw = SingleExpertExportModule._expert_linear_quant_weight(expert, "down_proj")
        else:
            raise RuntimeError(f"Unsupported expert storage in mlp={type(mlp)}")

        device = gate_w.device
        dtype = gate_w.dtype
        hidden_dim = int(gate_w.shape[1])
        intermediate_dim = int(gate_w.shape[0])
        down_out_dim = int(down_w.shape[0])

        gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=gate_b is not None, device=device, dtype=dtype)
        up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=up_b is not None, device=device, dtype=dtype)
        down_proj = nn.Linear(intermediate_dim, down_out_dim, bias=down_b is not None, device=device, dtype=dtype)
        gate_proj.weight.data.copy_(gate_w.to(device=device, dtype=dtype))
        up_proj.weight.data.copy_(up_w.to(device=device, dtype=dtype))
        down_proj.weight.data.copy_(down_w.to(device=device, dtype=dtype))
        SingleExpertExportModule._maybe_copy_bias(gate_proj, gate_b, gate_b_idx)
        SingleExpertExportModule._maybe_copy_bias(up_proj, up_b, up_b_idx)
        SingleExpertExportModule._maybe_copy_bias(down_proj, down_b, down_b_idx)
        SingleExpertExportModule._maybe_copy_quant_weight(gate_proj, gate_qw, gate_qw_idx)
        SingleExpertExportModule._maybe_copy_quant_weight(up_proj, up_qw, up_qw_idx)
        SingleExpertExportModule._maybe_copy_quant_weight(down_proj, down_qw, down_qw_idx)
        return gate_proj, up_proj, down_proj

    def forward(self, expert_input: torch.Tensor):
        return self.down_proj(F.silu(self.gate_proj(expert_input)) * self.up_proj(expert_input))


class PostMoENPUAggModule(nn.Module):
    """Decode-only NPU aggregation graph shared by the whole network."""

    def __init__(self, top_k: int):
        super().__init__()
        self.top_k = int(top_k)

    def forward(
        self,
        expert_out_0: torch.Tensor,
        expert_out_1: torch.Tensor,
        expert_out_2: torch.Tensor,
        expert_out_3: torch.Tensor,
        expert_out_4: torch.Tensor,
        expert_out_5: torch.Tensor,
        expert_out_6: torch.Tensor,
        expert_out_7: torch.Tensor,
        topk_gate: torch.Tensor,
        shared_out: torch.Tensor,
        residual1: torch.Tensor,
    ):
        expert_outs = (
            expert_out_0,
            expert_out_1,
            expert_out_2,
            expert_out_3,
            expert_out_4,
            expert_out_5,
            expert_out_6,
            expert_out_7,
        )
        routed_sum = expert_outs[0] * topk_gate[..., 0:1]
        for idx in range(1, self.top_k):
            routed_sum = routed_sum + expert_outs[idx] * topk_gate[..., idx : idx + 1]
        return residual1 + shared_out + routed_sum


class HeadExportModule(nn.Module):
    """Final language-model head graph: model.norm + lm_head."""

    def __init__(self, norm: nn.Module, lm_head: nn.Module):
        super().__init__()
        self.norm = deepcopy(norm)
        self.lm_head = deepcopy(lm_head)

    def forward(self, hidden_in: torch.Tensor):
        return self.lm_head(self.norm(hidden_in))


def _parse_index_spec(spec: str, max_count: int, name: str) -> list[int]:
    spec = str(spec).strip().lower()
    if spec in {"all", "*"}:
        return list(range(max_count))
    result: set[int] = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid {name} range: {raw_part!r}")
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    invalid = sorted(idx for idx in result if idx < 0 or idx >= max_count)
    if invalid:
        raise ValueError(f"{name} index out of range [0, {max_count}): {invalid}")
    return sorted(result)


def _parse_export_parts(spec: str) -> set[str]:
    valid_parts = {"premoe", "experts", "postmoe", "head"}
    spec = str(spec).strip().lower()
    if spec in {"all", "*"}:
        return set(valid_parts)
    parts = {part.strip() for part in spec.split(",") if part.strip()}
    invalid = sorted(parts - valid_parts)
    if invalid:
        raise ValueError(f"Invalid export part(s): {invalid}; valid parts are {sorted(valid_parts)}")
    return parts


def _derive_expert_quant_type(premoe_quant_type: str) -> str:
    premoe_quant_type = str(premoe_quant_type).strip()
    match = re.match(r"^w\d+", premoe_quant_type)
    if match is None:
        return "w4a8h0_ssfp"
    return _normalize_expert_quant_type("w4" + premoe_quant_type[match.end() :])


def _normalize_expert_quant_type(expert_quant_type: str) -> str:
    expert_quant_type = str(expert_quant_type).strip()
    if expert_quant_type.startswith("w4") and expert_quant_type.endswith("_sefp"):
        return expert_quant_type[: -len("_sefp")] + "_ssfp"
    return expert_quant_type


def _resolve_split_quant_types(args: argparse.Namespace) -> tuple[str, str, str]:
    premoe_quant_type = args.premoe_quant_type or args.quant_type
    if args.expert_quant_type:
        expert_quant_type = _normalize_expert_quant_type(args.expert_quant_type)
    else:
        expert_quant_type = _derive_expert_quant_type(premoe_quant_type)
    head_quant_type = args.head_quant_type or premoe_quant_type
    return premoe_quant_type, expert_quant_type, head_quant_type


def _resolve_gptq_restore_layer_spec(args: argparse.Namespace, export_parts: set[str]) -> Optional[str]:
    explicit_spec = getattr(args, "gptq_restore_expert_layers", None)
    if explicit_spec is not None:
        explicit_spec = str(explicit_spec).strip()
        if explicit_spec:
            return explicit_spec
    if "experts" in export_parts and not args.skip_experts:
        return str(args.export_layers).strip() or "all"
    return None


def _count_quant_weight_buffers(module: nn.Module) -> int:
    return sum(1 for child in module.modules() if getattr(child, "quant_weight", None) is not None)


def _count_quant_weight_tensors(module: nn.Module) -> int:
    total = 0
    for child in module.modules():
        if getattr(child, "quant_weight", None) is not None:
            total += 1
        for attr_name in (
            "expert_gate_proj_quant_weight",
            "expert_up_proj_quant_weight",
            "expert_down_proj_quant_weight",
            "gate_proj_quant_weight",
            "up_proj_quant_weight",
            "down_proj_quant_weight",
            "gate_up_proj_quant_weight",
        ):
            if getattr(child, attr_name, None) is not None:
                total += 1
    return total


def _extract_quant_method_from_config_dir(model_dir: str) -> Optional[str]:
    model_dir_path = Path(model_dir)
    cfg_paths = [model_dir_path / "config.json", model_dir_path / "quantization_config.json"]
    for cfg_path in cfg_paths:
        if not cfg_path.exists():
            continue
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        quant_cfg = cfg.get("quantization_config", cfg)
        if not isinstance(quant_cfg, dict):
            continue
        quant_method = quant_cfg.get("quant_method") or quant_cfg.get("checkpoint_format") or quant_cfg.get("provider")
        if quant_method is not None:
            return str(quant_method)
    return None


def _source_quant_method(native_model, loaded_model_path: str) -> Optional[str]:
    quant_method = _extract_quant_method(native_model.config)
    if quant_method is not None:
        return quant_method
    return _extract_quant_method_from_config_dir(loaded_model_path)


def _has_complete_output(output_path: Path) -> bool:
    try:
        return output_path.is_file() and output_path.stat().st_size > 0
    except OSError:
        return False


def _get_wrapped_text_model(wrapped_model: nn.Module) -> nn.Module:
    model = getattr(wrapped_model, "model", wrapped_model)
    return getattr(model, "language_model", model)


def _get_lm_head(wrapped_model: nn.Module, wrapped_text_model: nn.Module) -> nn.Module:
    for candidate in (wrapped_model, getattr(wrapped_model, "model", None), wrapped_text_model):
        if candidate is not None and hasattr(candidate, "lm_head"):
            return candidate.lm_head
    raise AttributeError("Failed to locate lm_head on wrapped_model/model/language_model")


def _copy_hf_config(hf_model_path: str, work_dir: Path) -> Path:
    hf_model_path = osp.normpath(osp.abspath(hf_model_path))
    hf_config_dir = work_dir / "hf_config"
    hf_config_dir.mkdir(exist_ok=True, parents=True)
    for cfg_file in (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "merges.txt",
        "tokenizer.model",
        "processor_config.json",
        "quantization_config.json",
    ):
        src = Path(hf_model_path) / cfg_file
        if src.exists():
            shutil.copyfile(src, hf_config_dir / cfg_file)
    return hf_config_dir


def _apply_update_cfg(module: nn.Module, cfg: Config) -> None:
    if hasattr(module, "_update_cfg"):
        module._update_cfg(cfg)


def _build_quant_config_for_module(
    converter: Qwen3_5MoeConverterXH2a,
    module: nn.Module,
    quant_type: str,
):
    old_quant_scheme = converter.config.quant_scheme
    converter.config.quant_scheme = QuantScheme(
        target_device=DeviceType.XH2a,
        quant_type=quant_type,
    )
    try:
        return converter._build_quant_config(module)
    finally:
        converter.config.quant_scheme = old_quant_scheme


def _build_convert_config(args: argparse.Namespace) -> Qwen3_5MoeConvertConfig:
    quant_scheme = QuantScheme(
        target_device=DeviceType.XH2a,
        quant_type=args.quant_type,
    )
    return Qwen3_5MoeConvertConfig(
        batch_size=args.batch_size,
        context_length=args.context_length,
        input_sequence_length=args.decode_sequence_length,
        quant_scheme=quant_scheme,
        quant_weight=args.quant_weight,
        num_logits_to_keep=0,
        linear_attention_mode=args.linear_attention_mode,
        linear_chunk_size=args.linear_chunk_size,
        split_conv_cache=args.split_conv_cache,
        normalize_force_fp32=args.normalize_force_fp32,
        use_manual_depthwise_conv1d=args.use_manual_depthwise_conv1d,
        fuse_gdr_ops=args.fuse_gdr_ops,
    )


def _parse_restore_layer_spec(layer_spec: Optional[str], num_layers: int) -> set[int]:
    if layer_spec is None:
        return set(range(num_layers))
    spec = str(layer_spec).strip().lower()
    if not spec or spec in {"all", "*"}:
        return set(range(num_layers))
    layers: set[int] = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid GPTQ restore layer range: {raw_part!r}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    invalid = sorted(layer for layer in layers if layer < 0 or layer >= num_layers)
    if invalid:
        raise ValueError(f"gptq_restore_expert_layers has out-of-range layer ids: {invalid}")
    return layers


def _set_quant_weight_buffer(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
    if name in module._buffers:
        module._buffers[name] = tensor
    else:
        module.register_buffer(name, tensor, persistent=False)


def _unpack_gptqmodel_qweight(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
    expected_shape: tuple[int, int],
) -> torch.Tensor:
    out_features, in_features = expected_shape
    if qweight.ndim != 2 or qzeros.ndim != 2:
        raise ValueError(f"Expected 2D qweight/qzeros, got {tuple(qweight.shape)} and {tuple(qzeros.shape)}")
    if qweight.shape[1] < out_features:
        raise ValueError(f"qweight output dim {qweight.shape[1]} is smaller than expected {out_features}")
    if in_features % qweight.shape[0] != 0:
        raise ValueError(
            f"Cannot infer GPTQ pack factor from qweight={tuple(qweight.shape)}, expected={expected_shape}"
        )

    pack_factor = in_features // qweight.shape[0]
    if pack_factor not in {4, 8, 16}:
        raise ValueError(f"Unsupported GPTQ pack factor {pack_factor} for qweight={tuple(qweight.shape)}")
    bits = 32 // pack_factor
    maxq = (1 << bits) - 1
    shifts = torch.arange(0, 32, bits, dtype=torch.int32, device=qweight.device)

    zeros = torch.bitwise_and(
        torch.bitwise_right_shift(qzeros.to(torch.int32).unsqueeze(2).expand(-1, -1, pack_factor), shifts),
        maxq,
    ).reshape(qzeros.shape[0], qzeros.shape[1] * pack_factor)
    if zeros.shape[1] < out_features:
        raise ValueError(f"qzeros output dim {zeros.shape[1]} is smaller than expected {out_features}")

    unpacked = torch.bitwise_and(
        torch.bitwise_right_shift(
            qweight.to(torch.int32).unsqueeze(1).expand(-1, pack_factor, -1),
            shifts.view(1, -1, 1),
        ),
        maxq,
    ).reshape(qweight.shape[0] * pack_factor, qweight.shape[1])
    unpacked = unpacked[:in_features, :out_features]

    g_idx = g_idx.to(torch.long)[:in_features]
    if g_idx.numel() != in_features:
        raise ValueError(f"g_idx length {g_idx.numel()} does not match expected in_features {in_features}")
    quant_weight = (unpacked - zeros[g_idx, :out_features]).t().contiguous()
    if tuple(quant_weight.shape) != expected_shape:
        raise ValueError(f"Unpacked quant_weight shape {tuple(quant_weight.shape)} != expected {expected_shape}")

    min_val = -(1 << (bits - 1))
    max_val = (1 << (bits - 1)) - 1
    if quant_weight.min() < min_val or quant_weight.max() > max_val:
        raise ValueError(
            f"Unpacked GPTQ quant_weight outside signed {bits}-bit range: "
            f"min={int(quant_weight.min())}, max={int(quant_weight.max())}"
        )
    return quant_weight.to(torch.int8)


def _dequantize_gptqmodel_weight(
    quant_weight: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    expected_shape: tuple[int, int],
    dtype: torch.dtype,
) -> torch.Tensor:
    out_features, in_features = expected_shape
    if scales.ndim != 2:
        raise ValueError(f"Expected 2D GPTQ scales, got {tuple(scales.shape)}")
    if scales.shape[1] < out_features:
        raise ValueError(f"scales output dim {scales.shape[1]} is smaller than expected {out_features}")

    g_idx = g_idx.to(torch.long)[:in_features]
    if g_idx.numel() != in_features:
        raise ValueError(f"g_idx length {g_idx.numel()} does not match expected in_features {in_features}")
    if int(g_idx.max()) >= scales.shape[0] or int(g_idx.min()) < 0:
        raise ValueError(
            f"g_idx range [{int(g_idx.min())}, {int(g_idx.max())}] is outside scales groups {scales.shape[0]}"
        )

    dequant_weight = quant_weight.float() * scales[g_idx, :out_features].t().float()
    if tuple(dequant_weight.shape) != expected_shape:
        raise ValueError(f"Dequantized weight shape {tuple(dequant_weight.shape)} != expected {expected_shape}")
    return dequant_weight.to(dtype=dtype).contiguous()


def _build_gptq_tensor_index(model_dir: Path) -> dict[str, Path]:
    from safetensors.torch import safe_open

    safetensors_files = sorted(model_dir.glob("model*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No model*.safetensors files found in {model_dir}")
    tensor_index: dict[str, Path] = {}
    for safetensors_file in safetensors_files:
        with safe_open(str(safetensors_file), framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if ".mlp.experts." not in key:
                    continue
                if key.endswith((".qweight", ".qzeros", ".scales", ".g_idx")):
                    tensor_index[key] = safetensors_file
    return tensor_index


def _read_indexed_gptq_tensor(tensor_index: dict[str, Path], key: str) -> torch.Tensor:
    from safetensors.torch import safe_open

    safetensors_file = tensor_index.get(key)
    if safetensors_file is None:
        raise KeyError(f"Missing GPTQ tensor {key}")
    with safe_open(str(safetensors_file), framework="pt", device="cpu") as reader:
        return reader.get_tensor(key)


def _packed_expert_weight_shape(experts: nn.Module, expert_idx: int, linear_name: str) -> tuple[int, int]:
    if linear_name in {"gate_proj", "up_proj"}:
        intermediate_dim = int(experts.intermediate_dim)
        hidden_dim = int(experts.gate_up_proj.shape[-1])
        return (intermediate_dim, hidden_dim)
    return tuple(int(dim) for dim in experts.down_proj[expert_idx].shape)


def _restore_gptq_quant_weights_from_checkpoint(
    model_dir: str,
    native_model: nn.Module,
    logger,
    layer_spec: Optional[str] = None,
) -> int:
    """Restore Qwen3.5-MoE routed expert quant weights from GPTQModel shards."""
    try:
        model_path = Path(model_dir)
        tensor_index = _build_gptq_tensor_index(model_path)
    except Exception as exc:
        logger.warning(f"Cannot index GPTQModel expert tensors from {model_dir}: {exc}")
        return 0

    text_model = _get_text_model(native_model)
    num_layers = int(getattr(text_model.config, "num_hidden_layers"))
    num_experts = int(getattr(text_model.config, "num_experts"))
    restore_layers = _parse_restore_layer_spec(layer_spec, num_layers)
    packed_qweights: dict[int, dict[str, dict[int, torch.Tensor]]] = {
        layer_idx: {"gate_proj": {}, "up_proj": {}, "down_proj": {}}
        for layer_idx in restore_layers
    }
    packed_dequant_weights: dict[int, dict[str, dict[int, torch.Tensor]]] = {
        layer_idx: {"gate_proj": {}, "up_proj": {}, "down_proj": {}}
        for layer_idx in restore_layers
    }
    modulelist_restored = 0
    unpack_failures = 0
    prefixes = ("model.language_model.layers.", "model.layers.")

    for key in sorted(tensor_index):
        if not key.endswith(".qweight"):
            continue
        prefix = next((item for item in prefixes if key.startswith(item)), None)
        if prefix is None:
            continue
        parts = key[len(prefix):].split(".")
        if len(parts) != 6 or parts[1:3] != ["mlp", "experts"]:
            continue
        try:
            layer_idx = int(parts[0])
            expert_idx = int(parts[3])
        except ValueError:
            continue
        linear_name = parts[4]
        if (
            layer_idx < 0
            or layer_idx >= num_layers
            or layer_idx not in restore_layers
            or expert_idx < 0
            or expert_idx >= num_experts
            or linear_name not in {"gate_proj", "up_proj", "down_proj"}
        ):
            continue

        key_prefix = key[: -len(".qweight")]
        try:
            qweight = _read_indexed_gptq_tensor(tensor_index, key)
            qzeros = _read_indexed_gptq_tensor(tensor_index, f"{key_prefix}.qzeros")
            g_idx = _read_indexed_gptq_tensor(tensor_index, f"{key_prefix}.g_idx")
            scales = _read_indexed_gptq_tensor(tensor_index, f"{key_prefix}.scales")
        except Exception as exc:
            unpack_failures += 1
            if unpack_failures <= 5:
                logger.warning(f"Missing GPTQ expert tensor for {key}: {exc}")
            continue

        experts = text_model.layers[layer_idx].mlp.experts
        if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
            expected_shape = _packed_expert_weight_shape(experts, expert_idx, linear_name)
            try:
                tensor = _unpack_gptqmodel_qweight(qweight, qzeros, g_idx, expected_shape)
                packed_qweights[layer_idx][linear_name][expert_idx] = tensor
                packed_dequant_weights[layer_idx][linear_name][expert_idx] = _dequantize_gptqmodel_weight(
                    tensor,
                    scales,
                    g_idx,
                    expected_shape,
                    experts.gate_up_proj.dtype,
                )
            except Exception as exc:
                unpack_failures += 1
                if unpack_failures <= 5:
                    logger.warning(f"Failed to unpack GPTQ expert tensor {key}: {exc}")
            continue

        try:
            expert = experts[expert_idx] if hasattr(experts, "__getitem__") else experts.get_submodule(str(expert_idx))
            linear = getattr(expert, linear_name)
        except Exception:
            continue
        if isinstance(linear, nn.Linear):
            try:
                expected_shape = tuple(int(dim) for dim in linear.weight.shape)
                tensor = _unpack_gptqmodel_qweight(qweight, qzeros, g_idx, expected_shape)
                _set_quant_weight_buffer(linear, "quant_weight", tensor.to(device=linear.weight.device))
                linear.weight.data.copy_(
                    _dequantize_gptqmodel_weight(
                        tensor,
                        scales,
                        g_idx,
                        expected_shape,
                        linear.weight.dtype,
                    ).to(device=linear.weight.device)
                )
                modulelist_restored += 1
            except Exception as exc:
                unpack_failures += 1
                if unpack_failures <= 5:
                    logger.warning(f"Failed to restore GPTQ expert linear {key}: {exc}")

    packed_restored = 0
    packed_fp_restored = 0
    for layer_idx, layer_qweights in packed_qweights.items():
        experts = text_model.layers[layer_idx].mlp.experts
        if not (hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj")):
            continue
        for linear_name in ("gate_proj", "up_proj", "down_proj"):
            per_expert = layer_qweights[linear_name]
            if len(per_expert) != num_experts:
                continue
            stacked = torch.stack(
                [per_expert[expert_idx] for expert_idx in range(num_experts)],
                dim=0,
            ).contiguous()
            _set_quant_weight_buffer(
                experts,
                f"{linear_name}_quant_weight",
                stacked.to(device=experts.gate_up_proj.device),
            )
            packed_restored += num_experts

        layer_fp_weights = packed_dequant_weights[layer_idx]
        if all(len(layer_fp_weights[name]) == num_experts for name in ("gate_proj", "up_proj", "down_proj")):
            gate_weight = torch.stack(
                [layer_fp_weights["gate_proj"][expert_idx] for expert_idx in range(num_experts)],
                dim=0,
            )
            up_weight = torch.stack(
                [layer_fp_weights["up_proj"][expert_idx] for expert_idx in range(num_experts)],
                dim=0,
            )
            down_weight = torch.stack(
                [layer_fp_weights["down_proj"][expert_idx] for expert_idx in range(num_experts)],
                dim=0,
            )
            experts.gate_up_proj.data.copy_(
                torch.cat([gate_weight, up_weight], dim=1).to(
                    device=experts.gate_up_proj.device,
                    dtype=experts.gate_up_proj.dtype,
                )
            )
            experts.down_proj.data.copy_(
                down_weight.to(device=experts.down_proj.device, dtype=experts.down_proj.dtype)
            )
            packed_fp_restored += num_experts * 3

    restored = modulelist_restored + packed_restored
    logger.info(
        "Restored GPTQ expert quant_weight tensors in split export: "
        f"modulelist={modulelist_restored}, packed={packed_restored}, "
        f"total={restored}, fp_weights={packed_fp_restored}, unpack_failures={unpack_failures}"
    )
    return restored


def _load_models(args: argparse.Namespace, config: Qwen3_5MoeConvertConfig):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    quant_weight = config.quant_weight
    if quant_weight is not None:
        quant_weight = osp.normpath(osp.abspath(quant_weight))
        config.quant_weight = quant_weight

    converter = Qwen3_5MoeConverterXH2a(config)
    logger = get_root_logger()
    restore_layer_spec = getattr(args, "gptq_restore_expert_layers", None)

    loaded_model_path = hf_model_path
    if quant_weight is not None and Path(quant_weight).is_dir():
        loaded_model_path = quant_weight
        logger.info(f"Loading GPTQ/pre-quantized model directory from --quant-weight: {loaded_model_path}")
        native_model = converter.get_hf_model(
            loaded_model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
        )
    else:
        is_gptq = Qwen3_5MoeConverterXH2a._is_gptqmodel_checkpoint(hf_model_path)
        if is_gptq:
            logger.info(f"Detected GPTQModel checkpoint in --model: {hf_model_path}")
        native_model = converter.get_hf_model(
            hf_model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
        )
        if is_gptq and restore_layer_spec is not None and _count_quant_weight_tensors(native_model) == 0:
            _restore_gptq_quant_weights_from_checkpoint(
                hf_model_path,
                native_model,
                logger,
                layer_spec=restore_layer_spec,
            )
        if quant_weight is not None:
            converter.load_quant_weight(quant_weight, native_model)

    wrapped_model, wrap_cfg = converter._prepare_wrap_model(native_model)
    wrap_cfg.num_logits_to_keep = 0
    wrapped_model.to(dtype=torch.float16)
    wrapped_model.eval()

    if Qwen3_5MoeConverterXH2a._is_gptqmodel_checkpoint(loaded_model_path) and restore_layer_spec is not None:
        quant_tensor_count = _count_quant_weight_tensors(wrapped_model)
        logger.info(f"GPTQModel quant_weight tensor count after wrapping: {quant_tensor_count}")
        if quant_tensor_count == 0:
            message = (
                "Detected a GPTQModel checkpoint, but no quant_weight tensors were preserved after wrapping. "
                "Expert ONNX export may produce zero qweights before the direct ONNX repair step."
            )
            if getattr(args, "repair_gptq_expert_onnx", True):
                logger.warning(message)
            else:
                raise RuntimeError(message)

    return converter, native_model, wrapped_model, wrap_cfg, loaded_model_path


def _text_dtype_device(text_model: nn.Module) -> tuple[torch.dtype, torch.device]:
    embedding = text_model.get_input_embeddings() if hasattr(text_model, "get_input_embeddings") else None
    if embedding is not None:
        weight = embedding.weight
        return weight.dtype, weight.device
    first_param = next(text_model.parameters())
    return first_param.dtype, first_param.device


def _make_hidden_inputs(batch_size: int, seq_len: int, hidden_size: int, dtype: torch.dtype, device: torch.device):
    return torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)


def _make_premoe_inputs(
    layer,
    layer_type: str,
    args: argparse.Namespace,
    text_config,
    dtype: torch.dtype,
    device: torch.device,
    seq_len: int | None = None,
):
    batch_size = int(args.batch_size)
    if seq_len is None:
        seq_len = int(args.decode_sequence_length)
    hidden_size = int(text_config.hidden_size)
    hidden_in = _make_hidden_inputs(batch_size, seq_len, hidden_size, dtype, device)
    current_input_length = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

    if layer_type == "full_attention":
        position_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        past_seq_length = torch.zeros(batch_size, dtype=torch.int32, device=device)
        kv_shape = [batch_size, text_config.num_key_value_heads, args.context_length, text_config.head_dim]
        past_key_cache = CacheTensor(torch.zeros(kv_shape, dtype=dtype, device=device))
        past_value_cache = CacheTensor(torch.zeros(kv_shape, dtype=dtype, device=device))
        inputs = (
            hidden_in,
            position_ids,
            position_ids,
            position_ids,
            past_seq_length,
            current_input_length,
            past_key_cache,
            past_value_cache,
        )
        input_names = [
            "hidden_in",
            "time_position_ids",
            "hight_position_ids",
            "width_position_ids",
            "past_seq_length",
            "current_input_length",
            "past_key_cache",
            "past_value_cache",
        ]
        output_names = ["moe_input", "topk_id", "topk_gate", "shared_out", "residual1"]
        cache_meta = {
            "kv_shape": kv_shape,
            "input_names": ["past_key_cache", "past_value_cache"],
            "output_names": [],
        }
        return inputs, input_names, output_names, cache_meta

    linear_attn_mask = torch.ones(batch_size, seq_len, dtype=dtype, device=device)
    cache_dtype = dtype
    if args.split_conv_cache:
        cache_dtype = (
            layer.linear_attn.conv1d_q.weight.dtype
            if hasattr(layer.linear_attn, "conv1d_q")
            else layer.linear_attn.conv1d.weight.dtype
        )
        q_dim, k_dim, v_dim = _linear_split_conv_dims(layer.linear_attn)
        conv_shapes = [
            [batch_size, q_dim, layer.linear_attn.conv_kernel_size],
            [batch_size, k_dim, layer.linear_attn.conv_kernel_size],
            [batch_size, v_dim, layer.linear_attn.conv_kernel_size],
        ]
        conv_inputs = tuple(CacheTensor(torch.zeros(shape, dtype=cache_dtype, device=device)) for shape in conv_shapes)
        conv_input_names = [f"past_conv_cache_{branch}" for branch in _LINEAR_CONV_CACHE_BRANCHES]
        conv_output_names = [f"conv_cache_out_{branch}" for branch in _LINEAR_CONV_CACHE_BRANCHES]
    else:
        cache_dtype = layer.linear_attn.conv1d.weight.dtype
        conv_shapes = [[batch_size, layer.linear_attn.conv_dim, layer.linear_attn.conv_kernel_size]]
        conv_inputs = (CacheTensor(torch.zeros(conv_shapes[0], dtype=cache_dtype, device=device)),)
        # The LinearAttentionPreMoEExportModule signature always has q/k/v slots;
        # k/v are unused in merged-cache mode but supplied as harmless placeholders.
        conv_inputs = conv_inputs + conv_inputs + conv_inputs
        conv_input_names = ["past_conv_cache_q", "past_conv_cache_k", "past_conv_cache_v"]
        conv_output_names = ["conv_cache_out"]
    recurrent_shape = [
        batch_size,
        layer.linear_attn.num_v_heads,
        layer.linear_attn.head_k_dim,
        layer.linear_attn.head_v_dim,
    ]
    recurrent_state = CacheTensor(torch.zeros(recurrent_shape, dtype=cache_dtype, device=device))
    inputs = (
        hidden_in,
        current_input_length,
        linear_attn_mask,
        *conv_inputs,
        recurrent_state,
    )
    input_names = [
        "hidden_in",
        "current_input_length",
        "linear_attn_mask",
        *conv_input_names,
        "past_recurrent_state",
    ]
    output_names = [
        "moe_input",
        "topk_id",
        "topk_gate",
        "shared_out",
        "residual1",
        *conv_output_names,
        "recurrent_state_out",
    ]
    cache_meta = {
        "conv_shapes": conv_shapes,
        "recurrent_shape": recurrent_shape,
        "input_names": [*conv_input_names, "past_recurrent_state"],
        "output_names": [*conv_output_names, "recurrent_state_out"],
    }
    return inputs, input_names, output_names, cache_meta


def _export_one_hmonnx(
    converter: Qwen3_5MoeConverterXH2a,
    module: nn.Module,
    inputs,
    output_path: Path,
    input_names: list[str],
    output_names: list[str],
    logger,
    quant_type: str,
    overwrite: bool = False,
) -> bool:
    output_path.parent.mkdir(exist_ok=True, parents=True)
    if not overwrite and _has_complete_output(output_path):
        logger.info(f"Skip existing HMONNX: {output_path} (use --overwrite to regenerate)")
        return False
    if output_path.exists():
        output_path.unlink()
    logger.info(f"HMONNX quant_type={quant_type}: {output_path}")
    quant_config = _build_quant_config_for_module(converter, module, quant_type)
    quanted_model = convert_fx_model_to_quanted_model(
        module,
        inputs,
        DeviceType.XH2a,
        quant_config=quant_config,
    )
    convert_quanted_model_to_hmonnx(
        quanted_model,
        inputs,
        str(output_path),
        BaseConverter.xh1_hmonnx_compatible(input_names),
        output_names,
    )
    patched_adds = _patch_hmonnx_standard_add_ops(output_path)
    if patched_adds:
        logger.info(f"Patched {patched_adds} standard Add node(s) in {output_path}")
    logger.info(f"Exported {output_path}")
    del quanted_model
    return True


def _export_premoe_layers(
    args: argparse.Namespace,
    converter: Qwen3_5MoeConverterXH2a,
    wrapped_text_model: nn.Module,
    text_config,
    layer_indices: Iterable[int],
    work_dir: Path,
    logger,
    quant_type: str,
    seq_len: int | None = None,
    mode: str = "decode",
) -> list[dict[str, Any]]:
    dtype, device = _text_dtype_device(wrapped_text_model)
    layer_types = list(text_config.layer_types)
    top_k = int(text_config.num_experts_per_tok)
    if seq_len is None:
        seq_len = int(args.decode_sequence_length)
    subdir = "premoe_prefill" if mode == "prefill" else "premoe"
    records = []
    for layer_idx in layer_indices:
        layer_type = layer_types[layer_idx]
        if layer_type == "full_attention":
            module = FullAttentionPreMoEExportModule(wrapped_text_model, layer_idx, top_k).eval()
        elif layer_type == "linear_attention":
            module = LinearAttentionPreMoEExportModule(
                wrapped_text_model,
                layer_idx,
                top_k,
                split_conv_cache=args.split_conv_cache,
            ).eval()
        else:
            raise ValueError(f"Unsupported layer_type for layer {layer_idx}: {layer_type}")
        layer = module.layer
        inputs, input_names, output_names, cache_meta = _make_premoe_inputs(
            layer,
            layer_type,
            args,
            text_config,
            dtype,
            device,
            seq_len=seq_len,
        )
        output_path = work_dir / "hmonnx" / subdir / f"layer_{layer_idx:03d}_premoe.onnx"
        logger.info(f"Exporting PreMoE ({mode}) layer={layer_idx} type={layer_type} seq_len={seq_len}")
        logger.info(f"PreMoE ({mode}) layer={layer_idx} quant_weight_buffers={_count_quant_weight_buffers(module)}")
        _export_one_hmonnx(
            converter,
            module,
            inputs,
            output_path,
            input_names,
            output_names,
            logger,
            quant_type=quant_type,
            overwrite=args.overwrite,
        )
        records.append(
            {
                "layer_idx": layer_idx,
                "layer_type": layer_type,
                "onnx": str(output_path.relative_to(work_dir)),
                "input_names": input_names,
                "output_names": output_names,
                "cache": cache_meta,
            }
        )
    return records


def _make_expert_input(args: argparse.Namespace, text_config, dtype: torch.dtype, device: torch.device):
    return _make_hidden_inputs(
        int(args.batch_size),
        int(args.expert_sequence_length),
        int(text_config.hidden_size),
        dtype,
        device,
    )


def _export_expert_layers(
    args: argparse.Namespace,
    converter: Qwen3_5MoeConverterXH2a,
    wrapped_text_model: nn.Module,
    text_config,
    layer_indices: Iterable[int],
    expert_indices: Iterable[int],
    work_dir: Path,
    logger,
    quant_type: str,
) -> list[dict[str, Any]]:
    dtype, device = _text_dtype_device(wrapped_text_model)
    expert_input = _make_expert_input(args, text_config, dtype, device)
    input_names = ["expert_input"]
    output_names = ["expert_out"]
    records = []
    for layer_idx in layer_indices:
        layer_records = []
        mlp = wrapped_text_model.layers[layer_idx].mlp
        for expert_id in expert_indices:
            module = SingleExpertExportModule(mlp, expert_id).eval()
            output_path = work_dir / "hmonnx" / "experts" / f"layer_{layer_idx:03d}" / f"expert_{expert_id:03d}.onnx"
            logger.info(f"Exporting Expert layer={layer_idx} expert={expert_id}")
            quant_weight_buffers = _count_quant_weight_buffers(module)
            logger.info(f"Expert layer={layer_idx} expert={expert_id} quant_weight_buffers={quant_weight_buffers}")
            _export_one_hmonnx(
                converter,
                module,
                (expert_input,),
                output_path,
                input_names,
                output_names,
                logger,
                quant_type=quant_type,
                overwrite=args.overwrite,
            )
            layer_records.append(
                {
                    "expert_id": expert_id,
                    "onnx": str(output_path.relative_to(work_dir)),
                }
            )
            del module
        records.append(
            {
                "layer_idx": layer_idx,
                "input_names": input_names,
                "output_names": output_names,
                "experts": layer_records,
            }
        )
    return records


def _expert_onnx_path(work_dir: Path, layer_idx: int, expert_id: int) -> Path:
    return work_dir / "hmonnx" / "experts" / f"layer_{layer_idx:03d}" / f"expert_{expert_id:03d}.onnx"


def _scan_bad_expert_onnx(
    work_dir: Path,
    layer_indices: Iterable[int],
    expert_indices: Iterable[int],
    logger,
) -> list[tuple[Path, list[str]]]:
    import numpy as np
    import onnx
    from onnx import numpy_helper

    bad_files: list[tuple[Path, list[str]]] = []
    checked = 0
    for layer_idx in layer_indices:
        for expert_id in expert_indices:
            onnx_path = _expert_onnx_path(work_dir, layer_idx, expert_id)
            issues: list[str] = []
            if not onnx_path.exists():
                bad_files.append((onnx_path, ["missing"]))
                continue
            model = onnx.load(str(onnx_path), load_external_data=True)
            checked += 1
            qweight_inits = [init for init in model.graph.initializer if init.name.endswith(".qweight")]
            if not qweight_inits:
                issues.append("missing_qweight")
            else:
                zero_names = [
                    init.name
                    for init in qweight_inits
                    if np.count_nonzero(numpy_helper.to_array(init)) == 0
                ]
                if zero_names:
                    issues.append(f"zero_qweight={zero_names}")
            non_ssfp_nodes = []
            for node in model.graph.node:
                if node.op_type != "Linear":
                    continue
                mode_attr = next((attr for attr in node.attribute if attr.name == "mode"), None)
                mode = "" if mode_attr is None else mode_attr.s.decode(errors="replace").lower()
                if mode != "ssfp":
                    non_ssfp_nodes.append(node.name or "<unnamed>")
            if non_ssfp_nodes:
                issues.append(f"non_ssfp_linear={non_ssfp_nodes}")
            if issues:
                bad_files.append((onnx_path, issues))
    logger.info(f"Validated {checked} expert ONNX file(s): bad={len(bad_files)}")
    return bad_files


def _onnx_tensor_dtype(np_dtype) -> int:
    import numpy as np
    import onnx

    if np_dtype == np.dtype("float16"):
        return onnx.TensorProto.FLOAT16
    if np_dtype == np.dtype("float32"):
        return onnx.TensorProto.FLOAT
    if np_dtype == np.dtype("int8"):
        return onnx.TensorProto.INT8
    raise TypeError(f"Unsupported dtype: {np_dtype}")


def _replace_onnx_initializer(model, name: str, array) -> None:
    import numpy as np
    from onnx import numpy_helper

    for idx, init in enumerate(model.graph.initializer):
        if init.name != name:
            continue
        replacement = numpy_helper.from_array(np.ascontiguousarray(array), name=name)
        replacement.data_type = _onnx_tensor_dtype(array.dtype)
        del model.graph.initializer[idx]
        model.graph.initializer.insert(idx, replacement)
        return
    raise KeyError(f"Initializer {name!r} not found")


def _set_onnx_linear_mode_ssfp(model) -> None:
    for node in model.graph.node:
        if node.op_type != "Linear":
            continue
        mode_attr = next((attr for attr in node.attribute if attr.name == "mode"), None)
        if mode_attr is None:
            mode_attr = node.attribute.add()
            mode_attr.name = "mode"
        mode_attr.s = b"ssfp"


def _expected_shape_from_onnx_qweight(model, linear_name: str) -> tuple[int, int]:
    init = next(item for item in model.graph.initializer if item.name == f"{linear_name}.qweight")
    dims = tuple(int(dim) for dim in init.dims)
    if len(dims) != 3 or dims[1] != 64:
        raise ValueError(f"Unsupported {linear_name}.qweight shape: {dims}")
    return dims[2], dims[0] * dims[1]


def _find_gptq_tensor_prefix(tensor_index: dict[str, Path], layer_idx: int, expert_idx: int, linear_name: str) -> str:
    suffix = f"{layer_idx}.mlp.experts.{expert_idx}.{linear_name}"
    for prefix in ("model.language_model.layers.", "model.layers."):
        key_prefix = f"{prefix}{suffix}"
        if f"{key_prefix}.qweight" in tensor_index:
            return key_prefix
    raise KeyError(f"Missing GPTQ tensors for layer={layer_idx} expert={expert_idx} {linear_name}")


def _make_hmonnx_expert_initializers(
    tensor_index: dict[str, Path],
    key_prefix: str,
    expected_shape: tuple[int, int],
    qweight_shape: tuple[int, int, int],
) -> tuple[Any, Any]:
    import numpy as np

    qweight = _read_indexed_gptq_tensor(tensor_index, f"{key_prefix}.qweight")
    qzeros = _read_indexed_gptq_tensor(tensor_index, f"{key_prefix}.qzeros")
    scales = _read_indexed_gptq_tensor(tensor_index, f"{key_prefix}.scales")
    g_idx = _read_indexed_gptq_tensor(tensor_index, f"{key_prefix}.g_idx")

    unpacked = _unpack_gptqmodel_qweight(qweight, qzeros, g_idx, expected_shape)
    q_np = unpacked.numpy().astype(np.int16, copy=False)
    scales_np = scales.numpy()
    g_idx_np = g_idx.numpy().astype(np.int64, copy=False)[: expected_shape[1]]

    sign = np.sign(scales_np).astype(np.int16, copy=False)
    sign[sign == 0] = 1
    signed = q_np * sign[g_idx_np, : expected_shape[0]].T
    signed = np.clip(signed, -8, 7).astype(np.int8, copy=False)
    hmonnx_qweight = signed.reshape(expected_shape[0], expected_shape[1] // 64, 64).transpose(1, 2, 0)
    if tuple(hmonnx_qweight.shape) != tuple(qweight_shape):
        raise ValueError(f"Packed qweight shape {hmonnx_qweight.shape} != expected {qweight_shape}")

    scale_or_exp = np.abs(scales_np.astype(np.float32, copy=False))
    scale_or_exp[scale_or_exp <= 1.1e-5] = 1.0
    scale_or_exp = scale_or_exp.astype(np.float16).reshape(qweight_shape[0], 1, qweight_shape[2])
    return hmonnx_qweight, scale_or_exp


def _expert_onnx_needs_repair(model) -> bool:
    import numpy as np
    from onnx import numpy_helper

    for init in model.graph.initializer:
        if init.name.endswith(".qweight") and np.count_nonzero(numpy_helper.to_array(init)) == 0:
            return True
    for node in model.graph.node:
        if node.op_type != "Linear":
            continue
        mode_attr = next((attr for attr in node.attribute if attr.name == "mode"), None)
        mode = "" if mode_attr is None else mode_attr.s.decode(errors="replace").lower()
        if mode != "ssfp":
            return True
    return False


def _patch_one_gptq_expert_onnx(
    onnx_path: Path,
    tensor_index: dict[str, Path],
    layer_idx: int,
    expert_idx: int,
) -> bool:
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(onnx_path), load_external_data=True)
    if not _expert_onnx_needs_repair(model):
        return False

    updated = {}
    for linear_name in ("gate_proj", "up_proj", "down_proj"):
        q_init = next(item for item in model.graph.initializer if item.name == f"{linear_name}.qweight")
        qweight_shape = tuple(int(dim) for dim in q_init.dims)
        expected_shape = _expected_shape_from_onnx_qweight(model, linear_name)
        key_prefix = _find_gptq_tensor_prefix(tensor_index, layer_idx, expert_idx, linear_name)
        updated[linear_name] = _make_hmonnx_expert_initializers(
            tensor_index,
            key_prefix,
            expected_shape,
            qweight_shape,
        )

    for linear_name, (qweight, scale_or_exp) in updated.items():
        _replace_onnx_initializer(model, f"{linear_name}.qweight", qweight)
        _replace_onnx_initializer(model, f"{linear_name}.scale_or_exp", scale_or_exp)
    _set_onnx_linear_mode_ssfp(model)
    onnx.save(model, str(onnx_path))

    repaired = onnx.load(str(onnx_path), load_external_data=True)
    zero_qweights = [
        init.name
        for init in repaired.graph.initializer
        if init.name.endswith(".qweight")
        and numpy_helper.to_array(init).max() == 0
        and numpy_helper.to_array(init).min() == 0
    ]
    if zero_qweights:
        raise RuntimeError(f"Repaired expert ONNX still has zero qweights: {zero_qweights}")
    return True


def _repair_gptq_expert_onnx(
    model_dir: str,
    work_dir: Path,
    layer_indices: Iterable[int],
    expert_indices: Iterable[int],
    logger,
) -> int:
    model_path = Path(model_dir)
    tensor_index = _build_gptq_tensor_index(model_path)
    patched = 0
    failures: list[tuple[Path, str]] = []
    for layer_idx in layer_indices:
        for expert_id in expert_indices:
            onnx_path = _expert_onnx_path(work_dir, layer_idx, expert_id)
            if not onnx_path.exists():
                failures.append((onnx_path, "missing"))
                continue
            try:
                if _patch_one_gptq_expert_onnx(
                    onnx_path,
                    tensor_index,
                    layer_idx,
                    expert_id,
                ):
                    patched += 1
            except Exception as exc:
                failures.append((onnx_path, str(exc)))
    if failures:
        preview = "; ".join(f"{path}: {reason}" for path, reason in failures[:5])
        raise RuntimeError(f"Failed to repair {len(failures)} expert ONNX file(s): {preview}")
    logger.info(f"Repaired {patched} GPTQ expert ONNX file(s) with direct qweight/mode patch")
    return patched


def _validate_and_repair_expert_onnx(
    args: argparse.Namespace,
    loaded_model_path: str,
    work_dir: Path,
    layer_indices: Iterable[int],
    expert_indices: Iterable[int],
    logger,
) -> None:
    bad_files = _scan_bad_expert_onnx(work_dir, layer_indices, expert_indices, logger)
    if not bad_files:
        return
    is_gptq = Qwen3_5MoeConverterXH2a._is_gptqmodel_checkpoint(loaded_model_path)
    if is_gptq and args.repair_gptq_expert_onnx:
        logger.warning(f"Found {len(bad_files)} bad expert ONNX file(s); repairing from GPTQModel tensors.")
        _repair_gptq_expert_onnx(loaded_model_path, work_dir, layer_indices, expert_indices, logger)
        bad_files = _scan_bad_expert_onnx(work_dir, layer_indices, expert_indices, logger)
        if not bad_files:
            return
    preview = "; ".join(f"{path}: {issues}" for path, issues in bad_files[:5])
    raise RuntimeError(
        "Expert ONNX validation failed. Expected nonzero qweight initializers and Linear.mode=ssfp. "
        f"bad_files={len(bad_files)} preview={preview}"
    )


def _export_postmoe(
    args: argparse.Namespace,
    converter: Qwen3_5MoeConverterXH2a,
    text_config,
    dtype: torch.dtype,
    device: torch.device,
    work_dir: Path,
    logger,
    quant_type: str,
) -> dict[str, Any]:
    top_k = int(text_config.num_experts_per_tok)
    if top_k != 8:
        raise ValueError(f"PostMoENPUAggModule currently has a fixed 8-expert ABI, got top_k={top_k}")
    module = PostMoENPUAggModule(top_k=top_k).eval()
    shape = [int(args.batch_size), int(args.decode_sequence_length), int(text_config.hidden_size)]
    expert_outs = tuple(torch.randn(shape, dtype=dtype, device=device) for _ in range(top_k))
    topk_gate = torch.rand(
        int(args.batch_size),
        int(args.decode_sequence_length),
        top_k,
        dtype=dtype,
        device=device,
    )
    topk_gate = topk_gate / topk_gate.sum(dim=-1, keepdim=True)
    shared_out = torch.randn(shape, dtype=dtype, device=device)
    residual1 = torch.randn(shape, dtype=dtype, device=device)
    inputs = (*expert_outs, topk_gate, shared_out, residual1)
    input_names = [*(f"expert_out_{idx}" for idx in range(top_k)), "topk_gate", "shared_out", "residual1"]
    output_names = ["hidden_out"]
    output_path = work_dir / "hmonnx" / "postmoe" / "postmoe_decode_npu_agg.onnx"
    logger.info("Exporting shared PostMoE decode NPU aggregation graph")
    logger.info("PostMoE has no model weights; using PreMoE quant_type for activation-only graph")
    _export_one_hmonnx(
        converter,
        module,
        inputs,
        output_path,
        input_names,
        output_names,
        logger,
        quant_type=quant_type,
        overwrite=args.overwrite,
    )
    return {
        "onnx": str(output_path.relative_to(work_dir)),
        "input_names": input_names,
        "output_names": output_names,
    }


def _export_head(
    args: argparse.Namespace,
    converter: Qwen3_5MoeConverterXH2a,
    wrapped_model: nn.Module,
    wrapped_text_model: nn.Module,
    text_config,
    work_dir: Path,
    logger,
    quant_type: str,
) -> dict[str, Any]:
    dtype, device = _text_dtype_device(wrapped_text_model)
    seq_len = int(args.head_sequence_length or args.decode_sequence_length)
    if seq_len < 1:
        raise ValueError("Head sequence length must be positive.")
    module = HeadExportModule(wrapped_text_model.norm, _get_lm_head(wrapped_model, wrapped_text_model)).eval()
    hidden_in = _make_hidden_inputs(
        int(args.batch_size),
        seq_len,
        int(text_config.hidden_size),
        dtype,
        device,
    )
    input_names = ["hidden_in"]
    output_names = ["logits"]
    output_path = work_dir / "hmonnx" / "head" / "head.onnx"
    logger.info(f"Exporting Head final norm + lm_head seq_len={seq_len}")
    logger.info(f"Head quant_weight_buffers={_count_quant_weight_buffers(module)}")
    _export_one_hmonnx(
        converter,
        module,
        (hidden_in,),
        output_path,
        input_names,
        output_names,
        logger,
        quant_type=quant_type,
        overwrite=args.overwrite,
    )
    return {
        "onnx": str(output_path.relative_to(work_dir)),
        "input_names": input_names,
        "output_names": output_names,
        "input_shape": [int(args.batch_size), seq_len, int(text_config.hidden_size)],
        "output_shape": [int(args.batch_size), seq_len, int(text_config.vocab_size)],
        "sequence_length": seq_len,
    }


def _write_meta(
    args: argparse.Namespace,
    native_model,
    text_config,
    config: Qwen3_5MoeConvertConfig,
    work_dir: Path,
    wrap_cfg: Config,
    layer_indices: list[int],
    expert_indices: list[int],
    premoe_prefill_records: list[dict[str, Any]],
    premoe_decode_records: list[dict[str, Any]],
    expert_records: list[dict[str, Any]],
    postmoe_record: Optional[dict[str, Any]],
    head_record: Optional[dict[str, Any]],
    export_parts: set[str],
    loaded_model_path: str,
) -> None:
    hf_config_dir = _copy_hf_config(args.model, work_dir)
    text_model = _get_text_model(native_model)
    token_embedding = text_model.get_input_embeddings()
    token_embedding_file = work_dir / "token_embedding.pt"
    torch.save(token_embedding.state_dict(), str(token_embedding_file))

    meta = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "architecture": "Qwen3_5MoeSplitMoEDecode",
        "model_name": Path(args.model).name,
        "hf_model_path": osp.normpath(osp.abspath(args.model)),
        "loaded_model_path": osp.normpath(osp.abspath(loaded_model_path)),
        "device": str(DeviceType.XH2a),
        "quant_type": args.quant_type,
        "quant_scheme": config.quant_scheme.to_dict(),
        "split_quant_types": {
            "premoe": args.premoe_quant_type,
            "experts": args.expert_quant_type,
            "head": args.head_quant_type,
        },
        "quant_weight": args.quant_weight,
        "gptq_restore_expert_layers": getattr(args, "gptq_restore_expert_layers", None),
        "repair_gptq_expert_onnx": bool(getattr(args, "repair_gptq_expert_onnx", False)),
        "source_quant_method": _source_quant_method(native_model, loaded_model_path),
        "hf_config": str(hf_config_dir.relative_to(work_dir)),
        "token_embedding_file": str(token_embedding_file.relative_to(work_dir)),
        "quant_embedding": str(token_embedding_file.relative_to(work_dir)),
        "batch_size": int(args.batch_size),
        "max_context_tokens": int(args.context_length),
        "prefill_sequence_length": int(args.prefill_sequence_length),
        "decode_sequence_length": int(args.decode_sequence_length),
        "expert_sequence_length": int(args.expert_sequence_length),
        "hidden_size": int(text_config.hidden_size),
        "num_hidden_layers": int(text_config.num_hidden_layers),
        "num_experts": int(text_config.num_experts),
        "num_experts_per_tok": int(text_config.num_experts_per_tok),
        "layer_types": list(text_config.layer_types),
        "export_parts": sorted(export_parts),
        "exported_layers": layer_indices,
        "exported_experts": expert_indices,
        "wrap_cfg": wrap_cfg.to_dict(),
        "premoe": {
            "description": "Per-layer attention + router + shared expert graphs.",
            "prefill": {
                "sequence_length": int(args.prefill_sequence_length),
                "layers": premoe_prefill_records,
            },
            "decode": {
                "sequence_length": int(args.decode_sequence_length),
                "layers": premoe_decode_records,
            },
        },
        "experts": {
            "description": "One routed expert MLP graph per layer/expert id.",
            "input_shape": [int(args.batch_size), int(args.expert_sequence_length), int(text_config.hidden_size)],
            "layers": expert_records,
        },
        "postmoe": postmoe_record,
        "head": head_record,
    }
    meta_path = work_dir / "split_moe_meta.json"
    if meta_path.exists():
        try:
            old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            old_meta = None
        if isinstance(old_meta, dict) and old_meta.get("architecture") == meta["architecture"]:
            for key in ("premoe", "experts", "postmoe", "head"):
                value = meta.get(key)
                missing_value = value is None
                if key == "premoe" and isinstance(value, dict):
                    missing_value = not value.get("prefill", {}).get("layers") and not value.get("decode", {}).get(
                        "layers"
                    )
                elif key == "experts" and isinstance(value, dict):
                    missing_value = not value.get("layers")
                if missing_value and old_meta.get(key) is not None:
                    meta[key] = old_meta[key]
            meta["export_parts"] = sorted(set(old_meta.get("export_parts", [])) | set(meta["export_parts"]))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main(args: argparse.Namespace) -> None:
    if args.batch_size != 1:
        raise ValueError("The first split-MoE export ABI supports --batch-size 1 only.")
    if args.decode_sequence_length < 1 or args.expert_sequence_length < 1:
        raise ValueError("Sequence lengths must be positive.")

    export_parts = _parse_export_parts(args.export_parts)
    args.premoe_quant_type, args.expert_quant_type, args.head_quant_type = _resolve_split_quant_types(args)
    args.quant_type = args.premoe_quant_type
    args.gptq_restore_expert_layers = _resolve_gptq_restore_layer_spec(args, export_parts)

    args.model = osp.normpath(osp.abspath(args.model))
    if args.quant_weight is not None:
        args.quant_weight = osp.normpath(osp.abspath(args.quant_weight))

    hf_model_path = args.model
    model_name = Path(hf_model_path).name
    split_quant_tag = f"premoe-{args.premoe_quant_type}-experts-{args.expert_quant_type}"
    if args.head_quant_type != args.premoe_quant_type:
        split_quant_tag = f"{split_quant_tag}-head-{args.head_quant_type}"
    work_dir = Path(args.work_dir) if args.work_dir else Path("work_dirs") / f"{model_name}-split-moe-{split_quant_tag}"
    work_dir = work_dir.resolve()
    work_dir.mkdir(exist_ok=True, parents=True)
    xhquant_init(work_dir / "split_moe_convert.log", debug=args.debug)
    logger = get_root_logger()
    logger.info(f"model: {hf_model_path}")
    logger.info(f"output: {work_dir}")
    logger.info(f"quant_type: {args.quant_type}")
    logger.info(f"premoe_quant_type: {args.premoe_quant_type}")
    logger.info(f"expert_quant_type: {args.expert_quant_type}")
    logger.info(f"head_quant_type: {args.head_quant_type}")
    logger.info(f"export_parts: {sorted(export_parts)}")
    logger.info(f"quant_weight: {args.quant_weight}")
    logger.info(f"gptq_restore_expert_layers: {args.gptq_restore_expert_layers}")

    config = _build_convert_config(args)
    converter, native_model, wrapped_model, wrap_cfg, loaded_model_path = _load_models(args, config)
    text_config = _get_text_config(native_model)

    layer_indices = _parse_index_spec(args.export_layers, int(text_config.num_hidden_layers), "layer")
    expert_indices = _parse_index_spec(args.export_experts, int(text_config.num_experts), "expert")
    logger.info(f"export_layers: {layer_indices}")
    logger.info(f"export_experts: {expert_indices}")

    do_premoe = "premoe" in export_parts and not args.skip_premoe
    do_experts = "experts" in export_parts and not args.skip_experts
    do_postmoe = "postmoe" in export_parts and not args.skip_postmoe
    do_head = "head" in export_parts and not args.skip_head
    do_prefill = do_premoe and not args.skip_premoe_prefill and args.prefill_sequence_length > 0
    do_decode = do_premoe and not args.skip_premoe_decode
    premoe_graph_count = len(layer_indices) * int(do_prefill) + len(layer_indices) * int(do_decode)
    expert_graph_count = len(layer_indices) * len(expert_indices) if do_experts else 0
    postmoe_graph_count = 1 if do_postmoe else 0
    head_graph_count = 1 if do_head else 0
    total_graph_count = premoe_graph_count + expert_graph_count + postmoe_graph_count + head_graph_count
    logger.info(
        "planned split exports: "
        f"premoe={premoe_graph_count}, experts={expert_graph_count}, "
        f"postmoe={postmoe_graph_count}, head={head_graph_count}, "
        f"total={total_graph_count}, overwrite={args.overwrite}"
    )
    if expert_graph_count > 512:
        logger.warning(
            "Expert export is one HMONNX per layer/expert and can be very slow: "
            f"layers={len(layer_indices)}, experts={len(expert_indices)}, graphs={expert_graph_count}. "
            "Use --export-layers/--export-experts for smoke runs; "
            "existing outputs are skipped unless --overwrite is set."
        )

    # Prepare prefill model copy (deepcopy before applying cfg, since
    # convert_fx_model_to_quanted_model mutates in-place).
    if do_prefill:
        logger.info(f"Preparing prefill model (seq_len={args.prefill_sequence_length})...")
        wrap_cfg_prefill = deepcopy(wrap_cfg)
        wrap_cfg_prefill.input_sequence_length = int(args.prefill_sequence_length)
        wrap_cfg_prefill.linear_attention_mode = "chunk"
        wrapped_model_prefill = deepcopy(wrapped_model)
        wrapped_model_prefill.apply(lambda m: _apply_update_cfg(m, wrap_cfg_prefill))
        wrapped_model_prefill.eval()

    # Prepare decode model (applied to the original wrapped_model).
    logger.info(f"Preparing decode model (seq_len={args.decode_sequence_length})...")
    wrap_cfg_decode = deepcopy(wrap_cfg)
    wrap_cfg_decode.input_sequence_length = int(args.decode_sequence_length)
    wrapped_model.apply(lambda m: _apply_update_cfg(m, wrap_cfg_decode))
    wrapped_model.eval()

    wrapped_text_model_decode = _get_wrapped_text_model(wrapped_model)
    dtype, device = _text_dtype_device(wrapped_text_model_decode)

    premoe_prefill_records: list[dict[str, Any]] = []
    premoe_decode_records: list[dict[str, Any]] = []
    expert_records: list[dict[str, Any]] = []
    postmoe_record: Optional[dict[str, Any]] = None
    head_record: Optional[dict[str, Any]] = None

    with TimeProfiler("split-moe export", logger), MemoryTracker("cuda:0", "split-moe export", logger):
        if do_prefill:
            wrapped_text_model_prefill = _get_wrapped_text_model(wrapped_model_prefill)
            premoe_prefill_records = _export_premoe_layers(
                args,
                converter,
                wrapped_text_model_prefill,
                text_config,
                layer_indices,
                work_dir,
                logger,
                quant_type=args.premoe_quant_type,
                seq_len=args.prefill_sequence_length,
                mode="prefill",
            )
            del wrapped_model_prefill, wrapped_text_model_prefill

        if do_decode:
            premoe_decode_records = _export_premoe_layers(
                args,
                converter,
                wrapped_text_model_decode,
                text_config,
                layer_indices,
                work_dir,
                logger,
                quant_type=args.premoe_quant_type,
                seq_len=args.decode_sequence_length,
                mode="decode",
            )

        if do_experts:
            expert_records = _export_expert_layers(
                args,
                converter,
                wrapped_text_model_decode,
                text_config,
                layer_indices,
                expert_indices,
                work_dir,
                logger,
                quant_type=args.expert_quant_type,
            )
            _validate_and_repair_expert_onnx(
                args,
                loaded_model_path,
                work_dir,
                layer_indices,
                expert_indices,
                logger,
            )
        if do_postmoe:
            postmoe_record = _export_postmoe(
                args,
                converter,
                text_config,
                dtype,
                device,
                work_dir,
                logger,
                quant_type=args.premoe_quant_type,
            )
        if do_head:
            head_record = _export_head(
                args,
                converter,
                wrapped_model,
                wrapped_text_model_decode,
                text_config,
                work_dir,
                logger,
                quant_type=args.head_quant_type,
            )

    _write_meta(
        args,
        native_model,
        text_config,
        config,
        work_dir,
        wrap_cfg_decode,
        layer_indices,
        expert_indices,
        premoe_prefill_records,
        premoe_decode_records,
        expert_records,
        postmoe_record,
        head_record,
        export_parts,
        loaded_model_path,
    )
    logger.info(f"Done. Split-MoE artifacts in: {work_dir}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Qwen3.5-MoE split PreMoE/Expert/PostMoE HMONNX graphs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="weights/Qwen3.5-35B-A3B", help="HuggingFace model directory")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size. First ABI supports 1 only")
    parser.add_argument("--context-length", type=int, default=2048, help="KV cache context length")
    parser.add_argument(
        "--prefill-sequence-length",
        type=int,
        default=256,
        help="PreMoE prefill sequence length (0 to skip prefill export)",
    )
    parser.add_argument("--decode-sequence-length", type=int, default=1, help="PreMoE/PostMoE decode sequence length")
    parser.add_argument("--expert-sequence-length", type=int, default=1, help="Expert graph sequence length")
    parser.add_argument("--quant-type", type=str, default="w8a8h0_sefp", help="Base/PreMoE quantisation type string")
    parser.add_argument("--premoe-quant-type", type=str, default=None, help="PreMoE attention/router/shared quant type")
    parser.add_argument(
        "--expert-quant-type",
        type=str,
        default="w4a8h0_ssfp",
        help="Expert MLP quant type; W4 sefp values are normalized to ssfp",
    )
    parser.add_argument(
        "--head-quant-type",
        type=str,
        default=None,
        help="Final norm + lm_head quant type; defaults to PreMoE quant type",
    )
    parser.add_argument(
        "--quant-weight", type=str, default=None, help="Path to GPTQModel quantised weights or weight file"
    )
    parser.add_argument(
        "--gptq-restore-expert-layers",
        type=str,
        default=None,
        help="GPTQModel routed expert layers to restore; defaults to --export-layers when exporting experts",
    )
    parser.add_argument(
        "--work-dir", "--work_dir", dest="work_dir", type=str, default=None, help="Output work directory"
    )
    parser.add_argument(
        "--linear-attention-mode",
        type=str,
        default="recurrent",
        choices=["auto", "chunk", "recurrent"],
        help="Linear attention computation mode for decode split graphs",
    )
    parser.add_argument("--linear-chunk-size", type=int, default=64, help="Chunk size for linear attention")
    parser.set_defaults(split_conv_cache=True)
    parser.add_argument(
        "--split-conv-cache", dest="split_conv_cache", action="store_true", help="Use q/k/v split conv cache"
    )
    parser.add_argument(
        "--no-split-conv-cache", dest="split_conv_cache", action="store_false", help="Use legacy merged conv cache"
    )
    parser.add_argument("--normalize-force-fp32", dest="normalize_force_fp32", action="store_true", default=False)
    parser.add_argument(
        "--use-manual-depthwise-conv1d", dest="use_manual_depthwise_conv1d", action="store_true", default=False
    )
    parser.add_argument("--fuse-gdr-ops", dest="fuse_gdr_ops", action="store_true", default=False)
    parser.add_argument(
        "--export-parts",
        type=str,
        default="all",
        help="Comma separated split parts to export: all, premoe, experts, postmoe, head",
    )
    parser.add_argument("--export-layers", type=str, default="all", help="Layer ids: all, 0, 0,1,2 or 0-3")
    parser.add_argument("--export-experts", type=str, default="all", help="Expert ids: all, 0, 0,1,2 or 0-7")
    parser.add_argument(
        "--head-sequence-length",
        type=int,
        default=0,
        help="Head graph sequence length; 0 means use --decode-sequence-length",
    )
    parser.add_argument("--skip-premoe", action="store_true", help="Skip PreMoE export")
    parser.add_argument("--skip-premoe-prefill", action="store_true", help="Skip PreMoE prefill export only")
    parser.add_argument("--skip-premoe-decode", action="store_true", help="Skip PreMoE decode export only")
    parser.add_argument("--skip-experts", action="store_true", help="Skip expert MLP export")
    parser.add_argument("--skip-postmoe", action="store_true", help="Skip PostMoE export")
    parser.add_argument("--skip-head", action="store_true", help="Skip final norm + lm_head export")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing HMONNX files instead of resuming")
    parser.set_defaults(repair_gptq_expert_onnx=True)
    parser.add_argument(
        "--repair-gptq-expert-onnx",
        dest="repair_gptq_expert_onnx",
        action="store_true",
        help="Repair bad GPTQ expert ONNX qweight/mode after export validation",
    )
    parser.add_argument(
        "--no-repair-gptq-expert-onnx",
        dest="repair_gptq_expert_onnx",
        action="store_false",
        help="Fail instead of repairing bad GPTQ expert ONNX qweight/mode after export validation",
    )
    parser.add_argument("--debug", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_argparser().parse_args())
