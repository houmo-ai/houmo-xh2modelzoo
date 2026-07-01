"""Gemma4 Series MTP assistant draft export model.

This module is intentionally owned by ``gemma4_series``.  Legacy gemma4,
gemma4e, and gemma4_moe MTP helpers are read-only references only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm, Gemma4TextModel

import xhquant.nn as xhnn
from xh_model_zoo.xh_llm.models.base_model import BaseModel
from xhquant.api import ConfigDict, get_xhquant_logger
from xhquant.nn import LLMCacheV2, MaskedAdd, SoftmaxPlus
from xhquant.nn import RMSNorm as XHRMSNorm


DTYPE_MAP = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def resolve_torch_dtype(dtype_name: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype_name, torch.dtype):
        return dtype_name
    key = str(dtype_name).strip().lower()
    if key not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return DTYPE_MAP[key]


def aligned(size: int, align: int) -> int:
    return ((size + align - 1) // align) * align


def load_json_file(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _is_present(value: Any) -> bool:
    return value is not None and value != ""


def _parse_positive_int(source: str, value: Any) -> int | None:
    if not _is_present(value):
        return None
    if isinstance(value, bool):
        raise ValueError(f"Gemma4 Series MTP {source} must be a positive integer: {value!r}")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"Gemma4 Series MTP {source} must be a positive integer: {value!r}")
        parsed = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped.isdecimal():
            raise ValueError(f"Gemma4 Series MTP {source} must be a positive integer: {value!r}")
        parsed = int(stripped)
    else:
        raise ValueError(f"Gemma4 Series MTP {source} must be a positive integer: {value!r}")
    if parsed <= 0:
        raise ValueError(f"Gemma4 Series MTP {source} must be positive, got {parsed}")
    return parsed


def _nested_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_hf_max_pe_length(target_config: dict[str, Any]) -> tuple[int | None, str | None]:
    target_text_config = _nested_dict(target_config.get("text_config"))
    candidates: list[tuple[str, Any]] = [
        ("target_config.max_pe_length", target_config.get("max_pe_length")),
        (
            "target_config.text_config.max_position_embeddings",
            target_text_config.get("max_position_embeddings"),
        ),
        ("target_config.max_position_embeddings", target_config.get("max_position_embeddings")),
    ]
    for source, raw_value in candidates:
        value = _parse_positive_int(source, raw_value)
        if value is not None:
            return value, source
    return None, None


def _emit_max_pe_length_override_warning(yaml_value: int, hf_value: int, hf_source: str) -> None:
    message = (
        "Gemma4 Series max_pe_length YAML override differs from HF config: "
        f"model_config.max_pe_length={yaml_value}, {hf_source}={hf_value}; using YAML override."
    )
    logging.getLogger(__name__).warning(message)
    try:
        get_xhquant_logger().warning(message)
    except Exception:  # pragma: no cover - logger setup is environment-specific
        pass


def _format_max_pe_length_result(
    value: int,
    source: str,
    *,
    return_source: bool,
    return_metadata: bool,
    hf_value: int | None = None,
    hf_source: str | None = None,
) -> int | tuple[int, str] | dict[str, Any]:
    if return_metadata:
        metadata: dict[str, Any] = {"value": value, "source": source}
        if hf_value is not None:
            metadata["hf_value"] = hf_value
        if hf_source is not None:
            metadata["hf_source"] = hf_source
        return metadata
    return (value, source) if return_source else value


def resolve_target_max_pe_length(
    target_model_dir: str | Path,
    model_cfg: dict[str, Any] | ConfigDict | None = None,
    *,
    max_pe_length_explicit: bool = False,
    return_source: bool = False,
    return_metadata: bool = False,
) -> int | tuple[int, str] | dict[str, Any]:
    """Resolve target/base Gemma4 MPE for MTP draft RoPE tables.

    ``BaseLLMModelConfig.max_pe_length`` defaults to 32768 for legacy/internal
    reasons.  Gemma4 Series must only treat ``model_cfg.max_pe_length`` as a
    YAML/user override when ``max_pe_length_explicit`` is true; otherwise the HF
    config owns the MPE.
    """

    target_config_path = Path(target_model_dir) / "config.json"
    target_config = load_json_file(target_config_path)
    hf_value, hf_source = _first_hf_max_pe_length(target_config)

    model_cfg_dict = dict(model_cfg or {})
    yaml_source = "model_config.max_pe_length"
    if max_pe_length_explicit:
        yaml_value = _parse_positive_int(yaml_source, model_cfg_dict.get("max_pe_length"))
        if yaml_value is None:
            raise ValueError("Gemma4 Series max_pe_length was marked explicit but model_config.max_pe_length is empty")
        if hf_value is not None and hf_value != yaml_value:
            _emit_max_pe_length_override_warning(yaml_value, hf_value, str(hf_source))
        return _format_max_pe_length_result(
            yaml_value,
            yaml_source,
            return_source=return_source,
            return_metadata=return_metadata,
            hf_value=hf_value,
            hf_source=hf_source,
        )

    if hf_value is not None and hf_source is not None:
        return _format_max_pe_length_result(
            hf_value,
            hf_source,
            return_source=return_source,
            return_metadata=return_metadata,
            hf_value=hf_value,
            hf_source=hf_source,
        )

    # Intentionally do not fall back to model_cfg.max_pe_length here.  When
    # max_pe_length_explicit is false, that field may be BaseLLMModelConfig's
    # legacy default (32768), not user YAML intent.
    raise ValueError(
        "Gemma4 Series MTP requires target max_pe_length from HF config or explicit YAML override. "
        f"Please check target model config: {target_config_path}"
    )


def _compute_rotary_cache(
    inv_freq: torch.Tensor,
    attention_scaling: float,
    max_seq_len: int,
    *,
    partial_rotary_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(max_seq_len, device=inv_freq.device, dtype=torch.float32).view(max_seq_len, 1)
    freqs = positions * inv_freq.float().view(1, -1)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling
    if partial_rotary_factor < 1.0:
        head_dim = int(cos.shape[-1])
        rotary_dim = int(head_dim * partial_rotary_factor)
        cos[:, head_dim // 2 : head_dim // 2 + rotary_dim] = 1.0
        sin[:, head_dim // 2 : head_dim // 2 + rotary_dim] = 0.0
    return cos.to(dtype=inv_freq.dtype), sin.to(dtype=inv_freq.dtype)


def _convert_gemma4_rmsnorm(hf_norm: nn.Module) -> XHRMSNorm:
    if isinstance(hf_norm, XHRMSNorm):
        return hf_norm
    if not isinstance(hf_norm, Gemma4RMSNorm):
        raise TypeError(f"Expected Gemma4RMSNorm, got {type(hf_norm).__name__}")
    hidden_size = int(hf_norm.weight.shape[0])
    eps = float(getattr(hf_norm, "eps", 1e-6))
    new_norm = XHRMSNorm(hidden_size, eps)
    with torch.no_grad():
        new_norm.weight.data.copy_(hf_norm.weight.data.detach().to(new_norm.weight.dtype))
    new_norm.weight.requires_grad_(False)
    return new_norm


class Gemma4AssistantMaskedEmbedder(nn.Module):
    """Ordered-embedding lm head used by E2B/E4B assistants.

    This mirrors Transformers ``Gemma4AssistantMaskedEmbedder`` but avoids
    Python scalar ``.item()`` so export keeps the mask value in the graph.
    """

    def __init__(self, assistant_config: dict[str, Any], text_config: Gemma4TextConfig):
        super().__init__()
        self.centroid_intermediate_top_k = int(assistant_config["centroid_intermediate_top_k"])
        self.hidden_size = int(text_config.hidden_size)
        self.num_centroids = int(assistant_config["num_centroids"])
        self.vocab_size = int(text_config.vocab_size)
        self.vocab_size_per_centroid = self.vocab_size // self.num_centroids
        self.centroids = nn.Linear(self.hidden_size, self.num_centroids, bias=False)
        self.register_buffer("token_ordering", torch.empty(self.vocab_size, dtype=torch.long))

    def forward(self, hidden_states: torch.Tensor, lm_head_weight: torch.Tensor) -> torch.Tensor:
        batch, seq_len = hidden_states.shape[:2]
        centroid_logits = self.centroids(hidden_states)
        _, top_k_indices = torch.topk(centroid_logits, k=self.centroid_intermediate_top_k, dim=-1)
        top_k_indices = top_k_indices.to(torch.long)

        canonical_positions_per_cluster = self.token_ordering.long().view(
            self.num_centroids,
            self.vocab_size_per_centroid,
        )
        selected_canonical = canonical_positions_per_cluster[top_k_indices]
        selected_flat = selected_canonical.reshape(-1)
        selected_embeddings = lm_head_weight[selected_flat].view(
            batch,
            seq_len,
            self.centroid_intermediate_top_k * self.vocab_size_per_centroid,
            self.hidden_size,
        )
        selected_logits = (hidden_states.unsqueeze(-2) @ selected_embeddings.transpose(-1, -2)).squeeze(-2)
        # HMONNX graph runtime does not support ReduceMin.  A fixed fp16
        # negative sentinel is sufficient for the ordered-embedding masked
        # logits: unselected vocab entries only need to be impossible under
        # greedy/top-k selection, not data-dependent.
        output = hidden_states.new_full((batch, seq_len, self.vocab_size), torch.finfo(torch.float16).min)
        scatter_idx = selected_canonical.view(batch, seq_len, -1)
        return output.scatter(dim=-1, index=scatter_idx, src=selected_logits)


class Gemma4AssistantSelfAttention(nn.Module):
    def __init__(self, hf_attn: nn.Module, layer_type: str, cache_axis: int = 2):
        super().__init__()
        self.layer_type = layer_type
        self.config = hf_attn.config
        self.q_proj = hf_attn.q_proj
        self.q_norm = hf_attn.q_norm
        self.o_proj = hf_attn.o_proj
        global_head_dim = getattr(self.config, "global_head_dim", None) or self.config.head_dim
        default_head_dim = global_head_dim if layer_type == "full_attention" else self.config.head_dim
        self.head_dim = int(getattr(hf_attn, "head_dim", default_head_dim))
        self.num_attention_heads = int(self.config.num_attention_heads)
        if layer_type == "full_attention":
            self.num_key_value_heads = int(
                getattr(self.config, "num_global_key_value_heads", None) or self.config.num_key_value_heads
            )
        else:
            self.num_key_value_heads = int(self.config.num_key_value_heads)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.rope = xhnn.Rope()
        self.k_old_cache = LLMCacheV2(axis=cache_axis, only_handle_old_cache=True)
        self.v_old_cache = LLMCacheV2(axis=cache_axis, only_handle_old_cache=True)
        self.k_repeat_interleave = xhnn.RepeatInterleave()
        self.v_repeat_interleave = xhnn.RepeatInterleave()
        self.qk_matmul = xhnn.MatMul()
        self.pv_matmul = xhnn.MatMul()
        self.masked_add = MaskedAdd()
        self.softmax = SoftmaxPlus(dim=-1)
        self.attn_compute_cast = xhnn.Cast(torch.float16).to(dtype=torch.float16)
        self.attn_output_cast = xhnn.Cast(self.o_proj.weight.dtype).to(dtype=self.o_proj.weight.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        past_seq_length: torch.Tensor,
        current_input_length: torch.Tensor,
        shared_key_cache: torch.Tensor,
        shared_value_cache: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_length = hidden_states.shape[:2]
        cos, sin = position_embeddings
        query_states = self.q_proj(hidden_states).reshape(
            batch_size,
            seq_length,
            self.num_attention_heads,
            self.head_dim,
        )
        query_states = self.q_norm(query_states).transpose(1, 2)
        query_states = self.attn_compute_cast(query_states)
        query_states = self.attn_compute_cast(self.rope(query_states, cos, sin))

        # MTP draft consumes target KV as read-only external inputs.  Keep this
        # as an explicit LLMCache only-handle-old-cache op in the export graph so HMONNX
        # can identify KV reuse without mutating the target cache.
        key_states = self.k_old_cache(
            shared_key_cache,
            past_seq_length,
            current_input_length,
            shared_key_cache,
        )
        value_states = self.v_old_cache(
            shared_value_cache,
            past_seq_length,
            current_input_length,
            shared_value_cache,
        )

        key_states = self.k_repeat_interleave(key_states.transpose(2, 3), self.num_key_value_groups, 1)
        value_states = self.v_repeat_interleave(value_states, self.num_key_value_groups, 1)
        attn_weights = self.qk_matmul(query_states, key_states)
        attn_weights = self.masked_add(attn_weights, attention_mask)
        attn_weights = self.softmax(attn_weights).to(query_states.dtype)
        attn_output = self.pv_matmul(attn_weights, value_states).transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.attn_output_cast(attn_output)
        return self.o_proj(attn_output)


class Gemma4AssistantDecoderLayer(nn.Module):
    def __init__(self, hf_layer: nn.Module, layer_type: str, cache_axis: int = 2):
        super().__init__()
        self.layer_type = layer_type
        self.self_attn = Gemma4AssistantSelfAttention(hf_layer.self_attn, layer_type, cache_axis=cache_axis)
        self.input_layernorm = hf_layer.input_layernorm
        self.post_attention_layernorm = hf_layer.post_attention_layernorm
        self.pre_feedforward_layernorm = hf_layer.pre_feedforward_layernorm
        self.post_feedforward_layernorm = hf_layer.post_feedforward_layernorm
        self.mlp = hf_layer.mlp
        self.register_buffer("layer_scalar", hf_layer.layer_scalar.detach().clone())

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        past_seq_length: torch.Tensor,
        current_input_length: torch.Tensor,
        shared_key_cache: torch.Tensor,
        shared_value_cache: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            shared_key_cache=shared_key_cache,
            shared_value_cache=shared_value_cache,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states * self.layer_scalar.to(dtype=hidden_states.dtype)


class Gemma4AssistantBackbone(nn.Module):
    def __init__(
        self,
        text_model: Gemma4TextModel,
        max_position_embeddings: int,
        input_sequence_length: int,
        cache_axis: int = 2,
    ):
        super().__init__()
        self.embed_tokens = text_model.embed_tokens
        self.layers = nn.ModuleList(
            [
                Gemma4AssistantDecoderLayer(layer, text_model.config.layer_types[index], cache_axis=cache_axis)
                for index, layer in enumerate(text_model.layers[: text_model.config.num_hidden_layers])
            ]
        )
        self.norm = text_model.norm
        self.layer_types = list(text_model.config.layer_types)
        self.input_sequence_length = int(input_sequence_length)

        rotary_emb = text_model.rotary_emb
        for layer_type in ("sliding_attention", "full_attention"):
            inv_freq = getattr(rotary_emb, f"{layer_type}_inv_freq", None)
            attention_scaling = getattr(rotary_emb, f"{layer_type}_attention_scaling", None)
            if inv_freq is None or attention_scaling is None:
                continue
            rope_params = text_model.config.rope_parameters.get(layer_type, {})
            cos, sin = _compute_rotary_cache(
                inv_freq,
                attention_scaling,
                max_position_embeddings,
                partial_rotary_factor=float(rope_params.get("partial_rotary_factor", 1.0)),
            )
            self.register_buffer(f"{layer_type}_cos_cached", cos.unsqueeze(0).unsqueeze(0), persistent=False)
            self.register_buffer(f"{layer_type}_sin_cached", sin.unsqueeze(0).unsqueeze(0), persistent=False)

        self.sliding_cos_slice = xhnn.DynamicSlice([self.input_sequence_length], [2], [1])
        self.sliding_sin_slice = xhnn.DynamicSlice([self.input_sequence_length], [2], [1])
        self.full_cos_slice = xhnn.DynamicSlice([self.input_sequence_length], [2], [1])
        self.full_sin_slice = xhnn.DynamicSlice([self.input_sequence_length], [2], [1])

    def _get_position_embeddings(
        self,
        past_seq_length: torch.Tensor,
        layer_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_cache = getattr(self, f"{layer_type}_cos_cached")
        sin_cache = getattr(self, f"{layer_type}_sin_cached")
        if layer_type == "full_attention":
            return self.full_cos_slice(cos_cache, past_seq_length), self.full_sin_slice(sin_cache, past_seq_length)
        return self.sliding_cos_slice(cos_cache, past_seq_length), self.sliding_sin_slice(sin_cache, past_seq_length)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_seq_length: torch.Tensor,
        current_input_length: torch.Tensor,
        sliding_attention_mask: torch.Tensor,
        full_attention_mask: torch.Tensor,
        shared_key_cache_sliding: torch.Tensor,
        shared_value_cache_sliding: torch.Tensor,
        shared_key_cache_full: torch.Tensor,
        shared_value_cache_full: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        full_pos = self._get_position_embeddings(past_seq_length, "full_attention")
        sliding_pos = self._get_position_embeddings(past_seq_length, "sliding_attention")
        for layer in self.layers:
            if layer.layer_type == "full_attention":
                attention_mask = full_attention_mask
                position_embeddings = full_pos
                shared_key_cache = shared_key_cache_full
                shared_value_cache = shared_value_cache_full
            else:
                attention_mask = sliding_attention_mask
                position_embeddings = sliding_pos
                shared_key_cache = shared_key_cache_sliding
                shared_value_cache = shared_value_cache_sliding
            hidden_states = layer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                shared_key_cache=shared_key_cache,
                shared_value_cache=shared_value_cache,
            )
        return self.norm(hidden_states)


class Gemma4AssistantDraftModule(nn.Module):
    def __init__(
        self,
        assistant_model_dir: str,
        target_model_dir: str,
        max_position_embeddings: int,
        input_sequence_length: int = 1,
        cache_axis: int = 2,
    ):
        super().__init__()
        assistant_model_dir = str(Path(assistant_model_dir).resolve())
        target_model_dir = str(Path(target_model_dir).resolve())
        self.assistant_model_dir = assistant_model_dir
        self.target_model_dir = target_model_dir
        self.assistant_config_dict = load_json_file(Path(assistant_model_dir) / "config.json")
        self.target_config_dict = load_json_file(Path(target_model_dir) / "config.json")

        text_config = Gemma4TextConfig(**self.assistant_config_dict["text_config"])
        self.text_config = text_config
        self.backbone_hidden_size = int(self.assistant_config_dict["backbone_hidden_size"])
        self.use_ordered_embeddings = bool(self.assistant_config_dict.get("use_ordered_embeddings", False))

        text_model_config = Gemma4TextConfig(**self.assistant_config_dict["text_config"])
        # Build local K/V parameters so the local Transformers version can
        # instantiate the model; the draft attention ignores those K/V weights
        # and consumes target shared KV by layer_type instead.
        text_model_config.num_kv_shared_layers = 0
        text_model = Gemma4TextModel(text_model_config)
        self.model = Gemma4AssistantBackbone(
            text_model,
            max_position_embeddings=max_position_embeddings,
            input_sequence_length=input_sequence_length,
            cache_axis=cache_axis,
        )
        self.pre_projection = nn.Linear(2 * self.backbone_hidden_size, text_config.hidden_size, bias=False)
        self.post_projection = nn.Linear(text_config.hidden_size, self.backbone_hidden_size, bias=False)
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight
        self.masked_embedding = (
            Gemma4AssistantMaskedEmbedder(self.assistant_config_dict, text_config)
            if self.use_ordered_embeddings
            else None
        )

        self._load_weights(Path(assistant_model_dir) / "model.safetensors")
        self._convert_norms_for_export()

    def _convert_norms_for_export(self) -> None:
        for layer in self.model.layers:
            attn = layer.self_attn
            if isinstance(attn.q_norm, Gemma4RMSNorm):
                attn.q_norm = _convert_gemma4_rmsnorm(attn.q_norm)
            for attr in (
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ):
                module = getattr(layer, attr, None)
                if isinstance(module, Gemma4RMSNorm):
                    setattr(layer, attr, _convert_gemma4_rmsnorm(module))
        if isinstance(self.model.norm, Gemma4RMSNorm):
            self.model.norm = _convert_gemma4_rmsnorm(self.model.norm)
        self.to(dtype=self.lm_head.weight.dtype)

    def _load_weights(self, weight_path: Path) -> None:
        state_dict = load_file(str(weight_path), device="cpu")
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        allowed_missing_substrings = (
            ".self_attn.k_proj.",
            ".self_attn.v_proj.",
            ".self_attn.k_norm.",
            ".self_attn.v_norm.",
        )
        allowed_missing_exact = {"lm_head.weight"}
        bad_missing = [
            key for key in missing_keys
            if key not in allowed_missing_exact and not any(part in key for part in allowed_missing_substrings)
        ]
        if bad_missing or unexpected_keys:
            raise RuntimeError(
                "Unexpected Gemma4 assistant checkpoint mismatch: "
                f"missing={bad_missing}, unexpected={unexpected_keys}"
            )
        get_xhquant_logger().info(f"Loaded Gemma4 Series assistant weights from {weight_path}")

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_seq_length: torch.Tensor,
        current_input_length: torch.Tensor,
        sliding_attention_mask: torch.Tensor,
        full_attention_mask: torch.Tensor,
        shared_key_cache_sliding: torch.Tensor,
        shared_value_cache_sliding: torch.Tensor,
        shared_key_cache_full: torch.Tensor,
        shared_value_cache_full: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.pre_projection(inputs_embeds)
        hidden_states = self.model(
            inputs_embeds=hidden_states,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            sliding_attention_mask=sliding_attention_mask,
            full_attention_mask=full_attention_mask,
            shared_key_cache_sliding=shared_key_cache_sliding,
            shared_value_cache_sliding=shared_value_cache_sliding,
            shared_key_cache_full=shared_key_cache_full,
            shared_value_cache_full=shared_value_cache_full,
        )
        if self.masked_embedding is not None:
            logits = self.masked_embedding(hidden_states, self.lm_head.weight)
        else:
            logits = self.lm_head(hidden_states)
        assistant_hidden_state = self.post_projection(hidden_states)
        return logits, assistant_hidden_state


class XHGemma4SeriesAssistantDraftModel(BaseModel):
    """BaseModel wrapper for exporting the Gemma4 Series MTP draft graph."""

    def __init__(
        self,
        assistant_model_dir: str,
        target_model_dir: str,
        wrap_cfg: ConfigDict,
        quant_config: ConfigDict,
        frontend_type: str = "TorchFX",
        export_cfg: ConfigDict | None = None,
    ):
        self.assistant_model_dir = str(Path(assistant_model_dir).resolve())
        self.target_model_dir = str(Path(target_model_dir).resolve())
        export_cfg = export_cfg or ConfigDict(
            input_names=[
                "inputs_embeds",
                "past_seq_length",
                "current_input_length",
                "sliding_attention_mask",
                "full_attention_mask",
                "shared_key_cache_sliding",
                "shared_value_cache_sliding",
                "shared_key_cache_full",
                "shared_value_cache_full",
            ],
            output_names=["logits", "assistant_hidden_state"],
        )
        super().__init__(
            hf_model=self.assistant_model_dir,
            wrap_cfg=wrap_cfg,
            quant_config=quant_config,
            frontend_type=frontend_type,
            allow_quant=True,
            export_cfg=export_cfg,
        )

    def get_tokenizer(self, **kwargs):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.target_model_dir, trust_remote_code=True, **kwargs)

    def init_wrap_model(self, hf_model=None):
        del hf_model
        max_position_embeddings = _parse_positive_int(
            "wrap_cfg.target_max_pe_length",
            self.wrap_cfg.get("target_max_pe_length"),
        )
        if max_position_embeddings is None:
            max_position_embeddings = resolve_target_max_pe_length(
                self.target_model_dir,
                dict(self.wrap_cfg.get("model_config") or {}),
            )
        self._wrap_model = Gemma4AssistantDraftModule(
            assistant_model_dir=self.assistant_model_dir,
            target_model_dir=self.target_model_dir,
            max_position_embeddings=int(max_position_embeddings),
            input_sequence_length=int(self.wrap_cfg.get("input_sequence_length", 1)),
            cache_axis=int(self.wrap_cfg.get("cache_axis", 2)),
        )
        self._wrap_model = self._wrap_model.to(dtype=resolve_torch_dtype(self.wrap_cfg.get("dtype", "float16")))
        self._wrap_model.eval()
        return self._wrap_model

    def prepare_inputs(self, data=None):
        if self._wrap_model is None:
            self.init_wrap_model()
        target_text_cfg = self._wrap_model.target_config_dict["text_config"]
        input_sequence_length = int(self.wrap_cfg.get("input_sequence_length", 1))
        context_length = int(self.wrap_cfg.get("context_length", 2048))
        hidden_size = int(self._wrap_model.backbone_hidden_size)
        dtype = resolve_torch_dtype(self.wrap_cfg.get("dtype", "float16"))
        default_sliding_cache_length = aligned(
            int(target_text_cfg.get("sliding_window", 1024)) + input_sequence_length - 1,
            16,
        )
        shared_sliding_cache_length = int(
            self.wrap_cfg.get("shared_sliding_cache_length", default_sliding_cache_length)
        )
        shared_full_cache_length = int(self.wrap_cfg.get("shared_full_cache_length", context_length))
        full_heads = int(target_text_cfg.get("num_global_key_value_heads") or target_text_cfg["num_key_value_heads"])
        full_head_dim = int(target_text_cfg.get("global_head_dim") or target_text_cfg["head_dim"])

        if data is None:
            data = dict(
                inputs_embeds=torch.zeros((1, input_sequence_length, hidden_size * 2), dtype=dtype),
                past_seq_length=torch.zeros((1,), dtype=torch.int32),
                current_input_length=torch.full((1,), input_sequence_length, dtype=torch.int32),
                sliding_attention_mask=torch.zeros(
                    (1, 1, input_sequence_length, shared_sliding_cache_length),
                    dtype=dtype,
                ),
                full_attention_mask=torch.zeros(
                    (1, 1, input_sequence_length, shared_full_cache_length),
                    dtype=dtype,
                ),
                shared_key_cache_sliding=torch.zeros(
                    (
                        1,
                        int(target_text_cfg["num_key_value_heads"]),
                        shared_sliding_cache_length,
                        int(target_text_cfg["head_dim"]),
                    ),
                    dtype=dtype,
                ),
                shared_value_cache_sliding=torch.zeros(
                    (
                        1,
                        int(target_text_cfg["num_key_value_heads"]),
                        shared_sliding_cache_length,
                        int(target_text_cfg["head_dim"]),
                    ),
                    dtype=dtype,
                ),
                shared_key_cache_full=torch.zeros(
                    (1, full_heads, shared_full_cache_length, full_head_dim),
                    dtype=dtype,
                ),
                shared_value_cache_full=torch.zeros(
                    (1, full_heads, shared_full_cache_length, full_head_dim),
                    dtype=dtype,
                ),
            )

        return (
            data["inputs_embeds"],
            data["past_seq_length"],
            data["current_input_length"],
            data["sliding_attention_mask"],
            data["full_attention_mask"],
            data["shared_key_cache_sliding"],
            data["shared_value_cache_sliding"],
            data["shared_key_cache_full"],
            data["shared_value_cache_full"],
        )

    @property
    def need_quant(self):
        return True

    def get_empty_hf_model(self, device_map="cpu", **kwargs):
        del device_map, kwargs
        return None

    def get_hf_model(self, device_map="cpu", **kwargs):
        del device_map, kwargs
        return None


# Backward-compatible short alias under the new series package only.
XHGemma4AssistantDraftModel = XHGemma4SeriesAssistantDraftModel


__all__ = [
    "Gemma4AssistantDraftModule",
    "XHGemma4SeriesAssistantDraftModel",
    "XHGemma4AssistantDraftModel",
    "aligned",
    "resolve_target_max_pe_length",
    "resolve_torch_dtype",
]
