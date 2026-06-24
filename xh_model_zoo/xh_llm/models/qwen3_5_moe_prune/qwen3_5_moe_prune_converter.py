import json
import re
import types
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from xhquant.api import Config, get_root_logger

from ..builder import wrap_llm_model
from ..qwen3_5_moe import Qwen3_5MoeConverterXH2a
from ..qwen3_5_moe.qwen3_5_moe_converter import (
    _flatten_cache_outputs,
    _get_text_config,
    _load_dflash_target_layer_ids,
)
from .qwen3_5_moe_prune_convert_config import Qwen3_5MoePruneConvertConfig


class Qwen3_5MoePruneConverterXH2a(Qwen3_5MoeConverterXH2a):
    def __init__(self, config: Qwen3_5MoePruneConvertConfig):
        super().__init__(config)
        self.config: Qwen3_5MoePruneConvertConfig = config

    @staticmethod
    def _force_float16(module):
        module.to(dtype=torch.float16)
        return module

    def load_hf_model(self, hf_model_dir: str, **kwargs):
        kwargs.setdefault("torch_dtype", torch.float16)
        native_model = super().load_hf_model(hf_model_dir, **kwargs)
        get_root_logger().info("Casting Qwen3.5-MoE prune model to float16 for XH2a export")
        return self._force_float16(native_model)

    def get_hf_model(self, hf_model_dir: str, **kwargs):
        native_model = super().get_hf_model(hf_model_dir, **kwargs)
        get_root_logger().info("Normalizing Qwen3.5-MoE prune HF model floating tensors to float16")
        return self._force_float16(native_model)

    @staticmethod
    def _load_s_scalars(s_scalar_path: str) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        s_scalar_file = Path(s_scalar_path)
        if s_scalar_file.suffix.lower() == ".json":
            with s_scalar_file.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                raise TypeError(f"JSON s_scalar payload must be a layer mapping, got {type(payload)}")
            return {str(key): torch.as_tensor(value, dtype=torch.float32) for key, value in payload.items()}
        return torch.load(s_scalar_file, map_location="cpu", weights_only=True)

    def _prepare_wrap_model(self, native_model):
        from ._moe_model_prune import register_wrap_modules as qwen3_5_moe_prune_register_wrap_modules

        qwen3_5_moe_prune_register_wrap_modules()
        text_config = _get_text_config(native_model)
        max_layers = getattr(self.config, "max_layers", None)
        if max_layers is not None:
            max_layers = int(max_layers)
            if max_layers <= 0:
                raise ValueError(f"max_layers must be positive, got {max_layers}")
            if max_layers > text_config.num_hidden_layers:
                raise ValueError(
                    f"max_layers={max_layers} exceeds model num_hidden_layers={text_config.num_hidden_layers}"
                )
        spec_decode_mode = getattr(self.config, "spec_decode_mode", None)
        output_post_norm_hidden = spec_decode_mode == "mtp"
        output_hidden_state_indices = None
        if spec_decode_mode == "dflash":
            dflash_model_dir = getattr(self.config, "dflash_model_dir", None)
            if not dflash_model_dir:
                raise ValueError("dflash_model_dir is required when spec_decode_mode='dflash'")
            output_hidden_state_indices = _load_dflash_target_layer_ids(dflash_model_dir)
        wrap_cfg = Config(
            dict(
                batch_size=self.config.batch_size,
                max_sequence_length=self.config.context_length,
                input_sequence_length=self.config.input_sequence_length,
                use_cache=True,
                max_layers=max_layers,
                num_logits_to_keep=self.config.num_logits_to_keep,
                linear_attention_mode=self.config.linear_attention_mode,
                linear_chunk_size=self.config.linear_chunk_size,
                enable_rope=self.config.enable_rope,
                max_pe_length=getattr(self.config, "max_pe_length", 262144),
                support_long_context_over_fp16_limit=getattr(
                    self.config, "support_long_context_over_fp16_limit", True
                ),
                alpha_scaling_layers=list(self.config.alpha_scaling_layers),
                chunk_inverse_alpha=self.config.chunk_inverse_alpha,
                output_hidden_state_indices=output_hidden_state_indices,
                output_post_norm_hidden=output_post_norm_hidden,
                split_conv_cache=self.config.split_conv_cache,
                use_manual_depthwise_conv1d=self.config.use_manual_depthwise_conv1d,
                fuse_gdr_ops=getattr(self.config, "fuse_gdr_ops", False),
                kv_cache=dict(
                    cache_axis=2,
                ),
            )
        )
        wraped_model = wrap_llm_model(native_model, wrap_cfg)
        self._apply_prune_config(wraped_model)
        self._force_float16(wraped_model)
        if not hasattr(wraped_model, "_qwen3_5_moe_original_forward"):
            wraped_model._qwen3_5_moe_original_forward = wraped_model.forward
            wraped_model.forward = types.MethodType(_flatten_cache_outputs, wraped_model)
        return wraped_model, wrap_cfg

    def _apply_prune_config(self, wraped_model):
        logger = get_root_logger()
        threshold = self.config.threshold
        s_scalars = None
        if self.config.s_scalar_path is not None:
            s_scalars = self._load_s_scalars(self.config.s_scalar_path)

        prune_modules_found = 0
        for name, module in wraped_model.named_modules():
            if hasattr(module, "prune_router") and hasattr(module, "s_scalar") and hasattr(module, "moeblock_prune"):
                module.threshold = threshold
                prune_modules_found += 1
                if s_scalars is None:
                    continue

                layer_match = re.search(r"layers\.(\d+)", name)
                layer_idx = int(layer_match.group(1)) if layer_match else prune_modules_found - 1
                selected_s_scalar = None
                if isinstance(s_scalars, dict):
                    for key in (name, layer_idx, str(layer_idx), f"layer_{layer_idx}", f"layers.{layer_idx}"):
                        if key in s_scalars:
                            selected_s_scalar = s_scalars[key]
                            break
                    if selected_s_scalar is None:
                        logger.warning(
                            f"s_scalar for prune block '{name}' (layer {layer_idx}) not found in "
                            f"{self.config.s_scalar_path}, using ones()"
                        )
                        continue
                elif isinstance(s_scalars, torch.Tensor):
                    if s_scalars.dim() == 1:
                        selected_s_scalar = s_scalars
                    elif s_scalars.dim() >= 2 and s_scalars.shape[0] > layer_idx:
                        selected_s_scalar = s_scalars[layer_idx]
                    else:
                        raise ValueError(
                            f"s_scalar tensor must have shape [num_experts] or [num_layers, num_experts], "
                            f"got {tuple(s_scalars.shape)}"
                        )
                else:
                    raise TypeError(f"Unsupported s_scalar payload type: {type(s_scalars)}")

                if selected_s_scalar.numel() != module.s_scalar.numel():
                    raise ValueError(
                        f"s_scalar for prune block '{name}' has {selected_s_scalar.numel()} entries, "
                        f"expected {module.s_scalar.numel()}"
                    )
                module.s_scalar = selected_s_scalar.to(module.s_scalar.device, dtype=module.s_scalar.dtype)

        logger.info(f"Set threshold={threshold} on {prune_modules_found} Qwen3.5 prune MoE blocks")
        if prune_modules_found == 0:
            raise RuntimeError("No Qwen3.5 MoE prune blocks found; dynamic pruning wrapper registration failed")
