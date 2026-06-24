# Copyright 2025 HOUMO AI
#
# File: qwen3_5_moe_converter.py
# Description:
#   Qwen3.5-MoE Converter implementation.
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

import copy
import json
import shutil
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import onnx
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
)

from xhquant.api import CacheTensor

from ..base_converter import BaseConverter, HFTransfromersConverter
from ..builder import wrap_llm_model
from .qwen3_5_moe_convert_config import Qwen3_5MoeConvertConfig


from xhquant.api import (  # type: ignore # isort:skip
    Config,
    DeviceType,
    ConfigDict,
    convert_fx_model_to_quanted_model,
    convert_quanted_model_to_hmonnx,
    create_quant_config,
    get_root_logger,
)


def _extract_quant_method(config: AutoConfig) -> Optional[str]:
    quantization_config = getattr(config, "quantization_config", None)
    quant_method = getattr(quantization_config, "quant_method", None)
    if isinstance(quantization_config, dict):
        quant_method = quantization_config.get("quant_method", quant_method)
    return None if quant_method is None else str(quant_method).lower()




def _linear_split_conv_dims(linear_attn) -> tuple[int, int, int]:
    def _module_channels(module_name: str, fallback: int) -> int:
        module = getattr(linear_attn, module_name, None)
        if module is not None:
            if hasattr(module, "in_channels"):
                return int(module.in_channels)
            if hasattr(module, "out_features"):
                return int(module.out_features)
        return int(fallback)

    qk_fallback = int(linear_attn.head_k_dim) * int(linear_attn.num_v_heads)
    v_fallback = int(linear_attn.head_v_dim) * int(linear_attn.num_v_heads)
    return (
        _module_channels("conv1d_q", qk_fallback),
        _module_channels("conv1d_k", qk_fallback),
        _module_channels("conv1d_v", v_fallback),
    )


def _split_batch_tensor(tensor: torch.Tensor, batch_size: int) -> List[torch.Tensor]:
    return [tensor[i : i + 1] for i in range(batch_size)]


def _patch_hmonnx_standard_add_ops(onnx_file: Path) -> int:
    """Move stray standard ONNX Add nodes into the XH2a domain.

    HMONNX registers Add as ``XH2a::Add``.  The export path already emits
    residual Add nodes in that domain, but the gather-index tail can leave one
    default-domain Add.  Normalize it after export so HMONNX session creation
    succeeds while preserving external-data tensors.
    """

    model = onnx.load(str(onnx_file), load_external_data=False)
    changed = 0
    for node in model.graph.node:
        if node.op_type == "Add" and (node.domain or "") == "":
            node.domain = "ai.houmo.xh2a"
            changed += 1
    if changed:
        onnx.save_model(model, str(onnx_file))
    return changed


def _flatten_cache_outputs(self: nn.Module, *args, **kwargs):
    result = self._qwen3_5_moe_original_forward(*args, **kwargs)
    batch_size = int(getattr(getattr(self, "cfg", None), "get", lambda *_: 1)("batch_size", 1))

    def _append_batch(outputs: List[torch.Tensor], value):
        if isinstance(value, (list, tuple)):
            outputs.extend(value)
            return
        if batch_size > 1:
            # During FX/export, values are Proxy-like objects rather than
            # torch.Tensor instances.  Split by the fixed exported batch size
            # unconditionally so graph outputs remain single-batch.
            outputs.extend(value[batch_idx : batch_idx + 1] for batch_idx in range(batch_size))
        else:
            outputs.append(value)

    # Handle 3-tuple (logits, conv_caches, recurrent_states) and 4-tuple (+ post_norm_hidden).
    if len(result) == 4:
        logits, conv_cache_out_list, recurrent_state_out_list, post_norm_hidden = result
        outputs: List[torch.Tensor] = []
        _append_batch(outputs, logits)
        if conv_cache_out_list is not None:
            outputs.extend(list(conv_cache_out_list))
        if recurrent_state_out_list is not None:
            outputs.extend(list(recurrent_state_out_list))
        _append_batch(outputs, post_norm_hidden)
    else:
        logits, conv_cache_out_list, recurrent_state_out_list = result
        outputs: List[torch.Tensor] = []
        _append_batch(outputs, logits)
        if conv_cache_out_list is not None:
            outputs.extend(list(conv_cache_out_list))
        if recurrent_state_out_list is not None:
            outputs.extend(list(recurrent_state_out_list))
    return tuple(outputs)


def _get_text_model(native_model):
    """Get the text model from either ForConditionalGeneration or ForCausalLM."""
    if isinstance(native_model, Qwen3_5MoeForConditionalGeneration):
        return native_model.model.language_model
    elif isinstance(native_model, Qwen3_5MoeForCausalLM):
        return native_model.model
    else:
        raise ValueError(f"Unsupported model type: {type(native_model)}")


def _get_text_config(native_model):
    """Get text_config from the model config."""
    config = native_model.config
    if hasattr(config, "text_config"):
        return config.text_config
    return config


def _load_dflash_target_layer_ids(dflash_model_dir: str) -> List[int]:
    config_path = Path(dflash_model_dir) / "config.json"
    with open(config_path, encoding="utf-8") as f:
        dflash_config = json.load(f)
    target_layer_ids = dflash_config.get("dflash_config", {}).get("target_layer_ids")
    if not target_layer_ids:
        raise ValueError(f"Failed to read dflash_config.target_layer_ids from {config_path}")
    return list(target_layer_ids)


DRAFT_BASE_QUANT_TYPE = "w8a8h1_sefp"
_LINEAR_CONV_CACHE_BRANCHES = ("q", "k", "v")


def _build_spec_draft_quant_config(head_weight_bits: int) -> ConfigDict:
    if head_weight_bits == 8:
        return ConfigDict(quant_type=DRAFT_BASE_QUANT_TYPE)
    if head_weight_bits != 4:
        raise ValueError(f"Unsupported spec draft head weight bits: {head_weight_bits}. Expected 4 or 8.")
    return ConfigDict(
        quant_type=DRAFT_BASE_QUANT_TYPE,
        nodes_cfg=dict(
            lm_head=dict(
                w_schema=dict(
                    bits=4,
                    fp_mode="ssfp",
                    hidden_bit=False,
                )
            )
        ),
    )


class Qwen3_5MoeConverterXH2a(HFTransfromersConverter):
    target_device = DeviceType.XH2a

    def __init__(self, config: Qwen3_5MoeConvertConfig):
        super().__init__()
        self.config = config
        self.hf_model_path: Optional[str] = None

    @staticmethod
    def _is_gptqmodel_checkpoint(hf_model_dir: str) -> bool:
        """Detect gptqmodel-format checkpoints by quant_method or checkpoint_format."""
        cfg_path = Path(hf_model_dir) / "config.json"
        if not cfg_path.exists():
            return False
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            qc = cfg.get("quantization_config", {})
            if not isinstance(qc, dict):
                return False
            quant_method = str(qc.get("quant_method", "")).lower()
            checkpoint_format = str(qc.get("checkpoint_format", "")).lower()
            return quant_method == "gptq" or checkpoint_format.startswith("gptq")
        except Exception:
            return False

    @staticmethod
    def _register_gptqmodel_qwen3_5_moe() -> None:
        """Register Qwen3.5-MoE with GPTQModel when the installed package does not know it.

        GPTQModel falls back to BaseQModel for unknown ``model_type`` values.  BaseQModel
        uses AutoModelForCausalLM and tries to construct the multimodal
        Qwen3_5MoeConfig as a text config, which fails because text fields such as
        ``vocab_size`` live under ``text_config``.  The checkpoint keys also use the
        multimodal prefix ``model.language_model.layers``.  This local definition combines
        the Qwen3-VL loader/prefix with the Qwen3-MoE expert module tree.
        """
        try:
            from gptqmodel.models import auto as gptq_auto  # type: ignore
            from gptqmodel.models.base import BaseQModel  # type: ignore
            from transformers import AutoModelForImageTextToText
        except ImportError:
            return

        if gptq_auto.MODEL_MAP.get("qwen3_5_moe") is not None:
            return

        class Qwen3_5MoeQModel(BaseQModel):  # noqa: N801
            loader = AutoModelForImageTextToText
            layer_modules_strict = False
            dynamic_expert_index = "num_experts"
            pre_lm_head_norm_module = "model.language_model.norm"

            module_tree = [
                "model",
                "language_model",
                "layers",
                "#",
                {
                    "input_layernorm": ("input_layernorm:!",),
                    "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
                    "linear_attn": ("in_proj_qkv:0", "in_proj_z:0", "out_proj:1"),
                    "post_attention_layernorm": ("post_attention_layernorm:!",),
                    "mlp": {
                        "gate": ("gate:!",),
                        "shared_expert_gate": ("shared_expert_gate:!",),
                        "shared_expert": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                        "experts": {
                            "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                        },
                    },
                },
            ]

        supported_models = gptq_auto.SUPPORTED_MODELS
        if isinstance(supported_models, set):
            supported_models.add("qwen3_5_moe")
        elif "qwen3_5_moe" not in supported_models:
            supported_models.append("qwen3_5_moe")
        gptq_auto.MODEL_MAP["qwen3_5_moe"] = Qwen3_5MoeQModel

    @staticmethod
    def _set_quant_weight_buffer(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
        if name in module._buffers:
            module._buffers[name] = tensor
        else:
            module.register_buffer(name, tensor, persistent=False)

    @staticmethod
    def _unpack_gptqmodel_qweight(
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        g_idx: torch.Tensor,
        expected_shape: tuple[int, int],
    ) -> torch.Tensor:
        """Unpack GPTQModel qweight to XH2a's signed per-element quant_weight layout."""
        out_features, in_features = expected_shape
        if qweight.ndim != 2 or qzeros.ndim != 2:
            raise ValueError(f"Expected 2D qweight/qzeros, got {tuple(qweight.shape)} and {tuple(qzeros.shape)}")
        if qweight.shape[1] < out_features:
            raise ValueError(f"qweight output dim {qweight.shape[1]} is smaller than expected {out_features}")
        if in_features % qweight.shape[0] != 0:
            raise ValueError(f"Cannot infer GPTQ pack factor from qweight={tuple(qweight.shape)}, expected={expected_shape}")

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
            torch.bitwise_right_shift(qweight.to(torch.int32).unsqueeze(1).expand(-1, pack_factor, -1), shifts.view(1, -1, 1)),
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

    @staticmethod
    def _dequantize_gptqmodel_weight(
        quant_weight: torch.Tensor,
        scales: torch.Tensor,
        g_idx: torch.Tensor,
        expected_shape: tuple[int, int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Reconstruct fp weights from GPTQModel per-element qweight and group scales."""
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

    @staticmethod
    def _packed_expert_weight_shape(experts: nn.Module, expert_idx: int, linear_name: str) -> tuple[int, int]:
        if linear_name in {"gate_proj", "up_proj"}:
            intermediate_dim = int(experts.intermediate_dim)
            hidden_dim = int(experts.gate_up_proj.shape[-1])
            return (intermediate_dim, hidden_dim)
        return tuple(int(dim) for dim in experts.down_proj[expert_idx].shape)

    @staticmethod
    def _restore_gptqmodel_expert_quant_weights(
        hf_model_dir: str,
        native_model: nn.Module,
        logger=None,
        layer_spec: Optional[str] = None,
    ) -> int:
        """Restore expert qweight tensors that GPTQModel leaves as unused checkpoint keys.

        Qwen3.5-MoE stores HF experts as packed ``experts.gate_up_proj`` and
        ``experts.down_proj`` parameters, while GPTQModel checkpoint keys are
        defused per expert/per projection.  GPTQModel dequantizes attention and
        shared-expert linears, but those packed routed-expert qweights are not
        attached to the model.  Keep them as explicit quant_weight buffers so
        XH2a W4 expert export uses the original GPTQ quantized weights instead
        of recalibrating from dequantized fp16 weights.
        """
        try:
            from safetensors.torch import safe_open
        except ImportError:
            if logger is not None:
                logger.warning("safetensors is not available; cannot restore GPTQ expert quant_weights.")
            return 0

        model_dir = Path(hf_model_dir)
        safetensors_files = sorted(model_dir.glob("model*.safetensors"))
        if not safetensors_files:
            if logger is not None:
                logger.warning(f"No model*.safetensors files found in {model_dir}; cannot restore expert quant_weights.")
            return 0

        text_model = _get_text_model(native_model)
        num_layers = int(getattr(text_model.config, "num_hidden_layers"))
        num_experts = int(getattr(text_model.config, "num_experts"))
        restore_layers = Qwen3_5MoeConverterXH2a._parse_restore_layer_spec(layer_spec, num_layers)
        packed_qweights: Dict[int, Dict[str, Dict[int, torch.Tensor]]] = {
            layer_idx: {"gate_proj": {}, "up_proj": {}, "down_proj": {}}
            for layer_idx in restore_layers
        }
        packed_dequant_weights: Dict[int, Dict[str, Dict[int, torch.Tensor]]] = {
            layer_idx: {"gate_proj": {}, "up_proj": {}, "down_proj": {}}
            for layer_idx in restore_layers
        }
        modulelist_restored = 0
        unpack_failures = 0
        prefixes = ("model.language_model.layers.", "model.layers.")

        for safetensors_file in safetensors_files:
            try:
                safetensors_reader = safe_open(str(safetensors_file), framework="pt", device="cpu")
            except Exception as exc:
                if logger is not None:
                    logger.warning(f"Failed to load safetensors {safetensors_file}: {exc}")
                continue

            with safetensors_reader:
                for key in safetensors_reader.keys():
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

                    experts = text_model.layers[layer_idx].mlp.experts
                    key_prefix = key[: -len(".qweight")]
                    qzeros_key = f"{key_prefix}.qzeros"
                    scales_key = f"{key_prefix}.scales"
                    g_idx_key = f"{key_prefix}.g_idx"
                    if qzeros_key not in safetensors_reader.keys() or g_idx_key not in safetensors_reader.keys():
                        unpack_failures += 1
                        continue
                    qweight = safetensors_reader.get_tensor(key)
                    qzeros = safetensors_reader.get_tensor(qzeros_key)
                    g_idx = safetensors_reader.get_tensor(g_idx_key)
                    scales = safetensors_reader.get_tensor(scales_key) if scales_key in safetensors_reader.keys() else None
                    if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
                        expected_shape = Qwen3_5MoeConverterXH2a._packed_expert_weight_shape(
                            experts, expert_idx, linear_name
                        )
                        try:
                            tensor = Qwen3_5MoeConverterXH2a._unpack_gptqmodel_qweight(
                                qweight, qzeros, g_idx, expected_shape
                            )
                        except Exception as exc:
                            unpack_failures += 1
                            if logger is not None and unpack_failures <= 5:
                                logger.warning(f"Failed to unpack GPTQ expert qweight {key}: {exc}")
                            continue
                        packed_qweights[layer_idx][linear_name][expert_idx] = tensor
                        if scales is not None:
                            try:
                                packed_dequant_weights[layer_idx][linear_name][expert_idx] = (
                                    Qwen3_5MoeConverterXH2a._dequantize_gptqmodel_weight(
                                        tensor,
                                        scales,
                                        g_idx,
                                        expected_shape,
                                        experts.gate_up_proj.dtype,
                                    )
                                )
                            except Exception as exc:
                                unpack_failures += 1
                                if logger is not None and unpack_failures <= 5:
                                    logger.warning(f"Failed to dequantize GPTQ expert weight {key}: {exc}")
                        continue

                    try:
                        expert = (
                            experts[expert_idx]
                            if hasattr(experts, "__getitem__")
                            else experts.get_submodule(str(expert_idx))
                        )
                        linear = getattr(expert, linear_name)
                    except Exception:
                        continue
                    if isinstance(linear, nn.Linear):
                        try:
                            tensor = Qwen3_5MoeConverterXH2a._unpack_gptqmodel_qweight(
                                qweight,
                                qzeros,
                                g_idx,
                                tuple(int(dim) for dim in linear.weight.shape),
                            )
                        except Exception as exc:
                            unpack_failures += 1
                            if logger is not None and unpack_failures <= 5:
                                logger.warning(f"Failed to unpack GPTQ expert qweight {key}: {exc}")
                            continue
                        Qwen3_5MoeConverterXH2a._set_quant_weight_buffer(
                            linear,
                            "quant_weight",
                            tensor.to(device=linear.weight.device),
                        )
                        if scales is not None:
                            try:
                                linear.weight.data.copy_(
                                    Qwen3_5MoeConverterXH2a._dequantize_gptqmodel_weight(
                                        tensor,
                                        scales,
                                        g_idx,
                                        tuple(int(dim) for dim in linear.weight.shape),
                                        linear.weight.dtype,
                                    ).to(device=linear.weight.device)
                                )
                            except Exception as exc:
                                unpack_failures += 1
                                if logger is not None and unpack_failures <= 5:
                                    logger.warning(f"Failed to restore GPTQ expert fp weight {key}: {exc}")
                        modulelist_restored += 1

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
                Qwen3_5MoeConverterXH2a._set_quant_weight_buffer(
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
                    torch.cat([gate_weight, up_weight], dim=1)
                    .to(device=experts.gate_up_proj.device, dtype=experts.gate_up_proj.dtype)
                )
                experts.down_proj.data.copy_(
                    down_weight.to(device=experts.down_proj.device, dtype=experts.down_proj.dtype)
                )
                packed_fp_restored += num_experts * 3

        restored = modulelist_restored + packed_restored
        if logger is not None:
            logger.info(
                "Restored GPTQ expert quant_weight tensors: "
                f"modulelist={modulelist_restored}, packed={packed_restored}, "
                f"total={restored}, fp_weights={packed_fp_restored}, unpack_failures={unpack_failures}"
            )
        return restored

    @staticmethod
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
                layers.update(range(start, end + 1))
            else:
                layers.add(int(part))
        invalid = sorted(layer for layer in layers if layer < 0 or layer >= num_layers)
        if invalid:
            raise ValueError(f"gptq_restore_expert_layer_spec has out-of-range layer ids: {invalid}")
        return layers

    def load_hf_model(self, hf_model_dir: str, **kwargs):
        logger = get_root_logger()

        if self._is_gptqmodel_checkpoint(hf_model_dir):
            # gptqmodel-format GPTQ checkpoints store weights as packed qweight/scales/qzeros.
            # AutoModelForCausalLM.from_pretrained() treats those as UNEXPECTED keys and
            # leaves Linear weights uninitialized, producing garbage outputs.
            # GPTQModel.load() properly unpacks and dequantizes back to float16 nn.Linear.
            try:
                from gptqmodel import BACKEND, GPTQModel  # type: ignore

                logger.info(f"Detected gptqmodel checkpoint; using GPTQModel.load() for dequantization: {hf_model_dir}")
                self._register_gptqmodel_qwen3_5_moe()
                torch_dtype = kwargs.get("torch_dtype", torch.float16)
                # BACKEND.TORCH ensures all packed linears become TorchQuantLinear, which
                # the base ``_dequantize_gptq_hf_model`` already knows how to dequantize.
                qmodel = GPTQModel.load(
                    hf_model_dir,
                    device="cpu",
                    dtype=torch_dtype,
                    backend=BACKEND.TORCH,
                )
                # GPTQModel.load() wraps the HF model in a BaseQModel; extract the inner model.
                from gptqmodel.models.base import BaseQModel  # type: ignore

                native_model = qmodel.model if isinstance(qmodel, BaseQModel) else qmodel
                logger.info(f"Extracted inner HF model: {type(native_model).__name__}")
                # Unpack TorchQuantLinear -> nn.Linear via the existing base helper, then
                # clear quantization metadata so the downstream dequantize_hf_model() is a no-op.
                native_model = self._dequantize_gptq_hf_model(native_model)
                if hasattr(native_model, "config"):
                    native_model.config.quantization_config = None
                restore_layer_spec = getattr(self.config, "gptq_restore_expert_layer_spec", None)
                if restore_layer_spec is not None:
                    self._restore_gptqmodel_expert_quant_weights(
                        hf_model_dir,
                        native_model,
                        logger,
                        layer_spec=restore_layer_spec,
                    )
            except ImportError:
                logger.warning("gptqmodel not available; falling back to AutoModelForCausalLM")
                native_model = AutoModelForCausalLM.from_pretrained(
                    hf_model_dir,
                    trust_remote_code=True,
                    **kwargs,
                )
                native_model = self.dequantize_hf_model(native_model)
        else:
            native_model = AutoModelForCausalLM.from_pretrained(
                hf_model_dir,
                trust_remote_code=True,
                **kwargs,
            )
            native_model = self.dequantize_hf_model(native_model)
        assert isinstance(native_model, (Qwen3_5MoeForConditionalGeneration, Qwen3_5MoeForCausalLM)), (
            f"Expected Qwen3_5MoeForConditionalGeneration or Qwen3_5MoeForCausalLM, got {type(native_model)}"
        )
        native_model = native_model.eval()

        # Handle tied embeddings
        tie_word_embeddings = getattr(native_model.config, "tie_word_embeddings", False)
        text_config = _get_text_config(native_model)
        if not tie_word_embeddings:
            tie_word_embeddings = getattr(text_config, "tie_word_embeddings", False)
        if tie_word_embeddings:
            old_torchscript = native_model.config.torchscript
            native_model.config.torchscript = True
            native_model.tie_weights()
            native_model.config.tie_word_embeddings = False
            if hasattr(text_config, "tie_word_embeddings"):
                text_config.tie_word_embeddings = False
            native_model.config.torchscript = old_torchscript

        self.hf_model_path = hf_model_dir
        return native_model

    def _copy_hf_config(self, hf_model_path: str, work_dir: Path):
        logger = get_root_logger()
        hf_config_dir = work_dir / "hf_config"
        hf_config_dir.mkdir(exist_ok=True, parents=True)
        hf_config_files = [
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.json",
            "special_tokens_map.json",
            "chat_template.jinja",
            "merges.txt",
            "tokenizer.model",
        ]
        for cfg_file in hf_config_files:
            src_file = Path(hf_model_path) / cfg_file
            dst_file = hf_config_dir / cfg_file
            if src_file.exists():
                shutil.copyfile(src_file, dst_file)
            else:
                logger.warning(f"{src_file} not exists, skip copy")
        return hf_config_dir

    def _build_quant_config(self, wraped_model: nn.Module):
        quant_config = ConfigDict(create_quant_config(self.config.quant_scheme))
        quant_config.setdefault("inputs", {})
        quant_config["inputs"].setdefault(
            "linear_attn_mask",
            dict(
                quantizer=dict(
                    qspec=dict(fake_dtype="float16"),
                )
            ),
        )

        cumsum_quant_cfg = self.config.cumsum_matmul_quant_config
        if cumsum_quant_cfg is None:
            cumsum_quant_cfg = dict(
                act_schema=dict(
                    fp_mode="sefp",
                    man_bit=16,
                ),
                act_schema_2=dict(
                    fp_mode="fp16",
                    man_bit=8,
                ),
            )

        quant_config.setdefault("nodes_cfg", {})
        for name, module in wraped_model.named_modules():
            if isinstance(module, nn.Linear) and name == "lm_head" and not hasattr(module, "quant_weight"):
                quant_config["nodes_cfg"].setdefault(
                    name,
                    dict(
                        w_schema=dict(
                            bits=8,
                            fp_mode="sefp",
                        )
                    ),
                )

            if "cumsum_matmul" not in name:
                continue
            key_candidates = {
                name,
                name.replace(".", "_"),
            }
            for key in key_candidates:
                quant_config["nodes_cfg"][key] = dict(cumsum_quant_cfg)

        quant_config.setdefault("ops_cfg", {})
        quant_config["ops_cfg"]["Normalize"] = dict(force_fp32=self.config.normalize_force_fp32)

        return quant_config

    def _prepare_wrap_model(self, native_model):
        from ._moe_model import register_wrap_modules as qwen3_5_moe_register_wrap_modules

        qwen3_5_moe_register_wrap_modules()
        spec_decode_mode = getattr(self.config, "spec_decode_mode", None)
        output_post_norm_hidden = bool(getattr(self.config, "output_post_norm_hidden", False))
        output_hidden_state_indices = getattr(self.config, "output_hidden_state_indices", None)
        if spec_decode_mode == "dflash":
            dflash_model_dir = getattr(self.config, "dflash_model_dir", None)
            if not dflash_model_dir:
                raise ValueError("dflash_model_dir is required when spec_decode_mode='dflash'")
            output_hidden_state_indices = _load_dflash_target_layer_ids(dflash_model_dir)
        elif spec_decode_mode == "mtp":
            output_post_norm_hidden = True
        wrap_cfg = Config(
            dict(
                batch_size=self.config.batch_size,
                max_sequence_length=self.config.context_length,
                input_sequence_length=self.config.input_sequence_length,
                use_cache=True,
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
                fuse_gdr_block_recurrent_ops=getattr(
                    self.config,
                    "fuse_gdr_block_recurrent_ops",
                    False,
                ),
                kv_cache=dict(
                    cache_axis=2,
                ),
            )
        )
        wraped_model = wrap_llm_model(native_model, wrap_cfg)
        if not hasattr(wraped_model, "_qwen3_5_moe_original_forward"):
            wraped_model._qwen3_5_moe_original_forward = wraped_model.forward
            wraped_model.forward = types.MethodType(_flatten_cache_outputs, wraped_model)
        return wraped_model, wrap_cfg

    def _build_cache_inputs(
        self,
        wraped_model: nn.Module,
        native_model,
        context_length: int,
    ):
        text_config = _get_text_config(native_model)
        text_model = _get_text_model(wraped_model)
        layer_types = list(text_config.layer_types)
        full_attention_layer_indices = [i for i, layer_type in enumerate(layer_types) if layer_type == "full_attention"]
        linear_attention_layer_indices = [
            i for i, layer_type in enumerate(layer_types) if layer_type == "linear_attention"
        ]

        head_dim = text_config.head_dim
        batch_size = int(self.config.batch_size)
        if batch_size > 1:
            # Continuous-batch export contract: every HMONNX graph input is
            # single-batch.  Full-attention KV cache inputs therefore become a
            # flat per-layer/per-batch list of [1, Hkv, ctx, D] tensors.
            kv_cache_shape = [1, text_config.num_key_value_heads, context_length, head_dim]
            past_key_caches = []
            past_value_caches = []
            for _ in full_attention_layer_indices:
                for _batch_idx in range(batch_size):
                    past_key_caches.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)))
                    past_value_caches.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)))
        else:
            kv_cache_shape = [
                batch_size,
                text_config.num_key_value_heads,
                context_length,
                head_dim,
            ]
            past_key_caches = [
                CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)) for _ in full_attention_layer_indices
            ]
            past_value_caches = [
                CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)) for _ in full_attention_layer_indices
            ]

        past_conv_caches = []
        past_recurrent_states = []
        linear_cache_meta = []
        for layer_idx in linear_attention_layer_indices:
            linear_attn = text_model.layers[layer_idx].linear_attn
            if self.config.split_conv_cache:
                cache_dtype = (
                    linear_attn.conv1d_q.weight.dtype
                    if hasattr(linear_attn, "conv1d_q")
                    else linear_attn.conv1d.weight.dtype
                )
                q_dim, k_dim, v_dim = _linear_split_conv_dims(linear_attn)
                conv_shapes = [
                    [1 if batch_size > 1 else batch_size, q_dim, linear_attn.conv_kernel_size],
                    [1 if batch_size > 1 else batch_size, k_dim, linear_attn.conv_kernel_size],
                    [1 if batch_size > 1 else batch_size, v_dim, linear_attn.conv_kernel_size],
                ]
                if batch_size > 1:
                    expanded_conv_shapes = []
                    for shape in conv_shapes:
                        for _batch_idx in range(batch_size):
                            past_conv_caches.append(CacheTensor(torch.zeros(shape, dtype=cache_dtype)))
                            expanded_conv_shapes.append(list(shape))
                else:
                    expanded_conv_shapes = conv_shapes
                    for shape in conv_shapes:
                        past_conv_caches.append(CacheTensor(torch.zeros(shape, dtype=cache_dtype)))
            else:
                cache_dtype = linear_attn.conv1d.weight.dtype
                conv_shape = [
                    1 if batch_size > 1 else batch_size,
                    linear_attn.conv_dim,
                    linear_attn.conv_kernel_size,
                ]
                if batch_size > 1:
                    expanded_conv_shapes = []
                    for _batch_idx in range(batch_size):
                        past_conv_caches.append(CacheTensor(torch.zeros(conv_shape, dtype=cache_dtype)))
                        expanded_conv_shapes.append(list(conv_shape))
                else:
                    expanded_conv_shapes = [conv_shape]
                    past_conv_caches.append(CacheTensor(torch.zeros(conv_shape, dtype=cache_dtype)))
            recurrent_shape = [
                1 if batch_size > 1 else batch_size,
                linear_attn.num_v_heads,
                linear_attn.head_k_dim,
                linear_attn.head_v_dim,
            ]
            if batch_size > 1:
                recurrent_shapes = []
                for _batch_idx in range(batch_size):
                    past_recurrent_states.append(CacheTensor(torch.zeros(recurrent_shape, dtype=cache_dtype)))
                    recurrent_shapes.append(list(recurrent_shape))
            else:
                recurrent_shapes = [recurrent_shape]
                past_recurrent_states.append(CacheTensor(torch.zeros(recurrent_shape, dtype=cache_dtype)))
            if self.config.split_conv_cache:
                linear_cache_meta.append(
                    dict(
                        layer_idx=layer_idx,
                        conv_shapes=expanded_conv_shapes,
                        recurrent_shapes=recurrent_shapes,
                        recurrent_shape=recurrent_shape,
                        per_batch=batch_size > 1,
                    )
                )
            else:
                linear_cache_meta.append(
                    dict(
                        layer_idx=layer_idx,
                        conv_shapes=expanded_conv_shapes,
                        conv_shape=conv_shape,
                        recurrent_shapes=recurrent_shapes,
                        recurrent_shape=recurrent_shape,
                        per_batch=batch_size > 1,
                    )
                )

        return (
            full_attention_layer_indices,
            linear_attention_layer_indices,
            kv_cache_shape,
            past_key_caches,
            past_value_caches,
            past_conv_caches,
            past_recurrent_states,
            linear_cache_meta,
        )

    def _convert(self, hf_model_path: str, output_dir: str):
        logger = get_root_logger()
        # When quant_weight is a directory, it is a pre-quantized GPTQ model.
        # Load from that directory (dequantize_hf_model is called inside get_hf_model).
        # When quant_weight is a single file (.safetensors / .pt), load base model
        # first and then apply the weight file via load_quant_weight.
        quant_weight = self.config.quant_weight
        if quant_weight is not None and Path(quant_weight).is_dir():
            logger.info(f"Loading GPTQ model from: {quant_weight}")
            native_model = self.get_hf_model(
                quant_weight,
                torch_dtype=torch.float16,
                device_map="cpu",
            )
        else:
            native_model = self.get_hf_model(
                hf_model_path,
                torch_dtype=torch.float16,
                device_map="cpu",
            )
            if quant_weight is not None:
                self.load_quant_weight(quant_weight, native_model)

        work_dir = Path(output_dir)
        work_dir.mkdir(exist_ok=True, parents=True)
        model_name = Path(hf_model_path).name
        context_length = self.config.context_length
        input_sequence_length = self.config.input_sequence_length
        quant_type = self.config.quant_scheme.quant_type
        target_device = self.config.quant_scheme.target_device
        text_config = _get_text_config(native_model)

        # Determine architecture string
        if isinstance(native_model, Qwen3_5MoeForConditionalGeneration):
            architecture_str = "Qwen3_5MoeForConditionalGeneration"
        else:
            architecture_str = "Qwen3_5MoeForCausalLM"

        meta_info: Dict[str, Any] = dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            device=str(target_device),
            model_name=model_name,
            hf_model_path=hf_model_path,
            architecture=architecture_str,
            quant_scheme=self.config.quant_scheme.to_dict(),
            quant_weight=self.config.quant_weight,
            source_quant_method=_extract_quant_method(native_model.config),
            pad_token_id=getattr(native_model.config, "eos_token_id", None)
            or getattr(text_config, "eos_token_id", 151645),
            max_context_tokens=context_length,
        )

        hf_config_dir = self._copy_hf_config(hf_model_path, work_dir)
        meta_info["hf_config"] = str(hf_config_dir.relative_to(work_dir))

        text_model = _get_text_model(native_model)
        token_embedding = text_model.get_input_embeddings()
        token_embedding_file = work_dir / "token_embedding.pt"
        torch.save(token_embedding.state_dict(), str(token_embedding_file))
        meta_info["token_embedding_file"] = str(token_embedding_file.relative_to(work_dir))

        wraped_model, wrap_cfg = self._prepare_wrap_model(native_model)
        meta_info["wrap_cfg"] = wrap_cfg.to_dict()

        (
            full_attention_layer_indices,
            linear_attention_layer_indices,
            kv_cache_shape,
            past_key_caches,
            past_value_caches,
            past_conv_caches,
            past_recurrent_states,
            linear_cache_meta,
        ) = self._build_cache_inputs(wraped_model, native_model, context_length)

        meta_info["kv_cache"] = dict(
            shape=kv_cache_shape,
            num_decoder_layers=len(full_attention_layer_indices),
            layer_indices=full_attention_layer_indices,
        )
        meta_info["linear_cache"] = dict(
            num_decoder_layers=len(linear_attention_layer_indices),
            layer_indices=linear_attention_layer_indices,
            layers=linear_cache_meta,
            num_conv_caches=len(past_conv_caches),
        )

        # Build calibration input_ids: use real text if tokenizer is available
        # (essential for W4A8 quantization — random tokens give wrong activation scales).
        try:
            from transformers import AutoTokenizer as _AutoTokenizer

            _tok = _AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
            _calib_text = (
                "The Qwen3.5-MoE model is a large language model based on the Mixture-of-Experts "
                "architecture. It combines dense attention layers with sparse MoE feed-forward blocks. "
                "Each token is routed to a small subset of experts, reducing compute cost while "
                "maintaining high model capacity. The model supports both Chinese and English. "
                "这是一个基于专家混合架构的大语言模型，支持中英双语，并具备强大的推理能力。"
            )
            _enc = _tok(
                _calib_text,
                return_tensors="pt",
                truncation=True,
                max_length=input_sequence_length,
            )
            raw_ids = _enc["input_ids"][0]
            seq_actual = raw_ids.shape[0]
            if seq_actual < input_sequence_length:
                # Repeat + truncate to fill the window
                repeat_times = (input_sequence_length + seq_actual - 1) // seq_actual
                raw_ids = raw_ids.repeat(repeat_times)[:input_sequence_length]
            else:
                raw_ids = raw_ids[:input_sequence_length]
            input_ids_t = raw_ids.unsqueeze(0).expand(self.config.batch_size, -1)
            del _tok, _calib_text, _enc, raw_ids, seq_actual
        except Exception:
            input_ids_t = torch.randint(0, 1000, (self.config.batch_size, input_sequence_length), dtype=torch.long)
        inputs_embeds = token_embedding(input_ids_t)
        past_seq_length_t = torch.zeros(self.config.batch_size, dtype=torch.int32)
        current_input_length_t = torch.full(
            (self.config.batch_size,),
            input_sequence_length,
            dtype=torch.int32,
        )
        linear_attn_mask_t = torch.ones(
            self.config.batch_size,
            input_sequence_length,
            dtype=inputs_embeds.dtype,
        )

        # M-RoPE position IDs: for text-only, all three are sequential
        position_ids = (
            torch.arange(input_sequence_length, dtype=torch.long).unsqueeze(0).expand(self.config.batch_size, -1)
        )
        time_position_ids = position_ids
        hight_position_ids = position_ids
        width_position_ids = position_ids

        batch_size = int(self.config.batch_size)
        if batch_size > 1:
            # All exported graph inputs are single-batch.  The wrapper receives
            # lists, concatenates them inside the graph, and splits batched
            # outputs back to one tensor per batch item.
            graph_inputs_embeds = _split_batch_tensor(inputs_embeds, batch_size)
            graph_time_position_ids = _split_batch_tensor(time_position_ids, batch_size)
            graph_hight_position_ids = _split_batch_tensor(hight_position_ids, batch_size)
            graph_width_position_ids = _split_batch_tensor(width_position_ids, batch_size)
            graph_past_seq_length = _split_batch_tensor(past_seq_length_t, batch_size)
            graph_current_input_length = _split_batch_tensor(current_input_length_t, batch_size)
            graph_linear_attn_mask = _split_batch_tensor(linear_attn_mask_t, batch_size)
        else:
            graph_inputs_embeds = inputs_embeds
            graph_time_position_ids = time_position_ids
            graph_hight_position_ids = hight_position_ids
            graph_width_position_ids = width_position_ids
            graph_past_seq_length = past_seq_length_t
            graph_current_input_length = current_input_length_t
            graph_linear_attn_mask = linear_attn_mask_t

        inputs = (
            graph_inputs_embeds,
            graph_time_position_ids,
            graph_hight_position_ids,
            graph_width_position_ids,
            graph_past_seq_length,
            graph_current_input_length,
            graph_linear_attn_mask,
            past_key_caches,
            past_value_caches,
            past_conv_caches,
            past_recurrent_states,
        )

        if batch_size > 1:
            input_names = []
            for base_name in (
                "inputs_embeds",
                "time_position_ids",
                "hight_position_ids",
                "width_position_ids",
                "past_seq_length",
                "current_input_length",
                "linear_attn_mask",
            ):
                for batch_idx in range(batch_size):
                    input_names.append(f"{base_name}_batch_{batch_idx}")
        else:
            input_names = [
                "inputs_embeds",
                "time_position_ids",
                "hight_position_ids",
                "width_position_ids",
                "past_seq_length",
                "current_input_length",
                "linear_attn_mask",
            ]
        for layer_idx in range(len(full_attention_layer_indices)):
            if batch_size > 1:
                for batch_idx in range(batch_size):
                    input_names.append(f"past_key_cache_{layer_idx}_batch_{batch_idx}")
            else:
                input_names.append(f"past_key_cache_{layer_idx}")
        for layer_idx in range(len(full_attention_layer_indices)):
            if batch_size > 1:
                for batch_idx in range(batch_size):
                    input_names.append(f"past_value_cache_{layer_idx}_batch_{batch_idx}")
            else:
                input_names.append(f"past_value_cache_{layer_idx}")
        for layer_idx in range(len(linear_attention_layer_indices)):
            if self.config.split_conv_cache:
                for branch in _LINEAR_CONV_CACHE_BRANCHES:
                    if batch_size > 1:
                        for batch_idx in range(batch_size):
                            input_names.append(f"past_conv_cache_{branch}_{layer_idx}_batch_{batch_idx}")
                    else:
                        input_names.append(f"past_conv_cache_{branch}_{layer_idx}")
            else:
                if batch_size > 1:
                    for batch_idx in range(batch_size):
                        input_names.append(f"past_conv_cache_{layer_idx}_batch_{batch_idx}")
                else:
                    input_names.append(f"past_conv_cache_{layer_idx}")
        for layer_idx in range(len(linear_attention_layer_indices)):
            if batch_size > 1:
                for batch_idx in range(batch_size):
                    input_names.append(f"past_recurrent_state_{layer_idx}_batch_{batch_idx}")
            else:
                input_names.append(f"past_recurrent_state_{layer_idx}")

        output_names_base = (
            [f"logits_batch_{batch_idx}" for batch_idx in range(batch_size)]
            if batch_size > 1
            else ["logits"]
        )
        for layer_idx in range(len(linear_attention_layer_indices)):
            if self.config.split_conv_cache:
                for branch in _LINEAR_CONV_CACHE_BRANCHES:
                    if batch_size > 1:
                        for batch_idx in range(batch_size):
                            output_names_base.append(f"conv_cache_out_{branch}_{layer_idx}_batch_{batch_idx}")
                    else:
                        output_names_base.append(f"conv_cache_out_{branch}_{layer_idx}")
            else:
                if batch_size > 1:
                    for batch_idx in range(batch_size):
                        output_names_base.append(f"conv_cache_out_{layer_idx}_batch_{batch_idx}")
                else:
                    output_names_base.append(f"conv_cache_out_{layer_idx}")
        for layer_idx in range(len(linear_attention_layer_indices)):
            if batch_size > 1:
                for batch_idx in range(batch_size):
                    output_names_base.append(f"recurrent_state_out_{layer_idx}_batch_{batch_idx}")
            else:
                output_names_base.append(f"recurrent_state_out_{layer_idx}")

        # Diagnostic/spec hidden output is appended last so existing cache-update
        # indexing in Qwen3_5MoeInference._forward() is unaffected.
        spec_decode_mode = getattr(self.config, "spec_decode_mode", None)
        num_draft_tokens = getattr(self.config, "num_draft_tokens", 4)
        verify_length = num_draft_tokens + 1
        output_hidden_state_indices = getattr(self.config, "output_hidden_state_indices", None)
        if spec_decode_mode == "dflash":
            dflash_model_dir = getattr(self.config, "dflash_model_dir", None)
            if not dflash_model_dir:
                raise ValueError("dflash_model_dir is required when spec_decode_mode='dflash'")
            output_hidden_state_indices = _load_dflash_target_layer_ids(dflash_model_dir)
        output_post_norm_hidden = bool(getattr(self.config, "output_post_norm_hidden", False))
        if spec_decode_mode == "mtp":
            output_post_norm_hidden = True
        extra_hidden_output_name = None
        if output_hidden_state_indices is not None:
            extra_hidden_output_name = "target_hidden"
        elif output_post_norm_hidden:
            extra_hidden_output_name = "post_norm_hidden"

        prefill_output_names = list(output_names_base)
        # Decode in spec mode emits per-step verify intermediates (mirrors dense
        # ``qwen3_5_llm_model.py:284-316``): ``conv_cache_out_{l}_{t}`` and
        # ``recurrent_state_out_{l}_{t}`` for ``t in 0..verify_length-1``.
        if spec_decode_mode:
            decode_output_names = (
                [f"logits_batch_{batch_idx}" for batch_idx in range(batch_size)]
                if batch_size > 1
                else ["logits"]
            )
            for layer_idx in range(len(linear_attention_layer_indices)):
                if self.config.split_conv_cache:
                    for branch in _LINEAR_CONV_CACHE_BRANCHES:
                        for step_idx in range(verify_length):
                            if batch_size > 1:
                                for batch_idx in range(batch_size):
                                    decode_output_names.append(
                                        f"conv_cache_out_{branch}_{layer_idx}_{step_idx}_batch_{batch_idx}"
                                    )
                            else:
                                decode_output_names.append(f"conv_cache_out_{branch}_{layer_idx}_{step_idx}")
                else:
                    for step_idx in range(verify_length):
                        if batch_size > 1:
                            for batch_idx in range(batch_size):
                                decode_output_names.append(
                                    f"conv_cache_out_{layer_idx}_{step_idx}_batch_{batch_idx}"
                                )
                        else:
                            decode_output_names.append(f"conv_cache_out_{layer_idx}_{step_idx}")
            for layer_idx in range(len(linear_attention_layer_indices)):
                for step_idx in range(verify_length):
                    if batch_size > 1:
                        for batch_idx in range(batch_size):
                            decode_output_names.append(
                                f"recurrent_state_out_{layer_idx}_{step_idx}_batch_{batch_idx}"
                            )
                    else:
                        decode_output_names.append(f"recurrent_state_out_{layer_idx}_{step_idx}")
        else:
            decode_output_names = list(output_names_base)
        if extra_hidden_output_name is not None:
            if batch_size > 1:
                extra_hidden_output_names = [
                    f"{extra_hidden_output_name}_batch_{batch_idx}"
                    for batch_idx in range(batch_size)
                ]
                prefill_output_names.extend(extra_hidden_output_names)
                decode_output_names.extend(extra_hidden_output_names)
            else:
                prefill_output_names.append(extra_hidden_output_name)
                decode_output_names.append(extra_hidden_output_name)

        # ── PREFILL quantisation & export ────────────────────────────────────
        # convert_fx_model_to_quanted_model modifies wraped_model in-place
        # (FrontendGraph is built on top of wraped_model; PTQ inserts QBaseModules
        # directly into the shared module tree). We deepcopy wraped_model for
        # prefill so the original remains clean for decode re-trace.
        quant_config = self._build_quant_config(wraped_model)
        prefix = f"{model_name}-{target_device}-{context_length // 1024}k-{quant_type}"
        if getattr(self.config, "export_prefill", True):
            wraped_model_prefill = copy.deepcopy(wraped_model)
            if spec_decode_mode:
                wrap_cfg_prefill = copy.deepcopy(wrap_cfg)
                wrap_cfg_prefill.num_logits_to_keep = 0

                def _apply_prefill_update_cfg(module):
                    if hasattr(module, "_update_cfg"):
                        module._update_cfg(wrap_cfg_prefill)

                wraped_model_prefill.apply(_apply_prefill_update_cfg)
            quanted_prefill_model = convert_fx_model_to_quanted_model(
                wraped_model_prefill,
                inputs,
                target_device,
                quant_config=quant_config,
            )

            prefill_onnx_file = work_dir / "hmonnx" / "prefill" / f"{prefix}_prefill.onnx"
            prefill_onnx_file.parent.mkdir(exist_ok=True, parents=True)
            meta_info["prefill_onnx"] = str(prefill_onnx_file.relative_to(work_dir))
            logger.info("********************* start export prefill model *********************")
            convert_quanted_model_to_hmonnx(
                quanted_prefill_model,
                inputs,
                str(prefill_onnx_file),
                BaseConverter.xh1_hmonnx_compatible(input_names),
                prefill_output_names,
            )
            patched_adds = _patch_hmonnx_standard_add_ops(prefill_onnx_file)
            if patched_adds:
                logger.info(f"Patched {patched_adds} standard Add node(s) to XH2a domain in {prefill_onnx_file}")
            logger.info(f"Export Prefill model to {prefill_onnx_file}")
            del quanted_prefill_model, wraped_model_prefill
        else:
            meta_info["prefill_onnx"] = None
            logger.info("********************* skip prefill export by config.export_prefill=False *********************")

        # ── DECODE quantisation & export ─────────────────────────────────────
        # Re-trace from clean wraped_model after updating to decode mode so that
        # baked-in FX slice constants use seq_len appropriately.
        #
        # For spec decode mode, decode session processes verify_length tokens
        # (draft tokens + 1 current) so the verifier can run in a single pass.
        decode_seq_len = verify_length if spec_decode_mode else 1
        wrap_cfg.input_sequence_length = decode_seq_len
        if spec_decode_mode:
            wrap_cfg.num_logits_to_keep = 0  # return all positions for verify
            # Per-step conv/recurrent verify intermediates (Phase 3, mirrors dense).
            wrap_cfg.verify_output_intermediates = True
            # Force recurrent mode so the per-step verify branch in
            # _Qwen3_5MoeGatedDeltaNet.forward executes (it requires use_recurrent=True).
            # Mirrors dense set_linear_attention_mode("recurrent") for spec decode.
            wrap_cfg.linear_attention_mode = "recurrent"

        def _apply_update_cfg(module):
            if hasattr(module, "_update_cfg"):
                module._update_cfg(wrap_cfg)

        wraped_model.apply(_apply_update_cfg)

        decode_position_ids = torch.zeros(self.config.batch_size, decode_seq_len, dtype=torch.long)
        decode_current_input_length_t = torch.full_like(current_input_length_t, decode_seq_len)
        decode_linear_attn_mask_t = torch.ones(batch_size, decode_seq_len, dtype=inputs_embeds.dtype)
        if batch_size > 1:
            decode_graph_inputs_embeds = _split_batch_tensor(inputs_embeds[:, :decode_seq_len, :], batch_size)
            decode_graph_position_ids = _split_batch_tensor(decode_position_ids, batch_size)
            decode_graph_past_seq_length = _split_batch_tensor(past_seq_length_t, batch_size)
            decode_graph_current_input_length = _split_batch_tensor(decode_current_input_length_t, batch_size)
            decode_graph_linear_attn_mask = _split_batch_tensor(decode_linear_attn_mask_t, batch_size)
        else:
            decode_graph_inputs_embeds = inputs_embeds[:, :decode_seq_len, :]
            decode_graph_position_ids = decode_position_ids
            decode_graph_past_seq_length = past_seq_length_t
            decode_graph_current_input_length = decode_current_input_length_t
            decode_graph_linear_attn_mask = decode_linear_attn_mask_t
        decode_inputs = (
            decode_graph_inputs_embeds,
            decode_graph_position_ids,
            decode_graph_position_ids,
            decode_graph_position_ids,
            decode_graph_past_seq_length,
            decode_graph_current_input_length,
            decode_graph_linear_attn_mask,
            past_key_caches,
            past_value_caches,
            past_conv_caches,
            past_recurrent_states,
        )

        quanted_decode_model = convert_fx_model_to_quanted_model(
            wraped_model,
            decode_inputs,
            target_device,
            quant_config=quant_config,
        )

        decode_onnx_file = work_dir / "hmonnx" / "decode" / f"{prefix}_decoder.onnx"
        decode_onnx_file.parent.mkdir(exist_ok=True, parents=True)
        meta_info["decode_onnx"] = str(decode_onnx_file.relative_to(work_dir))
        logger.info("********************* start export decode model *********************")
        convert_quanted_model_to_hmonnx(
            quanted_decode_model,
            decode_inputs,
            str(decode_onnx_file),
            BaseConverter.xh1_hmonnx_compatible(input_names),
            decode_output_names,
        )
        patched_adds = _patch_hmonnx_standard_add_ops(decode_onnx_file)
        if patched_adds:
            logger.info(f"Patched {patched_adds} standard Add node(s) to XH2a domain in {decode_onnx_file}")
        logger.info(f"Export decode model to {decode_onnx_file}")

        # ── MTP draft model export ────────────────────────────────────────────
        if spec_decode_mode == "mtp":
            draft_onnx_files = self._export_mtp_draft_model(
                hf_model_path=hf_model_path,
                work_dir=work_dir,
                prefix=prefix,
                context_length=context_length,
                verify_length=verify_length,
                target_device=target_device,
            )
            for key, value in draft_onnx_files.items():
                meta_info[f"{key}_file"] = str(Path(value).relative_to(work_dir))
            if "draft_decode_onnx" in draft_onnx_files:
                meta_info["draft_onnx_file"] = str(Path(draft_onnx_files["draft_decode_onnx"]).relative_to(work_dir))
            meta_info["spec_decode_mode"] = spec_decode_mode
            meta_info["spec_decode_block_size"] = num_draft_tokens
            meta_info["spec_decode_hidden_output_name"] = "post_norm_hidden"
            meta_info["spec_decode_verify_length"] = verify_length
            meta_info["spec_decode_draft_head_weight_bits"] = self.config.spec_draft_head_weight_bits
        elif spec_decode_mode == "dflash":
            draft_onnx_files = self._export_dflash_draft_model(
                hf_model_path=hf_model_path,
                dflash_model_dir=self.config.dflash_model_dir,
                work_dir=work_dir,
                prefix=prefix,
                context_length=context_length,
                verify_length=verify_length,
                target_device=target_device,
            )
            for key, value in draft_onnx_files.items():
                meta_info[f"{key}_file"] = str(Path(value).relative_to(work_dir))
            if "draft_decode_onnx" in draft_onnx_files:
                meta_info["draft_onnx_file"] = str(Path(draft_onnx_files["draft_decode_onnx"]).relative_to(work_dir))
            meta_info["spec_decode_mode"] = spec_decode_mode
            meta_info["spec_decode_block_size"] = num_draft_tokens
            meta_info["spec_decode_hidden_output_name"] = "target_hidden"
            meta_info["spec_decode_verify_length"] = verify_length
            meta_info["spec_decode_draft_head_weight_bits"] = self.config.spec_draft_head_weight_bits

        with open(work_dir / "meta.json", "w", encoding="utf-8") as fout:
            json.dump(meta_info, fout, ensure_ascii=False, indent=4)

    def _export_mtp_draft_model(
        self,
        hf_model_path: str,
        work_dir: Path,
        prefix: str,
        context_length: int,
        verify_length: int,
        target_device: str,
    ) -> dict:
        """Export MTP draft model for MoE spec decode.

        Reuses XHMTPDraftModel from the dense qwen3_5 package.  The MTP head
        structure (single attention layer + MLP + norms) is identical for both
        dense and MoE targets; only the backbone differs.

        Returns:
            dict mapping {"draft_prefill_onnx": path, "draft_decode_onnx": path}
        """
        import xh_model_zoo.xh_llm.models.qwen3_5.qwen3_5_mtp_model  # noqa: F401
        from xh_model_zoo.xh_llm.models.builder import MODELS
        from xhquant.api import (  # type: ignore
            ConfigDict,
            PrecisionMode,
            ptq_quantize,
        )

        logger = get_root_logger()
        draft_onnx_dir = work_dir / "draft_onnx"
        draft_onnx_dir.mkdir(exist_ok=True, parents=True)
        max_pe_length = 262144
        draft_head_weight_bits = int(getattr(self.config, "spec_draft_head_weight_bits", 4))
        logger.info(f"MTP draft quant config: base={DRAFT_BASE_QUANT_TYPE}, lm_head_w_bits={draft_head_weight_bits}")

        def _export_one(wrap_cfg_extra: dict, name_suffix: str) -> str:
            model_cfg = dict(
                type="XHMTPDraftModel",
                hf_model=None,
                wrap_cfg=ConfigDict(
                    max_sequence_length=context_length,
                    max_pe_length=max_pe_length,
                    dtype="float16",
                    batch_size=1,
                    **wrap_cfg_extra,
                ),
                quant_config=_build_spec_draft_quant_config(draft_head_weight_bits),
                export_cfg=ConfigDict(),
                target_model_dir=hf_model_path,
            )
            draft_model = MODELS.build(model_cfg)
            draft_model.init_wrap_model()
            logger.info(
                f"[{name_suffix}] MTP draft params: "
                f"{sum(p.numel() for p in draft_model._wrap_model.parameters()) / 1e6:.1f}M"
            )
            dummy_data = draft_model.prepare_inputs(None)
            draft_model.convert_to_fronted_graph(dummy_data)
            draft_model.convert_to_quant_graph(target_device)
            ptq_quantize(
                draft_model._quanted_model,
                [draft_model.prepare_inputs(None)],
                PrecisionMode.ALIGNED,
                [torch.device("cpu")],
            )
            draft_model.convert_to_export_graph(dummy_data)
            onnx_file = draft_model.to_export_onnx(dummy_data, str(draft_onnx_dir), f"{prefix}_{name_suffix}")[0]
            draft_model.release_exported_model()
            draft_model.release_quanted_model()
            draft_model.release_frontend_model()
            draft_model.release_wraped_model()
            del draft_model
            return onnx_file

        prefill_isl = self.config.input_sequence_length
        prefill_onnx = _export_one({"input_sequence_length": prefill_isl}, "mtp_prefill")
        decode_onnx = _export_one({"input_sequence_length": 1}, "mtp_decode")
        return {
            "draft_prefill_onnx": prefill_onnx,
            "draft_decode_onnx": decode_onnx,
        }

    def _export_dflash_draft_model(
        self,
        hf_model_path: str,
        dflash_model_dir: str,
        work_dir: Path,
        prefix: str,
        context_length: int,
        verify_length: int,
        target_device: str,
    ) -> dict:
        import xh_model_zoo.xh_llm.models.qwen3_5.qwen3_5_dflash_model  # noqa: F401
        from xh_model_zoo.xh_llm.models.builder import MODELS
        from xhquant.api import (  # type: ignore
            ConfigDict,
            PrecisionMode,
            ptq_quantize,
        )

        logger = get_root_logger()
        draft_onnx_dir = work_dir / "draft_onnx"
        draft_onnx_dir.mkdir(exist_ok=True, parents=True)
        max_pe_length = 262144
        draft_decode_seq_len = int(verify_length)
        draft_head_weight_bits = int(getattr(self.config, "spec_draft_head_weight_bits", 4))
        logger.info(f"DFlash draft quant config: base={DRAFT_BASE_QUANT_TYPE}, lm_head_w_bits={draft_head_weight_bits}")
        with open(Path(dflash_model_dir) / "config.json", encoding="utf-8") as f:
            dflash_cfg = json.load(f)
        model_block_size = int(dflash_cfg.get("block_size", draft_decode_seq_len))
        if draft_decode_seq_len > model_block_size:
            raise ValueError(
                f"DFlash draft decode input length ({draft_decode_seq_len} = num_draft_tokens + 1) exceeds "
                f"model block_size ({model_block_size}) from {Path(dflash_model_dir) / 'config.json'}"
            )
        if draft_decode_seq_len < model_block_size:
            logger.info(
                "DFlash draft decode input_sequence_length reduced from model "
                f"block_size={model_block_size} to verify_length={draft_decode_seq_len}"
            )

        def _export_one(mode: str, input_sequence_length: int, name_suffix: str) -> str:
            model_cfg = dict(
                type="XHDFlashDraftModel",
                hf_model=None,
                wrap_cfg=ConfigDict(
                    mode=mode,
                    input_sequence_length=input_sequence_length,
                    max_sequence_length=context_length,
                    max_pe_length=max_pe_length,
                    dtype="float16",
                    batch_size=1,
                ),
                quant_config=_build_spec_draft_quant_config(draft_head_weight_bits),
                export_cfg=ConfigDict(),
                dflash_model_dir=dflash_model_dir,
                target_model_dir=hf_model_path,
            )
            draft_model = MODELS.build(model_cfg)
            draft_model.init_wrap_model()
            logger.info(
                f"[{name_suffix}] DFlash draft params: "
                f"{sum(p.numel() for p in draft_model._wrap_model.parameters()) / 1e6:.1f}M"
            )
            dummy_data = draft_model.prepare_inputs(None)
            draft_model.convert_to_fronted_graph(dummy_data)
            draft_model.convert_to_quant_graph(target_device)
            ptq_quantize(
                draft_model._quanted_model,
                [draft_model.prepare_inputs(None)],
                PrecisionMode.ALIGNED,
                [torch.device("cpu")],
            )
            draft_model.convert_to_export_graph(dummy_data)
            onnx_file = draft_model.to_export_onnx(dummy_data, str(draft_onnx_dir), f"{prefix}_{name_suffix}")[0]
            draft_model.release_exported_model()
            draft_model.release_quanted_model()
            draft_model.release_frontend_model()
            draft_model.release_wraped_model()
            del draft_model
            return onnx_file

        return {
            "draft_context_onnx": _export_one("context", self.config.input_sequence_length, "dflash_context"),
            "draft_context_decode_onnx": _export_one("context", draft_decode_seq_len, "dflash_context_decode"),
            "draft_decode_onnx": _export_one(
                "decode",
                draft_decode_seq_len,
                "dflash_decode",
            ),
        }

    @staticmethod
    def _resolve_existing_meta_path(existing_work_dir: Path, meta_info: Dict[str, Any], *keys: str) -> Path:
        for key in keys:
            path_value = meta_info.get(key)
            if path_value:
                path = Path(str(path_value))
                if not path.is_absolute():
                    path = (existing_work_dir / path).resolve()
                if not path.exists():
                    raise FileNotFoundError(f"Resolved {key} does not exist: {path}")
                return path
        raise FileNotFoundError(f"None of {keys} found in {existing_work_dir / 'meta.json'}")

    def export_draft_only(self, hf_model_path: str, existing_work_dir: str, output_dir: str):
        logger = get_root_logger()
        existing_work_dir_path = Path(existing_work_dir).resolve()
        work_dir = Path(output_dir).resolve()
        meta_path = existing_work_dir_path / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found in existing work_dir: {meta_path}")

        with meta_path.open("r", encoding="utf-8") as file:
            existing_meta = json.load(file)

        spec_decode_mode = getattr(self.config, "spec_decode_mode", None)
        if spec_decode_mode not in {"mtp", "dflash"}:
            raise ValueError("draft-only MoE export requires spec_decode_mode to be one of {'mtp', 'dflash'}")
        if spec_decode_mode == "dflash" and not getattr(self.config, "dflash_model_dir", None):
            raise ValueError("dflash_model_dir is required when spec_decode_mode='dflash'")

        work_dir.mkdir(exist_ok=True, parents=True)
        hf_model_path = str(existing_meta.get("hf_model_path") or hf_model_path)
        context_length = int(existing_meta.get("max_context_tokens", self.config.context_length))
        wrap_cfg = existing_meta.get("wrap_cfg", {})
        if isinstance(wrap_cfg, dict) and wrap_cfg.get("input_sequence_length") is not None:
            self.config.input_sequence_length = int(wrap_cfg["input_sequence_length"])
        self.config.context_length = context_length

        quant_scheme = existing_meta.get("quant_scheme", {})
        quant_type = (
            quant_scheme.get("quant_type", self.config.quant_scheme.quant_type)
            if isinstance(quant_scheme, dict)
            else self.config.quant_scheme.quant_type
        )
        target_device = self.config.quant_scheme.target_device
        model_name = str(existing_meta.get("model_name") or Path(hf_model_path).name)
        prefix = f"{model_name}-{target_device}-{context_length // 1024}k-{quant_type}"
        verify_length = int(getattr(self.config, "num_draft_tokens", 4)) + 1

        logger.info(f"Draft-only MoE export: reusing target work_dir={existing_work_dir_path}")
        prefill_onnx = self._resolve_existing_meta_path(
            existing_work_dir_path, existing_meta, "prefill_onnx", "prefill_onnx_file"
        )
        decode_onnx = self._resolve_existing_meta_path(
            existing_work_dir_path, existing_meta, "decode_onnx", "decode_onnx_file"
        )
        hf_config = self._resolve_existing_meta_path(existing_work_dir_path, existing_meta, "hf_config")
        token_embedding = self._resolve_existing_meta_path(
            existing_work_dir_path, existing_meta, "token_embedding_file"
        )
        logger.info(f"Reused target prefill ONNX: {prefill_onnx}")
        logger.info(f"Reused target decode ONNX: {decode_onnx}")

        if spec_decode_mode == "mtp":
            draft_onnx_files = self._export_mtp_draft_model(
                hf_model_path=hf_model_path,
                work_dir=work_dir,
                prefix=prefix,
                context_length=context_length,
                verify_length=verify_length,
                target_device=target_device,
            )
            hidden_output_name = "post_norm_hidden"
        else:
            draft_onnx_files = self._export_dflash_draft_model(
                hf_model_path=hf_model_path,
                dflash_model_dir=self.config.dflash_model_dir,
                work_dir=work_dir,
                prefix=prefix,
                context_length=context_length,
                verify_length=verify_length,
                target_device=target_device,
            )
            hidden_output_name = "target_hidden"

        new_meta = copy.deepcopy(existing_meta)
        new_meta.update(
            dict(
                create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                hf_model_path=hf_model_path,
                prefill_onnx=str(prefill_onnx),
                decode_onnx=str(decode_onnx),
                hf_config=str(hf_config),
                token_embedding_file=str(token_embedding),
                spec_decode_mode=spec_decode_mode,
                spec_decode_block_size=int(getattr(self.config, "num_draft_tokens", 4)),
                spec_decode_hidden_output_name=hidden_output_name,
                spec_decode_verify_length=verify_length,
                spec_decode_draft_head_weight_bits=int(getattr(self.config, "spec_draft_head_weight_bits", 4)),
            )
        )
        for key, value in draft_onnx_files.items():
            new_meta[f"{key}_file"] = str(Path(value).relative_to(work_dir))
        if "draft_decode_onnx" in draft_onnx_files:
            new_meta["draft_onnx_file"] = str(Path(draft_onnx_files["draft_decode_onnx"]).relative_to(work_dir))

        with (work_dir / "meta.json").open("w", encoding="utf-8") as fout:
            json.dump(new_meta, fout, ensure_ascii=False, indent=4)
        with (work_dir / "export_meta_info.json").open("w", encoding="utf-8") as fout:
            json.dump(new_meta, fout, ensure_ascii=False, indent=4)
        logger.info(f"Draft-only MoE export done. New artifacts in: {work_dir}")

    @classmethod
    def convert(cls, hf_model_path: str, config: Qwen3_5MoeConvertConfig, output_dir: str):
        Qwen3_5MoeConverterXH2a(config)._convert(hf_model_path, output_dir)
