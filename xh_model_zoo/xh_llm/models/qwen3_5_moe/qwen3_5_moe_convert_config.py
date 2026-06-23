# Copyright 2025 HOUMO AI
#
# File: qwen3_5_moe_convert_config.py
# Description:
#   Qwen3.5-MoE Convert Config configuration.
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

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ...llm_convert_config import LLMConvertConfig


@dataclass
class Qwen3_5MoeConvertConfig(LLMConvertConfig):
    mix_search: Optional[str] = None
    num_logits_to_keep: Optional[int] = 1
    linear_attention_mode: str = "auto"
    linear_chunk_size: int = 64
    enable_rope: bool = True
    max_pe_length: int = 262144
    support_long_context_over_fp16_limit: bool = True
    alpha_scaling_layers: List[int] = field(default_factory=lambda: [8, 20])
    chunk_inverse_alpha: float = 0.5
    cumsum_matmul_quant_config: Optional[Dict] = None
    # Speculative decoding
    spec_decode_mode: Optional[str] = None  # "mtp", "dflash", or None
    num_draft_tokens: int = 4
    dflash_model_dir: Optional[str] = None
    spec_draft_head_weight_bits: int = 4
    split_conv_cache: bool = True
    normalize_force_fp32: bool = False
    use_manual_depthwise_conv1d: bool = False
    fuse_gdr_ops: bool = False
    output_hidden_state_indices: Optional[List[int]] = None
    output_post_norm_hidden: bool = False
    gptq_restore_expert_layer_spec: Optional[str] = None
    export_prefill: bool = True
