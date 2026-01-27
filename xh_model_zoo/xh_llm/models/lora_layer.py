# Copyright 2025 HOUMO AI
#
# File: lora_layer.py
# Description:
#   LoRA (Low-Rank Adaptation) layer implementations.
#   This module provides LoRALinear and LoRALayer classes for
#   efficient parameter fine-tuning.
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
from copy import deepcopy
from typing import Dict

import torch
import torch.fx as fx
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
from xhquant.api import FrontendGraph, get_xhquant_logger
from xhquant.frontend.torchfx import xh_fx
from xhquant.nn import FX_LEAF_MODULES


@FX_LEAF_MODULES.register_module()
class LoRALinear(nn.Linear):
    _is_lora_: bool = True


class LoRALayer(nn.Module):
    def __init__(self, linear: nn.Module, lora_A_weight, lora_B_weight, scaling=None):
        super().__init__()
        # 获取维度
        r, in_dim = lora_A_weight.shape
        out_dim, r_check = lora_B_weight.shape
        assert r == r_check, "Rank dimension mismatch"

        self.scaling = scaling / r if scaling is not None else None

        # 定义层 A: (in -> r)
        self.lora_A = LoRALinear(in_dim, r, bias=False)

        # 定义层 B: (r -> out)
        self.lora_B = LoRALinear(r, out_dim, bias=False)

        self.linear = deepcopy(linear)  # 原始线性层

        # 加载权重
        with torch.no_grad():
            self.lora_A.weight.copy_(lora_A_weight)
            self.lora_B.weight.copy_(lora_B_weight)

    def forward(self, x, lora_mask: Tensor) -> Tensor:
        # 典型的 LoRA 前向传播: B(A(x)) * scale
        src = self.linear(x)
        lora = self.lora_A(x)
        lora = self.lora_B(lora)
        scale = lora_mask
        if self.scaling is not None:
            scale = scale * self.scaling
        out = src + lora * scale
        return out


def replace_linear_with_lora(
    fronted_model: FrontendGraph,
    node: fx.Node,
    lora_graph_module: fx.GraphModule,
    lora_mask_node: fx.Node,
) -> FrontendGraph:
    matched_placeholders = [node.args[0], lora_mask_node]
    replacement_placeholders = [n for n in lora_graph_module.graph.nodes if n.op == "placeholder"]
    assert len(matched_placeholders) == len(replacement_placeholders)

    val_map: Dict[fx.Node, fx.Node] = {}
    for matched, replacement in zip(matched_placeholders, replacement_placeholders):
        val_map[replacement] = matched

    lora_linear_m_target = f"{node.target}_lora"
    fronted_model.add_submodule(lora_linear_m_target, nn.Module())
    last_insert_node = node
    for replace_node in lora_graph_module.graph.nodes:
        if replace_node.op in ["placeholder"]:
            continue
        if replace_node.op == "output":
            replaced_out_node = replace_node.args[0]
            continue
        args = fx.map_arg(replace_node.args, lambda n: val_map[n])
        kwargs = fx.map_arg(replace_node.kwargs, lambda n: val_map[n])
        with fronted_model.graph.inserting_after(last_insert_node):
            if replace_node.op == "call_module":
                sub_m = lora_graph_module.get_submodule(replace_node.target)
                new_target = f"{lora_linear_m_target}.{replace_node.target}"
                replace_node.target = new_target
                fronted_model.add_submodule(new_target, sub_m)
                new_node = fronted_model.graph.call_module(new_target, args, kwargs)
                val_map[replace_node] = new_node
            elif replace_node.op == "call_function":
                new_node = fronted_model.graph.call_function(replace_node.target, args, kwargs)
                val_map[replace_node] = new_node
            else:
                raise NotImplementedError(f"Unsupported op type: {replace_node.op}")
            last_insert_node = new_node
    node.replace_all_uses_with(val_map[replaced_out_node])
    fronted_model.graph.lint()
    fronted_model.graph.eliminate_dead_code()
    val_map[replaced_out_node].name = node.name
    fronted_model.recompile()
    return fronted_model


def apply_lora_to_linear(fronted_model: FrontendGraph, inputs, lora_scale: float = None) -> FrontendGraph:
    logger = get_xhquant_logger()
    lora_nodes: Dict[str, fx.Node] = {}
    for node in fronted_model.graph.nodes:
        if node.op == "call_module":
            m = fronted_model.get_submodule(node.target)
            if isinstance(m, nn.Linear):
                if hasattr(m, "weight_lora_a") and hasattr(m, "weight_lora_b"):
                    if str(node.target) in lora_nodes:
                        assert False, f"Duplicate LoRA module found: {node.target}"
                    lora_nodes.setdefault(str(node.target), node)

    has_lora = len(lora_nodes) > 0
    if not has_lora:
        return fronted_model
    last_placeholder_node = None
    for node in fronted_model.graph.nodes:
        if node.op == "placeholder":
            assert node.name != "lora_mask"
            last_placeholder_node = node

    assert last_placeholder_node is not None, "No placeholder node found in the graph"
    with fronted_model.graph.inserting_after(last_placeholder_node):
        lora_mask_node = fronted_model.graph.placeholder("lora_mask", default_value=1.0)
    # inputs.append(torch.tensor([1.0]))  # 输入增加lora_mask

    logger.info("********************** apply lora **********************")
    pbar = tqdm(list(lora_nodes.values()))
    for node in pbar:
        m = fronted_model.get_submodule(node.target)
        lora_a = m.weight_lora_a
        lora_b = m.weight_lora_b

        if lora_a.shape[0] != lora_b.shape[1]:
            lora_a = lora_a.T
            lora_b = lora_b.T
        pbar.set_description(f"Applying lora for module {node.target}, {lora_a.shape}, {lora_b.shape}")
        lora_layer = LoRALayer(m, lora_a, lora_b, scaling=lora_scale)
        lora_graph_module = xh_fx.symbolic_trace(lora_layer)
        replace_linear_with_lora(fronted_model, node, lora_graph_module, lora_mask_node)
        fronted_model.cpu()
