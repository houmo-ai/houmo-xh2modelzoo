"""Frontend LoRA layers and graph rewriting utilities for Merak LLM models."""

from __future__ import annotations

from typing import Any

import torch
import torch.fx as fx
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from xhquant.api import FrontendGraph, get_xhquant_logger
from xhquant.frontend.fx_transform import NormalizerTransform
from xhquant.frontend.torchfx import xh_fx
from xhquant.nn import FX_LEAF_MODULES


@FX_LEAF_MODULES.register_module()
class LoRALinear(nn.Linear):
    """Linear used by LoRA A/B branches so quantization can identify them."""

    _is_lora_: bool = True


class LoRALayer(nn.Module):
    """Runtime-mask LoRA wrapper preserving the complete base Linear."""

    def __init__(
        self,
        linear: nn.Linear,
        lora_a_weight: Tensor,
        lora_b_weight: Tensor,
        scaling: float | Tensor | None = None,
    ) -> None:
        super().__init__()
        rank, in_dim = lora_a_weight.shape
        out_dim, b_rank = lora_b_weight.shape
        if rank != b_rank:
            raise ValueError(f"LoRA rank mismatch: A={tuple(lora_a_weight.shape)}, B={tuple(lora_b_weight.shape)}")

        self.scaling = scaling / rank if scaling is not None else None
        if isinstance(self.scaling, Tensor):
            self.scaling = self.scaling.item()

        linear_device = linear.weight.device
        linear_dtype = linear.weight.dtype
        self.lora_A = LoRALinear(
            in_dim,
            rank,
            bias=False,
            device=linear_device,
            dtype=linear_dtype,
        )
        self.lora_B = LoRALinear(
            rank,
            out_dim,
            bias=False,
            device=linear_device,
            dtype=linear_dtype,
        )

        # Keep the original Linear object intact. Quantized HF models may
        # carry auxiliary buffers such as ``quant_weight`` which xhquant needs
        # when converting the base branch.
        self.linear = linear
        with torch.no_grad():
            self.lora_A.weight.copy_(lora_a_weight)
            self.lora_B.weight.copy_(lora_b_weight)

    def forward(self, x: Tensor, lora_mask: Tensor) -> Tensor:
        src = self.linear(x)
        lora = self.lora_B(self.lora_A(x))
        scale = lora_mask
        if self.scaling is not None:
            scale = scale * self.scaling
        return src + lora * scale


class LoRAStaticLayer(LoRALayer):
    """Static LoRA wrapper used by keep-LoRA HMONNX exports."""

    def forward(self, x: Tensor) -> Tensor:
        src = self.linear(x)
        lora = self.lora_B(self.lora_A(x))
        scale = self.scaling if self.scaling is not None else 1.0
        return src + lora * scale


def replace_linear_with_lora(
    fronted_model: FrontendGraph,
    node: fx.Node,
    lora_graph_module: fx.GraphModule,
    lora_mask_node: fx.Node | None = None,
) -> FrontendGraph:
    """Replace one Linear call node with the traced LoRA wrapper graph."""

    matched_placeholders = [node.args[0]]
    if lora_mask_node is not None:
        matched_placeholders.append(lora_mask_node)
    replacement_placeholders = [
        replacement_node for replacement_node in lora_graph_module.graph.nodes if replacement_node.op == "placeholder"
    ]
    if len(matched_placeholders) != len(replacement_placeholders):
        raise RuntimeError(
            "LoRA replacement placeholder mismatch: "
            f"target={len(matched_placeholders)}, replacement={len(replacement_placeholders)}"
        )

    val_map: dict[fx.Node, fx.Node] = dict(zip(replacement_placeholders, matched_placeholders))
    lora_target = f"{node.target}_lora"
    fronted_model.add_submodule(lora_target, nn.Module())
    last_insert_node = node
    replaced_out_node: fx.Node | None = None
    for replacement_node in lora_graph_module.graph.nodes:
        if replacement_node.op == "placeholder":
            continue
        if replacement_node.op == "output":
            replaced_out_node = replacement_node.args[0]
            continue

        args = fx.map_arg(replacement_node.args, lambda mapped_node: val_map[mapped_node])
        kwargs = fx.map_arg(replacement_node.kwargs, lambda mapped_node: val_map[mapped_node])
        with fronted_model.graph.inserting_after(last_insert_node):
            if replacement_node.op == "call_module":
                submodule = lora_graph_module.get_submodule(str(replacement_node.target))
                new_target = f"{lora_target}.{replacement_node.target}"
                fronted_model.add_submodule(new_target, submodule)
                new_node = fronted_model.graph.call_module(new_target, args, kwargs)
            elif replacement_node.op == "call_function":
                new_node = fronted_model.graph.call_function(replacement_node.target, args, kwargs)
            else:
                raise NotImplementedError(f"Unsupported LoRA replacement op: {replacement_node.op}")
            val_map[replacement_node] = new_node
            last_insert_node = new_node

    if replaced_out_node is None:
        raise RuntimeError("Traced LoRA wrapper graph has no output node")
    new_output = val_map[replaced_out_node]
    original_target = str(node.target)
    node.replace_all_uses_with(new_output)
    fronted_model.graph.eliminate_dead_code()
    fronted_model.graph.lint()

    # ``LoRALayer.linear`` reuses the original module. Remove its obsolete
    # registration after the original node is gone, leaving a single owner at
    # ``<target>_lora.linear`` and exporting its buffers only once.
    if not any(
        graph_node.op == "call_module" and graph_node.target == original_target
        for graph_node in fronted_model.graph.nodes
    ):
        fronted_model.delete_submodule(original_target)
    new_output.name = node.name
    fronted_model.recompile()
    return fronted_model


def apply_lora_to_linear(
    fronted_model: FrontendGraph,
    inputs: list[Any],
    lora_scale: float | Tensor | None = None,
    runtime_mask: bool = False,
) -> FrontendGraph:
    """Rewrite every Linear carrying ``weight_lora_a/b`` frontend buffers."""

    logger = get_xhquant_logger()
    lora_nodes: dict[str, fx.Node] = {}
    for node in fronted_model.graph.nodes:
        if node.op != "call_module":
            continue
        module = fronted_model.get_submodule(str(node.target))
        if not isinstance(module, nn.Linear):
            continue
        if not hasattr(module, "weight_lora_a") or not hasattr(module, "weight_lora_b"):
            continue
        target = str(node.target)
        if target in lora_nodes:
            raise ValueError(f"Duplicate LoRA module in frontend graph: {target}")
        lora_nodes[target] = node

    if not lora_nodes:
        return fronted_model

    last_placeholder_node: fx.Node | None = None
    for node in fronted_model.graph.nodes:
        if node.op == "placeholder":
            if node.name == "lora_mask":
                raise ValueError("Frontend graph already contains a lora_mask input")
            last_placeholder_node = node

    lora_mask_node: fx.Node | None = None
    if runtime_mask:
        if last_placeholder_node is None:
            raise RuntimeError("Cannot add lora_mask: frontend graph has no placeholder")
        with fronted_model.graph.inserting_after(last_placeholder_node):
            lora_mask_node = fronted_model.graph.placeholder("lora_mask", default_value=None)
        inputs.append(torch.tensor([1.0]))

    logger.info("********************** apply lora **********************")
    progress = tqdm(list(lora_nodes.values()))
    for node in progress:
        module = fronted_model.get_submodule(str(node.target))
        lora_a = module.weight_lora_a
        lora_b = module.weight_lora_b
        if lora_a.shape[0] != lora_b.shape[1]:
            lora_a = lora_a.T
            lora_b = lora_b.T
        progress.set_description(f"Applying lora for module {node.target}, {lora_a.shape}, {lora_b.shape}")
        if runtime_mask:
            lora_layer = LoRALayer(module, lora_a, lora_b, scaling=lora_scale)
        else:
            lora_layer = LoRAStaticLayer(module, lora_a, lora_b, scaling=lora_scale)
        lora_graph_module = xh_fx.symbolic_trace(lora_layer)
        replace_linear_with_lora(fronted_model, node, lora_graph_module, lora_mask_node)
        fronted_model.cpu()

    # Turn raw add/mul call_function nodes into supported frontend modules.
    # ScalarMul stores scale as an independent buffer; A/B remain unchanged.
    return NormalizerTransform()(fronted_model)
