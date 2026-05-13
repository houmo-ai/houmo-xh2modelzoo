import gc
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)



def resolve_native_glm_moe_cls() -> type[nn.Module]:
    import transformers.models.glm4_moe_lite.modeling_glm4_moe_lite as glm_module

    return glm_module.Glm4MoeLiteNaiveMoe


# =============================================================================
# ACTIVE PATH
# 该函数仍然需要：GLM 通过 XHBaseModel._postprocess_gptqmodel_structure()
# 在公共 GPTQModel 反量化后调用它，完成 split-MoE 到 fused-MoE 的结构归一。
# =============================================================================
def convert_gptqmodel_moe_structure(
    hf_model: nn.Module,
    target_moe_cls: type[nn.Module] | None = None,
) -> int:
    # XHGlm4MoeLiteModel.get_hf_model 的结构归一化步骤。
    # 输入前提：base_model.py 中的公共 GPTQModel 反量化逻辑已经把专家 qlinear 转成 nn.Linear。
    # 输出契约：恢复为原生 Glm4MoeLiteNaiveMoe，且暴露 gate_up_proj/down_proj 参数供 Merak wrapper 使用。
    if target_moe_cls is None:
        target_moe_cls = resolve_native_glm_moe_cls()

    def _is_split_layout_module(module: nn.Module) -> bool:
        return (
            hasattr(module, "gate_proj")
            and isinstance(getattr(module, "gate_proj"), nn.ModuleList)
            and hasattr(module, "up_proj")
            and isinstance(getattr(module, "up_proj"), nn.ModuleList)
            and hasattr(module, "down_proj")
            and isinstance(getattr(module, "down_proj"), nn.ModuleList)
        )

    modules_to_convert = [(name, module) for name, module in hf_model.named_modules() if _is_split_layout_module(module)]

    for name, module in modules_to_convert:
        gate_modules: nn.ModuleList = getattr(module, "gate_proj")
        up_modules: nn.ModuleList = getattr(module, "up_proj")
        down_modules: nn.ModuleList = getattr(module, "down_proj")
        num_experts = len(gate_modules)
        logger.info("Converting GPTQModel split MoE to fused structure: %s (%d experts)", name, num_experts)

        ref = gate_modules[0].weight
        intermediate = ref.shape[0]
        hidden = ref.shape[1]
        dtype = ref.dtype
        device = ref.device
        has_gate_quant = all(getattr(gate_modules[i], "quant_weight", None) is not None for i in range(num_experts))
        has_up_quant = all(getattr(up_modules[i], "quant_weight", None) is not None for i in range(num_experts))
        has_down_quant = all(getattr(down_modules[i], "quant_weight", None) is not None for i in range(num_experts))

        offload_to_cpu = device.type == "cuda"
        if offload_to_cpu:
            gate_src = [gate_modules[i].weight.detach().to(device="cpu", copy=True) for i in range(num_experts)]
            up_src = [up_modules[i].weight.detach().to(device="cpu", copy=True) for i in range(num_experts)]
            down_src = [down_modules[i].weight.detach().to(device="cpu", copy=True) for i in range(num_experts)]
            gate_quant_src = (
                [gate_modules[i].quant_weight.detach().to(device="cpu", copy=True) for i in range(num_experts)]
                if has_gate_quant
                else None
            )
            up_quant_src = (
                [up_modules[i].quant_weight.detach().to(device="cpu", copy=True) for i in range(num_experts)]
                if has_up_quant
                else None
            )
            down_quant_src = (
                [down_modules[i].quant_weight.detach().to(device="cpu", copy=True) for i in range(num_experts)]
                if has_down_quant
                else None
            )
            del module._modules["gate_proj"]
            del module._modules["up_proj"]
            del module._modules["down_proj"]
            del gate_modules
            del up_modules
            del down_modules
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            gate_src = [gate_modules[i].weight.data for i in range(num_experts)]
            up_src = [up_modules[i].weight.data for i in range(num_experts)]
            down_src = [down_modules[i].weight.data for i in range(num_experts)]
            gate_quant_src = [gate_modules[i].quant_weight.data for i in range(num_experts)] if has_gate_quant else None
            up_quant_src = [up_modules[i].quant_weight.data for i in range(num_experts)] if has_up_quant else None
            down_quant_src = [down_modules[i].quant_weight.data for i in range(num_experts)] if has_down_quant else None

        gate_up_proj = torch.empty(num_experts, 2 * intermediate, hidden, dtype=dtype, device=device)
        down_proj = torch.empty(num_experts, hidden, intermediate, dtype=dtype, device=device)
        gate_up_proj_quant_weight = None
        if has_gate_quant and has_up_quant:
            gate_quant_ref = gate_quant_src[0]
            gate_up_proj_quant_weight = torch.empty(
                num_experts,
                2 * intermediate,
                hidden,
                dtype=gate_quant_ref.dtype,
                device=device,
            )
        down_proj_quant_weight = None
        if has_down_quant:
            down_quant_ref = down_quant_src[0]
            down_proj_quant_weight = torch.empty(
                num_experts,
                hidden,
                intermediate,
                dtype=down_quant_ref.dtype,
                device=device,
            )
        with torch.no_grad():
            for i in range(num_experts):
                gate_w = gate_src[i]
                up_w = up_src[i]
                down_w = down_src[i]
                if gate_w.device != device or gate_w.dtype != dtype:
                    gate_w = gate_w.to(device=device, dtype=dtype)
                if up_w.device != device or up_w.dtype != dtype:
                    up_w = up_w.to(device=device, dtype=dtype)
                if down_w.device != device or down_w.dtype != dtype:
                    down_w = down_w.to(device=device, dtype=dtype)
                gate_up_proj[i, :intermediate, :].copy_(gate_w)
                gate_up_proj[i, intermediate:, :].copy_(up_w)
                down_proj[i].copy_(down_w)
                if gate_up_proj_quant_weight is not None and gate_quant_src is not None and up_quant_src is not None:
                    gate_q = gate_quant_src[i]
                    up_q = up_quant_src[i]
                    if gate_q.device != device:
                        gate_q = gate_q.to(device=device)
                    if up_q.device != device:
                        up_q = up_q.to(device=device)
                    gate_up_proj_quant_weight[i, :intermediate, :].copy_(gate_q)
                    gate_up_proj_quant_weight[i, intermediate:, :].copy_(up_q)
                if down_proj_quant_weight is not None and down_quant_src is not None:
                    down_q = down_quant_src[i]
                    if down_q.device != device:
                        down_q = down_q.to(device=device)
                    down_proj_quant_weight[i].copy_(down_q)

        if not offload_to_cpu:
            del module._modules["gate_proj"]
            del module._modules["up_proj"]
            del module._modules["down_proj"]
        else:
            del gate_src
            del up_src
            del down_src
            if gate_quant_src is not None:
                del gate_quant_src
            if up_quant_src is not None:
                del up_quant_src
            if down_quant_src is not None:
                del down_quant_src

        module.register_parameter("gate_up_proj", nn.Parameter(gate_up_proj, requires_grad=False))
        module.register_parameter("down_proj", nn.Parameter(down_proj, requires_grad=False))
        module._buffers.pop("gate_up_proj_quant_weight", None)
        module._buffers.pop("down_proj_quant_weight", None)
        if gate_up_proj_quant_weight is not None:
            module.register_buffer("gate_up_proj_quant_weight", gate_up_proj_quant_weight)
        if down_proj_quant_weight is not None:
            module.register_buffer("down_proj_quant_weight", down_proj_quant_weight)
        module.__class__ = target_moe_cls

    return len(modules_to_convert)
