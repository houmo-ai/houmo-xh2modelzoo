from contextlib import contextmanager

import torch.nn as nn

from .float_checkpoint_compat import fuse_split_experts, split_expert_checkpoint_loader


def register_laguna_gptqmodel() -> bool:
    """Register Laguna's split-expert module tree with GPTQModel when available."""

    try:
        from gptqmodel.models import auto as gptq_auto
        from gptqmodel.models.base import BaseQModel
        from transformers import AutoModelForCausalLM
    except ImportError:
        return False

    if gptq_auto.MODEL_MAP.get("laguna") is not None:
        return True

    class LagunaQModel(BaseQModel):
        loader = AutoModelForCausalLM
        require_trust_remote_code = True
        require_fast_init = False
        layer_modules_strict = False
        dynamic_expert_index = "num_experts"
        pre_lm_head_norm_module = "model.norm"

        module_tree = [
            "model",
            "layers",
            "#",
            {
                "input_layernorm": ("input_layernorm:!",),
                "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "g_proj:0", "o_proj:1"),
                "post_attention_layernorm": ("post_attention_layernorm:!",),
                "mlp": {
                    "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                    "gate": ("gate:!",),
                    "shared_expert": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                    "experts": {
                        "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                    },
                },
            },
        ]

    supported_models = gptq_auto.SUPPORTED_MODELS
    if isinstance(supported_models, set):
        supported_models.add("laguna")
    elif "laguna" not in supported_models:
        supported_models.append("laguna")
    gptq_auto.MODEL_MAP["laguna"] = LagunaQModel
    return True


@contextmanager
def laguna_gptqmodel_loader(model_dir: str):
    """Expose split experts while GPTQModel constructs or quantizes Laguna."""

    if not register_laguna_gptqmodel():
        raise ImportError("Laguna GPTQModel support requires the optional gptqmodel package")
    with split_expert_checkpoint_loader(model_dir) as native_experts_cls:
        yield native_experts_cls


def restore_gptqmodel_moe_structure(
    model: nn.Module,
    model_dir: str,
    native_experts_cls: type[nn.Module] | None = None,
) -> int:
    """Restore dequantized GPTQModel split experts to Laguna's fused layout."""

    if native_experts_cls is None:
        with split_expert_checkpoint_loader(model_dir) as native_experts_cls:
            pass
    return fuse_split_experts(model, native_experts_cls)


__all__ = [
    "laguna_gptqmodel_loader",
    "register_laguna_gptqmodel",
    "restore_gptqmodel_moe_structure",
]