"""Qwen3.5-MoE vision wrappers shared with the dense Qwen3.5 ViT.

The dense and MoE language models use different Transformers vision class
identities, but their vision tower implementation and weights have the same
contract. Register the MoE identities against the common patch-token wrappers
so dense and MoE gear exports cannot drift apart.
"""

from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeVisionAttention,
    Qwen3_5MoeVisionBlock,
    Qwen3_5MoeVisionModel,
    Qwen3_5MoeVisionPatchEmbed,
    Qwen3_5MoeVisionPatchMerger,
)

from ...register import XHLLM_TRACEABLE_MODULES
from ..qwen3_5._vision_model_impl import (
    _Qwen3_5VisionAttention,
    _Qwen3_5VisionBlock,
    _Qwen3_5VisionModel,
    _Qwen3_5VisionPatchEmbed,
    _Qwen3_5VisionPatchMerger,
)


@XHLLM_TRACEABLE_MODULES.register_module(
    {Qwen3_5MoeVisionAttention: "Qwen3_5MoeVisionAttention"}
)
class _Qwen3_5MoeVisionAttention(_Qwen3_5VisionAttention):  # noqa: N801
    pass


@XHLLM_TRACEABLE_MODULES.register_module(
    {Qwen3_5MoeVisionBlock: "Qwen3_5MoeVisionBlock"}
)
class _Qwen3_5MoeVisionBlock(_Qwen3_5VisionBlock):  # noqa: N801
    pass


@XHLLM_TRACEABLE_MODULES.register_module(
    {Qwen3_5MoeVisionPatchEmbed: "Qwen3_5MoeVisionPatchEmbed"}
)
class _Qwen3_5MoeVisionPatchEmbed(_Qwen3_5VisionPatchEmbed):  # noqa: N801
    pass


@XHLLM_TRACEABLE_MODULES.register_module(
    {Qwen3_5MoeVisionPatchMerger: "Qwen3_5MoeVisionPatchMerger"}
)
class _Qwen3_5MoeVisionPatchMerger(_Qwen3_5VisionPatchMerger):  # noqa: N801
    pass


@XHLLM_TRACEABLE_MODULES.register_module(
    {Qwen3_5MoeVisionModel: "Qwen3_5MoeVisionModel"}
)
class _Qwen3_5MoeVisionModel(_Qwen3_5VisionModel):  # noqa: N801
    pass


def register_wrap_cls(hf_model):
    """Import side effects above register all MoE vision wrapper classes."""
