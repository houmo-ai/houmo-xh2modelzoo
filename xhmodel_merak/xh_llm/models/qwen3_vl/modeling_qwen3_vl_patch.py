from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLTextAttention,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextMLP,
    Qwen3VLTextModel,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionAttention,
    Qwen3VLVisionBlock,
    Qwen3VLVisionMLP,
    Qwen3VLVisionModel,
    Qwen3VLVisionPatchEmbed,
    Qwen3VLVisionPatchMerger,
    Qwen3VLVisionRotaryEmbedding,
)

from xhquant.api import get_xhquant_logger


def qwen3_vl_patch(hf_model: Qwen3VLForConditionalGeneration) -> Qwen3VLForConditionalGeneration:
    from .modeling_qwen3_vl import Qwen3VLForConditionalGeneration as XHQwen3VLForConditionalGeneration
    from .modeling_qwen3_vl import Qwen3VLModel as XHQwen3VLModel
    from .modeling_qwen3_vl import Qwen3VLTextAttention as XHQwen3VLTextAttention
    from .modeling_qwen3_vl import Qwen3VLTextDecoderLayer as XHQwen3VLTextDecoderLayer
    from .modeling_qwen3_vl import Qwen3VLTextMLP as XHQwen3VLTextMLP
    from .modeling_qwen3_vl import Qwen3VLTextModel as XHQwen3VLTextModel
    from .modeling_qwen3_vl import Qwen3VLTextRMSNorm as XHQwen3VLTextRMSNorm
    from .modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding as XHQwen3VLTextRotaryEmbedding
    from .modeling_qwen3_vl import Qwen3VLVisionAttention as XHQwen3VLVisionAttention
    from .modeling_qwen3_vl import Qwen3VLVisionBlock as XHQwen3VLVisionBlock
    from .modeling_qwen3_vl import Qwen3VLVisionMLP as XHQwen3VLVisionMLP
    from .modeling_qwen3_vl import Qwen3VLVisionModel as XHQwen3VLVisionModel
    from .modeling_qwen3_vl import Qwen3VLVisionPatchEmbed as XHQwen3VLVisionPatchEmbed
    from .modeling_qwen3_vl import Qwen3VLVisionPatchMerger as XHQwen3VLVisionPatchMerger
    from .modeling_qwen3_vl import Qwen3VLVisionRotaryEmbedding as XHQwen3VLVisionRotaryEmbedding

    patch_mapping = {
        Qwen3VLForConditionalGeneration: XHQwen3VLForConditionalGeneration,
        Qwen3VLModel: XHQwen3VLModel,
        Qwen3VLTextModel: XHQwen3VLTextModel,
        Qwen3VLVisionModel: XHQwen3VLVisionModel,
        Qwen3VLVisionBlock: XHQwen3VLVisionBlock,
        Qwen3VLVisionAttention: XHQwen3VLVisionAttention,
        Qwen3VLTextRotaryEmbedding: XHQwen3VLTextRotaryEmbedding,
        Qwen3VLVisionPatchEmbed: XHQwen3VLVisionPatchEmbed,
        Qwen3VLVisionMLP: XHQwen3VLVisionMLP,
        Qwen3VLVisionRotaryEmbedding: XHQwen3VLVisionRotaryEmbedding,
        Qwen3VLVisionPatchMerger: XHQwen3VLVisionPatchMerger,
        Qwen3VLTextRMSNorm: XHQwen3VLTextRMSNorm,
        Qwen3VLTextAttention: XHQwen3VLTextAttention,
        Qwen3VLTextMLP: XHQwen3VLTextMLP,
        Qwen3VLTextDecoderLayer: XHQwen3VLTextDecoderLayer,
    }
    logger = get_xhquant_logger()
    for name, m in hf_model.named_modules():
        orig_cls = type(m)
        if orig_cls in patch_mapping:
            m.__class__ = patch_mapping[orig_cls]
        logger.debug(f"Convert {name}[{orig_cls}]")
    return hf_model
    # from accelerate import init_empty_weights

    # with init_empty_weights():
    #     hf_model_patch = XHQwen3VLForConditionalGeneration(hf_model.config)
    #     hf_model.cpu()
    # hf_model_patch.load_state_dict(hf_model.state_dict(), strict=False, assign=True)

    # return hf_model_patch
