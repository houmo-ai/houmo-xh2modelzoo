from typing import Optional

import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5GatedDeltaNet,
    Qwen3_5MLP,
    Qwen3_5Model,
    Qwen3_5RMSNorm,
    Qwen3_5RMSNormGated,
    Qwen3_5TextModel,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5VisionAttention,
    Qwen3_5VisionBlock,
    Qwen3_5VisionMLP,
    Qwen3_5VisionModel,
    Qwen3_5VisionPatchEmbed,
    Qwen3_5VisionPatchMerger,
    Qwen3_5VisionRotaryEmbedding,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from xhquant.api import get_xhquant_logger


def qwen3_5_patch(hf_model: Qwen3_5ForConditionalGeneration) -> Qwen3_5ForConditionalGeneration:
    from .modeling_qwen3_5 import Qwen3_5Attention as XHQwen3_5Attention
    from .modeling_qwen3_5 import Qwen3_5DecoderLayer as XHQwen3_5DecoderLayer
    from .modeling_qwen3_5 import Qwen3_5ForCausalLM as XHQwen3_5ForCausalLM
    from .modeling_qwen3_5 import Qwen3_5ForConditionalGeneration as XHQwen3_5ForConditionalGeneration
    from .modeling_qwen3_5 import Qwen3_5GatedDeltaNet as XHQwen3_5GatedDeltaNet
    from .modeling_qwen3_5 import Qwen3_5MLP as XHQwen3_5MLP
    from .modeling_qwen3_5 import Qwen3_5Model as XHQwen3_5Model
    from .modeling_qwen3_5 import Qwen3_5RMSNorm as XHQwen3_5RMSNorm
    from .modeling_qwen3_5 import Qwen3_5RMSNormGated as XHQwen3_5RMSNormGated
    from .modeling_qwen3_5 import Qwen3_5TextModel as XHQwen3_5TextModel
    from .modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding as XHQwen3_5TextRotaryEmbedding
    from .modeling_qwen3_5 import Qwen3_5VisionAttention as XHQwen3_5VisionAttention
    from .modeling_qwen3_5 import Qwen3_5VisionBlock as XHQwen3_5VisionBlock
    from .modeling_qwen3_5 import Qwen3_5VisionMLP as XHQwen3_5VisionMLP
    from .modeling_qwen3_5 import Qwen3_5VisionModel as XHQwen3_5VisionModel
    from .modeling_qwen3_5 import Qwen3_5VisionPatchEmbed as XHQwen3_5VisionPatchEmbed
    from .modeling_qwen3_5 import Qwen3_5VisionPatchMerger as XHQwen3_5VisionPatchMerger
    from .modeling_qwen3_5 import Qwen3_5VisionRotaryEmbedding as XHQwen3_5VisionRotaryEmbedding

    patch_mapping = {
        Qwen3_5ForConditionalGeneration: XHQwen3_5ForConditionalGeneration,
        Qwen3_5Attention: XHQwen3_5Attention,
        Qwen3_5Model: XHQwen3_5Model,
        Qwen3_5TextModel: XHQwen3_5TextModel,
        Qwen3_5VisionModel: XHQwen3_5VisionModel,
        Qwen3_5DecoderLayer: XHQwen3_5DecoderLayer,
        Qwen3_5ForCausalLM: XHQwen3_5ForCausalLM,
        Qwen3_5GatedDeltaNet: XHQwen3_5GatedDeltaNet,
        Qwen3_5VisionAttention: XHQwen3_5VisionAttention,
        Qwen3_5VisionBlock: XHQwen3_5VisionBlock,
        Qwen3_5VisionMLP: XHQwen3_5VisionMLP,
        Qwen3_5VisionPatchEmbed: XHQwen3_5VisionPatchEmbed,
        Qwen3_5VisionPatchMerger: XHQwen3_5VisionPatchMerger,
        Qwen3_5VisionRotaryEmbedding: XHQwen3_5VisionRotaryEmbedding,
        Qwen3_5MLP: XHQwen3_5MLP,
        Qwen3_5RMSNorm: XHQwen3_5RMSNorm,
        Qwen3_5RMSNormGated: XHQwen3_5RMSNormGated,
        Qwen3_5TextRotaryEmbedding: XHQwen3_5TextRotaryEmbedding,
    }
    logger = get_xhquant_logger()
    for name, m in hf_model.named_modules():
        orig_cls = type(m)
        if orig_cls in patch_mapping:
            m.__class__ = patch_mapping[orig_cls]
        logger.debug(f"Convert {name}[{orig_cls}]")
    return hf_model


def get_image_features(
    self,
    pixel_values: list[torch.FloatTensor],
    image_grid_thw: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
):
    """
    Encodes images into continuous embeddings that can be forwarded to the language model.

    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input images.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
    """
    assert len(pixel_values) == 1
    assert isinstance(pixel_values, (tuple, list))
    pixel_values = pixel_values[0]
    pixel_values = pixel_values.type(self.visual.dtype)
    image_embeds = self.visual(pixel_values)
    vision_output = BaseModelOutputWithPooling(
        pooler_output=image_embeds,
    )
    if len(image_embeds.shape) == 3:
        image_embeds = image_embeds[0]
    split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
    image_embeds = torch.split(image_embeds, split_sizes)
    vision_output.pooler_output = image_embeds

    return vision_output
