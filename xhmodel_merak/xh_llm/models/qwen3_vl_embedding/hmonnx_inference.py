from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from xhmodel_merak.xh_llm.infer_mixin import (
    LLMInferenceContextManager,
)
from xhmodel_merak.xh_llm.hmonnx.hmonnx_model import (
    HMONNXBaseModel,
    HMONNXModel,
)
from xhmodel_merak.xh_llm.kv_cache_mixin import KVCacheMixin
from xhmodel_merak.xh_llm.models.qwen3_vl import (
    XHQwen3VLModel,
)
from xhmodel_merak.xh_llm.models.qwen3_vl.qwen3_vl_hmonnx_inference import (
    VisualHMONNXModel,
    XHQwen3VLHMONNXModel,
)
from xhmodel_merak.xh_llm.types import (
    KVCacheConfig,
    LLMModelMeta,
)


class XHQwen3VLEmbeddingHMONNXModel(XHQwen3VLHMONNXModel):
    LLM_MODEL_CLS = XHQwen3VLModel

    def __init__(
        self,
        meta_info: LLMModelMeta,
        enable_cuda_graph: bool = False,
        enable_auto_offload: bool = False,
        enable_golden: bool = False,
        enable_page_attention: bool = False,
        **kwargs: Any,
    ):
        if enable_page_attention:
            raise ValueError(
                "Qwen3-VL-Embedding is Prefill-only and does not "
                "support page attention"
            )

        HMONNXBaseModel.__init__(self, **kwargs)
        self.meta_info = meta_info
        self.hf_model_dir = meta_info.hf_config
        self.hf_compatible_model = None
        self.embed_tokens = self._build_embed_tokens_from_meta(
            meta_info
        )
        self._llm_prefill = True
        self.kvcache_config = (
            meta_info.kv_cache
            if isinstance(meta_info.kv_cache, KVCacheConfig)
            else KVCacheConfig(**meta_info.kv_cache)
        )
        self.use_cache = self.kvcache_config.num_layers > 0
        self.enable_page_attention = False
        self.prefill_model = HMONNXModel(
            meta_info.prefill_hmonnx,
            enable_cuda_graph=enable_cuda_graph,
            enable_auto_offload=enable_auto_offload,
            enable_golden=enable_golden,
            device_map=self._valid_devices,
        )
        self.decode_model = None
        self._data_processor = None
        self._kvcache_mixin = KVCacheMixin(self.kvcache_config)
        self._sync_page_attention_mode_to_kvcache()
        self.pad_token_id = meta_info.pad_token_id

        self.visual_meta = meta_info.visual_config
        self.visual = VisualHMONNXModel(
            self.visual_meta.hmonnx,
            enable_auto_offload=enable_auto_offload,
            enable_golden=enable_golden,
            device_map=self._valid_devices,
        )

    def set_decode(self):
        raise RuntimeError(
            "Qwen3-VL-Embedding exports Prefill only; Decode is "
            "not available"
        )

    @staticmethod
    def _build_messages(item: Mapping[str, Any]) -> list[dict[str, Any]]:
        text = item.get("text")
        image = item.get("image")
        if text is None and image is None:
            raise ValueError(
                "Each embedding input must contain 'text' or 'image'"
            )

        content: list[dict[str, Any]] = []
        if image is not None:
            content.append({"type": "image", "image": image})
        if text is not None:
            content.append({"type": "text", "text": str(text)})
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Represent the user's input.",
                    }
                ],
            },
            {"role": "user", "content": content},
        ]

    def _extract_visual_features(
        self,
        model_inputs: Mapping[str, Any],
    ) -> tuple[torch.Tensor | None, list[torch.Tensor] | None]:
        pixel_values = model_inputs.get("hm_pixel_values")
        if pixel_values is None:
            return None, None

        if isinstance(pixel_values, torch.Tensor):
            pixel_values = [pixel_values]
        image_embeds = []
        deepstack_groups = []
        for pixel_values_i in pixel_values:
            image_embeds_i, deepstack_i = self.visual(
                pixel_values_i.to(
                    device=self.device,
                    dtype=self.dtype,
                )
            )
            image_embeds.append(image_embeds_i)
            deepstack_groups.append(deepstack_i)

        merged_image_embeds = torch.cat(image_embeds, dim=0).squeeze(0)
        merged_deepstack = [
            torch.cat(
                [group[layer_idx] for group in deepstack_groups],
                dim=0,
            ).squeeze(0)
            for layer_idx in range(3)
        ]
        return merged_image_embeds, merged_deepstack

    def embed(
        self,
        item: Mapping[str, Any],
        processor: Any = None,
    ) -> torch.Tensor:
        if not isinstance(item, Mapping):
            raise TypeError(
                "Embedding input must be a mapping with text/image fields"
            )
        if processor is None:
            processor = self.get_tf_processor()

        messages = self._build_messages(item)
        model_inputs = processor.apply_chat_template(messages)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is None:
            input_length = int(input_ids.shape[-1])
        else:
            input_length = int(attention_mask.sum().item())

        max_length = int(
            self.meta_info.model_config.prefill_chunk_length
        )
        if input_length > max_length:
            raise ValueError(
                f"Processed input length {input_length} exceeds the "
                f"exported prefill length {max_length}"
            )

        self.set_prefill()
        image_embeds, deepstack_image_embeds = (
            self._extract_visual_features(model_inputs)
        )
        data_prefill = {
            "input_ids": input_ids,
            "past_seq_length": 0,
            "image_grid_thw": model_inputs.get("image_grid_thw"),
            "image_embeds": image_embeds,
            "deepstack_image_embeds": deepstack_image_embeds,
        }
        data_processor = self.get_data_preprocessor()
        prefill_inputs = data_processor(data_prefill)
        hidden_states = self.forward(*prefill_inputs)
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(
                "Qwen3-VL-Embedding prefill must return a tensor"
            )
        if hidden_states.ndim != 3:
            raise ValueError(
                "Qwen3-VL-Embedding prefill must return a [B, T, H] "
                f"tensor, got shape {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[1] < input_length:
            raise ValueError(
                f"Prefill output length {hidden_states.shape[1]} is "
                f"shorter than input length {input_length}"
            )

        embedding = hidden_states[:, input_length - 1, :]
        embedding = F.normalize(embedding.float(), p=2, dim=-1)
        return embedding

    def embed_items(
        self,
        items: Sequence[Mapping[str, Any]],
        processor: Any = None,
    ) -> torch.Tensor:
        if not items:
            raise ValueError("At least one embedding input is required")
        if processor is None:
            processor = self.get_tf_processor()

        embeddings = []
        with LLMInferenceContextManager(self):
            for item in items:
                embeddings.append(
                    self.embed(
                        item,
                        processor=processor,
                    )
                )
        return torch.cat(embeddings, dim=0)

    def embed_texts(
        self,
        texts: Sequence[str],
        processor: Any = None,
    ) -> torch.Tensor:
        return self.embed_items(
            [{"text": text} for text in texts],
            processor=processor,
        )

    def embed_images(
        self,
        images: Sequence[Any],
        processor: Any = None,
    ) -> torch.Tensor:
        return self.embed_items(
            [{"image": image} for image in images],
            processor=processor,
        )
