from typing import Any

from qwen_vl_utils import process_vision_info
from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor

from xhmodel_merak.configuration_utils import BaseConfig


class Qwen2VLProcessorConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.patch_size: int | None = None
        self.max_size_h: int | None = None
        self.max_size_w: int | None = None


class XHQwen2VLProcessor(Qwen2VLProcessor):
    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
            **kwargs,
        )
        self.config = Qwen2VLProcessorConfig()

    def _set_fixed_visual_size(self, messages: list[dict[str, Any]]) -> None:
        for message in messages:
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                if (
                    "image" in item
                    or "image_url" in item
                    or "video" in item
                    or item.get("type", "text") in ("image", "image_url", "video")
                ):
                    item["resized_height"] = self.config.max_size_h
                    item["resized_width"] = self.config.max_size_w

    def apply_chat_template(self, messages: list[dict[str, Any]], enable_thinking: bool = False):
        text = super().apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        self._set_fixed_visual_size(messages)
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=self.config.patch_size)
        self.image_processor.max_pixels = max(
            self.config.max_size_w * self.config.max_size_h + 1,
            self.image_processor.max_pixels,
        )
        model_inputs = self(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if "pixel_values" in model_inputs:
            model_inputs["hm_pixel_values"] = [model_inputs["pixel_values"].contiguous().float()]
        return model_inputs
