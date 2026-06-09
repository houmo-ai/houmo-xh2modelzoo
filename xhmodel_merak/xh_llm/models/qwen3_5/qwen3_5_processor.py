from qwen_vl_utils import process_vision_info

from xhmodel_merak.configuration_utils import BaseConfig

from .processing_qwen3_5 import Qwen3_5Processor


class Qwen3_5VLProcessorConfig(BaseConfig):  # noqa: N801
    def __init__(self):
        super().__init__()
        self.patch_size: int = 16
        self.max_size_h: int = 224
        self.max_size_w: int = 224


class XHQwen3_5Processor(Qwen3_5Processor):  # noqa: N801
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        **kwargs,
    ):
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
            **kwargs,
        )
        self.config = Qwen3_5VLProcessorConfig()

    def apply_chat_template(self, messages: list[dict[str, str]], enable_thinking: bool = False):
        text = super().apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        for message in messages:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele.get("type", "text") in ("image", "image_url", "video")
                    ):
                        ele["resized_height"] = self.config.max_size_h
                        ele["resized_width"] = self.config.max_size_w

        image_inputs, video_inputs = process_vision_info(
            messages,
            image_patch_size=self.config.patch_size,
        )
        self.image_processor.max_pixels = max(
            self.config.max_size_w * self.config.max_size_h + 1, self.image_processor.max_pixels
        )

        model_inputs = self(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if "hm_pixel_values" in model_inputs:
            model_inputs["pixel_values"] = [model_inputs["hm_pixel_values"][0].half()]
            model_inputs.pop("hm_pixel_values")
        return model_inputs
