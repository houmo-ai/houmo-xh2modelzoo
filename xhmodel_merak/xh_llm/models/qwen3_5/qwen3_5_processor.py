from qwen_vl_utils import process_vision_info

from xhmodel_merak.configuration_utils import BaseConfig

from .processing_qwen3_5 import Qwen3_5Processor
from .visual_token_gears import VISUAL_INPUT_PATCHES


class Qwen3_5VLProcessorConfig(BaseConfig):  # noqa: N801
    def __init__(self):
        super().__init__()
        self.patch_size: int = 16
        self.visual_input_mode: str = VISUAL_INPUT_PATCHES
        self.max_pixels: int | None = None


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

        image_inputs, video_inputs = process_vision_info(
            messages,
            image_patch_size=self.config.patch_size,
        )
        if self.config.max_pixels is not None:
            self.image_processor.max_pixels = int(self.config.max_pixels)

        model_inputs = self(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if "hm_pixel_values" in model_inputs:
            model_inputs.pop("hm_pixel_values")
        return model_inputs
