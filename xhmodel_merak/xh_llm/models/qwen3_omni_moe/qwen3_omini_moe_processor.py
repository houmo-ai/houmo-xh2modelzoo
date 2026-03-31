import torch
from qwen_omni_utils import process_mm_info
from qwen_vl_utils import process_vision_info

from xhmodel_merak.configuration_utils import BaseConfig

from .processing_qwen3_omni_moe import Qwen3OmniMoeProcessor


class Qwen3OmniMoeProcessorConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.patch_size: int = None
        self.max_size_h: int = None
        self.max_size_w: int = None


class XHQwen3OmniMoeProcessor(Qwen3OmniMoeProcessor):
    def __init__(
        self,
        image_processor=None,
        video_processor=None,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
            feature_extractor=feature_extractor,
            **kwargs,
        )
        self.config = Qwen3OmniMoeProcessorConfig()

    def apply_chat_template(self, messages: list[dict[str, str]], enable_thinking: bool = False):
        if enable_thinking:
            text = super().apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
            )
        else:
            text = super().apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        # 与原始的 Qwen3OmniMoeProcessor相比，将图像的归一化合并到卷积权重中，因此在处理输入时不再进行归一化。
        # pixel_values的shape:[1,3, 2, 224,224], 2是时间维度，代表视频的两帧，图片则为1帧
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

        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
        self.image_processor.max_pixels = max(
            self.config.max_size_w * self.config.max_size_h + 1, self.image_processor.max_pixels
        )
        model_inputs = self(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True,
        )

        model_inputs["pixel_values"] = model_inputs["hm_pixel_values"][0]
        model_inputs.pop("hm_pixel_values")
        model_inputs["input_features"] = model_inputs["input_features"].to(torch.float16)
        return model_inputs
