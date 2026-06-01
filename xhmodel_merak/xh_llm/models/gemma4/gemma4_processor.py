from __future__ import annotations

from transformers import AutoProcessor


class XHGemma4Processor:  # noqa: N801
    def __init__(self, processor):
        self.processor = processor

    def __getattr__(self, name):
        return getattr(self.processor, name)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, trust_remote_code: bool = True):
        processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, trust_remote_code=trust_remote_code)
        return cls(processor)

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        return_dict: bool = True,
        return_tensors: str = "pt",
        **kwargs,
    ):
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            return_dict=return_dict,
            return_tensors=return_tensors,
            **kwargs,
        )
