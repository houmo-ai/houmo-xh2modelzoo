from typing import Protocol, runtime_checkable

import torch

from .kv_cache_mixin import KVCacheContextManager, KVCacheMixin, KVCacheWithLinearMixin


@runtime_checkable
class InferenceMixin(Protocol):
    """Inference Mixin Protocol类，定义了模型推理相关的方法接口。"""

    def prepare_for_inference(self):
        """准备推理模型，例如加载权重、设置为eval模式等。"""
        raise NotImplementedError("prepare_for_inference method should be implemented in the compatible module")

    def release_inference_model(self):
        """释放推理模型占用的资源，例如显存等。"""
        raise NotImplementedError("release_inference_model method should be implemented in the compatible module")


@runtime_checkable
class LLMInferenceMixin(InferenceMixin, Protocol):
    """LLM Inference Mixin Protocol类，定义了LLM模型推理相关的方法接口。"""

    pass


@runtime_checkable
class TextLLMInferenceMixin(LLMInferenceMixin, Protocol):
    """
    Inference mixin for text LLM models.
    This mixin provides a forward method that is compatible with Hugging Face's generation utilities.
    """

    def get_kvcache_mixin(self) -> KVCacheMixin | KVCacheWithLinearMixin:
        raise NotImplementedError("get_kvcache_mixin method should be implemented in the compatible module")

    def get_input_embeddings(self):
        raise NotImplementedError("get_input_embeddings method should be implemented in the compatible module")

    def set_prefill(self):
        raise NotImplementedError("set_prefill method should be implemented in the compatible module")

    def set_decode(self):
        raise NotImplementedError("set_decode method should be implemented in the compatible module")

    def set_input_sequence_length():
        raise NotImplementedError("set_input_sequence_length method should be implemented in the compatible module")

    def get_input_sequence_length():
        raise NotImplementedError("get_input_sequence_length method should be implemented in the compatible module")

    def forward(self, *args, **kwargs):
        raise NotImplementedError("forward method should be implemented in the compatible module")

    def is_support_dynamic_input(self):
        raise NotImplementedError("is_prefill method should be implemented in the compatible module")

    def is_prefill(self):
        raise NotImplementedError("is_prefill method should be implemented in the compatible module")

    def is_decode(self):
        raise NotImplementedError("is_decode method should be implemented in the compatible module")

    def apply_chat_template(self, prompts):
        raise NotImplementedError("apply_chat_template method should be implemented in the compatible module")


class GoldenMixin:
    def enable_golden(self, enable: bool) -> None:
        self._enable_golden = enable

    def _set_enable_golden(self, enable: bool) -> None:
        self._enable_golden = enable


class LLMInferenceContextManager(KVCacheContextManager):
    """LLM Inference 上下文管理器类，提供更灵活的使用方式。"""

    def __init__(
        self, llm_model: TextLLMInferenceMixin | KVCacheMixin, devices: list[torch.device | str] | None = None
    ):
        super().__init__(llm_model, devices)

    def __enter__(self):
        super().__enter__()
        self._model.prepare_for_inference()

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        self._model.release_inference_model()
