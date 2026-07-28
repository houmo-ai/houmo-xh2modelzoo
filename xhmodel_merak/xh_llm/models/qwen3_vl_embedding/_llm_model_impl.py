from collections.abc import Generator
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union

from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPast

from xhquant.api import ConfigDict

from ...register import XHLLM_TRACEABLE_MODULES
from ..qwen3_vl._llm_model_impl import (
    _Qwen3VLForConditionalGeneration,
)
from ..qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
)


class _Qwen3VLEmbeddingForConditionalGeneration(
    _Qwen3VLForConditionalGeneration
):
    """Return sequence hidden states for embedding export."""

    def _setup(self, cfg: ConfigDict):
        self.cfg = cfg
        self.output_hidden_states_for_export = bool(
            cfg.get("output_hidden_states_for_export", True)
        )

    def graph_forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        time_position_ids: Optional[Tensor] = None,
        hight_position_ids: Optional[Tensor] = None,
        width_position_ids: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        deepstack_visual_embed_0: Optional[Tensor] = None,
        deepstack_visual_embed_1: Optional[Tensor] = None,
        deepstack_visual_embed_2: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        outputs = self.model.language_model.forward(
            input_embeds=inputs_embeds,
            time_position_ids=time_position_ids,
            hight_position_ids=hight_position_ids,
            width_position_ids=width_position_ids,
            deepstack_visual_embed_0=deepstack_visual_embed_0,
            deepstack_visual_embed_1=deepstack_visual_embed_1,
            deepstack_visual_embed_2=deepstack_visual_embed_2,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )
        hidden_states = outputs[0]
        if self.output_hidden_states_for_export:
            return hidden_states
        return self.lm_head(hidden_states)


@contextmanager
def embedding_wrap_cls_scope() -> Generator[None, None, None]:
    """Temporarily install the embedding-specific outer wrapper."""
    registry = XHLLM_TRACEABLE_MODULES
    model_cls = Qwen3VLForConditionalGeneration
    previous_wrapper = registry._registry.get(model_cls)
    previous_key = registry._key_registry.get(model_cls)
    had_dynamic_cls = model_cls in registry._dynamic_classes
    previous_dynamic_cls = registry._dynamic_classes.get(model_cls)

    registry._registry[model_cls] = (
        _Qwen3VLEmbeddingForConditionalGeneration
    )
    registry._key_registry[model_cls] = model_cls.__name__
    registry._dynamic_classes.pop(model_cls, None)
    try:
        yield
    finally:
        if previous_wrapper is None:
            registry._registry.pop(model_cls, None)
        else:
            registry._registry[model_cls] = previous_wrapper
        if previous_key is None:
            registry._key_registry.pop(model_cls, None)
        else:
            registry._key_registry[model_cls] = previous_key
        if had_dynamic_cls:
            registry._dynamic_classes[model_cls] = (
                previous_dynamic_cls
            )
        else:
            registry._dynamic_classes.pop(model_cls, None)
