from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..base_llm_model import LLMBaseModel
from ..builder import MODELS
from .modeling_glm_ocr import GlmOcrForConditionalGeneration
from .utils import get_rope_index, scatter_image_embeds


@MODELS.register_module()
class XHGlmOcrLLMModel(LLMBaseModel):
    def __init__(
        self,
        hf_model: str,
        wrap_cfg,
        quant_config,
        frontend_type="TorchFX",
        allow_quant=True,
        export_cfg=None,
    ):
        super().__init__(
            hf_model,
            wrap_cfg,
            quant_config,
            frontend_type,
            allow_quant=allow_quant,
            export_cfg=export_cfg,
        )
        self.rope_deltas = None

    def get_hf_model(self, device_map="cpu", **kwargs) -> GlmOcrForConditionalGeneration:
        assert self.hf_model_dir is not None
        attn_implementation = kwargs.get("attn_implementation", "eager")
        if attn_implementation is not None:
            attn_implementation = attn_implementation.lower()

        hf_model = GlmOcrForConditionalGeneration.from_pretrained(
            self.hf_model_dir,
            torch_dtype=torch.float16,
            device_map=device_map,
            attn_implementation=attn_implementation,
        ).eval()

        if hf_model.config.tie_word_embeddings:
            hf_model.config.torchscript = True
            hf_model.tie_weights()
            hf_model.config.tie_word_embeddings = False
            hf_model.config.torchscript = False
        return hf_model

    def init_wrap_model(self, hf_model=None):
        if hf_model is None:
            hf_model = self.get_hf_model()

        from ._llm_model_impl import register_wrap_cls as llm_register_wrap_cls

        llm_register_wrap_cls(hf_model)
        self.config = hf_model.model.language_model.config
        self.model_config = hf_model.config
        self.token_embedding = hf_model.get_input_embeddings()
        wraped_model = super().init_wrap_model(hf_model)
        hf_model = wraped_model

        self.generation_config = hf_model.generation_config
        self.num_hidden_layers = self.config.num_hidden_layers

        if hasattr(self.config, "head_dim"):
            head_dim = self.config.head_dim
        else:
            head_dim = self.config.hidden_size // self.config.num_attention_heads

        batch_size = 1
        if self.use_cache:
            num_decoder_layers = self.num_hidden_layers
            only_first_block = self.wrap_cfg.get("only_first_block", False)
            if only_first_block:
                num_decoder_layers = 1
            self.prepare_kv_cache(
                num_decoder_layers,
                [batch_size, self.config.num_key_value_heads, self.cache_length, head_dim],
            )

        hf_model = None
        return wraped_model

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            spatial_merge_size=self.model_config.vision_config.spatial_merge_size,
            image_token_id=self.model_config.image_token_id,
            video_start_token_id=self.model_config.video_start_token_id,
            video_end_token_id=self.model_config.video_end_token_id,
        )

    def prepare_inputs(self, data: Union[dict, tuple, list]):
        device = self.execution_device

        input_ids = data["input_ids"].to(device)
        seq_length = input_ids.shape[1]

        assert self.token_embedding is not None, "Token embedding is not available."
        assert input_ids.shape[0] == 1, "Batch size should be 1 in inference mode."
        assert seq_length <= self.input_sequence_length, (
            "Input sequence length is too long. "
            f"max input sequence length is {self.input_sequence_length} but got {seq_length}"
        )

        attention_mask = data.get("attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones((input_ids.shape[0], seq_length), dtype=torch.long, device=device)
        else:
            attention_mask = attention_mask.to(device)

        if self.input_sequence_length > seq_length:
            pad_len = self.input_sequence_length - seq_length
            padding_input_ids = torch.zeros((1, pad_len), dtype=torch.long, device=device)
            padding_input_ids.fill_(self.pad_token_id)
            input_ids = torch.cat([input_ids, padding_input_ids], dim=-1)

            padding_attention_mask = torch.zeros((1, pad_len), dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([attention_mask, padding_attention_mask], dim=-1)

        inputs_embeds = self.token_embedding.to(device)(input_ids)

        n_image_tokens = torch.sum(input_ids == self.model_config.image_token_id).item()
        if n_image_tokens > 0 and data.get("image_embeds", None) is not None:
            image_embeds = data["image_embeds"].to(device)
            inputs_embeds = scatter_image_embeds(
                input_ids=input_ids,
                token_embeds=inputs_embeds,
                image_embeds=image_embeds,
                image_token_id=self.model_config.image_token_id,
            )

        past_seq_length = data["past_seq_length"]
        assert past_seq_length >= 0, "past_seq_length should be non-negative."

        if past_seq_length == 0:
            image_grid_thw = data.get("image_grid_thw", None)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(device)

            video_grid_thw = data.get("video_grid_thw", None)
            if video_grid_thw is not None:
                video_grid_thw = video_grid_thw.to(device)

            position_ids, rope_deltas = self.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
        else:
            assert self.rope_deltas is not None, f"rope_deltas is None, but past_seq_length is {past_seq_length}"
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = past_seq_length + self.rope_deltas
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        past_key_caches = self.past_key_caches
        past_value_caches = self.past_value_caches

        return (
            inputs_embeds.to(self.execution_device),
            position_ids.to(device=self.execution_device, dtype=torch.float16),
            torch.tensor([past_seq_length], dtype=torch.int32).to(self.execution_device),
            torch.tensor([seq_length], dtype=torch.int32).to(self.execution_device),
            past_key_caches,
            past_value_caches,
        )

    def prepare_inputs_for_graph(self, data: Union[dict, tuple, list]):
        input_ids = data["input_ids"]
        input_seq_len = input_ids.shape[-1]
        steps = (input_seq_len + self.input_sequence_length - 1) // self.input_sequence_length
        old_input_seq_len = self.input_sequence_length
        self.input_sequence_length = self.input_sequence_length * steps

        (
            inputs_embeds,
            position_ids,
            past_seq_length,
            seg_length,
            past_key_caches,
            past_value_caches,
        ) = self.prepare_inputs(data)
        self.input_sequence_length = old_input_seq_len

        return (
            inputs_embeds[:, : self.input_sequence_length, :],
            position_ids[:, :, : self.input_sequence_length],
            past_seq_length,
            seg_length,
            past_key_caches,
            past_value_caches,
        )

    def _forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_seq_length: Tensor = None,
        current_input_length: Tensor = None,
        past_key_caches: Optional[List[Tensor]] = None,
        past_value_caches: Optional[List[Tensor]] = None,
    ):
        logits = self(
            inputs_embeds,
            position_ids,
            past_seq_length,
            current_input_length,
            past_key_caches,
            past_value_caches,
        )
        return CausalLMOutputWithPast(logits=logits)

    @torch.no_grad()
    def test_step(self, data: Union[dict, tuple, list]):
        input_ids = data["input_ids"]
        input_seq_len = input_ids.shape[-1]
        steps = (input_seq_len + self.input_sequence_length - 1) // self.input_sequence_length
        old_input_seq_len = self.input_sequence_length
        self.input_sequence_length = self.input_sequence_length * steps
        inputs = self.prepare_inputs(data)
        self.input_sequence_length = old_input_seq_len

        (
            inputs_embeds,
            position_ids,
            past_seq_length,
            _,
            past_key_caches,
            past_value_caches,
        ) = inputs
        for i in range(steps):
            start = i * self.input_sequence_length
            end = (i + 1) * self.input_sequence_length
            current_input_length = min(end, input_seq_len) - start
            output = self._forward(
                inputs_embeds[:, start:end, :],
                position_ids[:, :, start:end],
                past_seq_length,
                torch.tensor([current_input_length], dtype=torch.int32).to(inputs_embeds.device),
                past_key_caches,
                past_value_caches,
            )
            past_seq_length += current_input_length

        return output
