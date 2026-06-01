from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModel, AutoTokenizer, Qwen3Model
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import no_init_weights
from xhquant.api import CacheTensor, HMONNXInference


def get_empty_hf_model(hf_model_dir, device_map="cpu", **kwargs) -> Qwen3Model:
    """
    Load model structure only (no weights), to avoid GPU memory usage.
    """
    config = AutoConfig.from_pretrained(hf_model_dir)
    with no_init_weights(), init_empty_weights():
        hf_model: Qwen3Model = AutoModel.from_config(
            config,
            torch_dtype=torch.float16,
            **kwargs,
        )
    return hf_model


class Qwen3EmbeddingInference:
    def __init__(
        self,
        model_config_file: str,
        fast_mode: bool = True,
        device: str = "cuda",
        execution_device: str = "cuda",
    ):
        self.fast_mode = fast_mode
        self._device = torch.device(device)
        self.execution_device = torch.device(execution_device)

        model_dir = Path(model_config_file).parent
        import json

        self.meta_info = json.load(open(model_config_file, "r"))

        self.prefill_onnx_file = model_dir / self.meta_info["prefill_onnx"]

        hf_model_config_dir = str(model_dir / self.meta_info["hf_config"])
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_config_dir)

        embedding_key = (
            "quant_embedding_file"
            if "quant_embedding_file" in self.meta_info
            else "token_embedding_file"
        )
        token_embedding_state_dict = torch.load(
            model_dir / self.meta_info[embedding_key],
            map_location="cpu",
            weights_only=True,
        )
        self.token_embedding = nn.Embedding(
            token_embedding_state_dict["weight"].shape[0],
            token_embedding_state_dict["weight"].shape[1],
        ).to(torch.float16)
        self.token_embedding.load_state_dict(token_embedding_state_dict)

        self.input_sequence_length = self.meta_info["wrap_cfg"]["input_sequence_length"]
        self.pad_token_id = self.tokenizer.eos_token_id
        self.prefill_session: Optional[HMONNXInference] = None
        self.kv_cache_shape = None
        self.num_decoder_layers = None
        if "kv_cache" in self.meta_info:
            self.kv_cache_shape = self.meta_info["kv_cache"]["shape"]
            self.num_decoder_layers = self.meta_info["kv_cache"]["num_decoder_layers"]

    @property
    def device(self):
        return self._device

    def init_prefill(self):
        if self.prefill_session is not None:
            return
        self.prefill_session = HMONNXInference(self.prefill_onnx_file)
        if self.fast_mode:
            self.prefill_session.to_fast_mode()
        self.prefill_session.exec_device = self.execution_device
        self.prefill_session.to(self._device)

    def _create_kv_cache(self):
        assert (
            self.kv_cache_shape is not None and self.num_decoder_layers is not None
        ), "kv_cache is not initialized"
        past_key_caches = []
        past_value_caches = []
        for _ in range(self.num_decoder_layers):
            past_key_caches.append(
                CacheTensor(torch.zeros(self.kv_cache_shape, dtype=torch.float16))
            )
            past_value_caches.append(
                CacheTensor(torch.zeros(self.kv_cache_shape, dtype=torch.float16))
            )
        return past_key_caches, past_value_caches

    def prepare_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        input_sequence_length: int,
    ):
        assert input_ids.shape[0] == 1, "Batch size should be 1 in inference mode."
        seq_length = attention_mask.sum(dim=1).item()
        input_ids = input_ids.to(self.execution_device)
        attention_mask = attention_mask.to(self.execution_device)
        if input_sequence_length > input_ids.shape[1]:
            pad_len = input_sequence_length - input_ids.shape[1]
            pad_ids = torch.full(
                (1, pad_len),
                self.pad_token_id,
                dtype=torch.long,
                device=self.execution_device,
            )
            input_ids = torch.cat([input_ids, pad_ids], dim=-1)
        inputs_embeds = self.token_embedding.to(self.execution_device)(input_ids)
        past_seq_length = torch.tensor(
            [0], dtype=torch.int32, device=self.execution_device
        )
        current_input_length = torch.tensor(
            [seq_length], dtype=torch.int32, device=self.execution_device
        )
        past_key_caches, past_value_caches = self._create_kv_cache()
        return (
            inputs_embeds,
            past_seq_length,
            current_input_length,
            past_key_caches,
            past_value_caches,
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        self.init_prefill()
        assert self.prefill_session is not None, "Prefill session is not initialized."
        (
            inputs_embeds,
            past_seq_length,
            current_input_length,
            past_key_caches,
            past_value_caches,
        ) = self.prepare_inputs(input_ids, attention_mask, self.input_sequence_length)
        seq_length = current_input_length.item()
        pad_len = (
            (seq_length + self.input_sequence_length - 1) // self.input_sequence_length
        ) * self.input_sequence_length
        if pad_len > inputs_embeds.shape[1]:
            pad_tokens = pad_len - inputs_embeds.shape[1]
            pad_embeds = torch.zeros(
                (1, pad_tokens, inputs_embeds.shape[-1]),
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
            inputs_embeds = torch.cat([inputs_embeds, pad_embeds], dim=1)

        steps = pad_len // self.input_sequence_length
        outputs = None
        for i in range(steps):
            start = i * self.input_sequence_length
            end = (i + 1) * self.input_sequence_length
            sub_inputs_embeds = inputs_embeds[:, start:end, :]
            sub_past_seq_length = past_seq_length + start
            sub_current_input_length = torch.tensor(
                [min(end, seq_length) - start],
                dtype=current_input_length.dtype,
                device=current_input_length.device,
            )
            outputs = self.prefill_session(
                sub_inputs_embeds.to(self._device),
                sub_past_seq_length.to(self._device),
                sub_current_input_length.to(self._device),
                *past_key_caches,
                *past_value_caches,
            )
        return outputs


class Qwen3EmbeddingHFCompatible(Qwen3Model):
    def setup(self, llm_model: Qwen3EmbeddingInference):
        self._llm_model = llm_model

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "DynamicModule cannot be initialized directly; use convert instead!"
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if input_ids is None:
            raise ValueError("input_ids is required for qwen3-embedding hf compatible")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.int32)
        if self._llm_model is None:
            raise ValueError("llm_model is not initialized")
        if input_ids.shape[0] == 1:
            hidden_states = self._llm_model.forward(input_ids, attention_mask)
        else:
            outputs = []
            for i in range(input_ids.shape[0]):
                outputs.append(
                    self._llm_model.forward(
                        input_ids[i : i + 1], attention_mask[i : i + 1]
                    )
                )
            hidden_states = torch.cat(outputs, dim=0)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)

    @classmethod
    def to_hf_compatible(
        cls,
        hf_model_or_path: Union[Qwen3Model, str],
        llm_model: Optional[Qwen3EmbeddingInference] = None,
    ) -> Qwen3Model:
        if isinstance(hf_model_or_path, str):
            hf_model = get_empty_hf_model(hf_model_or_path, device_map="auto")
        else:
            hf_model = hf_model_or_path

        assert llm_model is not None, "llm_model is required for hf compatible wrapper"
        hf_model.__class__ = cls
        hf_model.setup(llm_model)
        return hf_model
