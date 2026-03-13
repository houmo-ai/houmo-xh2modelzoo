import json
from pathlib import Path
from typing import List, Union

import torch
import torch.nn as nn
from torch import Tensor
from xhquant.api import ConfigDict

from ..builder import MODELS
from ..llm_onnx_model import LLMONNXModel

from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

from xhquant.api import HMONNXInference


@MODELS.register_module()
class XHGlmAsrHMONNXModel(LLMONNXModel):
    def __init__(self, model_dir: str):
        meta_info_file = Path(model_dir) / "export_meta_info.json"
        assert meta_info_file.exists(), f"export_meta_info.json not found in {model_dir}"
        meta_info = json.load(open(meta_info_file, "r"))
        meta_info = ConfigDict(meta_info)
        self.wrap_cfg = meta_info.wrap_cfg
        prefill = ConfigDict(
            dict(
                onnx=str(Path(model_dir) / meta_info.prefill_onnx_file),
                input_sequence_length=meta_info.wrap_cfg.input_sequence_length,
            )
        )

        decode = ConfigDict(
            dict(
                onnx=str(Path(model_dir) / meta_info.decode_onnx_file),
            )
        )
        kv_cache = ConfigDict(
            dict(
                num_hidden_layers=meta_info.num_hidden_layers,
                shape=meta_info.kv_cache_shape,
            )
        )
        super().__init__(prefill, decode, kv_cache)
        self.input_sequence_length = self.prefill_input_sequence_length

        # token embedding
        token_embedding_state_dict = torch.load(
            Path(model_dir) / meta_info["token_embedding_file"], map_location="cpu", weights_only=True
        )
        token_embedding = nn.Embedding(
            token_embedding_state_dict["weight"].shape[0],
            token_embedding_state_dict["weight"].shape[1],
        )

        token_embedding.load_state_dict(token_embedding_state_dict)

        self.set_input_embeddings(token_embedding)

        self.pad_token_id = 0
        self._phase_prefill = None

    def get_input_sequence_length(self):
        return self.input_sequence_length

    def set_input_sequence_length(self, input_sequence_length):
        self.input_sequence_length = input_sequence_length

    def prepare_inputs(self, data: Union[dict, tuple, list]):
        device = self._exec_device
        self.token_embedding.to(device)

        past_seq_length = data["past_seq_length"]
        assert isinstance(past_seq_length, list), f"past_seq_length must be list, but got {type(past_seq_length)}"

        if "input_ids" in data:
            input_ids = data["input_ids"]
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            inputs_embeds = self.token_embedding(input_ids)
        else:
            inputs_embeds = data["inputs_embeds"]
        seq_length = inputs_embeds.shape[1]

        assert seq_length <= self.input_sequence_length, f"input_sequence_length must be greater than {seq_length}"

        inputs_embeds = inputs_embeds.to(device)

        position_ids = torch.arange(past_seq_length[0], past_seq_length[0] + seq_length, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).to(device)

        current_input_length = [seq_length]
        current_input_length = torch.tensor(current_input_length, dtype=torch.int32).to(device)
        past_seq_length = torch.tensor(past_seq_length, dtype=torch.int32).to(device)
        assert torch.all(past_seq_length >= 0)
        if seq_length < self.input_sequence_length:
            padding_input_ids = torch.zeros((self.input_sequence_length - seq_length), dtype=torch.long, device=device)
            padding_input_ids = padding_input_ids.unsqueeze(0)
            position_ids = torch.cat([position_ids, padding_input_ids], dim=1)

            padding_input_ids = torch.zeros((self.input_sequence_length - seq_length), dtype=torch.long, device=device)
            padding_input_ids.fill_(self.pad_token_id)
            padding_input_ids = padding_input_ids.unsqueeze(0)

            inputs_embeds = torch.cat([inputs_embeds, self.token_embedding(padding_input_ids)], dim=1)

        past_key_caches = []
        past_value_caches = []
        for i in range(self.num_hidden_layers):
            past_key_caches.append(getattr(self, f"past_k_cache_{i}"))
            past_value_caches.append(getattr(self, f"past_v_cache_{i}"))

        return (
            inputs_embeds.to(device),
            past_seq_length.to(device),
            current_input_length,
            past_key_caches,
            past_value_caches,
        )

    def _forward(
        self,
        inputs_embeds: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        past_key_caches: List[Tensor],
        past_value_caches: List[Tensor],
    ):
        last_hidden_state = self(
            inputs_embeds,
            past_seq_length,
            current_input_length,
            past_key_caches,
            past_value_caches,
        )
        return BaseModelOutputWithPast(
            hidden_states=last_hidden_state,
        )

    def forward(
        self,
        inputs_embeds: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        past_key_caches: List,
        past_value_caches: List,
    ):
        self.prefill_session.to(self.device)

        out = self.prefill_session(
            inputs_embeds.to(torch.float16).to(self.device),
            past_seq_length.to(self.device),
            current_input_length.squeeze(),
            *past_key_caches,
            *past_value_caches,
        )

    @torch.no_grad()
    def prefill(self, data: Union[dict, tuple, list]):
        raise NotImplementedError("Prefill is not supported for XHGlmAsrHMONNXModel.")

    @torch.no_grad()
    def decode(self, data: Union[dict, tuple, list]):
        raise NotImplementedError("Decode is not supported for XHGlmAsrHMONNXModel.")
