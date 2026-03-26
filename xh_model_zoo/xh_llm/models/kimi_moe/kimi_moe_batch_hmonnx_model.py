from typing import List, Union

import torch

from ..builder import MODELS
from ..llm_onnx_model import LLMONNXModel


@MODELS.register_module()
class KimiMoeHMONNXModel(LLMONNXModel):
    @torch.no_grad()
    def prefill(self, data: Union[dict, tuple, list]):
        raw_input_ids: List[List[int]] = data["input_ids"]
        assert self.token_embedding is not None, "Token embedding is not available."

        device = self.device
        input_ids = []
        current_input_length = []
        position_ids = []
        for batch_idx, input_id in enumerate(raw_input_ids):
            input_id = torch.tensor(input_id, dtype=torch.long)
            seq_length = input_id.shape[0]
            past_seq_length = data["past_seq_length"][batch_idx]
            position_id = torch.arange(past_seq_length, past_seq_length + seq_length, dtype=torch.long)
            current_input_length.append(seq_length)
            assert (
                seq_length <= self.prefill_input_sequence_length
            ), f"Input sequence length is too long. max input sequence length is {self.prefill_input_sequence_length} but got {seq_length}"
            if self.prefill_input_sequence_length > seq_length:
                padding_input_ids = torch.zeros(
                    (self.prefill_input_sequence_length - seq_length), dtype=torch.long, device=input_id.device
                )
                # padding_input_ids.fill_(self.pad_token_id)
                input_id = torch.cat([input_id, padding_input_ids], dim=-1)

                padding_position_id = torch.ones(
                    (self.prefill_input_sequence_length - seq_length), dtype=torch.long, device=input_id.device
                )
                position_id = torch.cat([position_id, padding_position_id], dim=-1)

            input_id = input_id.unsqueeze(0)
            position_id = position_id.unsqueeze(0)
            position_ids.append(position_id)
            input_ids.append(input_id)

        input_ids = torch.cat(input_ids, dim=0).to(device)
        position_ids = torch.cat(position_ids, dim=0).to(device)

        current_input_length = torch.tensor(current_input_length, dtype=torch.int32).to(device)

        self.token_embedding.to(device)
        inputs_embeds = self.token_embedding(input_ids)
        past_seq_length = data["past_seq_length"]
        past_seq_length = torch.tensor(past_seq_length, dtype=torch.int32).to(device)
        assert torch.all(past_seq_length >= 0)

        past_key_caches = []
        past_value_caches = []
        for i in range(self.num_hidden_layers):
            past_key_caches.append(getattr(self, f"past_k_cache_{i}"))
            past_value_caches.append(getattr(self, f"past_v_cache_{i}"))

        return self.prefill_session(
            inputs_embeds.to(self.device),
            past_seq_length.to(self.device),
            current_input_length.to(self.device),
            position_ids.to(dtype=torch.int32).to(self.device),
            *past_key_caches,
            *past_value_caches,
        )

    @torch.no_grad()
    def decode(self, data: Union[dict, tuple, list]):
        input_ids = torch.tensor(data["input_ids"], dtype=torch.int32)
        batch_size = input_ids.shape[0]
        assert self.token_embedding is not None, "Token embedding is not available."

        seq_length = torch.tensor([input_ids.shape[1]] * batch_size, dtype=torch.int32)
        input_ids = input_ids.to(self.device)
        inputs_embeds = self.token_embedding(input_ids)
        past_seq_length = torch.tensor(data["past_seq_length"], dtype=torch.int32)
        assert torch.all(past_seq_length >= 0)
        position_id = past_seq_length + 1
        position_id = position_id.reshape(-1, 1)

        past_key_caches = []
        past_value_caches = []
        for i in range(self.num_hidden_layers):
            past_key_caches.append(getattr(self, f"past_k_cache_{i}"))
            past_value_caches.append(getattr(self, f"past_v_cache_{i}"))

        return self.decode_session(
            inputs_embeds.to(self.device),
            past_seq_length.to(self.device),
            seq_length.to(self.device),
            position_id,
            *past_key_caches,
            *past_value_caches,
        )
