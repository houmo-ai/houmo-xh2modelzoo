import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from xhquant.api import CacheTensor, HMONNXInference

from ...utils import decode_next_token


class Qwen2Inference:
    def __init__(self, model_config_file: str, device: str = "cuda", execution_device: str = "cuda"):
        model_dir = Path(model_config_file).parent
        meta_info = json.load(open(model_config_file, "r"))
        self.meta_info = meta_info

        # hmonnx files
        self.prefill_onnx_file = model_dir / meta_info["prefill_onnx"]
        self.decode_onnx_file = model_dir / meta_info["decode_onnx"]

        # kv cache
        kv_cache_shape = meta_info["kv_cache"]["shape"]
        num_decoder_layers = meta_info["kv_cache"]["num_decoder_layers"]

        # create kv cache
        past_key_caches = []
        past_value_caches = []
        for i in range(num_decoder_layers):
            past_k_cache = CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16))
            past_v_cache = CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16))
            past_key_caches.append(past_k_cache)
            past_value_caches.append(past_v_cache)

        self.past_key_caches = past_key_caches
        self.past_value_caches = past_value_caches

        # tokenizer
        hf_model_config_dir = str(model_dir / meta_info["hf_config"])
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_config_dir)

        # token embedding
        token_embedding_state_dict = torch.load(
            model_dir / meta_info["token_embedding_file"], map_location="cpu", weights_only=True
        )
        self.token_embedding = nn.Embedding(
            token_embedding_state_dict["weight"].shape[0],
            token_embedding_state_dict["weight"].shape[1],
        )
        self.token_embedding.load_state_dict(token_embedding_state_dict)

        self.batch_size = meta_info["wrap_cfg"]["batch_size"]
        self.device = torch.device(device)
        self.execution_device = torch.device(execution_device)

        self.token_embedding.to(execution_device)

        self.prefill_input_sequence_length = meta_info["wrap_cfg"]["input_sequence_length"]

    def prepare_inputs(self, data, input_sequence_length):
        raw_input_ids: List[List[int]] = data["input_ids"]
        assert self.token_embedding is not None, "Token embedding is not available."
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
                seq_length <= input_sequence_length
            ), f"Input sequence length is too long. max input sequence length is {input_sequence_length} but got {seq_length}"
            if input_sequence_length > seq_length:
                padding_input_ids = torch.zeros(
                    (input_sequence_length - seq_length), dtype=torch.long, device=input_id.device
                )
                # padding_input_ids.fill_(self.pad_token_id)
                input_id = torch.cat([input_id, padding_input_ids], dim=-1)

                padding_position_id = torch.ones(
                    (input_sequence_length - seq_length), dtype=torch.long, device=input_id.device
                )
                position_id = torch.cat([position_id, padding_position_id], dim=-1)

            input_id = input_id.unsqueeze(0)
            position_id = position_id.unsqueeze(0)
            position_ids.append(position_id)
            input_ids.append(input_id)

        input_ids = torch.cat(input_ids, dim=0).to(self.execution_device)
        position_ids = torch.cat(position_ids, dim=0).to(self.execution_device)

        current_input_length = torch.tensor(current_input_length, dtype=torch.int32).to(self.execution_device)

        self.token_embedding.to(self.execution_device)
        inputs_embeds = self.token_embedding(input_ids)
        past_seq_length = data["past_seq_length"]
        past_seq_length = torch.tensor(past_seq_length, dtype=torch.int32).to(self.execution_device)
        assert torch.all(past_seq_length >= 0)

        return (
            inputs_embeds.to(torch.float16).to(self.execution_device),
            past_seq_length.to(self.execution_device),
            current_input_length.to(self.execution_device),
            position_ids.to(dtype=torch.int32).to(self.execution_device),
            *self.past_key_caches,
            *self.past_value_caches,
        )

    @torch.no_grad()
    def forward(self, messages):
        assert len(messages) == self.batch_size

        texts = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        batch_input_ids = []
        for text in texts:
            model_inputs = self.tokenizer([text], padding=False, return_tensors="pt")
            batch_input_ids.append(model_inputs.input_ids.cpu().numpy().tolist()[0])

        data_prefill = {
            "input_ids": batch_input_ids,
            "past_seq_length": [0] * self.batch_size,
        }

        device = torch.device(self.device)
        execution_device = torch.device(self.execution_device)

        # prefill
        prefill_inputs = self.prepare_inputs(data_prefill, self.prefill_input_sequence_length)

        prefill_session = HMONNXInference(str(self.prefill_onnx_file))
        prefill_session.to(device)
        prefill_session.exec_device = execution_device

        prefill_logits = prefill_session(*prefill_inputs)
        prefill_next_token_id, prefill_next_token_text = decode_next_token(self.tokenizer, prefill_logits)

        del prefill_session
        torch.cuda.empty_cache()

        # decode
        past_seq_len = [len(input_ids) for input_ids in batch_input_ids]
        batch_input_ids = prefill_next_token_id.cpu().tolist()

        data_decode = {
            "input_ids": batch_input_ids,
            "past_seq_length": past_seq_len,
        }
        decode_inputs = self.prepare_inputs(data_decode, 1)

        decode_session = HMONNXInference(str(self.decode_onnx_file))
        decode_session.to(device)
        decode_session.exec_device = execution_device

        decode_logits = decode_session(*decode_inputs)
        decode_next_token_id, decode_next_token_text = decode_next_token(self.tokenizer, decode_logits)
        # logger.info(f"Decode next token: {decode_next_token_id} {decode_next_token_text}")
        generate_ids = torch.cat([prefill_next_token_id, decode_next_token_id], dim=1)

        generate_text = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
        return (generate_ids, generate_text)
