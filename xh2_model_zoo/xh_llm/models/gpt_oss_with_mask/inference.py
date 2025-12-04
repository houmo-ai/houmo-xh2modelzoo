import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoTokenizer
from xhquant.api import CacheTensor, GoldenMixin, HMONNXInference

from ....utils import DeviceDtypeMixin
from ....xh_llm.utils import decode_next_token


def aligned(size: int, align: int) -> int:
    return ((size + align - 1) // align) * align


def _gen_mask_v2(x: Tensor, valid_length, attention_max_length: int = -1):
    if isinstance(valid_length, int):
        valid_length = torch.tensor(valid_length).to(x.device)
    valid_length = valid_length.reshape(-1)
    bsz, nq, nk = x.size(0), x.size(-2), x.size(-1)
    masks = []
    for i in range(bsz):
        b_valid_length = int(valid_length[i].item())
        if attention_max_length > 0:
            b_valid_length = min(b_valid_length, attention_max_length - 1)
        attention_mask = torch.tril(
            torch.ones(nq, nk, dtype=torch.bool, device=x.device),
            diagonal=b_valid_length,
        ).logical_not()
        if attention_max_length > 0:
            sliding_window_mask = torch.tril(
                torch.ones_like(attention_mask, dtype=torch.bool),
                diagonal=b_valid_length - attention_max_length,
            )
            attention_mask = torch.where(sliding_window_mask, True, attention_mask)
        masks.append(attention_mask.unsqueeze(0).unsqueeze(0))
    return torch.cat(masks, dim=0)


def prepare_casual_mask(x: Tensor, valid_length, attention_max_length: int):
    mask = _gen_mask_v2(x, valid_length, attention_max_length)
    attention_mask = torch.zeros_like(mask, dtype=x.dtype, device=x.device)
    return attention_mask.masked_fill(mask, torch.finfo(x.dtype).min)


class GptOssWithMaskInference(DeviceDtypeMixin):
    def __init__(self, model_config_file: str, fast_mode=True, device: str = "cuda", execution_device: str = "cuda"):
        super().__init__()

        self.fast_mode = fast_mode
        self._device = torch.device(device)
        self._set_exec_device(torch.device(execution_device))

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
        past_key_caches: List[CacheTensor] = []
        past_value_caches: List[CacheTensor] = []
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
        ).to(torch.float16)
        self.token_embedding.load_state_dict(token_embedding_state_dict)

        self.batch_size = 1

        self.prefill_input_sequence_length = meta_info["wrap_cfg"]["input_sequence_length"]
        self.input_sequence_length = self.prefill_input_sequence_length
        self.pad_token_id = self.tokenizer.eos_token_id
        self._phase_prefill = True

        # sliding window config
        wrap_cfg = meta_info.get("wrap_cfg", {})
        self.sliding_window = wrap_cfg.get("sliding_window", -1)
        if self.sliding_window is None:
            self.sliding_window = -1
        self.sliding_window = int(self.sliding_window)

        # 检查是否有 local/global attention
        self.has_local_attention = False
        self.has_global_attention = True
        if self.sliding_window > 0:
            self.has_local_attention = True
        else:
            self.has_global_attention = True

        self.prefill_session: Optional[HMONNXInference] = None
        self.decode_session: Optional[HMONNXInference] = None

    def get_tokenizer(self,hf_model_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_dir)
        return self.tokenizer

    def set_phase_prefill(self, prefill: bool):
        self._phase_prefill = prefill
        if prefill:
            if self.prefill_session is None:
                self.init_prefill()
                self.input_sequence_length = self.prefill_input_sequence_length
        else:
            self.prefill_session = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if self.decode_session is None:
                self.init_decode()
            self.input_sequence_length = 1

    def init_prefill(self):
        if self.prefill_session is not None:
            return
        self.prefill_session = HMONNXInference(self.prefill_onnx_file)
        if self.fast_mode:
            self.prefill_session.to_fast_mode()
        self.prefill_session.exec_device = self.execution_device
        self.prefill_session.to(self._device)

    def init_decode(self):
        if self.decode_session is not None:
            return
        self.decode_session = HMONNXInference(self.decode_onnx_file)
        if self.fast_mode:
            self.decode_session.to_fast_mode()
        self.decode_session.exec_device = self.execution_device
        self.decode_session.to(self._device)

    def get_input_sequence_length(self):
        return self.input_sequence_length

    def set_input_sequence_length(self, input_sequence_length):
        self.input_sequence_length = input_sequence_length

    def prepare_inputs(
        self, data, input_sequence_length
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], List[CacheTensor], List[CacheTensor]]:
        input_ids = data["input_ids"]
        assert self.token_embedding is not None, "Token embedding is not available."
        assert input_ids.shape[0] == 1, "Batch size should be 1 in inference mode."
        seq_length = input_ids.shape[1]
        input_ids = input_ids.to(self.execution_device)
        assert (
            seq_length <= input_sequence_length
        ), f"Input sequence length is too long. max input sequence length is {input_sequence_length} but got {seq_length}"
        if input_sequence_length > seq_length:
            padding_input_ids = torch.zeros((1, input_sequence_length - seq_length), dtype=torch.long).to(
                self.execution_device
            )
            padding_input_ids.fill_(self.pad_token_id)
            input_ids = torch.cat([input_ids, padding_input_ids], dim=-1)
        inputs_embeds = self.token_embedding.to(self.execution_device)(input_ids)

        past_seq_length = data["past_seq_length"]
        assert past_seq_length >= 0, "past_seq_length should be non-negative."
        past_key_caches = self.past_key_caches
        past_value_caches = self.past_value_caches

        # 准备 attention masks
        bz, nq = inputs_embeds.shape[:2]
        local_attention_mask = None
        global_attention_mask = None

        if self.has_global_attention:
            width = self.past_key_caches[0].shape[2] if self.past_key_caches else 2048
            x = torch.empty((bz, nq, width), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            global_attention_mask = prepare_casual_mask(x, torch.tensor([past_seq_length], dtype=torch.int32), -1).to(
                self.execution_device
            )

        if self.has_local_attention and self.sliding_window > 0:
            local_window = self.sliding_window + nq - 1
            local_window = aligned(local_window, 16)
            x = torch.empty((bz, nq, local_window), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            local_attention_mask = prepare_casual_mask(
                x, torch.tensor([past_seq_length], dtype=torch.int32), self.sliding_window
            ).to(self.execution_device)

        return (
            inputs_embeds.to(self.execution_device),
            torch.tensor([past_seq_length], dtype=torch.int32).to(self.execution_device),
            torch.tensor([seq_length], dtype=torch.int32).to(self.execution_device),
            local_attention_mask,
            global_attention_mask,
            past_key_caches,
            past_value_caches,
        )

    def forward(
        self,
        inputs_embeds: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        local_attention_mask: Optional[Tensor],
        global_attention_mask: Optional[Tensor],
        past_key_caches: List[CacheTensor],
        past_value_caches: List[CacheTensor],
    ) -> torch.FloatTensor:
        if self._phase_prefill:
            self.init_prefill()
            assert self.prefill_session is not None, "Prefill session is not initialized."
            # 构建输入参数列表
            session_inputs = [
                inputs_embeds.to(self._device),
                past_seq_length.to(self._device),
                current_input_length.to(self._device),
            ]
            if local_attention_mask is not None:
                session_inputs.append(local_attention_mask.to(self._device))
            if global_attention_mask is not None:
                session_inputs.append(global_attention_mask.to(self._device))
            session_inputs.extend(past_key_caches)
            session_inputs.extend(past_value_caches)

            out = self.prefill_session(*session_inputs)
            if isinstance(self.prefill_session, GoldenMixin):
                self.prefill_session.update_step()
            return out
        else:
            self.init_decode()
            assert self.decode_session is not None, "Decode session is not initialized."
            # 构建输入参数列表
            session_inputs = [
                inputs_embeds.to(self._device),
                past_seq_length.to(self._device),
                current_input_length.to(self._device),
            ]
            if local_attention_mask is not None:
                session_inputs.append(local_attention_mask.to(self._device))
            if global_attention_mask is not None:
                session_inputs.append(global_attention_mask.to(self._device))
            session_inputs.extend(past_key_caches)
            session_inputs.extend(past_value_caches)

            out = self.decode_session(*session_inputs)
            if isinstance(self.decode_session, GoldenMixin):
                self.decode_session.update_step()
            return out

    @torch.no_grad()
    def _forward(self, messages, enable_thinking=False):
        assert self.batch_size == 1, "Batch size should be 1 in inference mode."
        assert len(messages) == self.batch_size

        # 部分 GPT-Oss tokenizer 没有 chat_template，这里做兼容处理
        if getattr(self.tokenizer, "chat_template", None):
            texts = self.tokenizer.apply_chat_template(
                messages, tokenize=False, enable_thinking=enable_thinking, add_generation_prompt=True
            )
        else:
            # 手动构造简单的对话 prompt
            texts = []
            for conv in messages:
                parts = []
                for msg in conv:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "system":
                        parts.append(f"[SYSTEM] {content}\n")
                    elif role == "user":
                        parts.append(f"[USER] {content}\n")
                    elif role == "assistant":
                        parts.append(f"[ASSISTANT] {content}\n")
                    else:
                        parts.append(f"[{role.upper()}] {content}\n")
                parts.append("[ASSISTANT] ")
                texts.append("".join(parts))

        batch_input_ids = []
        for text in texts:
            model_inputs = self.tokenizer([text], padding=False, return_tensors="pt")
            batch_input_ids.append(model_inputs.input_ids.cpu().numpy().tolist()[0])

        data_prefill = {
            "input_ids": torch.tensor(batch_input_ids),
            "past_seq_length": 0,
        }

        device = torch.device(self.device)
        execution_device = torch.device(self.execution_device)

        # prefill
        prefill_inputs = self.prepare_inputs(data_prefill, self.prefill_input_sequence_length)
        (
            inputs_embeds,
            past_seq_length,
            seq_length,
            local_attention_mask,
            global_attention_mask,
            past_key_caches,
            past_value_caches,
        ) = prefill_inputs

        prefill_session = HMONNXInference(str(self.prefill_onnx_file))
        prefill_session.to(device)
        prefill_session.exec_device = execution_device

        session_inputs = [inputs_embeds, past_seq_length, seq_length]
        if local_attention_mask is not None:
            session_inputs.append(local_attention_mask)
        if global_attention_mask is not None:
            session_inputs.append(global_attention_mask)
        session_inputs.extend(past_key_caches)
        session_inputs.extend(past_value_caches)

        prefill_logits = prefill_session(*session_inputs)
        prefill_next_token_id, prefill_next_token_text = decode_next_token(self.tokenizer, prefill_logits)

        del prefill_session
        torch.cuda.empty_cache()

        # decode
        past_seq_len = [len(input_ids) for input_ids in batch_input_ids]
        batch_input_ids = prefill_next_token_id.cpu().tolist()

        data_decode = {
            "input_ids": torch.tensor(batch_input_ids),
            "past_seq_length": past_seq_len[0],
        }
        decode_inputs = self.prepare_inputs(data_decode, 1)
        (
            inputs_embeds,
            past_seq_length,
            seq_length,
            local_attention_mask,
            global_attention_mask,
            past_key_caches,
            past_value_caches,
        ) = decode_inputs

        decode_session = HMONNXInference(str(self.decode_onnx_file))
        decode_session.to(device)
        decode_session.exec_device = execution_device

        session_inputs = [inputs_embeds, past_seq_length, seq_length]
        if local_attention_mask is not None:
            session_inputs.append(local_attention_mask)
        if global_attention_mask is not None:
            session_inputs.append(global_attention_mask)
        session_inputs.extend(past_key_caches)
        session_inputs.extend(past_value_caches)

        decode_logits = decode_session(*session_inputs)
        decode_next_token_id, decode_next_token_text = decode_next_token(self.tokenizer, decode_logits)
        generate_ids = torch.cat([prefill_next_token_id, decode_next_token_id], dim=1)

        generate_text = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
        return (generate_ids, generate_text)

