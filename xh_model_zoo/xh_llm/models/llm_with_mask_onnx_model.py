from typing import Any, Dict, Union

import torch
from torch import Tensor
from tqdm import tqdm

from xhquant.api import ConfigDict
from xhquant.api import HMONNXGoldenInference as HMONNXInference
from xhquant.api import Hook
from xhquant.core import CacheTensor

from .builder import MODELS
from .device_dtype_mixin import DeviceDtypeMixin


def aligned(size, align):
    return ((size + align - 1) // align) * align


def _prepare_window_attention_mask(inputs_tensor: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    nq, nk = inputs_tensor.size(-2), inputs_tensor.size(-1)
    attention_mask = torch.ones([1, nq, nk], device=inputs_tensor.device, dtype=torch.bool)
    for i in range(1, len(cu_seqlens)):
        attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
    return attention_mask


def _gen_mask_v2(x: Tensor, valid_length: Union[int, Tensor], attention_max_length: int = -1):
    if isinstance(valid_length, int):
        valid_length = torch.tensor(valid_length).to(x.device)
    valid_length = valid_length.reshape(-1)
    if x.shape[0] != valid_length.size().numel() or (valid_length[0].item() == 0 and valid_length.shape[0] == 2):
        return _prepare_window_attention_mask(x, valid_length)

    bsz, nq, nk = x.size(0), x.size(-2), x.size(-1)
    batch_attention_mask = []
    for i in range(bsz):
        batch_valid_length = int(valid_length[i].item())
        if attention_max_length > 0:
            batch_valid_length = min(batch_valid_length, attention_max_length - 1)
        attention_mask = torch.tril(
            torch.ones(nq, nk, dtype=torch.bool, device=x.device), diagonal=batch_valid_length
        ).logical_not()
        if attention_max_length > 0:
            sliding_window_mask = torch.tril(
                torch.ones_like(attention_mask, dtype=torch.bool), diagonal=batch_valid_length - attention_max_length
            )
            attention_mask = torch.where(sliding_window_mask, True, attention_mask)
        batch_attention_mask.append(attention_mask.unsqueeze(dim=0).unsqueeze(dim=0))

    return torch.cat(batch_attention_mask, dim=0)


@MODELS.register_module()
class LLMWithMaskONNXModel(DeviceDtypeMixin):
    def __init__(self, prefill, decode, kv_cache, sliding_window_cfg, pad_token_id=0):
        super().__init__()
        self._device = torch.device("cpu")
        self._dtype = torch.float16
        self._exec_device = torch.device("cpu")
        self.prefill_config = ConfigDict(prefill) if isinstance(prefill, dict) else prefill
        self.decode_config = ConfigDict(decode) if isinstance(decode, dict) else decode

        self.prefill_session = HMONNXInference(self.prefill_config.onnx)
        self.decode_session = HMONNXInference(self.decode_config.onnx)

        self.prefill_input_sequence_length = self.prefill_config.input_sequence_length
        kv_cache = ConfigDict(kv_cache) if isinstance(kv_cache, dict) else kv_cache
        self.kv_cache = kv_cache
        self.num_hidden_layers = kv_cache.num_hidden_layers
        kv_cache_shape = kv_cache.shape
        self.pad_token_id = pad_token_id
        self.sliding_window_cfg = ConfigDict(sliding_window_cfg)
        for i in range(self.num_hidden_layers):
            self.register_buffer(
                f"past_k_cache_{i}",
                CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)),
                persistent=False,
            )
            self.register_buffer(
                f"past_v_cache_{i}",
                CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)),
                persistent=False,
            )
        self.token_embedding = None
        self._hooks = []

    @staticmethod
    def _prepare_runtime_session(session) -> None:
        session.initialize()
        if session._session is None:
            return
        for module in session._session.node_modules:
            if getattr(module, "attention_max_length", 0) < 0 and hasattr(module, "fast_mode"):
                module.fast_mode = False

    def register_hook(self, hook: Hook):
        self._hooks.append(hook)

    def clear_hooks(self):
        self._hooks = []

    def _set_exec_device(self, device):
        super()._set_exec_device(device)
        if self.prefill_session is not None:
            self.prefill_session.exec_device = device
        if self.decode_session is not None:
            self.decode_session.exec_device = device

    def prepare_casual_mask(self, x: Tensor, valid_length: int, attention_max_length: int):
        mask = _gen_mask_v2(x, valid_length, attention_max_length)
        attention_mask = torch.zeros_like(mask, dtype=x.dtype, device=x.device)
        return attention_mask.masked_fill(mask, torch.finfo(x.dtype).min)

    def init_prefill(self):
        self.prefill_session = HMONNXInference(self.prefill_config.onnx)
        self._prepare_runtime_session(self.prefill_session)
        self.prefill_session.exec_device = self._exec_device
        self.prefill_session.to(self.device)

    def init_decode(self):
        self.decode_session = HMONNXInference(self.decode_config.onnx)
        self._prepare_runtime_session(self.decode_session)
        self.decode_session.exec_device = self._exec_device
        self.decode_session.to(self.device)

    def set_input_embeddings(self, value):
        self.token_embedding = value
        self.token_embedding.to(torch.float16)

    def _set_device(self, device):
        super()._set_device(device)
        if self.prefill_session is not None:
            self.prefill_session.to(device)
        if self.decode_session is not None:
            self.decode_session.to(device)
        self.token_embedding.to(device)
        for i in range(self.num_hidden_layers):
            setattr(self, f"past_k_cache_{i}", getattr(self, f"past_k_cache_{i}").to(device))
            setattr(self, f"past_v_cache_{i}", getattr(self, f"past_v_cache_{i}").to(device))
        return self

    def _set_dtype(self, dtype):
        super()._set_dtype(dtype)
        self.token_embedding.to(dtype)
        return self

    def save_prefill_golden(self, output_dir):
        self.prefill_session.step = 0
        self.prefill_session.save_golden = True
        self.prefill_session.golden_dir = output_dir

    def save_decode_golden(self, output_dir):
        self.decode_session.step = 0
        self.decode_session.save_golden = True
        self.decode_session.golden_dir = output_dir

    def _prepare_inputs(self, data: Dict[str, Any]):
        assert isinstance(data, Dict)
        input_ids = data.get("input_ids", None)
        inputs_embeds = data.get("inputs_embeds", None)
        input_sequence_length = data["input_sequence_length"]
        seq_length = -1
        if input_ids is not None:
            assert input_ids.shape[0] == 1, "Batch size should be 1 in inference mode."
            seq_length = input_ids.shape[1]
            input_ids = input_ids.to(self.exec_device)
            input_sequence_length = (
                (seq_length + input_sequence_length - 1) // input_sequence_length * input_sequence_length
            )

            if input_sequence_length > seq_length:
                padding_input_ids = torch.zeros((1, input_sequence_length - seq_length), dtype=torch.long).to(
                    self.exec_device
                )
                padding_input_ids.fill_(self.pad_token_id)
                input_ids = torch.cat([input_ids, padding_input_ids], dim=-1)
            inputs_embeds = self.token_embedding.to(self.exec_device)(input_ids)
        elif inputs_embeds is not None:
            assert inputs_embeds.shape[0] == 1, "Batch size should be 1 in inference mode."
            seq_length = inputs_embeds.shape[1]
            input_sequence_length = (
                (seq_length + input_sequence_length - 1) // input_sequence_length * input_sequence_length
            )
            inputs_embeds = inputs_embeds.to(self.exec_device)

            if input_sequence_length > seq_length:
                padding_token_id = self.pad_token_id
                padding_input_ids = (
                    torch.ones((1, input_sequence_length - seq_length), dtype=torch.long).to(self.exec_device)
                    * padding_token_id
                )
                padding_embedding = self.token_embedding.to(self.exec_device)(padding_input_ids)
                inputs_embeds = torch.cat([inputs_embeds, padding_embedding], dim=1)

        assert self.token_embedding is not None, "Token embedding is not available."

        past_seq_length = data["past_seq_length"]
        assert past_seq_length >= 0, "past_seq_length should be non-negative."

        past_key_caches = []
        past_value_caches = []
        for i in range(self.num_hidden_layers):
            past_key_caches.append(getattr(self, f"past_k_cache_{i}"))
            past_value_caches.append(getattr(self, f"past_v_cache_{i}"))

        return (
            inputs_embeds.to(self.exec_device),
            torch.tensor([past_seq_length], dtype=torch.int32).to(self.exec_device),
            torch.tensor([seq_length], dtype=torch.int32).to(self.exec_device),
            past_key_caches,
            past_value_caches,
        )

    def prepare_inputs(self, data: Dict[str, Any]):
        inputs_embeds, past_seq_length, seg_length, past_key_caches, past_value_caches = self._prepare_inputs(data)
        bz, nq = inputs_embeds.shape[:2]
        local_attention_mask = None
        global_attention_mask = None
        if self.sliding_window_cfg.has_global_attention:
            x = torch.empty(
                (bz, nq, self.sliding_window_cfg.global_attention_window_size),
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
            global_attention_mask = self.prepare_casual_mask(x, past_seq_length, -1)
            global_attention_mask.to(self.exec_device)
        if self.sliding_window_cfg.has_local_attention:
            local_attention_window_size = self.sliding_window_cfg.local_attention_window_size + nq - 1
            local_attention_window_size = aligned(local_attention_window_size, 16)
            x = torch.empty((bz, nq, local_attention_window_size), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            local_attention_mask = self.prepare_casual_mask(x, past_seq_length, self.sliding_window_cfg.sliding_window)
            local_attention_mask.to(self.exec_device)
        outputs = [inputs_embeds, past_seq_length, seg_length]
        outputs.append(local_attention_mask)
        outputs.append(global_attention_mask)
        outputs.append(past_key_caches)
        outputs.append(past_value_caches)
        return tuple(outputs)

    @torch.no_grad()
    def prefill(self, data: Dict[str, Any]):
        self._prepare_runtime_session(self.prefill_session)
        input_ids = data["input_ids"]
        current_seq_length = input_ids.shape[1]
        step_input_max_length = self.prefill_input_sequence_length
        past_seq_length = data["past_seq_length"]
        pad_input_seq_length = ((current_seq_length + step_input_max_length - 1) // step_input_max_length) * step_input_max_length

        steps = pad_input_seq_length // step_input_max_length

        if steps > 1:
            for i in tqdm(range(steps), desc="prefill"):
                start = i * step_input_max_length
                end = (i + 1) * step_input_max_length
                sub_past_seq_length = past_seq_length + start

                data_input = {
                    "past_seq_length": sub_past_seq_length,
                    "input_sequence_length": self.prefill_input_sequence_length,
                }
                sub_input_ids = input_ids[:, start:end]
                data_input["input_ids"] = sub_input_ids
                (
                    inputs_embeds,
                    input_past_seq_length,
                    input_seq_length,
                    input_local_attention_mask,
                    input_global_attention_mask,
                    input_past_key_caches,
                    input_past_value_caches,
                ) = self.prepare_inputs(data_input)
                inputs = [inputs_embeds, input_past_seq_length, input_seq_length]
                if input_local_attention_mask is not None:
                    inputs.append(input_local_attention_mask)
                if input_global_attention_mask is not None:
                    inputs.append(input_global_attention_mask)
                inputs += input_past_key_caches
                inputs += input_past_value_caches
                out = self.prefill_session(*inputs)
        else:
            data["input_sequence_length"] = self.prefill_input_sequence_length
            (
                inputs_embeds,
                past_seq_length,
                seq_length,
                local_attention_mask,
                global_attention_mask,
                past_key_caches,
                past_value_caches,
            ) = self.prepare_inputs(data)
            inputs = [inputs_embeds, past_seq_length, seq_length]
            if local_attention_mask is not None:
                inputs.append(local_attention_mask)
            if global_attention_mask is not None:
                inputs.append(global_attention_mask)
            inputs += past_key_caches
            inputs += past_value_caches
            out = self.prefill_session(*inputs)
        return out

    @torch.no_grad()
    def decode(self, data: Dict[str, Any]):
        self._prepare_runtime_session(self.decode_session)
        data["input_sequence_length"] = 1
        (
            inputs_embeds,
            past_seq_length,
            seq_length,
            local_attention_mask,
            global_attention_mask,
            past_key_caches,
            past_value_caches,
        ) = self.prepare_inputs(data)
        inputs = [inputs_embeds, past_seq_length, seq_length]
        if local_attention_mask is not None:
            inputs.append(local_attention_mask)
        if global_attention_mask is not None:
            inputs.append(global_attention_mask)
        inputs += past_key_caches
        inputs += past_value_caches
        return self.decode_session(*inputs)

    def release_prefill_session(self):
        self.prefill_session = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def release_decode_session(self):
        self.decode_session = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()