from typing import Union

import torch
# from xhquant.api import HMONNXGoldenInference as HMONNXInference
from xhquant.api import HMONNXInference
from xhquant.api import Hook
from xhquant.core import CacheTensor

from .builder import MODELS
from .device_dtype_mixin import DeviceDtypeMixin

#from xhquant.api import HMONNXInference

@MODELS.register_module()
class LLMONNXModel(DeviceDtypeMixin):
    def __init__(self, prefill, decode, kv_cache):
        super().__init__()
        self._device = torch.device("cuda")
        self._dtype = torch.float16
        self._exec_device = torch.device("cuda")
        self.prefill_config = prefill
        self.decode_config = decode

        self.prefill_session = HMONNXInference(self.prefill_config.onnx)
        self.decode_session = HMONNXInference(self.decode_config.onnx)

        self.prefill_input_sequence_length = prefill.input_sequence_length
        self.kv_cache = kv_cache
        self.num_hidden_layers = kv_cache.num_hidden_layers
        kv_cache_shape = kv_cache.shape
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

    def init_prefill(self):
        self.prefill_session = HMONNXInference(self.prefill_config.onnx)
        self.prefill_session.exec_device = self._exec_device
        self.prefill_session.to(self.device)

    def init_decode(self):
        self.decode_session = HMONNXInference(self.decode_config.onnx)
        self.decode_session.exec_device = self._exec_device
        self.decode_session.to(self.device)

    def set_input_embeddings(self, value):
        self.token_embedding = value
        self.token_embedding.to(torch.float16)

    def set_input_embeddings_speech(self, value):
        self.speech_embedding = value
        self.speech_embedding.to(torch.float16).to(self.device)

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

    @torch.no_grad()
    def prefill(self, data: Union[dict, tuple, list]):
        input_ids = data["input_ids"]
        assert self.token_embedding is not None, "Token embedding is not available."
        assert input_ids.shape[0] == 1, "Batch size should be 1 in inference mode."
        seq_length = input_ids.shape[1]
        input_ids = input_ids.to(self.device)
        if self.prefill_input_sequence_length > seq_length:
            padding_input_ids = torch.zeros((1, self.prefill_input_sequence_length - seq_length), dtype=torch.long).to(
                self.device
            )
            input_ids = torch.cat([input_ids, padding_input_ids], dim=-1)
        inputs_embeds = self.token_embedding(input_ids)
        past_seq_length = 0
        if "past_seq_length" in data:
            past_seq_length = data["past_seq_length"]
        assert past_seq_length >= 0, "past_seq_length should be non-negative."
        past_key_caches = []
        past_value_caches = []
        for i in range(self.num_hidden_layers):
            past_key_caches.append(getattr(self, f"past_k_cache_{i}"))
            past_value_caches.append(getattr(self, f"past_v_cache_{i}"))

        return self.prefill_session(
            inputs_embeds.to(self.device),
            torch.tensor([past_seq_length], dtype=torch.int32).to(self.device),
            torch.tensor([seq_length], dtype=torch.int32).to(self.device),
            *past_key_caches,
            *past_value_caches,
        )

    @torch.no_grad()
    def decode(self, data: Union[dict, tuple, list]):
        input_ids = data["input_ids"]
        assert self.token_embedding is not None, "Token embedding is not available."
        assert input_ids.shape[0] == 1, "Batch size should be 1 in inference mode."
        assert input_ids.shape[1] == 1, "Input sequence length should be 1 in decode mode."
        seq_length = input_ids.shape[1]
        input_ids = input_ids.to(self.device)
        inputs_embeds = self.token_embedding(input_ids)
        past_seq_length = data["past_seq_length"]
        assert past_seq_length > 0, "past_seq_length should be non-negative."
        past_key_caches = []
        past_value_caches = []
        for i in range(self.num_hidden_layers):
            past_key_caches.append(getattr(self, f"past_k_cache_{i}"))
            past_value_caches.append(getattr(self, f"past_v_cache_{i}"))

        out = self.decode_session(
            inputs_embeds.to(self.device),
            torch.tensor([past_seq_length], dtype=torch.int32).to(self.device),
            torch.tensor([seq_length], dtype=torch.int32).to(self.device),
            *past_key_caches,
            *past_value_caches,
        )
        return out

    def release_prefill_session(self):
        self.prefill_session = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def release_decode_session(self):
        self.decode_session = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
