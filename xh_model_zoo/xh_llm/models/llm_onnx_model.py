from typing import Any, Dict, Union

import torch
from tqdm import tqdm

from xhquant.api import Config, Hook
from xhquant.api import HMONNXGoldenInference as HMONNXInference
from xhquant.api import HMONNXGraphGoldenInference as HMONNXGrapInference
from xhquant.core import CacheTensor

from .builder import MODELS
from .device_dtype_mixin import DeviceDtypeMixin


@MODELS.register_module()
class LLMONNXModel(DeviceDtypeMixin):
    def __init__(self, prefill, decode, kv_cache, pad_token_id=0):
        super().__init__()
        self._device = torch.device("cpu")
        self._dtype = torch.float16
        self._exec_device = torch.device("cpu")
        self.prefill_config = Config(prefill)
        self.decode_config = Config(decode)
        kv_cache = Config(kv_cache)
        self.prefill_session = HMONNXInference(self.prefill_config.onnx)
        self.decode_session = HMONNXInference(self.decode_config.onnx)

        self.prefill_input_sequence_length = self.prefill_config.input_sequence_length
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
        self.pad_token_id = 0
        self._prefill = False

    def get_input_embeddings(self):
        return self.token_embedding

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
            # position_ids.to(self.device),
            inputs_embeds.to(self.exec_device),
            torch.tensor([past_seq_length], dtype=torch.int32).to(self.exec_device),
            torch.tensor([seq_length], dtype=torch.int32).to(self.exec_device),
            past_key_caches,
            past_value_caches,
        )

    def get_num_logits_to_keep(self):
        return 1

    @property
    def past_value_caches(self):
        past_value_caches = []
        for i in range(self.num_hidden_layers):
            past_value_caches.append(getattr(self, f"past_v_cache_{i}"))
        return past_value_caches

    @property
    def past_key_caches(self):
        past_key_caches = []
        for i in range(self.num_hidden_layers):
            past_key_caches.append(getattr(self, f"past_k_cache_{i}"))
        return past_key_caches

    def prepare_inputs(self, data: Dict[str, Any]):
        inputs_embeds, past_seq_length, seg_length, past_key_caches, past_value_caches = self._prepare_inputs(data)
        bz, nq = inputs_embeds.shape[:2]

        outputs = [
            inputs_embeds,
            past_seq_length,
            seg_length,
        ]
        outputs.append(past_key_caches)
        outputs.append(past_value_caches)
        return tuple(outputs)

    @torch.no_grad()
    def prefill(self, data: Union[dict, tuple, list]):
        input_ids = data["input_ids"]
        current_seq_length = input_ids.shape[1]
        step_input_max_length = self.prefill_input_sequence_length
        past_seq_length = data["past_seq_length"]
        pad_input_seq_length = (
            (current_seq_length + step_input_max_length - 1) // step_input_max_length
        ) * step_input_max_length

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
                    input_past_key_caches,
                    input_past_value_caches,
                    *extra_args,
                ) = self.prepare_inputs(data_input)
                inputs = [inputs_embeds, input_past_seq_length, input_seq_length]
                inputs += input_past_key_caches
                inputs += input_past_value_caches
                inputs += extra_args
                out = self.prefill_session(*inputs)
        else:
            data["input_sequence_length"] = self.prefill_input_sequence_length
            (inputs_embeds, past_seq_length, seq_length, past_key_caches, past_value_caches, *extra_args) = (
                self.prepare_inputs(data)
            )
            inputs = [inputs_embeds, past_seq_length, seq_length]
            inputs += past_key_caches
            inputs += past_value_caches
            inputs += extra_args
            out = self.prefill_session(*inputs)
        return out

    @torch.no_grad()
    def decode(self, data: Union[dict, tuple, list]):
        data["input_sequence_length"] = 1
        (inputs_embeds, past_seq_length, seq_length, past_key_caches, past_value_caches, *extra_args) = (
            self.prepare_inputs(data)
        )
        inputs = [inputs_embeds, past_seq_length, seq_length]
        inputs += past_key_caches
        inputs += past_value_caches
        inputs += extra_args
        return self.decode_session(*inputs)

    def release_prefill_session(self):
        self.prefill_session = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def release_decode_session(self):
        self.decode_session = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@MODELS.register_module()
class LLMLoRAONNXModel(LLMONNXModel):
    def __init__(self, prefill, decode, kv_cache, pad_token_id=0, use_lora_mask=True):
        super().__init__(prefill, decode, kv_cache, pad_token_id=pad_token_id)
        self.use_lora_mask = use_lora_mask

    def prepare_inputs(self, data: Dict[str, Any]):
        inputs = super().prepare_inputs(data)
        if not self.use_lora_mask:
            return inputs
        lora_mask = torch.tensor([1.0], dtype=torch.float16).to(self.exec_device)
        outputs = list(inputs)
        outputs.append(lora_mask)
        return tuple(outputs)

@MODELS.register_module()
class LLMONNXGraphModel(DeviceDtypeMixin):
    def __init__(self, prefill, decode, kv_cache, pad_token_id=0):
        super().__init__()
        self._device = torch.device("cpu")
        self._dtype = torch.float16
        self._exec_device = torch.device("cpu")
        self.prefill_config = Config(prefill)
        self.decode_config = Config(decode)
        kv_cache = Config(kv_cache)
        self.prefill_session = HMONNXGrapInference(self.prefill_config.onnx)
        self.decode_session = HMONNXGrapInference(self.decode_config.onnx)

        self.prefill_input_sequence_length = self.prefill_config.input_sequence_length
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
        self.pad_token_id = 0
        self._prefill = False

    def get_input_embeddings(self):
        return self.token_embedding

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
        self.prefill_session = HMONNXGrapInference(self.prefill_config.onnx)
        self.prefill_session.exec_device = self._exec_device
        self.prefill_session.to(self.device)

    def init_decode(self):
        self.decode_session = HMONNXGrapInference(self.decode_config.onnx)
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
                    torch.ones((1, self.input_sequence_length - seq_length), dtype=torch.long).to(self.exec_device)
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
            # position_ids.to(self.device),
            inputs_embeds.to(self.exec_device),
            torch.tensor([past_seq_length], dtype=torch.int32).to(self.exec_device),
            torch.tensor([seq_length], dtype=torch.int32).to(self.exec_device),
            past_key_caches,
            past_value_caches,
        )

    def get_num_logits_to_keep(self):
        return 1

    @property
    def past_value_caches(self):
        past_value_caches = []
        for i in range(self.num_hidden_layers):
            past_value_caches.append(getattr(self, f"past_v_cache_{i}"))
        return past_value_caches

    @property
    def past_key_caches(self):
        past_key_caches = []
        for i in range(self.num_hidden_layers):
            past_key_caches.append(getattr(self, f"past_k_cache_{i}"))
        return past_key_caches

    def prepare_inputs(self, data: Dict[str, Any]):
        inputs_embeds, past_seq_length, seg_length, past_key_caches, past_value_caches = self._prepare_inputs(data)
        bz, nq = inputs_embeds.shape[:2]

        outputs = [
            inputs_embeds,
            past_seq_length,
            seg_length,
        ]
        outputs.append(past_key_caches)
        outputs.append(past_value_caches)
        return tuple(outputs)

    @torch.no_grad()
    def prefill(self, data: Union[dict, tuple, list]):
        input_ids = data["input_ids"]
        current_seq_length = input_ids.shape[1]
        step_input_max_length = self.prefill_input_sequence_length
        past_seq_length = data["past_seq_length"]
        pad_input_seq_length = (
            (current_seq_length + step_input_max_length - 1) // step_input_max_length
        ) * step_input_max_length

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
                    input_past_key_caches,
                    input_past_value_caches,
                    *extra_args,
                ) = self.prepare_inputs(data_input)
                inputs = [inputs_embeds, input_past_seq_length, input_seq_length]
                inputs += input_past_key_caches
                inputs += input_past_value_caches
                inputs += extra_args
                out = self.prefill_session(*inputs)
        else:
            data["input_sequence_length"] = self.prefill_input_sequence_length
            (inputs_embeds, past_seq_length, seq_length, past_key_caches, past_value_caches, *extra_args) = (
                self.prepare_inputs(data)
            )
            inputs = [inputs_embeds, past_seq_length, seq_length]
            inputs += past_key_caches
            inputs += past_value_caches
            inputs += extra_args
            out = self.prefill_session(*inputs)
        return out

    @torch.no_grad()
    def decode(self, data: Union[dict, tuple, list]):
        data["input_sequence_length"] = 1
        (inputs_embeds, past_seq_length, seq_length, past_key_caches, past_value_caches, *extra_args) = (
            self.prepare_inputs(data)
        )
        inputs = [inputs_embeds, past_seq_length, seq_length]
        inputs += past_key_caches
        inputs += past_value_caches
        inputs += extra_args
        return self.decode_session(*inputs)

    def release_prefill_session(self):
        self.prefill_session = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def release_decode_session(self):
        self.decode_session = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
