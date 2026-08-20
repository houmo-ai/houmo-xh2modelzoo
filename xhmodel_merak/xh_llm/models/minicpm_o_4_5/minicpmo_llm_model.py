from copy import deepcopy
from typing import Union

import torch

from xhmodel_merak.xh_other_model.base_llm_model import LLMBaseModel

from ...builder import register_llm_model
from .minicpmo_base_model import XHMiniCPMOBaseModel


@register_llm_model("MiniCPMO45LLMModel", master=False)
class XHMiniCPMOLLMModel(LLMBaseModel, XHMiniCPMOBaseModel):
    def __init__(
        self,
        hf_model: str,
        wrap_cfg,
        quant_config,
        frontend_type,
        allow_quant=True,
        export_cfg=None,
    ):
        super().__init__(
            hf_model,
            wrap_cfg=wrap_cfg,
            quant_config=quant_config,
            frontend_type=frontend_type,
            allow_quant=allow_quant,
            export_cfg=export_cfg,
        )
        if "extra_cfg" in self.quant_cfg:
            self.extra_quant_cfg = self.quant_cfg.pop("extra_cfg")
        else:
            self.extra_quant_cfg = None

    def _set_dtype(self, dtype):
        self.token_embedding = self.token_embedding.to(dtype)
        return super()._set_dtype(dtype)

    def init_wrap_model(self, hf_model=None):
        if hf_model is None:
            hf_model = self.get_hf_model()

        # Lazy import is load-bearing: importing _llm_model_impl runs its top-level
        # @XHLLM_TRACEABLE_MODULES.register_module decorators (the only load point
        # of that module). Do not remove.
        from ._llm_model_impl import register_wrap_modules as llm_register_wrap_modules  # noqa: F401

        llm_module = hf_model.llm
        llm_register_wrap_modules(llm_module)

        restore_llm_model_ref = None
        if hasattr(llm_module, "_llm_model"):
            restore_llm_model_ref = llm_module._llm_model
            delattr(llm_module, "_llm_model")

        try:
            self._init_wrap_model_with_llm_registry(llm_module)
        finally:
            if restore_llm_model_ref is not None:
                llm_module._llm_model = restore_llm_model_ref
        # 4.5 uses projector_semantic; 2.6 used projector
        if hasattr(hf_model.tts, "projector_semantic"):
            self._wrap_model.chat_tts_projector = hf_model.tts.projector_semantic
            self._normalize_projected_hidden = getattr(hf_model.tts.config, "normalize_projected_hidden", True)
        elif hasattr(hf_model.tts, "projector"):
            self._wrap_model.chat_tts_projector = hf_model.tts.projector
            self._normalize_projected_hidden = False
        else:
            self._wrap_model.chat_tts_projector = None
            self._normalize_projected_hidden = False

        self.token_embedding = deepcopy(self._wrap_model.model.get_input_embeddings())
        self.generation_config = self._wrap_model.generation_config
        self.config = self._wrap_model.config
        self.num_hidden_layers = self._wrap_model.model.config.num_hidden_layers
        head_dim = self._wrap_model.model.layers[0].self_attn.head_dim
        self.pad_token_id = self._wrap_model.config.eos_token_id
        self.head_dim = head_dim
        if self.use_cache:
            num_decoder_layers = self.num_hidden_layers
            only_first_block = self.wrap_cfg.get("only_first_block", False)
            if only_first_block:
                num_decoder_layers = 1
            self.prepare_kv_cache(
                num_decoder_layers,
                [1, self._wrap_model.model.config.num_key_value_heads, self.cache_length, head_dim],
            )
        del hf_model

    def get_hf_model(self, device_map="cpu", **kwargs):
        hf_model = super().get_hf_model(device_map=device_map, **kwargs)
        return hf_model

    def prepare_inputs(self, data: Union[dict, tuple, list]):
        inputs_embeds = data["inputs_embeds"]
        assert self.token_embedding is not None, "Token embedding is not available."
        assert inputs_embeds.shape[0] == 1, "Batch size should be 1 in inference mode."
        seq_length = inputs_embeds.shape[1]
        inputs_embeds = inputs_embeds.to(self.execution_device)

        past_seq_length = data["past_seq_length"]
        assert past_seq_length >= 0, "past_seq_length should be non-negative."
        past_key_caches = self.past_key_caches
        past_value_caches = self.past_value_caches

        return (
            inputs_embeds.to(self.execution_device),
            torch.tensor([past_seq_length], dtype=torch.int32).to(self.execution_device),
            torch.tensor([seq_length], dtype=torch.int32).to(self.execution_device),
            past_key_caches,
            past_value_caches,
        )

    def prepare_inputs_for_graph(self, data: Union[dict, tuple, list]):
        inputs_embeds, past_seq_length, seg_length, past_key_caches, past_value_caches = self.prepare_inputs(data)
        return (
            inputs_embeds[:, : self.input_sequence_length, :],
            past_seq_length,
            torch.tensor([self.input_sequence_length], dtype=torch.int32).to(self.execution_device),
            past_key_caches,
            past_value_caches,
        )

    def test_step(self, data: Union[dict, tuple, list]):
        input_embeds = data["inputs_embeds"]
        input_seq_len = input_embeds.shape[1]
        raw_past_seq_length = data.get("past_seq_length", 0)
        if isinstance(raw_past_seq_length, torch.Tensor):
            decode_past_len = int(raw_past_seq_length.detach().cpu().reshape(-1)[0].item())
        else:
            decode_past_len = int(raw_past_seq_length)
        effective_input_sequence_length = input_seq_len if decode_past_len > 0 else self.input_sequence_length
        steps = (input_seq_len + effective_input_sequence_length - 1) // effective_input_sequence_length
        inputs = self.prepare_inputs(data)

        inputs_embeds, past_seq_length, seg_length, past_key_caches, past_value_caches = inputs

        input_sequence_length = effective_input_sequence_length
        hidden_states = list()
        last_next_logits = None

        def _to_tensor(value, name: str):
            if isinstance(value, torch.Tensor):
                return value
            if isinstance(value, (list, tuple)):
                if len(value) == 0:
                    raise TypeError(f"{name} is empty {type(value)}")
                for candidate in reversed(value):
                    if isinstance(candidate, torch.Tensor):
                        return candidate
                return _to_tensor(value[0], name)
            raise TypeError(f"Unsupported {name} output type: {type(value)}")

        def _normalize_hidden(hidden):
            hidden = _to_tensor(hidden, "hidden")
            if hidden.ndim == 4:
                hidden = hidden[-1]
            if hidden.ndim != 3:
                raise TypeError(f"Unsupported hidden ndim: {hidden.ndim}, shape={tuple(hidden.shape)}")
            return hidden

        def _resolve_logits_and_hidden(forward_output):
            raw_logits = getattr(forward_output, "logits", forward_output)

            if isinstance(raw_logits, (tuple, list)):
                next_logits = raw_logits[0]
                hidden = raw_logits[1] if len(raw_logits) > 1 else raw_logits[0]
                return _to_tensor(next_logits, "logits"), _normalize_hidden(hidden)

            if isinstance(raw_logits, torch.Tensor):
                return raw_logits, raw_logits

            next_logits = getattr(raw_logits, "logits", None)
            if next_logits is None:
                next_logits = getattr(raw_logits, "next_token_logits", None)
            if next_logits is None:
                raise TypeError(f"Unsupported logits output type: {type(raw_logits)}")

            hidden = getattr(raw_logits, "hidden_states", None)
            if hidden is None:
                hidden = getattr(raw_logits, "hidden_state", None)
            if hidden is None:
                hidden = next_logits

            return _to_tensor(next_logits, "logits"), _normalize_hidden(hidden)

        for i in range(steps):
            start = i * self.input_sequence_length
            end = (i + 1) * self.input_sequence_length
            current_input_length = min(end, input_seq_len) - start

            sub_inputs_embeds = inputs_embeds[:, start:end, :]
            if current_input_length != input_sequence_length:
                b, sl, hdim = sub_inputs_embeds.shape
                sub_padding_inputs_embeds = torch.zeros(
                    (b, input_sequence_length, hdim), dtype=sub_inputs_embeds.dtype, device=sub_inputs_embeds.device
                )
                sub_padding_inputs_embeds[:, :sl, :] = sub_inputs_embeds
                sub_inputs_embeds = sub_padding_inputs_embeds
            else:
                sl = current_input_length
            output = self._forward(
                sub_inputs_embeds,
                past_seq_length,
                torch.tensor([current_input_length], dtype=torch.int32).to(inputs_embeds.device),
                past_key_caches,
                past_value_caches,
            )
            step_logits, step_hidden = _resolve_logits_and_hidden(output)
            last_next_logits = step_logits
            hidden_states.append(step_hidden[:, :sl, :])
            past_seq_length += current_input_length

        return last_next_logits, torch.cat(hidden_states, dim=1)
