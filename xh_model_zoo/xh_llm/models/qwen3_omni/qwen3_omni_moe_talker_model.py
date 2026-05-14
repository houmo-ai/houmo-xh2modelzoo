from copy import deepcopy
from typing import List, Optional, Union, cast

import torch
from torch import Tensor
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..base_llm_model import LLMBaseModel
from ..builder import MODELS
from .modeling_qwen3_omni_moe import Qwen3OmniMoeTalkerForConditionalGeneration


@MODELS.register_module()
class XHQwen3OmniMoeTalkerModel(LLMBaseModel):
    """LLMBaseModel adapter for the fused-projection Qwen3-Omni talker.

    The wrap forward (``_Qwen3OmniMoeTalkerForConditionalGeneration`` in
    ``_talker_model.py``) folds ``hidden_projection`` and ``text_projection``
    into the talker graph and selects between them with arithmetic masks
    instead of HF's host-side control flow. This adapter feeds that
    signature.

    Expected ``data`` keys for ``prepare_inputs``:

    - ``hidden_state`` (or ``source``): per-batch list of
      ``[seq, thinker_hidden]`` fp16 tensors. mm positions hold
      ``thinker_hidden``, text positions hold ``thinker_embed``, codec /
      decode positions can be zeros.
    - ``role_mask``: per-batch list of ``[seq, 1]`` fp16. ``1.0`` =
      ``text_projection`` path, ``0.0`` = ``hidden_projection`` path.
    - ``bypass_embeds``: per-batch list of ``[seq, talker_hidden]`` fp16.
      Pre-mixed embeddings for positions where projection is bypassed
      (assistant tts/codec specials, decode codec embeds).
    - ``bypass_mask``: per-batch list of ``[seq, 1]`` fp16. ``1.0`` = use
      ``bypass_embeds``, ``0.0`` = use the projected result.
    - ``past_seq_length``: per-batch list of ``int``.

    Construction of these four tensors must follow HF semantics; see
    ``Qwen3OmniMoeForConditionalGeneration._get_talker_user_parts`` and
    ``_get_talker_assistant_parts`` in ``modeling_qwen3_omni_moe.py``.
    """

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

    def _set_dtype(self, dtype):
        self.token_embedding = self.token_embedding.to(dtype)
        return super()._set_dtype(dtype)

    def get_input_embeddings(self):
        return self.token_embedding

    def init_wrap_model(self, hf_model: Optional[Qwen3OmniMoeTalkerForConditionalGeneration] = None):
        from ._talker_model import register_wrap_modules as qwen3omni_register_wrap_modules

        qwen3omni_register_wrap_modules()

        super().init_wrap_model(hf_model)
        hf_model = cast(Qwen3OmniMoeTalkerForConditionalGeneration, self.wrap_model)

        # Codec embedding kept here so callers can build ``bypass_embeds`` for
        # decode via ``self.get_input_embeddings()(codec_token_id)``.
        self.token_embedding = deepcopy(hf_model.model.get_input_embeddings())
        self.generation_config = hf_model.generation_config
        self.config = hf_model.config
        self.num_hidden_layers = hf_model.model.config.num_hidden_layers
        self.head_dim = hf_model.model.layers[0].self_attn.head_dim
        pad_token_id = getattr(hf_model.config, "pad_token_id", None)
        self.pad_token_id = pad_token_id if pad_token_id is not None else hf_model.config.eos_token_id
        batch_size = self.wrap_cfg.batch_size
        if self.use_cache:
            num_decoder_layers = self.num_hidden_layers
            only_first_block = self.wrap_cfg.get("only_first_block", False)
            if only_first_block:
                num_decoder_layers = 1
            self.prepare_kv_cache(
                num_decoder_layers,
                [batch_size, hf_model.model.config.num_key_value_heads, self.cache_length, self.head_dim],
            )

        hf_model = None

    def _pad_to_seq_len(self, tensor, fill_value: float = 0.0) -> Tensor:
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor, dtype=torch.float16)
        seq_length = tensor.shape[0]
        assert seq_length <= self.input_sequence_length, (
            f"Input sequence length is too long. max input sequence length is "
            f"{self.input_sequence_length} but got {seq_length}"
        )
        if self.input_sequence_length > seq_length:
            pad_shape = (self.input_sequence_length - seq_length, *tensor.shape[1:])
            pad = torch.full(pad_shape, fill_value, dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat([tensor, pad], dim=0)
        return tensor.unsqueeze(0)

    def prepare_inputs(self, data: Union[dict, tuple, list]):
        device = self.execution_device

        raw_source = data.get("hidden_state", data.get("source"))
        assert raw_source is not None, (
            "XHQwen3OmniMoeTalkerModel requires `hidden_state` (or `source`) in data; "
            "the legacy `inputs_embeds` / `input_ids` path is no longer supported "
            "because the fused projection lives inside the wrapped graph."
        )
        raw_role_mask = data["role_mask"]
        raw_bypass_embeds = data["bypass_embeds"]
        raw_bypass_mask = data["bypass_mask"]
        raw_past_seq_length = data["past_seq_length"]

        batch_size = len(raw_source)
        assert (
            len(raw_role_mask) == batch_size
            and len(raw_bypass_embeds) == batch_size
            and len(raw_bypass_mask) == batch_size
            and len(raw_past_seq_length) == batch_size
        ), "All per-batch lists in `data` must share the same length."

        sources: List[Tensor] = []
        role_masks: List[Tensor] = []
        bypass_embeds_list: List[Tensor] = []
        bypass_masks: List[Tensor] = []
        current_input_length: List[int] = []

        for batch_idx in range(batch_size):
            src = raw_source[batch_idx]
            if not isinstance(src, torch.Tensor):
                src = torch.tensor(src, dtype=torch.float16)
            seq_length = src.shape[0]
            current_input_length.append(seq_length)
            sources.append(self._pad_to_seq_len(src))
            role_masks.append(self._pad_to_seq_len(raw_role_mask[batch_idx]))
            bypass_embeds_list.append(self._pad_to_seq_len(raw_bypass_embeds[batch_idx]))
            bypass_masks.append(self._pad_to_seq_len(raw_bypass_mask[batch_idx]))

        source = torch.cat(sources, dim=0).to(device)
        role_mask = torch.cat(role_masks, dim=0).to(device)
        bypass_embeds = torch.cat(bypass_embeds_list, dim=0).to(device)
        bypass_mask = torch.cat(bypass_masks, dim=0).to(device)
        current_input_length_t = torch.tensor(current_input_length, dtype=torch.int32).to(device)

        past_seq_length = torch.tensor(raw_past_seq_length, dtype=torch.int32).to(device)
        assert torch.all(past_seq_length >= 0)

        past_key_caches = self.past_key_caches
        past_value_caches = self.past_value_caches

        return (
            source,
            role_mask,
            bypass_embeds,
            bypass_mask,
            past_seq_length,
            current_input_length_t,
            past_key_caches,
            past_value_caches,
        )

    def prepare_inputs_for_graph(self, data: Union[dict, tuple, list]):
        # Override LLMBaseModel.prepare_inputs_for_graph, which assumes a
        # 5-tuple (inputs_embeds, past_seq_length, seg_length, past_key_caches,
        # past_value_caches). The fused-projection talker emits an 8-tuple, so
        # we just forward whatever prepare_inputs produced.
        return self.prepare_inputs(data)

    def _forward(
        self,
        source: Tensor,
        role_mask: Tensor,
        bypass_embeds: Tensor,
        bypass_mask: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        past_key_caches: List[Tensor],
        past_value_caches: List[Tensor],
    ):
        out = self(
            source,
            role_mask,
            bypass_embeds,
            bypass_mask,
            past_seq_length,
            current_input_length,
            past_key_caches,
            past_value_caches,
        )
        # Wrap forward returns ``(logits, hidden_states)``; exported HMONNX
        # mirrors the same output schema. Frontend / quant graphs preserve
        # the tuple. Only ``logits`` is needed for the CausalLMOutputWithPast
        # return contract used by `LLMBaseModel.test_step`.
        if isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out
        return CausalLMOutputWithPast(logits=logits)
