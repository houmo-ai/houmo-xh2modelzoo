"""MiniCPM-V-4.5 text-only adapter for the Merak Qwen3 stack.

MiniCPM-V-4.5 owns a SigLIP2 + Resampler Vision tower, but its ``llm`` is a
standard ``Qwen3ForCausalLM`` and its ``lm_head`` follows the Qwen3 ABI.  The
regular Qwen3 loader can therefore consume the composite MiniCPM checkpoint
directly: we load the host, expose its text branch through a Qwen3 shell, and
free the Vision tower before wrapping.
"""

from __future__ import annotations

import os
from typing import Any

import torch.nn as nn
from accelerate import init_empty_weights
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from xhmodel_merak.xh_llm.models.qwen3.qwen3_model import XHQwen3Model


class MiniCPMV45TextModel(XHQwen3Model):
    """Export the embedded Qwen3-8B backend without MiniCPM's Vision graph."""

    # MiniCPM-V-4.5 remote code requires transformers >= 4.57; the qwen3 stack
    # itself only needs >= 4.51. Encode the real constraint here.
    transformers_min_version = "4.57.1"

    # Real calibration prompt for the Qwen3-8B backbone, mirroring the
    # minicpm_o_4_5 / qwen3_5 family convention: the plain text prompt is
    # embedded through the real tokenizer/embedding so the activation
    # statistics seen by the quantizer reflect genuine text inputs.
    LLM_CALIBRATION_PROMPT = (
        "请用一句话描述这张图片的内容。图片中有一个女孩和一只金毛犬在海滩上互动，背景是蓝天和大海。"
    )

    def get_prefill_dummy_inputs(self) -> dict[str, object]:
        """Build prefill calibration inputs from a real prompt, not random ids."""
        import torch
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.hf_model_dir, trust_remote_code=True)
        token_ids = tokenizer(
            self.LLM_CALIBRATION_PROMPT,
            return_tensors="pt",
            add_special_tokens=True,
        )["input_ids"]
        embeds = self.embed_tokens(token_ids)
        prefill_length = int(self.config.prefill_chunk_length)
        if embeds.shape[1] > prefill_length:
            embeds = embeds[:, :prefill_length, :]
        elif embeds.shape[1] < prefill_length:
            pad = torch.zeros(
                (1, prefill_length - embeds.shape[1], embeds.shape[-1]),
                dtype=embeds.dtype,
                device=embeds.device,
            )
            embeds = torch.cat([embeds, pad], dim=1)
        return {"inputs_embeds": embeds, "past_seq_length": 0}

    @classmethod
    def get_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs: Any):
        """Load MiniCPM, then expose its text branch through a Qwen3 shell.

        ``quant_weight`` optionally points at the GPTQ-dequantized backbone
        state dict (Qwen3ForCausalLM keys) produced by ``quant_llm.py``; it is
        applied to the MiniCPM host's ``llm`` submodule.
        """
        import torch

        kwargs.setdefault("trust_remote_code", True)
        kwargs.setdefault("attn_implementation", "eager")
        full_model = super().get_hf_model(
            hf_model_dir,
            quant_weight=None,
            **kwargs,
        )
        if quant_weight:
            if not os.path.isfile(quant_weight):
                raise FileNotFoundError(f"quant_weight backbone not found: {quant_weight}")
            backbone_state = torch.load(quant_weight, weights_only=True, map_location="cpu")
            # GPTQModel's Qwen3 structure keeps zero-initialized bias keys even
            # though the checkpoint has attention_bias=false; the host llm has
            # no bias parameters, so drop them before the strict load.
            backbone_state = {key: value for key, value in backbone_state.items() if not key.endswith(".bias")}
            full_model.llm.load_state_dict(backbone_state, strict=True)
        llm = full_model.llm
        qwen_config = Qwen3Config(**llm.config.to_dict())
        with init_empty_weights():
            text_model = Qwen3ForCausalLM(qwen_config)

        text_model.model = llm.model
        text_model.lm_head = llm.lm_head
        text_model.generation_config = full_model.generation_config
        text_model.name_or_path = str(hf_model_dir)

        input_weight = text_model.model.get_input_embeddings().weight
        if text_model.lm_head.weight is input_weight:
            text_model.lm_head.weight = nn.Parameter(
                input_weight.detach().clone(),
                requires_grad=input_weight.requires_grad,
            )
        text_model.config.tie_word_embeddings = False
        del full_model
        return text_model.eval()


__all__ = ["MiniCPMV45TextModel"]
