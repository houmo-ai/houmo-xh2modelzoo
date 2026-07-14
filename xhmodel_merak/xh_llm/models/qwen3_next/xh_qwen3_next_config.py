from __future__ import annotations

from collections.abc import Mapping

from xhmodel_merak.configuration_utils import HFModelConfig

from ..qwen3_5.xh_qwen3_5_config import (
    XHQwen3_5ModelConfig,
    build_spec_draft_quant_scheme,
)


class XHQwen3NextMTPConfig(HFModelConfig):
    """Merak export configuration for the Qwen3-Next MTP draft graph."""

    def __init__(
        self,
        *,
        hidden_size: int = 2048,
        num_key_value_heads: int = 2,
        head_dim: int = 256,
        batch_size: int = 1,
        input_sequence_length: int = 1,
        context_max_length: int = 2048,
        max_pe_length: int = 262144,
        use_cache: bool = True,
        mtp_layer_index: int = 0,
        draft_head_weight_bits: int = 4,
        model_type: str = "Qwen3NextMTP",
        **kwargs,
    ):
        super().__init__(model_type=model_type, **kwargs)
        self.hidden_size = hidden_size
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.input_sequence_length = input_sequence_length
        self.context_max_length = context_max_length
        self.max_pe_length = max_pe_length
        self.use_cache = use_cache
        self.mtp_layer_index = mtp_layer_index
        self.draft_head_weight_bits = draft_head_weight_bits
        if getattr(self, "quant_scheme", None) is None:
            self.quant_scheme = build_spec_draft_quant_scheme(draft_head_weight_bits)

    @property
    def hf_model_dir(self) -> str:
        return self.hf_model


class XHQwen3NextModelConfig(XHQwen3_5ModelConfig):
    """Text-only Qwen3-Next config using the Merak hybrid-cache pipeline."""

    def __init__(
        self,
        *,
        model_name: str,
        model_type: str = "Qwen3NextForCausalLM",
        mtp_config: Mapping | XHQwen3NextMTPConfig | None = None,
        spec_decode_mode: str | None = None,
        **kwargs,
    ):
        # Qwen3-Next is text-only. Do not let the Qwen3.5 base materialize its
        # visual or Qwen3.5-specific MTP configs.
        kwargs.pop("visual_config", None)
        kwargs.pop("dflash_config", None)
        super().__init__(
            model_name=model_name,
            model_type=model_type,
            visual_config=None,
            mtp_config=None,
            dflash_config=None,
            spec_decode_mode=spec_decode_mode,
            **kwargs,
        )
        if hasattr(self, "visual_config"):
            del self.visual_config
        if isinstance(mtp_config, Mapping):
            raw = dict(mtp_config)
            raw.setdefault("model_name", f"{model_name}_mtp")
            raw.setdefault("hf_model", self.hf_model)
            raw.setdefault("draft_head_weight_bits", self.spec_draft_head_weight_bits)
            mtp_config = XHQwen3NextMTPConfig(**raw)
        self.mtp_config = mtp_config
        self.dflash_config = None
        if spec_decode_mode not in {None, "mtp"}:
            raise ValueError("Qwen3-Next supports only spec_decode_mode='mtp'")
