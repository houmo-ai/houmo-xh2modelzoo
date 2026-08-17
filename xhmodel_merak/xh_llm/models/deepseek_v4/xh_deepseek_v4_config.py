"""Merak conversion configuration for DeepSeek-V4 Flash."""

from __future__ import annotations

from ...types import TextLLMModelConfig


def resolve_deepseek_v4_pad_token_id(hf_config) -> int:
    """Resolve the static-padding token from an HF config or config-like object."""

    pad_token_id = getattr(hf_config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(hf_config, "eos_token_id", None)
    if isinstance(pad_token_id, (list, tuple)):
        pad_token_id = pad_token_id[0] if pad_token_id else None
    return 0 if pad_token_id is None else int(pad_token_id)


class XHDeepSeekV4ModelConfig(TextLLMModelConfig):
    """Fixed prefill-256/decode-1 export contract.

    ``max_layers`` optionally selects a diagnostic prefix. Omitting it exports
    the released 43-block network. The cache layout itself is fixed when the
    model is wrapped and therefore cannot be enlarged at runtime.
    """

    def __init__(
        self,
        *,
        model_name: str,
        model_type: str = "DeepseekV4ForCausalLM",
        batch_size: int = 1,
        context_max_length: int = 256 * 1024,
        prefill_chunk_length: int = 256,
        num_logits_to_keep: int = 1,
        use_cache: bool = True,
        enable_auto_offload: bool = False,
        packed_weight_only: bool = True,
        max_layers: int | None = None,
        **kwargs,
    ) -> None:
        if model_type != "DeepseekV4ForCausalLM":
            raise ValueError("DeepSeek-V4 Flash architecture must be DeepseekV4ForCausalLM")
        if int(batch_size) != 1:
            raise ValueError("the initial DeepSeek-V4 static runtime supports batch_size=1")
        if int(prefill_chunk_length) != 256:
            raise ValueError("DeepSeek-V4 export currently requires prefill_chunk_length=256")
        if int(context_max_length) <= 0 or int(context_max_length) > 256 * 1024:
            raise ValueError("context_max_length must be in [1, 262144]")
        if int(context_max_length) < 4 * 512:
            raise ValueError("context_max_length must provide at least 512 CSA entries")
        if int(num_logits_to_keep) != 1:
            raise ValueError("DeepSeek-V4 static export currently returns the last-token logit only")
        if not use_cache:
            raise ValueError("DeepSeek-V4 static export requires cache state")
        if max_layers is not None and not 1 <= int(max_layers) <= 43:
            raise ValueError("max_layers must be in [1, 43]")
        kwargs.pop("enable_prefill_chunk", None)
        kwargs.pop("max_pe_length", None)
        super().__init__(
            model_name=model_name,
            model_type=model_type,
            batch_size=1,
            context_max_length=int(context_max_length),
            prefill_chunk_length=256,
            num_logits_to_keep=1,
            use_cache=True,
            enable_auto_offload=bool(enable_auto_offload),
            enable_prefill_chunk=True,
            max_pe_length=int(context_max_length),
            max_layers=None if max_layers is None else int(max_layers),
            **kwargs,
        )
        self.packed_weight_only = bool(packed_weight_only)


__all__ = ["XHDeepSeekV4ModelConfig", "resolve_deepseek_v4_pad_token_id"]
