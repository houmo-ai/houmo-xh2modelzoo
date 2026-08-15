from __future__ import annotations

from collections.abc import Mapping

from ..qwen3_next.xh_qwen3_next_config import XHQwen3NextModelConfig


class XHLing3FlashModelConfig(XHQwen3NextModelConfig):
    """Merak export configuration for Ling-3-Flash.

    Ling keeps Qwen3.5's hybrid-cache optimizations, but replaces the scalar
    GDR decay with KDA's per-key-dimension decay. ``GDRChunkScan`` supports
    both layouts, treating scalar GDR as the broadcast key-dimension case, so
    the same fused operator is enabled by default.
    """

    def __init__(
        self,
        *,
        model_name: str,
        model_type: str = "BailingMoeV3ForCausalLM",
        linear_chunk_size: int = 64,
        split_conv_cache: bool = True,
        fuse_gdr_ops: bool = True,
        fuse_gdr_block_recurrent_ops: bool = True,
        flash_attention: Mapping | None = None,
        spec_decode_mode: str | None = None,
        **kwargs,
    ):
        if spec_decode_mode is not None:
            raise ValueError("Ling-3-Flash target export does not yet expose its checkpoint MTP layer")
        if not split_conv_cache:
            raise ValueError("Ling-3 KDA requires split q/k/v convolution caches")
        if linear_chunk_size <= 0 or linear_chunk_size % 8:
            raise ValueError("Ling KDA linear_chunk_size must be a positive multiple of 8")
        if flash_attention is not None and bool(flash_attention.get("enable", False)):
            unsupported = {
                name: int(flash_attention.get(name, 8))
                for name in ("q_bits", "k_bits", "v_bits", "s_bits", "p_bits")
                if int(flash_attention.get(name, 8)) not in (8, 16)
            }
            if unsupported:
                raise ValueError(f"Ling FlashAttention supports only 8/16-bit tensors: {unsupported}")

        kwargs.pop("mtp_config", None)
        super().__init__(
            model_name=model_name,
            model_type=model_type,
            mtp_config=None,
            spec_decode_mode=None,
            linear_chunk_size=linear_chunk_size,
            split_conv_cache=split_conv_cache,
            fuse_gdr_ops=fuse_gdr_ops,
            fuse_gdr_block_recurrent_ops=fuse_gdr_block_recurrent_ops,
            flash_attention=flash_attention,
            **kwargs,
        )


__all__ = ["XHLing3FlashModelConfig"]
