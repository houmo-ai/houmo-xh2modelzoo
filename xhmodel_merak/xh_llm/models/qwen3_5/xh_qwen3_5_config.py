from collections.abc import Mapping

from xhmodel_merak.configuration_utils import HFModelConfig
from xhquant.api import QuantScheme

from ...vision_llm_model import VisionLLMModelConfig


DRAFT_BASE_QUANT_TYPE = "w8a8h1_sefp"


def build_spec_draft_quant_scheme(head_weight_bits: int = 4) -> dict:
    """Build the default MTP/DFlash draft quant scheme.

    Draft graphs use the regular W8A8 base quantization, but the large logits
    head is configurable.  Keep the default at W4 for spec-decode draft heads
    unless callers provide an explicit per-draft ``quant_scheme``.
    """
    if head_weight_bits == 8:
        return dict(quant_type=DRAFT_BASE_QUANT_TYPE)
    if head_weight_bits != 4:
        raise ValueError(
            f"Unsupported spec draft head weight bits: {head_weight_bits}. Expected 4 or 8."
        )
    return dict(
        quant_type=DRAFT_BASE_QUANT_TYPE,
        nodes_cfg=dict(
            lm_head=dict(
                w_schema=dict(
                    bits=4,
                    fp_mode="ssfp",
                    hidden_bit=False,
                )
            )
        ),
    )


class XHQwen3_5_VisualConfig(HFModelConfig):  # noqa: N801
    def __init__(
        self,
        *,
        max_size_w: int,
        max_size_h: int,
        max_size_t: int = 2,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_size_w = max_size_w
        self.max_size_h = max_size_h
        self.max_size_t = max_size_t
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size


class XHQwen3_5_MTPConfig(HFModelConfig):  # noqa: N801
    def __init__(
        self,
        *,
        hidden_size: int = 3584,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        batch_size: int = 1,
        input_sequence_length: int = 1,
        context_max_length: int = 2048,
        max_pe_length: int = 262144,
        use_cache: bool = True,
        draft_head_weight_bits: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.input_sequence_length = input_sequence_length
        self.context_max_length = context_max_length
        self.max_pe_length = max_pe_length
        self.use_cache = use_cache
        self.draft_head_weight_bits = draft_head_weight_bits
        if getattr(self, "quant_scheme", None) is None:
            self.quant_scheme = build_spec_draft_quant_scheme(draft_head_weight_bits)

    @property
    def hf_model_dir(self) -> str:
        return self.hf_model


class XHQwen3_5_DFlashConfig(HFModelConfig):  # noqa: N801
    def __init__(
        self,
        *,
        mode: str = "decode",
        target_model_dir: str | None = None,
        hidden_size: int = 3584,
        num_attention_heads: int = 28,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        num_hidden_layers: int = 1,
        num_target_layers: int = 4,
        batch_size: int = 1,
        input_sequence_length: int = 1,
        max_pe_length: int = 262144,
        max_sequence_length: int = 256,
        draft_head_weight_bits: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode = mode
        self.target_model_dir = target_model_dir
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_target_layers = num_target_layers
        self.batch_size = batch_size
        self.input_sequence_length = input_sequence_length
        self.max_pe_length = max_pe_length
        self.max_sequence_length = max_sequence_length
        self.draft_head_weight_bits = draft_head_weight_bits
        if getattr(self, "quant_scheme", None) is None:
            self.quant_scheme = build_spec_draft_quant_scheme(draft_head_weight_bits)

    @property
    def dflash_model_dir(self) -> str:
        return self.hf_model


class XHQwen3_5ModelConfig(VisionLLMModelConfig):  # noqa: N801
    def __init__(
        self,
        *,
        model_name: str,
        chip_arch: str = "XH2a",
        model_type: str | None = None,
        quant_scheme: dict | QuantScheme | None = None,
        quant_weight: str | None = None,
        hf_model: str | None = None,
        batch_size: int = 1,
        context_max_length: int = 2048,
        prefill_chunk_length: int = 256,
        num_logits_to_keep: int | None = 1,
        mix_search: bool = False,
        use_cache: bool = True,
        linear_chunk_size: int = 64,
        fuse_gdr_ops: bool = False,
        fuse_gdr_block_recurrent_ops: bool = False,
        split_conv_cache: bool = True,
        normalize_force_fp32: bool = False,
        use_manual_depthwise_conv1d: bool = False,
        cumsum_matmul_quant_config: dict | None = None,
        visual_config: dict | XHQwen3_5_VisualConfig | None = None,
        spec_decode_mode: str | None = None,
        mtp_config: dict | XHQwen3_5_MTPConfig | None = None,
        dflash_config: dict | XHQwen3_5_DFlashConfig | None = None,
        num_draft_tokens: int = 4,
        spec_draft_head_weight_bits: int = 4,
        mtp_head_k: int | None = None,
        reranked_repo_dir: str | None = None,
        force_rerank: bool = False,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            chip_arch=chip_arch,
            model_type=model_type,
            hf_model=hf_model,
            quant_scheme=quant_scheme,
            quant_weight=quant_weight,
            batch_size=batch_size,
            context_max_length=context_max_length,
            prefill_chunk_length=prefill_chunk_length,
            num_logits_to_keep=num_logits_to_keep,
            mix_search=mix_search,
            use_cache=use_cache,
            **kwargs,
        )
        if isinstance(visual_config, Mapping):
            visual_config = dict(visual_config)
            if "model_name" not in visual_config:
                visual_config["model_name"] = f"{model_name}_visual"
            if "hf_model" not in visual_config:
                visual_config["hf_model"] = hf_model
            visual_config = XHQwen3_5_VisualConfig(**visual_config)
        self.visual_config = visual_config

        self.linear_chunk_size = linear_chunk_size
        self.fuse_gdr_ops = fuse_gdr_ops
        self.fuse_gdr_block_recurrent_ops = fuse_gdr_block_recurrent_ops
        self.split_conv_cache = split_conv_cache
        self.normalize_force_fp32 = normalize_force_fp32
        self.use_manual_depthwise_conv1d = use_manual_depthwise_conv1d
        self.cumsum_matmul_quant_config = cumsum_matmul_quant_config

        if spec_decode_mode in {"mtp", "dflash"} and int(batch_size) != 1:
            raise ValueError(
                "Qwen3.5 multi-batch export does not support "
                f"spec_decode_mode={spec_decode_mode!r}; use batch_size=1 for MTP/DFlash "
                "or disable spec_decode_mode for batch_size>1."
            )

        self.spec_decode_mode = spec_decode_mode
        self.num_draft_tokens = num_draft_tokens
        self.spec_draft_head_weight_bits = spec_draft_head_weight_bits
        self.mtp_head_k = mtp_head_k
        self.reranked_repo_dir = reranked_repo_dir
        self.force_rerank = force_rerank

        if isinstance(mtp_config, Mapping):
            mtp_config = dict(mtp_config)
            if "model_name" not in mtp_config:
                mtp_config["model_name"] = f"{model_name}_mtp"
            if "hf_model" not in mtp_config:
                mtp_config["hf_model"] = hf_model
            if "draft_head_weight_bits" not in mtp_config:
                mtp_config["draft_head_weight_bits"] = spec_draft_head_weight_bits
            mtp_config = XHQwen3_5_MTPConfig(**mtp_config)
        self.mtp_config = mtp_config

        if isinstance(dflash_config, Mapping):
            dflash_config = dict(dflash_config)
            if "model_name" not in dflash_config:
                dflash_config["model_name"] = f"{model_name}_dflash"
            if "hf_model" not in dflash_config:
                dflash_config["hf_model"] = hf_model
            if dflash_config.get("target_model_dir") is None:
                dflash_config["target_model_dir"] = hf_model
            if "draft_head_weight_bits" not in dflash_config:
                dflash_config["draft_head_weight_bits"] = spec_draft_head_weight_bits
            dflash_config = XHQwen3_5_DFlashConfig(**dflash_config)
        self.dflash_config = dflash_config
        if self.dflash_config is not None and getattr(self.dflash_config, "target_model_dir", None) is None:
            self.dflash_config.target_model_dir = hf_model

        self.image_token_id: int | None = None
        self.video_token_id: int | None = None
        self.vision_start_token_id: int | None = None

        self.vision_end_token_id: int | None = None
        self.spatial_merge_size: int | None = None
