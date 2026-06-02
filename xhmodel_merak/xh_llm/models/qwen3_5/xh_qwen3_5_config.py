from xhmodel_merak.configuration_utils import HFModelConfig
from xhquant.api import QuantScheme

from ...vision_llm_model import VisionLLMModelConfig


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
        split_conv_cache: bool = False,
        visual_config: dict | XHQwen3_5_VisualConfig | None = None,
        spec_decode_mode: str | None = None,
        mtp_config: dict | XHQwen3_5_MTPConfig | None = None,
        dflash_config: dict | XHQwen3_5_DFlashConfig | None = None,
        num_draft_tokens: int = 4,
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
        if isinstance(visual_config, dict):
            if "model_name" not in visual_config:
                visual_config["model_name"] = f"{model_name}_visual"
            if "hf_model" not in visual_config:
                visual_config["hf_model"] = hf_model
            visual_config = XHQwen3_5_VisualConfig(**visual_config)
        self.visual_config = visual_config

        self.linear_chunk_size = linear_chunk_size
        self.split_conv_cache = split_conv_cache

        self.spec_decode_mode = spec_decode_mode
        self.num_draft_tokens = num_draft_tokens

        if isinstance(mtp_config, dict):
            if "model_name" not in mtp_config:
                mtp_config["model_name"] = f"{model_name}_mtp"
            if "hf_model" not in mtp_config:
                mtp_config["hf_model"] = hf_model
            mtp_config = XHQwen3_5_MTPConfig(**mtp_config)
        self.mtp_config = mtp_config

        if isinstance(dflash_config, dict):
            if "model_name" not in dflash_config:
                dflash_config["model_name"] = f"{model_name}_dflash"
            if "hf_model" not in dflash_config:
                dflash_config["hf_model"] = hf_model
            if "target_model_dir" not in dflash_config:
                dflash_config["target_model_dir"] = hf_model
            dflash_config = XHQwen3_5_DFlashConfig(**dflash_config)
        self.dflash_config = dflash_config

        self.image_token_id: int | None = None
        self.video_token_id: int | None = None
        self.vision_start_token_id: int | None = None

        self.vision_end_token_id: int | None = None
        self.spatial_merge_size: int | None = None
