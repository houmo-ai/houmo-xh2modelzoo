from xhmodel_merak.configuration_utils import HFModelConfig
from xhquant.api import QuantScheme

from ...vision_llm_model import VisionLLMModelConfig


class XHGemma4VisualConfig(HFModelConfig):  # noqa: N801
    def __init__(
        self,
        *,
        image_seq_length: int = 280,
        patch_size: int = 16,
        pooling_kernel_size: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_seq_length = image_seq_length
        self.patch_size = patch_size
        self.pooling_kernel_size = pooling_kernel_size


class XHGemma4ModelConfig(VisionLLMModelConfig):  # noqa: N801
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
        visual_config: dict | XHGemma4VisualConfig | None = None,
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
            visual_config = XHGemma4VisualConfig(**visual_config)
        self.visual_config = visual_config

        self.image_token_id: int | None = None
        self.video_token_id: int | None = None
        self.audio_token_id: int | None = None
        self.boi_token_id: int | None = None
        self.eoi_token_id: int | None = None
        self.mm_token_type_ids_enabled: bool = True
