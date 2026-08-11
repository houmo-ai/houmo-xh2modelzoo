import os
from pathlib import Path

from xhmodel_merak.configuration_utils import HFModelConfig
from xhquant.api import QuantScheme

from ...vision_llm_model import VisionLLMModelConfig


def resolve_unlimited_ocr_path(path: str | None, *, env_var: str, must_exist: bool = False) -> str | None:
    resolved = os.environ.get(env_var) or path
    if resolved is None:
        return None
    resolved_path = Path(resolved).expanduser()
    if must_exist and not resolved_path.exists():
        raise FileNotFoundError(
            f"Unlimited-OCR path does not exist: {resolved_path}. "
            f"Set {env_var} to override the default path."
        )
    return str(resolved_path)


def resolve_unlimited_ocr_calib_config(calib_config: dict | None) -> dict | None:
    if calib_config is None:
        return None
    resolved = dict(calib_config)
    image_dir = resolve_unlimited_ocr_path(
        resolved.get("image_dir"), env_var="UNLIMITED_OCR_CALIB_IMAGE_DIR", must_exist=False
    )
    if image_dir is not None:
        resolved["image_dir"] = image_dir
    return resolved


class XHUnlimitedOCRVisualConfig(HFModelConfig):
    def __init__(
        self,
        *,
        export_mode: str = "base",
        image_size: int = 1024,
        base_size: int = 1024,
        crop_mode: bool = False,
        patch_size: int = 16,
        downsample_ratio: int = 4,
        image_token_id: int = 128815,
        image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize: bool = True,
        max_crop_num: int = 32,
        hmonnx_export: bool | None = None,
        **kwargs,
    ):
        kwargs["hf_model"] = resolve_unlimited_ocr_path(
            kwargs.get("hf_model"), env_var="UNLIMITED_OCR_HF_MODEL", must_exist=False
        )
        super().__init__(**kwargs)
        self.export_mode = export_mode
        self.image_size = image_size
        self.base_size = base_size
        self.crop_mode = crop_mode
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.image_token_id = image_token_id
        self.image_mean = image_mean
        self.image_std = image_std
        self.normalize = normalize
        self.max_crop_num = max_crop_num
        if hmonnx_export is None:
            hmonnx_export = export_mode == "base" and not crop_mode
        self.hmonnx_export = bool(hmonnx_export)


class XHUnlimitedOCRModelConfig(VisionLLMModelConfig):
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
        context_max_length: int = 32768,
        prefill_chunk_length: int = 256,
        num_logits_to_keep: int | None = 1,
        mix_search: bool = False,
        use_cache: bool = True,
        sliding_window_size: int = 128,
        image_token_id: int = 128815,
        visual_config: dict | XHUnlimitedOCRVisualConfig | None = None,
        calib_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            chip_arch=chip_arch,
            model_type=model_type,
            hf_model=resolve_unlimited_ocr_path(hf_model, env_var="UNLIMITED_OCR_HF_MODEL", must_exist=False),
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
        if visual_config is None:
            raise ValueError("visual_config must be provided for XHUnlimitedOCRModelConfig")
        hf_model = self.hf_model
        if isinstance(visual_config, dict):
            visual_config = dict(visual_config)
            if "model_name" not in visual_config:
                visual_config["model_name"] = f"{model_name}_visual"
            if "hf_model" not in visual_config:
                visual_config["hf_model"] = hf_model
            else:
                visual_config["hf_model"] = resolve_unlimited_ocr_path(
                    visual_config["hf_model"], env_var="UNLIMITED_OCR_HF_MODEL", must_exist=False
                )
            if "image_token_id" not in visual_config:
                visual_config["image_token_id"] = image_token_id
            visual_config = XHUnlimitedOCRVisualConfig(**visual_config)
        self.visual_config = visual_config
        self.sliding_window_size = sliding_window_size
        self.image_token_id = image_token_id
        # Optional real-image calibration. When None or disabled, PTQ falls back
        # to the text-only dummy calibration.
        self.calib_config = resolve_unlimited_ocr_calib_config(calib_config)
