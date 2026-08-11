"""Config + registration contract tests for Unlimited-OCR (CPU, no weights)."""

from pathlib import Path

import pytest

HF_MODEL = "./data/models/Unlimited-OCR"
BASE_LLM_CONFIG = "configs_merak/xh2a/llm_models/unlimited_ocr/base/unlimited_ocr_llm_base_xh2a_32k.py"
BASE_VISUAL_CONFIG = "configs_merak/xh2a/llm_models/unlimited_ocr/base/unlimited_ocr_visual_base_xh2a_32k.py"
GUNDAM_LLM_CONFIG = "configs_merak/xh2a/llm_models/unlimited_ocr/gundam/unlimited_ocr_llm_gundam_xh2a_32k.py"
GUNDAM_VISUAL_CONFIG = "configs_merak/xh2a/llm_models/unlimited_ocr/gundam/unlimited_ocr_visual_gundam_xh2a_32k.py"
CALIB_LLM_CONFIG = "configs_merak/xh2a/llm_models/unlimited_ocr/_unlimited_ocr_xh2a_calib.py"


def _base_visual_dict():
    return {
        "model_type": "UnlimitedOCRForCausalLM_visual",
        "hf_model": HF_MODEL,
        "export_mode": "base",
        "image_size": 1024,
        "base_size": 1024,
        "crop_mode": False,
        "patch_size": 16,
        "downsample_ratio": 4,
        "image_token_id": 128815,
    }


def _gundam_visual_dict():
    d = _base_visual_dict()
    d.update({"export_mode": "gundam", "image_size": 640, "crop_mode": True, "max_crop_num": 32})
    return d


def test_unlimited_ocr_model_config_builds_visual_subconfig():
    from xhmodel_merak.xh_llm.models.unlimited_ocr.xh_unlimited_ocr_config import (
        XHUnlimitedOCRModelConfig,
        XHUnlimitedOCRVisualConfig,
    )

    config = XHUnlimitedOCRModelConfig(
        model_name="unlimited_ocr_base",
        model_type="UnlimitedOCRForCausalLM",
        hf_model=HF_MODEL,
        visual_config=_base_visual_dict(),
    )
    assert isinstance(config.visual_config, XHUnlimitedOCRVisualConfig)
    assert config.image_token_id == 128815
    assert config.sliding_window_size == 128
    assert config.prefill_chunk_length == 256
    assert config.context_max_length == 32768
    assert config.visual_config.patch_size == 16
    assert config.visual_config.downsample_ratio == 4
    assert config.visual_config.image_size == 1024
    assert config.visual_config.base_size == 1024
    assert config.visual_config.crop_mode is False


def test_unlimited_ocr_model_config_requires_visual_config():
    from xhmodel_merak.xh_llm.models.unlimited_ocr.xh_unlimited_ocr_config import (
        XHUnlimitedOCRModelConfig,
    )

    with pytest.raises(ValueError):
        XHUnlimitedOCRModelConfig(
            model_name="unlimited_ocr_base",
            model_type="UnlimitedOCRForCausalLM",
            hf_model=HF_MODEL,
            visual_config=None,
        )


def test_unlimited_ocr_visual_subconfig_defaults_name_and_hf_model():
    from xhmodel_merak.xh_llm.models.unlimited_ocr.xh_unlimited_ocr_config import (
        XHUnlimitedOCRModelConfig,
    )

    config = XHUnlimitedOCRModelConfig(
        model_name="unlimited_ocr_base",
        model_type="UnlimitedOCRForCausalLM",
        hf_model=HF_MODEL,
        visual_config={"export_mode": "base"},
    )
    # sub-config name/hf_model/image_token_id are inherited from the parent.
    assert config.visual_config.model_name == "unlimited_ocr_base_visual"
    assert Path(config.visual_config.hf_model) == Path(HF_MODEL)
    assert config.visual_config.image_token_id == 128815


def test_unlimited_ocr_gundam_config_contract():
    from xhmodel_merak.xh_llm.models.unlimited_ocr.xh_unlimited_ocr_config import (
        XHUnlimitedOCRModelConfig,
    )

    config = XHUnlimitedOCRModelConfig(
        model_name="unlimited_ocr_gundam",
        model_type="UnlimitedOCRForCausalLM",
        hf_model=HF_MODEL,
        visual_config=_gundam_visual_dict(),
    )
    assert config.visual_config.crop_mode is True
    assert config.visual_config.image_size == 640
    assert config.visual_config.base_size == 1024
    assert config.visual_config.max_crop_num == 32
    assert config.visual_config.hmonnx_export is False


def test_unlimited_ocr_base_config_is_hmonnx_exportable():
    from xhmodel_merak.xh_llm.models.unlimited_ocr.xh_unlimited_ocr_config import (
        XHUnlimitedOCRModelConfig,
    )

    config = XHUnlimitedOCRModelConfig(
        model_name="unlimited_ocr_base",
        model_type="UnlimitedOCRForCausalLM",
        hf_model=HF_MODEL,
        visual_config=_base_visual_dict(),
    )
    assert config.visual_config.hmonnx_export is True


def test_unlimited_ocr_registration_resolves_model_classes():
    from xhmodel_merak.xh_llm.builder import get_model_class

    llm_cls = get_model_class(
        {"model_name": "unlimited_ocr", "model_type": "UnlimitedOCRForCausalLM", "hf_model": HF_MODEL}
    )
    visual_cls = get_model_class(
        {
            "model_name": "unlimited_ocr_visual",
            "model_type": "UnlimitedOCRForCausalLM_visual",
            "hf_model": HF_MODEL,
        }
    )
    assert llm_cls is not None and llm_cls.__name__ == "XHUnlimitedOCRModel"
    assert visual_cls is not None and visual_cls.__name__ == "XHUnlimitedOCRVisualModel"


def test_unlimited_ocr_auto_llm_config_parses_base_and_gundam():
    from xhmodel_merak.xh_llm import AutoLLMConfig
    from xhmodel_merak.xh_llm.models.unlimited_ocr.xh_unlimited_ocr_config import (
        XHUnlimitedOCRModelConfig,
        XHUnlimitedOCRVisualConfig,
    )

    for visual in (_base_visual_dict(), _gundam_visual_dict()):
        cfg = AutoLLMConfig.from_pretrained(
            {
                "model_name": "unlimited_ocr",
                "model_type": "UnlimitedOCRForCausalLM",
                "hf_model": HF_MODEL,
                "visual_config": visual,
            }
        )
        assert isinstance(cfg, XHUnlimitedOCRModelConfig)
        assert isinstance(cfg.visual_config, XHUnlimitedOCRVisualConfig)
        assert cfg.visual_config.model_type == "UnlimitedOCRForCausalLM_visual"


def test_unlimited_ocr_hf_model_path_can_be_overridden_by_env(monkeypatch, tmp_path):
    from xhmodel_merak.xh_llm.models.unlimited_ocr.xh_unlimited_ocr_config import (
        XHUnlimitedOCRModelConfig,
        XHUnlimitedOCRVisualConfig,
    )

    model_dir = tmp_path / "Unlimited-OCR"
    calib_dir = tmp_path / "calib"
    model_dir.mkdir()
    calib_dir.mkdir()
    monkeypatch.setenv("UNLIMITED_OCR_HF_MODEL", str(model_dir))
    monkeypatch.setenv("UNLIMITED_OCR_CALIB_IMAGE_DIR", str(calib_dir))
    config = XHUnlimitedOCRModelConfig(
        model_name="unlimited_ocr_base",
        model_type="UnlimitedOCRForCausalLM",
        hf_model=HF_MODEL,
        visual_config={"export_mode": "base", "hf_model": HF_MODEL},
        calib_config={"enable": True, "image_dir": "./data/calib_data/sampledata"},
    )
    visual_config = XHUnlimitedOCRVisualConfig(
        model_name="unlimited_ocr_visual",
        model_type="UnlimitedOCRForCausalLM_visual",
        hf_model=HF_MODEL,
        export_mode="base",
    )
    assert Path(config.hf_model) == model_dir
    assert Path(config.visual_config.hf_model) == model_dir
    assert Path(config.calib_config["image_dir"]) == calib_dir
    assert Path(visual_config.hf_model) == model_dir


def test_unlimited_ocr_empty_env_paths_fall_back_to_config(monkeypatch):
    from xhmodel_merak.xh_llm.models.unlimited_ocr.xh_unlimited_ocr_config import XHUnlimitedOCRModelConfig

    monkeypatch.setenv("UNLIMITED_OCR_HF_MODEL", "")
    monkeypatch.setenv("UNLIMITED_OCR_CALIB_IMAGE_DIR", "")
    config = XHUnlimitedOCRModelConfig(
        model_name="unlimited_ocr_base",
        model_type="UnlimitedOCRForCausalLM",
        hf_model=HF_MODEL,
        visual_config={"export_mode": "base"},
        calib_config={"enable": True, "image_dir": "./data/calib_data/sampledata"},
    )
    assert Path(config.hf_model) == Path(HF_MODEL)
    assert Path(config.visual_config.hf_model) == Path(HF_MODEL)
    assert Path(config.calib_config["image_dir"]) == Path("./data/calib_data/sampledata")


def test_unlimited_ocr_relative_env_paths_are_normalized(monkeypatch):
    from xhmodel_merak.xh_llm.models.unlimited_ocr.xh_unlimited_ocr_config import XHUnlimitedOCRModelConfig

    monkeypatch.setenv("UNLIMITED_OCR_HF_MODEL", "./data/models/Custom-OCR")
    monkeypatch.setenv("UNLIMITED_OCR_CALIB_IMAGE_DIR", "./data/calib_data/custom")
    config = XHUnlimitedOCRModelConfig(
        model_name="unlimited_ocr_base",
        model_type="UnlimitedOCRForCausalLM",
        hf_model=HF_MODEL,
        visual_config={"export_mode": "base"},
        calib_config={"enable": True, "image_dir": "./data/calib_data/sampledata"},
    )
    assert Path(config.hf_model) == Path("data/models/Custom-OCR")
    assert Path(config.visual_config.hf_model) == Path("data/models/Custom-OCR")
    assert Path(config.calib_config["image_dir"]) == Path("data/calib_data/custom")


def test_unlimited_ocr_missing_hf_model_path_fails_before_hmonnx_export(tmp_path):
    from xhmodel_merak.xh_llm.models.unlimited_ocr.unlimited_ocr_model import XHUnlimitedOCRModel
    from xhmodel_merak.xh_llm.models.unlimited_ocr.xh_unlimited_ocr_config import XHUnlimitedOCRModelConfig

    missing_model = tmp_path / "missing-model"
    config = XHUnlimitedOCRModelConfig(
        model_name="unlimited_ocr_base",
        model_type="UnlimitedOCRForCausalLM",
        hf_model=str(missing_model),
        visual_config=_base_visual_dict(),
    )
    model = XHUnlimitedOCRModel(config)
    with pytest.raises(FileNotFoundError, match="UNLIMITED_OCR_HF_MODEL"):
        model._ensure_hmonnx_export_supported()


def test_unlimited_ocr_missing_calib_image_dir_collects_no_images(tmp_path):
    from xhmodel_merak.xh_llm.models.unlimited_ocr.unlimited_ocr_model import XHUnlimitedOCRModel
    from xhmodel_merak.xh_llm.models.unlimited_ocr.xh_unlimited_ocr_config import XHUnlimitedOCRModelConfig

    missing_calib_dir = tmp_path / "missing-calib"
    config = XHUnlimitedOCRModelConfig(
        model_name="unlimited_ocr_base",
        model_type="UnlimitedOCRForCausalLM",
        hf_model=HF_MODEL,
        visual_config=_base_visual_dict(),
        calib_config={"enable": True, "image_dir": str(missing_calib_dir)},
    )
    model = XHUnlimitedOCRModel(config)
    assert model._collect_calib_images() == []


@pytest.mark.parametrize("config_path", [BASE_LLM_CONFIG, GUNDAM_LLM_CONFIG, CALIB_LLM_CONFIG])
def test_unlimited_ocr_file_llm_configs_apply_env_path_overrides(monkeypatch, tmp_path, config_path):
    from xhmodel_merak.xh_llm import AutoLLMConfig
    from xhquant.api import Config

    model_dir = tmp_path / "Unlimited-OCR"
    calib_dir = tmp_path / "calib"
    model_dir.mkdir()
    calib_dir.mkdir()
    monkeypatch.setenv("UNLIMITED_OCR_HF_MODEL", str(model_dir))
    monkeypatch.setenv("UNLIMITED_OCR_CALIB_IMAGE_DIR", str(calib_dir))

    cfg = Config.fromfile(config_path)
    model_config = AutoLLMConfig.from_pretrained(cfg.model)
    assert Path(model_config.hf_model) == model_dir
    assert Path(model_config.visual_config.hf_model) == model_dir
    if model_config.calib_config is not None:
        assert Path(model_config.calib_config["image_dir"]) == calib_dir


def test_unlimited_ocr_calib_config_uses_high_precision_llm_and_visual_quant():
    from xhmodel_merak.xh_llm import AutoLLMConfig
    from xhquant.api import Config

    model_config = AutoLLMConfig.from_pretrained(Config.fromfile(CALIB_LLM_CONFIG).model)
    assert model_config.quant_scheme.quant_type == "w16a16h0_sefp"
    assert model_config.quant_scheme.nodes["lm_head"].quant_type == "w16a16h0_sefp"
    assert model_config.visual_config.quant_scheme.quant_type == "w16a16h0_sefp"


@pytest.mark.parametrize("config_path", [BASE_VISUAL_CONFIG, GUNDAM_VISUAL_CONFIG])
def test_unlimited_ocr_file_visual_configs_apply_env_path_overrides(monkeypatch, tmp_path, config_path):
    from xhmodel_merak.xh_llm import AutoLLMConfig
    from xhmodel_merak.xh_llm.models.unlimited_ocr.xh_unlimited_ocr_config import XHUnlimitedOCRVisualConfig
    from xhquant.api import Config

    model_dir = tmp_path / "Unlimited-OCR"
    model_dir.mkdir()
    monkeypatch.setenv("UNLIMITED_OCR_HF_MODEL", str(model_dir))

    cfg = Config.fromfile(config_path)
    visual_config = AutoLLMConfig.from_pretrained(cfg.model)
    assert isinstance(visual_config, XHUnlimitedOCRVisualConfig)
    assert Path(visual_config.hf_model) == model_dir
    assert not hasattr(visual_config, "visual_config")


def test_unlimited_ocr_independent_visual_configs_do_not_inherit_llm_fields(monkeypatch, tmp_path):
    from xhmodel_merak.xh_llm import AutoLLMConfig
    from xhquant.api import Config

    model_dir = tmp_path / "Unlimited-OCR"
    model_dir.mkdir()
    monkeypatch.setenv("UNLIMITED_OCR_HF_MODEL", str(model_dir))

    base_visual = AutoLLMConfig.from_pretrained(Config.fromfile(BASE_VISUAL_CONFIG).model)
    gundam_visual = AutoLLMConfig.from_pretrained(Config.fromfile(GUNDAM_VISUAL_CONFIG).model)

    assert base_visual.hmonnx_export is True
    assert base_visual.crop_mode is False
    assert gundam_visual.hmonnx_export is False
    assert gundam_visual.crop_mode is True
    assert Path(base_visual.hf_model) == model_dir
    assert Path(gundam_visual.hf_model) == model_dir
    assert not hasattr(base_visual, "visual_config")
    assert not hasattr(gundam_visual, "visual_config")


def test_unlimited_ocr_gundam_hmonnx_export_is_rejected_before_weights(tmp_path):
    from xhmodel_merak.xh_llm.models.unlimited_ocr.unlimited_ocr_model import XHUnlimitedOCRModel
    from xhmodel_merak.xh_llm.models.unlimited_ocr.unlimited_ocr_visual_model import XHUnlimitedOCRVisualModel
    from xhmodel_merak.xh_llm.models.unlimited_ocr.xh_unlimited_ocr_config import (
        XHUnlimitedOCRModelConfig,
        XHUnlimitedOCRVisualConfig,
    )

    config = XHUnlimitedOCRModelConfig(
        model_name="unlimited_ocr_gundam",
        model_type="UnlimitedOCRForCausalLM",
        hf_model=HF_MODEL,
        visual_config=_gundam_visual_dict(),
    )
    model = XHUnlimitedOCRModel(config)
    with pytest.raises(NotImplementedError, match="export_mode='gundam'.*crop_mode=True.*hmonnx_export=False"):
        model._ensure_hmonnx_export_supported()

    visual_config = XHUnlimitedOCRVisualConfig(
        model_name="unlimited_ocr_gundam_visual",
        model_type="UnlimitedOCRForCausalLM_visual",
        hf_model=HF_MODEL,
        export_mode="gundam",
        crop_mode=True,
        hmonnx_export=False,
    )
    visual_model = XHUnlimitedOCRVisualModel(visual_config)
    with pytest.raises(NotImplementedError, match="export_mode='gundam'.*crop_mode=True.*hmonnx_export=False"):
        visual_model.export_hmonnx(str(tmp_path))
