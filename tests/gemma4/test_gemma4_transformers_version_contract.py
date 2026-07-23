from pathlib import Path


def test_gemma4_model_classes_own_their_transformers_versions():
    from xhmodel_merak.xh_llm.models.gemma4_series import (
        XHGemma4SeriesModel,
        XHGemma4UnifiedModel,
    )

    assert XHGemma4SeriesModel.transformers_min_version == "5.5.0"
    assert XHGemma4UnifiedModel.transformers_min_version == "5.13.0"
    assert issubclass(XHGemma4UnifiedModel, XHGemma4SeriesModel)


def test_gemma4_architectures_route_to_version_scoped_model_classes():
    from xhmodel_merak.xh_llm.builder import get_model_class

    legacy_cls = get_model_class(
        {
            "chip_arch": "XH2a",
            "model_type": "Gemma4ForConditionalGeneration",
            "model_name": "gemma4_legacy_version_contract",
        }
    )
    unified_cls = get_model_class(
        {
            "chip_arch": "XH2a",
            "model_type": "Gemma4UnifiedForConditionalGeneration",
            "model_name": "gemma4_12b_version_contract",
        }
    )

    assert legacy_cls.__name__ == "XHGemma4SeriesModel"
    assert legacy_cls.transformers_min_version == "5.5.0"
    assert unified_cls.__name__ == "XHGemma4UnifiedModel"
    assert unified_cls.transformers_min_version == "5.13.0"


def test_gemma4_12b_does_not_raise_the_project_transformers_pin():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    requirements = Path("requirements/requirements.txt").read_text(encoding="utf-8")

    assert "transformers (>=4.57.0,<4.58.0)" in pyproject
    assert "transformers==4.57.1" in requirements
