"""Structural import/export tests for the Unlimited-OCR merak adaptation.

CPU-only, no real weights: verifies package files exist and public symbols are
importable/exported. Mirrors tests/gemma4_merak_import_test.py.
"""

import importlib.util
from pathlib import Path

MODEL_DIR = Path("xhmodel_merak/xh_llm/models/unlimited_ocr")
EXAMPLE_DIR = Path("examples_merak/llm/unlimited_ocr")


def test_unlimited_ocr_module_files_exist_and_are_exported():
    expected_files = [
        "workflow.py",
        "__init__.py",
        "_llm_model_impl.py",
        "_visual_model_impl.py",
        "data_preprocess.py",
        "unlimited_ocr_processor.py",
        "unlimited_ocr_model.py",
        "unlimited_ocr_visual_model.py",
        "unlimited_ocr_hmonnx_inference.py",
        "xh_unlimited_ocr_config.py",
        "modeling_unlimitedocr.py",
        "modeling_deepseekv2.py",
        "deepencoder.py",
        "modeling_unlimitedocr_patch.py",
    ]
    for name in expected_files:
        assert (MODEL_DIR / name).exists(), f"missing {name}"

    init_text = (MODEL_DIR / "__init__.py").read_text(encoding="utf-8")
    for exported_name in [
        "UnlimitedOCRForCausalLM",
        "XHUnlimitedOCRModel",
        "XHUnlimitedOCRVisualModel",
        "XHUnlimitedOCRHMONNXModel",
        "XHUnlimitedOCRProcessor",
        "XHUnlimitedOCRModelConfig",
        "XHUnlimitedOCRVisualConfig",
        "UnlimitedOCRDataPreprocess",
    ]:
        assert exported_name in init_text, f"{exported_name} not exported in __init__.py"


def test_unlimited_ocr_public_symbols_import():
    from xhmodel_merak.xh_llm.models.unlimited_ocr import (
        UnlimitedOCRDataPreprocess,
        XHUnlimitedOCRHMONNXModel,
        XHUnlimitedOCRModel,
        XHUnlimitedOCRModelConfig,
        XHUnlimitedOCRProcessor,
        XHUnlimitedOCRVisualConfig,
        XHUnlimitedOCRVisualModel,
    )

    assert XHUnlimitedOCRModel.__name__ == "XHUnlimitedOCRModel"
    assert XHUnlimitedOCRVisualModel.__name__ == "XHUnlimitedOCRVisualModel"
    assert XHUnlimitedOCRHMONNXModel.__name__ == "XHUnlimitedOCRHMONNXModel"
    assert XHUnlimitedOCRProcessor.__name__ == "XHUnlimitedOCRProcessor"
    assert XHUnlimitedOCRModelConfig.__name__ == "XHUnlimitedOCRModelConfig"
    assert XHUnlimitedOCRVisualConfig.__name__ == "XHUnlimitedOCRVisualConfig"
    assert UnlimitedOCRDataPreprocess.__name__ == "UnlimitedOCRDataPreprocess"


def test_unlimited_ocr_class_wiring():
    from xhmodel_merak.xh_llm.models.unlimited_ocr import XHUnlimitedOCRModel
    from xhmodel_merak.xh_llm.models.unlimited_ocr.unlimited_ocr_hmonnx_inference import (
        XHUnlimitedOCRHMONNXModel,
    )
    from xhmodel_merak.xh_llm.types import VLLMModelMeta

    assert XHUnlimitedOCRModel.HMONNXINFERENCE_CLS is XHUnlimitedOCRHMONNXModel
    assert XHUnlimitedOCRModel.META_CLS is VLLMModelMeta
    assert callable(XHUnlimitedOCRModel.BUILD_HF_COMPATIBLE_FUNC)


def test_expected_unlimited_ocr_examples_exist():
    expected_files = [
        "unlimited_ocr_workflow.py",
        "unlimited_ocr_xh_hmonnx_generate.py",
        "README.md",
        "debug_scripts/native_unlimited_ocr_forward.py",
        "debug_scripts/unlimited_ocr_visual_xh_debug.py",
        "debug_scripts/unlimited_ocr_llm_xh_debug.py",
        "debug_scripts/unlimited_ocr_crop_smoke.py",
        "debug_scripts/README.md",
    ]
    for name in expected_files:
        assert (EXAMPLE_DIR / name).exists(), f"missing {name}"


def test_unlimited_ocr_generate_script_is_importable():
    script_path = EXAMPLE_DIR / "unlimited_ocr_xh_hmonnx_generate.py"
    spec = importlib.util.spec_from_file_location("unlimited_ocr_xh_hmonnx_generate", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "main")
