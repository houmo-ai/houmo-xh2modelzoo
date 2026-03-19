from pathlib import Path


def test_qwen3_omni_module_files_exist_and_exported():
    base = Path("xh_model_zoo/xh_llm/models/qwen3_omni")
    assert (base / "__init__.py").exists()
    assert (base / "modeling_qwen3_omni_moe.py").exists()
    assert (base / "processing_qwen3_omni_moe.py").exists()
    init_text = (base / "__init__.py").read_text(encoding="utf-8")
    assert "Qwen3OmniMoeForConditionalGeneration" in init_text


def test_llm_converter_supports_qwen3_omni_architecture():
    converter_file = Path("xh_model_zoo/xh_llm/llm_converter.py")
    text = converter_file.read_text(encoding="utf-8")
    assert "Qwen3OmniMoeForConditionalGeneration" in text
    assert "Qwen3OmniMoeConverterXH2a" in text


def test_expected_qwen3omni_examples_exist():
    base = Path("examples/llm/qwen3omni")
    expected = [
        "qwen3_omni_xh2a_export_text.py",
        "qwen3_omni_xh2a_export_vision.py",
        "qwen3_omni_xh2a_export_audio.py",
        "qwen3_omni_xh2a_export_talker_model.py",
        "qwen3_omni_xh2a_export_talker_prediction.py",
        "qwen3_omni_xh2a_export_talker_projection.py",
        "qwen3_omni_xh2a_export_code2wav.py",
        "qwen3_omni_onnx_golden.py",
        "qwen3_omni_demo.py",
        "qwen3_omni_hmonnx_forward.py",
    ]
    for name in expected:
        assert (base / name).exists(), f"missing {name}"
