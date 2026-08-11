from pathlib import Path

import pytest
import yaml
from PIL import Image

from xhmodel_merak.xh_llm.models.unlimited_ocr.workflow import XHUnlimitedOCRWorkflow
from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow, BaseLLMWorkflow


WORKFLOW_CONFIG = Path(
    "configs_merak/workflows/xh2a/llm_models/unlimited_ocr/base/"
    "unlimited_ocr_base_xh2a_w8a8.yaml"
)
CALIB_WORKFLOW_CONFIG = Path(
    "configs_merak/workflows/xh2a/llm_models/unlimited_ocr/base/"
    "unlimited_ocr_base_xh2a_w16a16_calib.yaml"
)


def test_unlimited_ocr_workflow_config_and_quant_contract(monkeypatch, tmp_path):
    monkeypatch.delenv("UNLIMITED_OCR_HF_MODEL", raising=False)
    model_dir = tmp_path / "Unlimited-OCR"
    model_dir.mkdir()
    workflow = AutoLLMWorkflow.from_config(
        model_dir=str(model_dir),
        config_path=str(WORKFLOW_CONFIG),
    )

    quant_result = workflow.quant(output_dir=str(tmp_path / "quant"), device="cpu")

    assert type(workflow) is XHUnlimitedOCRWorkflow
    assert quant_result.skipped is True
    assert quant_result.raw_model_dir == str(model_dir)


def test_unlimited_ocr_workflow_formats_model_name(monkeypatch, tmp_path):
    monkeypatch.delenv("UNLIMITED_OCR_HF_MODEL", raising=False)
    model_dir = tmp_path / "Unlimited-OCR"
    model_dir.mkdir()
    workflow = XHUnlimitedOCRWorkflow(str(model_dir), str(WORKFLOW_CONFIG))

    assert workflow._format_model_name(workflow.workflow_config, str(model_dir)) == (
        "xh2_unlimited_ocr_base_w8a8_256_32k_mpe32k"
    )


def test_unlimited_ocr_workflow_override_is_per_call(monkeypatch, tmp_path):
    monkeypatch.delenv("UNLIMITED_OCR_HF_MODEL", raising=False)
    model_dir = tmp_path / "Unlimited-OCR"
    model_dir.mkdir()
    workflow = XHUnlimitedOCRWorkflow(str(model_dir), str(WORKFLOW_CONFIG))

    overridden = workflow.workflow_config.with_overrides(
        {"export.model.calib_config.enable": True}
    )

    assert overridden.export["model"]["calib_config"]["enable"] is True
    assert workflow.workflow_config.export["model"]["calib_config"]["enable"] is False


def test_unlimited_ocr_workflow_override_requires_calibration_images(monkeypatch, tmp_path):
    model_dir = tmp_path / "Unlimited-OCR"
    model_dir.mkdir()
    missing_calib_dir = tmp_path / "missing-calib"
    monkeypatch.setenv("UNLIMITED_OCR_CALIB_IMAGE_DIR", str(missing_calib_dir))
    workflow = XHUnlimitedOCRWorkflow(str(model_dir), str(WORKFLOW_CONFIG))
    quant_result = workflow.quant(output_dir=str(tmp_path / "quant"), device="cpu")
    export_calls = []

    def fake_base_export(*args, **kwargs):
        export_calls.append((args, kwargs))

    monkeypatch.setattr(BaseLLMWorkflow, "export", fake_base_export)

    with pytest.raises(FileNotFoundError, match="no calibration images"):
        workflow.export(
            quant_result=quant_result,
            output_dir=str(tmp_path / "export"),
            device="cpu",
            config_overrides={"export.model.calib_config.enable": True},
        )
    assert export_calls == []


@pytest.mark.parametrize(
    ("config_path", "quant_type", "calib_enabled"),
    [
        (WORKFLOW_CONFIG, "w8a8h1_sefp", False),
        (CALIB_WORKFLOW_CONFIG, "w16a16h0_sefp", True),
    ],
)
def test_unlimited_ocr_workflow_yaml_contract(config_path, quant_type, calib_enabled):
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = data["export"]["model"]
    visual = model["visual_config"]

    assert data["quant"] is None
    assert model["model_type"] == "UnlimitedOCRForCausalLM"
    assert model["context_max_length"] == 32768
    assert model["prefill_chunk_length"] == 256
    assert model["max_pe_length"] == 32768
    assert model["quant_scheme"]["quant_type"] == quant_type
    assert visual["quant_scheme"]["quant_type"] == quant_type
    assert visual["export_mode"] == "base"
    assert visual["crop_mode"] is False
    assert visual["hmonnx_export"] is True
    assert "hf_model" not in visual
    assert model["calib_config"]["enable"] is calib_enabled


def test_unlimited_ocr_calibration_workflow_dispatches_with_real_images(monkeypatch, tmp_path):
    model_dir = tmp_path / "Unlimited-OCR"
    model_dir.mkdir()
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()
    Image.new("RGB", (16, 16), (128, 128, 128)).save(calib_dir / "sample.png")
    monkeypatch.setenv("UNLIMITED_OCR_CALIB_IMAGE_DIR", str(calib_dir))

    workflow = AutoLLMWorkflow.from_config(str(model_dir), str(CALIB_WORKFLOW_CONFIG))

    assert type(workflow) is XHUnlimitedOCRWorkflow


def test_unlimited_ocr_workflow_rejects_crop_export():
    with pytest.raises(NotImplementedError, match="base/no-crop"):
        XHUnlimitedOCRWorkflow._validate_export_mode(
            {
                "visual_config": {
                    "export_mode": "gundam",
                    "crop_mode": True,
                    "hmonnx_export": False,
                }
            }
        )


def test_unlimited_ocr_workflow_rejects_crop_during_initialization(monkeypatch, tmp_path):
    monkeypatch.delenv("UNLIMITED_OCR_HF_MODEL", raising=False)
    model_dir = tmp_path / "Unlimited-OCR"
    model_dir.mkdir()
    config_path = tmp_path / "gundam.yaml"
    config = yaml.safe_load(WORKFLOW_CONFIG.read_text(encoding="utf-8"))
    config["export"]["model"]["visual_config"].update(
        {"export_mode": "gundam", "crop_mode": True, "hmonnx_export": False}
    )
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(NotImplementedError, match="base/no-crop"):
        XHUnlimitedOCRWorkflow(str(model_dir), str(config_path))


def test_unlimited_ocr_workflow_rejects_conflicting_env_model_dir(monkeypatch, tmp_path):
    model_dir = tmp_path / "explicit-model"
    model_dir.mkdir()
    monkeypatch.setenv("UNLIMITED_OCR_HF_MODEL", str(tmp_path / "environment-model"))

    with pytest.raises(ValueError, match="conflicts with the workflow model_dir"):
        XHUnlimitedOCRWorkflow(str(model_dir), str(WORKFLOW_CONFIG))


def test_unlimited_ocr_workflow_ignores_empty_env_model_dir(monkeypatch, tmp_path):
    model_dir = tmp_path / "explicit-model"
    model_dir.mkdir()
    monkeypatch.setenv("UNLIMITED_OCR_HF_MODEL", "")

    workflow = XHUnlimitedOCRWorkflow(str(model_dir), str(WORKFLOW_CONFIG))

    assert workflow.model_dir == str(model_dir)


def test_unlimited_ocr_calibration_workflow_requires_images(monkeypatch, tmp_path):
    model_dir = tmp_path / "Unlimited-OCR"
    model_dir.mkdir()
    missing_calib_dir = tmp_path / "missing-calib"
    monkeypatch.setenv("UNLIMITED_OCR_CALIB_IMAGE_DIR", str(missing_calib_dir))

    with pytest.raises(FileNotFoundError, match="no calibration images"):
        XHUnlimitedOCRWorkflow(str(model_dir), str(CALIB_WORKFLOW_CONFIG))


def test_unlimited_ocr_workflow_builds_golden_input():
    prompt = "<image>\\nFree OCR. "

    image_path, resolved_prompt = XHUnlimitedOCRWorkflow.build_input(
        {"image": "document.png", "prompt": prompt}
    )

    assert image_path == "document.png"
    assert resolved_prompt == prompt
    assert "\\n" in resolved_prompt
    assert "\n" not in resolved_prompt


def test_unlimited_ocr_workflow_accepts_text_prompt_alias():
    assert XHUnlimitedOCRWorkflow.build_input(
        {"image": "document.png", "text": "<image>\\nFree OCR. "}
    ) == ("document.png", "<image>\\nFree OCR. ")


@pytest.mark.parametrize(
    "input_messages",
    [
        None,
        {},
        {"prompt": "<image>\\nFree OCR. "},
        {"image": "document.png"},
        {"image": "", "prompt": "<image>\\nFree OCR. "},
        {"image": "document.png", "prompt": ""},
    ],
)
def test_unlimited_ocr_workflow_rejects_invalid_golden_input(input_messages):
    with pytest.raises(ValueError, match="Unlimited-OCR"):
        XHUnlimitedOCRWorkflow.build_input(input_messages)