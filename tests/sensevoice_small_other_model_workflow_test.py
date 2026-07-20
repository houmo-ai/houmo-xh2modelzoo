import importlib
import json
import sys
from pathlib import Path

import yaml

from xhmodel_merak.workflows import AutoWorkflow
from xhmodel_merak.xh_other_model.workflows.result import QuantResult


CONFIG_PATH = Path("configs_merak/workflows/xh2a/other_models/sensevoice_small/sensevoice_small_xh2a.yaml")


def _make_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "SenseVoiceSmall"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text("frontend_conf: {}\n", encoding="utf-8")
    (model_dir / "am.mvn").write_text("dummy", encoding="utf-8")
    (model_dir / "model.pt").write_bytes(b"dummy")
    return model_dir


def test_default_yaml_matches_legacy_readme_parameters():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["quant"] is None
    export = config["export"]
    assert export["target_device"] == "XH2a"
    assert export["model"]["type"] == "XHSenseVoiceSmallModel"
    assert export["onnx"] == {
        "enabled": True,
        "device": "cpu",
        "max_seq_len": 512,
        "opset": 14,
        "static": True,
        "simplify": True,
        "layer_norm_scale": 32.0,
        "verbose": False,
    }
    hmonnx = export["hmonnx"]
    assert hmonnx["quant_type"] == "w8a8h1_sefp"
    assert hmonnx["calib_metric"] == "minmax"
    assert hmonnx["force_fp32_ops"] == ["LayerNorm"]
    assert hmonnx["calibration"] == {
        "source": "hf_dataset",
        "file": None,
        "hf_dataset": "openslr/librispeech_asr",
        "hf_config": "clean",
        "hf_split": "validation",
        "hf_audio_field": "audio",
        "hf_streaming": True,
        "samples": 128,
    }


def test_auto_workflow_routes_empty_shell_model(tmp_path):
    model_dir = _make_model_dir(tmp_path)
    workflow = AutoWorkflow.from_config(model_dir=str(model_dir), config_path=str(CONFIG_PATH))

    from xhmodel_merak.xh_other_model.models.sensevoice_small._model import XHSenseVoiceSmallModel
    from xhmodel_merak.xh_other_model.models.sensevoice_small.workflow import SenseVoiceSmallWorkflow

    assert isinstance(workflow, SenseVoiceSmallWorkflow)
    assert XHSenseVoiceSmallModel.__bases__ == (object,)
    assert XHSenseVoiceSmallModel.WORKFLOW_CLS.endswith("sensevoice_small.workflow:SenseVoiceSmallWorkflow")


def test_package_import_does_not_import_workflow():
    package_name = "xhmodel_merak.xh_other_model.models.sensevoice_small"
    workflow_module_name = f"{package_name}.workflow"
    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            del sys.modules[module_name]

    package = importlib.import_module(package_name)

    assert package.XHSenseVoiceSmallModel.__name__ == "XHSenseVoiceSmallModel"
    assert workflow_module_name not in sys.modules


def test_quant_stage_is_explicitly_skipped(tmp_path):
    model_dir = _make_model_dir(tmp_path)
    workflow = AutoWorkflow.from_config(model_dir=str(model_dir), config_path=str(CONFIG_PATH))

    result = workflow.quant(output_dir=str(tmp_path / "quant"), device="cpu")

    assert result.skipped is True
    assert result.raw_model_dir == str(model_dir.resolve())
    assert result.quanted_model_dir is None


def test_export_dispatches_in_process_helpers_and_writes_metadata(tmp_path, monkeypatch):
    model_dir = _make_model_dir(tmp_path)
    workflow = AutoWorkflow.from_config(model_dir=str(model_dir), config_path=str(CONFIG_PATH))
    calls = []

    import xhmodel_merak.xh_other_model.models.sensevoice_small.export_utils as export_utils

    def fake_export_fp32_onnx(**kwargs):
        calls.append(("onnx", kwargs))
        onnx_file = kwargs["work_dir"] / "onnx" / "model.onnx"
        onnx_file.parent.mkdir(parents=True)
        onnx_file.write_bytes(b"onnx")
        return {
            "onnx": str(onnx_file),
            "input_names": ["speech", "speech_lengths", "language", "textnorm"],
            "output_names": ["ctc_logits", "encoder_out_lens"],
            "input_shapes": {"speech": [1, 512, 560]},
            "max_seq_len": 512,
            "opset": 14,
            "static": True,
            "layer_norm_scale": 32.0,
        }

    def fake_export_hmonnx(**kwargs):
        calls.append(("hmonnx", kwargs))
        hmonnx_file = kwargs["work_dir"] / "hmonnx" / "sensevoice_small_XH2a_w8a8h1_sefp.onnx"
        hmonnx_file.parent.mkdir(parents=True)
        hmonnx_file.write_bytes(b"hmonnx")
        return {
            "hmonnx": str(hmonnx_file),
            "quant_type": kwargs["quant_type"],
            "calib_metric": kwargs["calib_metric"],
            "force_fp32_ops": list(kwargs["force_fp32_ops"]),
            "calibration_source": kwargs["calibration_cfg"]["source"],
            "calibration_samples": 2,
            "calibration": dict(kwargs["calibration_cfg"]),
        }

    monkeypatch.setattr(export_utils, "export_fp32_onnx", fake_export_fp32_onnx)
    monkeypatch.setattr(export_utils, "export_hmonnx", fake_export_hmonnx)
    output_dir = tmp_path / "export"
    result = workflow.export(
        quant_result=QuantResult(raw_model_dir=str(model_dir), skipped=True),
        output_dir=str(output_dir),
        device="cuda:0",
        config_overrides={"export.hmonnx.calibration.samples": 2},
    )

    assert [name for name, _ in calls] == ["onnx", "hmonnx"]
    assert calls[0][1]["model_dir"] == str(model_dir.resolve())
    assert calls[1][1]["quant_type"] == "w8a8h1_sefp"
    assert calls[1][1]["execution_device"] == "cuda:0"
    assert Path(result.config_file).is_file()
    meta = json.loads((output_dir / "export_meta_info.json").read_text(encoding="utf-8"))
    assert meta["model_type"] == "XHSenseVoiceSmallModel"
    assert meta["target_device"] == "XH2a"
    assert meta["sensevoice"]["calibration_samples"] == 2
    assert meta["sensevoice"]["hmonnx_file"].endswith("_XH2a_w8a8h1_sefp.onnx")


def test_migrated_package_has_no_legacy_runtime_dependency():
    package_dir = Path("xhmodel_merak/xh_other_model/models/sensevoice_small")
    source = "\n".join(path.read_text(encoding="utf-8") for path in package_dir.glob("*.py"))

    assert "examples.audio" not in source
    assert "xh_model_zoo" not in source
    assert "sensevoice_common" not in source
    assert "sensevoice_frontend" not in source
