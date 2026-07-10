import importlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest
import yaml

from xhmodel_merak.workflows import AutoWorkflow
from xhmodel_merak.xh_other_model.workflows import AutoOtherModelWorkflow
from xhmodel_merak.xh_other_model.workflows.result import ExportResult, QuantResult


def _write_qwen3_asr_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "qwen3_asr.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "quant": {
                    "quant_type": "w8a8_sefp",
                },
                "export": {
                    "target_device": "XH2a",
                    "model": {
                        "type": "XHQwen3ASRLLMModel",
                        "hf_model": None,
                        "wrap_cfg": {
                            "max_sequence_length": 2048,
                            "input_sequence_length": 216,
                            "use_cache": True,
                            "num_logits_to_keep": 1,
                            "kv_cache": {
                                "cache_axis": 2,
                            },
                        },
                        "quant_config": {},
                        "frontend_type": "TorchFX",
                        "export_cfg": {
                            "input_names": [
                                "inputs_embeds",
                                "past_seq_length",
                                "current_input_length",
                                "past_key_cache",
                                "past_value_cache",
                            ],
                            "output_names": ["last_hidden_state"],
                        },
                    },
                    "audio": {
                        "max_audio_length": 1500,
                    },
                    "components": {
                        "encoder": {
                            "enabled": True,
                            "quant_type": "w8a8_sefp",
                        },
                        "prefill_decode": {
                            "enabled": True,
                            "prefix_token_budget": 512,
                        },
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


def test_auto_other_model_workflow_binds_qwen3_asr(tmp_path):
    config_path = _write_qwen3_asr_config(tmp_path)

    workflow = AutoOtherModelWorkflow.from_config(
        model_dir="/models/Qwen3-ASR-0.6B",
        config_path=str(config_path),
    )

    from xhmodel_merak.xh_other_model.models.qwen3_asr.workflow import Qwen3ASRWorkflow

    assert isinstance(workflow, Qwen3ASRWorkflow)
    assert workflow.model_dir == "/models/Qwen3-ASR-0.6B"


def test_auto_workflow_entrypoints_only_accept_model_dir():
    assert "hf_model_dir" not in inspect.signature(AutoWorkflow.from_config).parameters
    assert "hf_model_dir" not in inspect.signature(AutoOtherModelWorkflow.from_config).parameters


def test_top_level_auto_workflow_routes_qwen3_asr(tmp_path):
    config_path = _write_qwen3_asr_config(tmp_path)

    workflow = AutoWorkflow.from_config(
        model_dir="/models/Qwen3-ASR-0.6B",
        config_path=str(config_path),
    )

    from xhmodel_merak.xh_other_model.models.qwen3_asr.workflow import Qwen3ASRWorkflow

    assert isinstance(workflow, Qwen3ASRWorkflow)


def test_qwen3_asr_hf_model_type_alias_is_registered(tmp_path):
    config_path = _write_qwen3_asr_config(tmp_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["export"]["model"]["type"] = "qwen3_asr"
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    workflow = AutoOtherModelWorkflow.from_config(
        model_dir="/models/Qwen3-ASR-0.6B",
        config_path=str(config_path),
    )

    from xhmodel_merak.xh_other_model.models.qwen3_asr.workflow import Qwen3ASRWorkflow

    assert isinstance(workflow, Qwen3ASRWorkflow)


def test_qwen3_asr_registration_points_to_migrated_model_files():
    from xhmodel_merak.xh_other_model.builder import get_model_class

    model_cls = get_model_class({"type": "XHQwen3ASRLLMModel"})

    assert model_cls.__module__ == "xhmodel_merak.xh_other_model.models.qwen3_asr._qwen3_asr_llm_model"
    assert model_cls.WORKFLOW_CLS.endswith("qwen3_asr.workflow:Qwen3ASRWorkflow")
    for file_name in [
        "modeling_qwen3_asr.py",
        "configuration_qwen3_asr.py",
        "_qwen3_asr_llm_model.py",
        "_llm_onnx_model.py",
        "_llm_model_impl.py",
        "_qwen3asr_hf_compatible.py",
    ]:
        assert (
            Path("xhmodel_merak/xh_other_model/models/qwen3_asr") / file_name
        ).exists()


def test_old_other_model_descriptor_registry_files_are_removed():
    base = Path("xhmodel_merak/xh_other_model")
    assert not (base / "configuration_auto.py").exists()
    assert not (base / "register.py").exists()
    assert not (base / "scan_model_types.py").exists()


def test_qwen3_asr_package_import_does_not_import_workflow():
    package_name = "xhmodel_merak.xh_other_model.models.qwen3_asr"
    workflow_module_name = f"{package_name}.workflow"
    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            del sys.modules[module_name]

    package = importlib.import_module(package_name)

    assert package.XHQwen3ASRLLMModel.__name__ == "XHQwen3ASRLLMModel"
    assert "XHQwen3ASRLLMModel" in package.__all__
    assert hasattr(package, "Qwen3ASRForConditionalGeneration")
    assert workflow_module_name not in sys.modules


def test_qwen3_asr_export_dispatches_in_process_exporters(tmp_path, monkeypatch):
    config_path = _write_qwen3_asr_config(tmp_path)
    workflow = AutoOtherModelWorkflow.from_config(
        model_dir="/models/Qwen3-ASR-0.6B",
        config_path=str(config_path),
        debug=True,
    )
    calls = []

    def fake_export_encoder(**kwargs):
        calls.append(("encoder", kwargs))
        return {"encoder": "ok"}

    def fake_export_prefill_decode(**kwargs):
        calls.append(("prefill_decode", kwargs))
        return {"prefill_decode": "ok"}

    import xhmodel_merak.xh_other_model.models.qwen3_asr.workflow as asr_workflow

    monkeypatch.setattr(asr_workflow, "export_qwen3_asr_encoder", fake_export_encoder)
    monkeypatch.setattr(asr_workflow, "export_qwen3_asr_prefill_decode", fake_export_prefill_decode)

    result = workflow.export(
        quant_result=QuantResult(raw_model_dir="/models/Qwen3-ASR-0.6B", skipped=True),
        output_dir=str(tmp_path / "out"),
        device="cuda:0",
    )

    assert len(calls) == 2
    encoder_call = calls[0][1]
    prefill_decode_call = calls[1][1]
    assert encoder_call["model_dir"] == "/models/Qwen3-ASR-0.6B"
    assert encoder_call["work_dir"] == tmp_path / "out"
    assert encoder_call["quant_type"] == "w8a8_sefp"
    assert encoder_call["max_audio_length"] == 1500
    assert prefill_decode_call["model_cfg"]["type"] == "XHQwen3ASRLLMModel"
    assert prefill_decode_call["model_cfg"]["hf_model"] == "/models/Qwen3-ASR-0.6B"
    assert prefill_decode_call["prefix_token_budget"] == 512
    assert result.work_dir == str(tmp_path / "out")
    assert Path(result.config_file).exists()
    assert result.meta["components"] == ["encoder", "prefill_decode"]
    effective_config = yaml.safe_load(Path(result.config_file).read_text(encoding="utf-8"))
    assert effective_config["export"]["model"]["type"] == "XHQwen3ASRLLMModel"
    assert effective_config["export"]["model"]["hf_model"] == "/models/Qwen3-ASR-0.6B"


def test_qwen3_asr_export_uses_resolved_quanted_model_dir(tmp_path, monkeypatch):
    config_path = _write_qwen3_asr_config(tmp_path)
    workflow = AutoOtherModelWorkflow.from_config(
        model_dir="/models/Qwen3-ASR-0.6B",
        config_path=str(config_path),
    )
    calls = []

    def fake_export_encoder(**kwargs):
        calls.append(("encoder", kwargs))
        return {}

    def fake_export_prefill_decode(**kwargs):
        calls.append(("prefill_decode", kwargs))
        return {}

    import xhmodel_merak.xh_other_model.models.qwen3_asr.workflow as asr_workflow

    monkeypatch.setattr(asr_workflow, "export_qwen3_asr_encoder", fake_export_encoder)
    monkeypatch.setattr(asr_workflow, "export_qwen3_asr_prefill_decode", fake_export_prefill_decode)

    quanted_model_dir = tmp_path / "quanted_model"
    workflow.export(
        quant_result=QuantResult(
            raw_model_dir="/models/Qwen3-ASR-0.6B",
            skipped=False,
            quanted_model_dir=str(quanted_model_dir),
        ),
        output_dir=str(tmp_path / "out"),
        device="cuda:0",
    )

    assert calls[0][1]["model_dir"] == str(quanted_model_dir)
    assert calls[1][1]["model_dir"] == str(quanted_model_dir)
    assert calls[1][1]["model_cfg"]["hf_model"] == str(quanted_model_dir)


def test_qwen3_asr_dump_golden_uses_export_result_work_dir(tmp_path, monkeypatch):
    config_path = _write_qwen3_asr_config(tmp_path)
    workflow = AutoOtherModelWorkflow.from_config(
        model_dir="/models/Qwen3-ASR-0.6B",
        config_path=str(config_path),
    )
    calls = []

    import xhmodel_merak.xh_other_model.models.qwen3_asr.workflow as asr_workflow
    import torch

    def fake_run_hmonnx_golden(hmonnx_file, golden_dir, device, inputs):
        calls.append(
            {
                "hmonnx_file": hmonnx_file,
                "golden_dir": golden_dir,
                "device": device,
                "num_inputs": len(inputs),
            }
        )

    work_dir = tmp_path / "out"
    (work_dir / "Encoder" / "hmonnx").mkdir(parents=True)
    (work_dir / "Prefill").mkdir()
    (work_dir / "Decoder").mkdir()
    torch.save({"weight": torch.zeros((8, 4), dtype=torch.float16)}, work_dir / "token_embedding.pt")
    (work_dir / "export_meta_info.json").write_text(
        json.dumps(
            {
                "encoder": {
                    "hmonnx_file": "Encoder/hmonnx/encoder.onnx",
                    "model_cfg": {
                        "fixed_max_audio_length": 10,
                        "num_mel_bins": 80,
                    },
                },
                "token_embedding_file": "token_embedding.pt",
                "prefill_input_sequence_length": 3,
                "kv_cache_shape": [1, 2, 4, 4],
                "num_hidden_layers": 1,
                "prefill_onnx_file": "Prefill/prefill.onnx",
                "decode_onnx_file": "Decoder/decode.onnx",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(asr_workflow, "_run_hmonnx_golden", fake_run_hmonnx_golden)

    result = workflow.dump_golden(
        export_result=ExportResult(
            work_dir=str(work_dir),
            config_file=str(tmp_path / "out" / "qwen3_asr_effective.yaml"),
        ),
        device="cuda:0",
        input_messages={"audio": "sample.wav"},
    )

    assert result == str(work_dir / "golden_meta_info.json")
    assert [call["hmonnx_file"] for call in calls] == [
        work_dir / "Encoder/hmonnx/encoder.onnx",
        work_dir / "Prefill/prefill.onnx",
        work_dir / "Decoder/decode.onnx",
    ]
    assert (work_dir / "golden_meta_info.json").exists()


def test_qwen3_asr_rejects_unknown_component(tmp_path):
    config_path = _write_qwen3_asr_config(tmp_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["export"]["components"]["decoder"] = {"enabled": True}
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    workflow = AutoOtherModelWorkflow.from_config(
        model_dir="/models/Qwen3-ASR-0.6B",
        config_path=str(config_path),
    )

    with pytest.raises(ValueError, match="Unsupported Qwen3-ASR export component"):
        workflow.export(
            quant_result=QuantResult(raw_model_dir="/models/Qwen3-ASR-0.6B", skipped=True),
            output_dir=str(tmp_path / "out"),
            device="cuda:0",
        )


def test_qwen3_asr_merak_example_uses_workflow_entrypoint():
    example_dir = Path("examples_merak/asr/qwen3_asr")
    script_path = example_dir / "qwen3_asr_workflow.py"
    readme_path = example_dir / "README.md"

    assert script_path.exists()
    assert readme_path.exists()

    source = script_path.read_text(encoding="utf-8")
    assert "from xhmodel_merak.xh_other_model.workflows import AutoOtherModelWorkflow" in source
    assert "AutoOtherModelWorkflow.from_config" in source
    assert "hmonnx_artifacts" in source
    assert "workflow_outputs" in source
    assert "examples/audio/qwen3_asr/hmonnx_export_encoder.py" not in source
    assert (example_dir / ".gitignore").read_text(encoding="utf-8") == (
        "hmonnx_artifacts/\nworkflow_outputs/\n"
    )

    spec = importlib.util.spec_from_file_location("qwen3_asr_workflow_example", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.parse_args)
    assert callable(module.main)

    readme = readme_path.read_text(encoding="utf-8")
    assert "xh2modelzoo_qwen3asr" in readme
    assert "AutoOtherModelWorkflow" in readme
    assert "examples_merak/asr/qwen3_asr/hmonnx_artifacts" in readme
    assert "configs_merak/workflows/xh2a/other_models/qwen3_asr/0_6b/qwen3_asr_full.yaml" in readme


def test_qwen3_asr_merak_hmonnx_demos_are_parameterized():
    example_dir = Path("examples_merak/asr/qwen3_asr")
    demo_path = example_dir / "hmonnx_demo.py"
    prefix_demo_path = example_dir / "hmonnx_demo_chunk_prefix.py"

    assert demo_path.exists()
    assert prefix_demo_path.exists()

    for module_name, script_path in [
        ("qwen3_asr_hmonnx_demo", demo_path),
        ("qwen3_asr_hmonnx_demo_chunk_prefix", prefix_demo_path),
    ]:
        source = script_path.read_text(encoding="utf-8")
        assert "examples/audio/qwen3_asr" not in source
        assert "DEFAULT_AUDIO = " not in source
        assert "WORK_DIR = " not in source
        assert "--work-dir" in source
        assert "--audio" in source
        assert "--device" in source

        spec = importlib.util.spec_from_file_location(module_name, script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert callable(module.parse_args)
        assert callable(module.main)
