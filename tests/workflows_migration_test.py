from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow, BaseHMONNXWorkflow
from xhmodel_merak.xh_llm.workflows.result import ExportResult, QuantResult


class DummyWorkflow(BaseHMONNXWorkflow):
    pass


def _write_workflow_config(path: Path, model_type: str, visual_buckets: bool = False) -> Path:
    export = {
        "model": {
            "chip_arch": "XH2a",
            "model_type": model_type,
            "hf_model": None,
            "model_name": "test_model",
        }
    }
    if visual_buckets:
        export["visual_buckets"] = {
            "model": {
                "chip_arch": "XH2a",
                "model_type": "Qwen2VLForConditionalGeneration_visual",
                "patch_size": 14,
                "quant_scheme": {
                    "quant_type": "w8a16h0_ssfp",
                    "ops": {},
                },
            },
            "buckets": [
                {
                    "max_size_h": 140,
                    "max_size_w": 392,
                }
            ],
        }

    path.write_text(
        yaml.safe_dump(
            {
                "quant": None,
                "export": export,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_auto_llm_workflow_loads_declared_workflow_class(monkeypatch, tmp_path):
    class DummyModel:
        WORKFLOW_CLS = "dummy_workflow_module:DummyWorkflow"

    captured_cfg = {}

    def fake_get_model_class(cfg):
        captured_cfg.update(cfg)
        return DummyModel

    monkeypatch.setattr("xhmodel_merak.xh_llm.workflows.auto.get_model_class", fake_get_model_class)
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.workflows.auto.importlib.import_module",
        lambda module_name: SimpleNamespace(DummyWorkflow=DummyWorkflow),
    )
    hf_model_dir = tmp_path / "hf"
    hf_model_dir.mkdir()
    config_path = _write_workflow_config(tmp_path / "workflow.yaml", "Qwen2VLForConditionalGeneration")

    workflow = AutoLLMWorkflow.from_config(str(hf_model_dir), str(config_path))

    assert type(workflow) is DummyWorkflow
    assert captured_cfg["hf_model"] == str(hf_model_dir)


def test_auto_llm_workflow_falls_back_to_base_workflow(monkeypatch, tmp_path):
    class DummyModel:
        pass

    monkeypatch.setattr("xhmodel_merak.xh_llm.workflows.auto.get_model_class", lambda cfg: DummyModel)
    hf_model_dir = tmp_path / "hf"
    hf_model_dir.mkdir()
    config_path = _write_workflow_config(tmp_path / "workflow.yaml", "DummyForCausalLM")

    workflow = AutoLLMWorkflow.from_config(str(hf_model_dir), str(config_path))

    assert type(workflow) is BaseHMONNXWorkflow


@pytest.mark.parametrize(
    ("visual_buckets", "expected_calls"),
    [
        (False, 0),
        (True, 1),
    ],
)
def test_qwen2_vl_workflow_dispatches_visual_buckets(monkeypatch, tmp_path, visual_buckets, expected_calls):
    from xhmodel_merak.xh_llm.models.qwen2_vl.workflow import XHQwen2VLHMONNXWorkflow

    def fake_export(self, quant_result, output_dir, device, config_overrides=None):
        return ExportResult(work_dir=str(tmp_path), config_file=str(tmp_path / "used.yaml"), meta=object())

    calls = []

    def fake_write_visual_bucket_manifest(self, export_result, visual_buckets_cfg):
        calls.append(visual_buckets_cfg)
        return str(tmp_path / "mineru_visual_buckets.json")

    monkeypatch.setattr(BaseHMONNXWorkflow, "export", fake_export)
    monkeypatch.setattr(XHQwen2VLHMONNXWorkflow, "_write_visual_bucket_manifest", fake_write_visual_bucket_manifest)

    hf_model_dir = tmp_path / "hf"
    hf_model_dir.mkdir()
    config_path = _write_workflow_config(
        tmp_path / "workflow.yaml",
        "Qwen2VLForConditionalGeneration",
        visual_buckets=visual_buckets,
    )
    workflow = XHQwen2VLHMONNXWorkflow(str(hf_model_dir), str(config_path))

    workflow.export(
        quant_result=QuantResult(hf_model_dir=str(hf_model_dir), skipped=True),
        output_dir=str(tmp_path / "export"),
        device="cpu",
    )

    assert len(calls) == expected_calls
