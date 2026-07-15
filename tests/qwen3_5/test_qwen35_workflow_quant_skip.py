from __future__ import annotations

from pathlib import Path

import yaml


def _write_workflow_config(path: Path, *, quant):
    data = {
        "quant": quant,
        "export": {
            "model": {
                "chip_arch": "XH2a",
                "model_type": "Qwen3_5ForConditionalGeneration",
                "hf_model": None,
                "model_name": "xh2_qwen3_5_test",
                "context_max_length": 2048,
                "prefill_chunk_length": 256,
                "quant_scheme": {"quant_type": "w8a8h1_sefp", "ops": {}},
            }
        },
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_qwen35_quant_null_in_workflow_config_skips_quantization(tmp_path: Path):
    from xhmodel_merak.xh_llm.models.qwen3_5.workflow import Qwen35Workflow

    hf_model_dir = tmp_path / "qwen35-hf"
    hf_model_dir.mkdir()
    config_path = _write_workflow_config(tmp_path / "qwen35_base.yaml", quant=None)
    workflow = Qwen35Workflow.from_config(str(hf_model_dir), str(config_path))

    quant_result = workflow.quant(output_dir=str(tmp_path / "quant"), device="cpu")

    assert quant_result.skipped is True
    assert quant_result.raw_model_dir == str(hf_model_dir.resolve())
    assert quant_result.quanted_model_dir is None


def test_qwen35_quant_null_override_skips_quantization(tmp_path: Path):
    from xhmodel_merak.xh_llm.models.qwen3_5.workflow import Qwen35Workflow

    hf_model_dir = tmp_path / "qwen35-hf"
    hf_model_dir.mkdir()
    config_path = _write_workflow_config(
        tmp_path / "qwen35_quant.yaml",
        quant={"algorithm": "gptqmodel", "method": "autoround", "group_size": 64},
    )
    workflow = Qwen35Workflow.from_config(str(hf_model_dir), str(config_path))

    quant_result = workflow.quant(
        output_dir=str(tmp_path / "quant"),
        device="cpu",
        config_overrides={"quant": None},
    )

    assert quant_result.skipped is True
    assert quant_result.raw_model_dir == str(hf_model_dir.resolve())
    assert quant_result.quanted_model_dir is None


def test_qwen35_workflow_context_length_override_updates_target_and_draft_configs():
    from examples_merak.llm.qwen3_5.qwen3_5_workflow import _add_context_length_overrides

    workflow_data = {
        "export": {
            "model": {
                "context_max_length": 2048,
                "mtp_config": {"context_max_length": 2048},
                "dflash_config": {"max_sequence_length": 2048},
            }
        }
    }
    overrides = {}

    _add_context_length_overrides(overrides, workflow_data, 8192)

    assert overrides == {
        "export.model.context_max_length": 8192,
        "export.model.mtp_config.context_max_length": 8192,
        "export.model.dflash_config.max_sequence_length": 8192,
    }


def test_qwen35_workflow_context_length_override_skips_missing_optional_draft_configs():
    from examples_merak.llm.qwen3_5.qwen3_5_workflow import _add_context_length_overrides

    workflow_data = {"export": {"model": {"context_max_length": 2048}}}
    overrides = {"export.model.visual_config.max_size_h": 896}

    _add_context_length_overrides(overrides, workflow_data, 4096)

    assert overrides == {
        "export.model.visual_config.max_size_h": 896,
        "export.model.context_max_length": 4096,
    }


def test_qwen35_workflow_parse_args_accepts_context_length_alias(monkeypatch):
    from examples_merak.llm.qwen3_5.qwen3_5_workflow import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "qwen3_5_workflow.py",
            "--model-dir",
            "hf",
            "--config-path",
            "config.yaml",
            "--context-length",
            "8192",
        ],
    )

    args = parse_args()

    assert args.context_max_length == 8192


def test_qwen35_workflow_spec_decode_golden_uses_real_generate_source(monkeypatch, tmp_path: Path):
    import json

    from xhmodel_merak.xh_llm.models.qwen3_5 import hmonnx_validation
    from xhmodel_merak.xh_llm.models.qwen3_5.workflow import Qwen35Workflow

    meta_file = tmp_path / "hmquant_fake" / "golden_meta_info.json"
    meta_file.parent.mkdir()
    meta_file.write_text(
        json.dumps({"spec_decode": {"mode": "dflash", "num_draft_tokens": 9}}),
        encoding="utf-8",
    )

    calls = []

    def fake_spec_decode_generate(**kwargs):
        calls.append(kwargs)
        return hmonnx_validation.HMONNXQuickTestResult(
            meta_file=str(kwargs["meta_file"]),
            output_text="ok",
            output_tokens=2,
            latency_s=1.0,
            tokens_per_second=2.0,
            spec_decode_mode="dflash",
        )

    class FakeLogger:
        def info(self, *args, **kwargs):
            pass

    monkeypatch.setattr(hmonnx_validation, "spec_decode_generate", fake_spec_decode_generate)
    workflow = Qwen35Workflow.__new__(Qwen35Workflow)
    workflow._dump_spec_decode_golden(
        str(meta_file),
        "cuda:1",
        [{"role": "user", "content": [{"type": "image", "image": "x.png"}, {"type": "text", "text": "hello"}]}],
        logger=FakeLogger(),
    )

    assert len(calls) == 1
    assert calls[0]["meta_file"] == str(meta_file)
    assert calls[0]["prompt"] == "hello"
    assert calls[0]["device"] == "cuda:1"
    assert calls[0]["exec_device"] == "cuda:1"
    assert calls[0]["max_new_tokens"] == 11
    assert calls[0]["warmup_runs"] == 0
    assert calls[0]["benchmark_runs"] == 1
    assert calls[0]["golden"] is True


def test_qwen35_workflow_mtp_spec_decode_golden_stays_minimal(monkeypatch, tmp_path: Path):
    import json

    from xhmodel_merak.xh_llm.models.qwen3_5 import hmonnx_validation
    from xhmodel_merak.xh_llm.models.qwen3_5.workflow import Qwen35Workflow

    meta_file = tmp_path / "hmquant_fake" / "golden_meta_info.json"
    meta_file.parent.mkdir()
    meta_file.write_text(
        json.dumps({"spec_decode": {"mode": "mtp", "num_draft_tokens": 4}}),
        encoding="utf-8",
    )

    calls = []

    def fake_spec_decode_generate(**kwargs):
        calls.append(kwargs)
        return hmonnx_validation.HMONNXQuickTestResult(
            meta_file=str(kwargs["meta_file"]),
            output_text="ok",
            output_tokens=2,
            latency_s=1.0,
            tokens_per_second=2.0,
            spec_decode_mode="mtp",
        )

    class FakeLogger:
        def info(self, *args, **kwargs):
            pass

    monkeypatch.setattr(hmonnx_validation, "spec_decode_generate", fake_spec_decode_generate)
    workflow = Qwen35Workflow.__new__(Qwen35Workflow)
    workflow._dump_spec_decode_golden(
        str(meta_file),
        "cuda:0",
        [{"role": "user", "content": "hello"}],
        logger=FakeLogger(),
    )

    assert len(calls) == 1
    assert calls[0]["max_new_tokens"] == 2
    assert calls[0]["golden"] is True


def test_qwen35_workflow_collects_base_and_lora_golden_metadata(tmp_path: Path):
    import json

    from xhmodel_merak.xh_llm.models.qwen3_5.workflow import Qwen35Workflow

    root_meta = tmp_path / "hmquant_model" / "golden_meta_info.json"
    adapter_meta = root_meta.parent / "lora" / "adapter" / "golden_meta_info.json"
    adapter_meta.parent.mkdir(parents=True)
    adapter_meta.write_text("{}", encoding="utf-8")
    root_meta.write_text(
        json.dumps({"lora_adapters": [{"name": "adapter", "meta_file": "lora/adapter/golden_meta_info.json"}]}),
        encoding="utf-8",
    )

    assert Qwen35Workflow._collect_golden_meta_files(str(root_meta)) == [
        str(root_meta),
        str(adapter_meta),
    ]


def test_qwen35_workflow_releases_each_golden_model_before_loading_next(monkeypatch):
    from xhmodel_merak.xh_llm.models.qwen3_5 import workflow as workflow_module

    events = []
    workflow = workflow_module.Qwen35Workflow.__new__(workflow_module.Qwen35Workflow)
    workflow._find_golden_meta_file = lambda _result: "root-meta"
    workflow.build_input_message = lambda _messages: []
    workflow._collect_golden_meta_files = lambda _root: ["base-meta", "lora-meta"]
    workflow._dump_golden_for_meta = lambda meta, *_args, **_kwargs: events.append(f"dump:{meta}")

    monkeypatch.setattr(workflow_module.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(workflow_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(workflow_module.torch.cuda, "empty_cache", lambda: events.append("empty_cache"))

    assert workflow.dump_golden(object(), "cuda:0", {}) == "root-meta"
    assert events == [
        "dump:base-meta",
        "gc",
        "empty_cache",
        "dump:lora-meta",
        "gc",
        "empty_cache",
    ]
