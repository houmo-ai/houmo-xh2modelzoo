from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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


def test_qwen35_workflow_context_length_override_has_one_source_of_truth():
    from examples_merak.llm.qwen3_5.qwen3_5_workflow import _add_context_length_overrides

    overrides = {}

    _add_context_length_overrides(overrides, 8192)

    assert overrides == {
        "export.model.context_max_length": 8192,
    }


def test_qwen35_workflow_context_length_override_skips_missing_optional_draft_configs():
    from examples_merak.llm.qwen3_5.qwen3_5_workflow import _add_context_length_overrides

    overrides = {"export.model.visual_config.image_token_capacity": 704}

    _add_context_length_overrides(overrides, 4096)

    assert overrides == {
        "export.model.visual_config.image_token_capacity": 704,
        "export.model.context_max_length": 4096,
    }


def test_qwen35_typed_config_derives_spec_cache_capacity(tmp_path):
    from xhmodel_merak.xh_llm.models.qwen3_5.xh_qwen3_5_config import (
        XHQwen3_5ModelConfig,
    )

    mtp = XHQwen3_5ModelConfig(
        model_name="mtp",
        hf_model="/target",
        context_max_length=8192,
        spec_decode_mode="mtp",
        mtp_config={},
    )
    assert mtp.mtp_config.hf_model == "/target"
    assert mtp.mtp_config.context_max_length == 8192

    assistant = tmp_path / "dflash"
    assistant.mkdir()
    (assistant / "config.json").write_text(
        '{"dflash_config": {"mask_token_id": 123}}',
        encoding="utf-8",
    )
    dflash = XHQwen3_5ModelConfig(
        model_name="dflash",
        hf_model="/target",
        context_max_length=8192,
        spec_decode_mode="dflash",
        dflash_config={
            "hf_model": str(assistant),
            "block_size": 16,
        },
    )
    assert dflash.dflash_config.target_model_dir == "/target"
    assert dflash.dflash_config.max_sequence_length == 8192
    assert dflash.num_draft_tokens == 9


def test_qwen35_spec_configs_accept_required_256k_flash_gdr_overrides():
    from examples_merak.llm.qwen3_5.qwen3_5_workflow import (
        _add_context_length_overrides,
    )
    from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig

    config_dir = Path(__file__).resolve().parents[2] / "configs_merak/workflows/xh2a/llm_models/qwen3_5/9b"
    expected_modes = {
        "qwen3_5_9b_full_mtp.yaml": "mtp",
        "qwen3_5_9b_full_dflash.yaml": "dflash",
    }
    for config_name, expected_mode in expected_modes.items():
        config_path = config_dir / config_name
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        overrides = {}
        _add_context_length_overrides(overrides, 262144)
        overrides["export.model.flash_attention.enable"] = True
        overrides["export.model.fuse_gdr_block_recurrent_ops"] = True
        resolved = WorkflowConfig(data, str(config_path)).with_overrides(overrides)
        model = resolved.data["export"]["model"]

        assert model["spec_decode_mode"] == expected_mode
        assert model["context_max_length"] == 262144
        if expected_mode == "mtp":
            assert "context_max_length" not in model["mtp_config"]
        else:
            assert "max_sequence_length" not in model["dflash_config"]
        assert model["flash_attention"]["enable"] is True
        assert model["fuse_gdr_block_recurrent_ops"] is True


def test_qwen35_mtp_export_contract_v2_is_runtime_complete():
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model import (
        build_qwen35_spec_decode_contract,
    )

    config = SimpleNamespace(
        spec_decode_mode="mtp",
        num_draft_tokens=4,
        spec_draft_head_weight_bits=4,
        mtp_config=SimpleNamespace(draft_head_weight_bits=4),
        dflash_config=None,
    )
    meta = SimpleNamespace(
        mtp_prefill_config=SimpleNamespace(hmonnx="mtp_draft_prefill/prefill.onnx"),
        mtp_decode_config=SimpleNamespace(hmonnx="mtp_draft_decode/decode.onnx"),
    )

    contract = build_qwen35_spec_decode_contract(config, meta)

    assert contract["runtime_contract_version"] == 2
    assert contract["verify_length"] == 5
    assert contract["target"] == {
        "hidden_output_name": "post_norm_hidden",
        "linear_state_outputs": "per_step",
    }
    assert contract["draft"] == {
        "abi": "qwen_mtp_paged_v2",
        "prefill_hmonnx": "mtp_draft_prefill/prefill.onnx",
        "decode_hmonnx": "mtp_draft_decode/decode.onnx",
        "cache_mutation": "page_attention",
        "cache_binding": "private_draft",
        "page_cache_block_size": 64,
    }


def test_qwen35_dflash_export_contract_v2_declares_shared_inplace_cache():
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model import (
        build_qwen35_spec_decode_contract,
    )

    config = SimpleNamespace(
        spec_decode_mode="dflash",
        num_draft_tokens=9,
        spec_draft_head_weight_bits=4,
        mtp_config=None,
        dflash_config=SimpleNamespace(
            draft_head_weight_bits=4,
            noise_token_id=248070,
            flash_attention={"enable": False},
        ),
    )
    meta = SimpleNamespace(
        dflash_context_config=SimpleNamespace(hmonnx="dflash_draft_context/context.onnx"),
        dflash_context_decode_config=SimpleNamespace(hmonnx="dflash_draft_context_decode/context_decode.onnx"),
        dflash_decode_config=SimpleNamespace(hmonnx="dflash_draft_decode/decode.onnx"),
    )

    contract = build_qwen35_spec_decode_contract(config, meta)

    assert contract["verify_length"] == 10
    assert contract["num_draft_tokens"] == 9
    assert contract["target"]["hidden_output_name"] == "target_hidden"
    assert contract["draft"] == {
        "abi": "qwen_dflash_v1",
        "context_hmonnx": "dflash_draft_context/context.onnx",
        "context_decode_hmonnx": ("dflash_draft_context_decode/context_decode.onnx"),
        "decode_hmonnx": "dflash_draft_decode/decode.onnx",
        "cache_mutation": "in_place",
        "cache_binding": "private_draft",
        "noise_token_id": 248070,
    }


def test_qwen35_dflash_flash_attention_uses_shared_paged_cache_contract():
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model import (
        build_qwen35_spec_decode_contract,
    )

    config = SimpleNamespace(
        spec_decode_mode="dflash",
        num_draft_tokens=9,
        spec_draft_head_weight_bits=4,
        mtp_config=None,
        dflash_config=SimpleNamespace(
            draft_head_weight_bits=4,
            noise_token_id=248070,
            flash_attention={"enable": True},
        ),
    )
    meta = SimpleNamespace(
        dflash_context_config=SimpleNamespace(hmonnx="context.onnx"),
        dflash_context_decode_config=SimpleNamespace(hmonnx="context_decode.onnx"),
        dflash_decode_config=SimpleNamespace(hmonnx="decode.onnx"),
    )

    contract = build_qwen35_spec_decode_contract(config, meta)

    assert contract["draft"] == {
        "abi": "qwen_dflash_paged_shared_v3",
        "context_hmonnx": "context.onnx",
        "context_decode_hmonnx": "context_decode.onnx",
        "decode_hmonnx": "decode.onnx",
        "cache_mutation": "page_attention",
        "cache_binding": "private_draft",
        "noise_token_id": 248070,
        "page_cache_block_size": 64,
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


def test_qwen35_workflow_parse_args_accepts_explicit_golden_device_map(
    monkeypatch,
):
    from examples_merak.llm.qwen3_5.qwen3_5_workflow import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "qwen3_5_workflow.py",
            "--model-dir",
            "hf",
            "--config-path",
            "config.yaml",
            "--dump-golden",
            "--golden-device-map",
            "cuda:0",
            "cuda:1",
        ],
    )

    args = parse_args()

    assert args.dump_golden is True
    assert args.golden_device_map == ["cuda:0", "cuda:1"]


def test_qwen35_workflow_parse_args_accepts_resource_tight_mode(monkeypatch):
    from examples_merak.llm.qwen3_5.qwen3_5_workflow import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "qwen3_5_workflow.py",
            "--model-dir",
            "hf",
            "--config-path",
            "config.yaml",
            "--resource-tight-mode",
        ],
    )

    assert parse_args().resource_tight_mode is True


def test_qwen35_workflow_dump_golden_auto_offloads_only_when_explicit(monkeypatch):
    import os

    from xhmodel_merak.xh_llm.models.qwen3_5 import workflow as workflow_module

    class FakeLogger:
        def info(self, *_args, **_kwargs):
            pass

    calls = []
    workflow = workflow_module.Qwen35Workflow.__new__(workflow_module.Qwen35Workflow)
    workflow._find_golden_meta_file = lambda _result: "root-meta"
    workflow.build_input_message = lambda _messages: []
    workflow._collect_golden_meta_files = lambda _root: ["root-meta"]

    def fake_dump(meta_file, device, messages, **kwargs):
        calls.append(
            {
                "meta_file": meta_file,
                "device": device,
                "messages": messages,
                "kwargs": kwargs,
                "inference_v2": os.environ.get("ENABLE_HMINFERENCE_V2"),
            }
        )

    workflow._dump_golden_for_meta = fake_dump
    monkeypatch.setattr("xhquant.api.get_xhquant_logger", lambda: FakeLogger())
    monkeypatch.setattr(workflow_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(workflow_module.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(workflow_module.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.delenv("ENABLE_HMINFERENCE_V2", raising=False)

    assert (
        workflow.dump_golden(
            object(),
            "cuda",
            {},
            device_map=[0, 1],
            auto_offload=True,
            use_v2=True,
        )
        == "root-meta"
    )
    assert calls == [
        {
            "meta_file": "root-meta",
            "device": "cuda",
            "messages": [],
            "kwargs": {
                "logger": calls[0]["kwargs"]["logger"],
                "auto_offload": True,
                "device_map": [0, 1],
                "resource_tight_mode": False,
            },
            "inference_v2": "1",
        }
    ]
    assert "ENABLE_HMINFERENCE_V2" not in os.environ


def test_qwen35_workflow_dump_golden_keeps_default_single_device(monkeypatch):
    import os

    from xhmodel_merak.xh_llm.models.qwen3_5 import workflow as workflow_module

    class FakeLogger:
        def info(self, *_args, **_kwargs):
            pass

    calls = []
    workflow = workflow_module.Qwen35Workflow.__new__(workflow_module.Qwen35Workflow)
    workflow._find_golden_meta_file = lambda _result: "root-meta"
    workflow.build_input_message = lambda _messages: []
    workflow._collect_golden_meta_files = lambda _root: ["root-meta"]
    workflow._dump_golden_for_meta = lambda meta, device, messages, **kwargs: calls.append(
        (meta, device, messages, kwargs, os.environ.get("ENABLE_HMINFERENCE_V2"))
    )
    monkeypatch.setattr("xhquant.api.get_xhquant_logger", lambda: FakeLogger())
    monkeypatch.setattr(workflow_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(workflow_module.torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(workflow_module.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.delenv("ENABLE_HMINFERENCE_V2", raising=False)

    assert workflow.dump_golden(object(), "cuda", {}) == "root-meta"
    assert calls[0][3]["auto_offload"] is False
    assert calls[0][3]["device_map"] is None
    assert calls[0][3]["resource_tight_mode"] is False
    assert calls[0][4] == "1"
    assert "ENABLE_HMINFERENCE_V2" not in os.environ


def test_qwen35_workflow_dumps_every_standalone_visual_gear(monkeypatch, tmp_path: Path):
    import os

    from xhmodel_merak.xh_llm.models.qwen3_5 import qwen3_5_hmonnx_inference
    from xhmodel_merak.xh_llm.models.qwen3_5 import workflow as workflow_module
    from xhmodel_merak.xh_llm.workflows.result import ExportResult

    meta_file = tmp_path / "visual_meta_info.json"
    meta_file.write_text("{}", encoding="utf-8")
    events = []

    class FakeLogger:
        def info(self, *_args, **_kwargs):
            pass

    class FakeVisualRuntime:
        def run_all_gears(self):
            events.append("run_all_gears")
            return {96: (1, 96, 32), 1536: (1, 1536, 32)}

    def fake_from_meta_file(path, *, device, enable_golden):
        events.append((str(path), str(device), enable_golden, os.environ.get("ENABLE_HMINFERENCE_V2")))
        return FakeVisualRuntime()

    monkeypatch.setattr(
        qwen3_5_hmonnx_inference.VisualTokenGearHMONNXModel,
        "from_meta_file",
        fake_from_meta_file,
    )
    monkeypatch.setattr("xhquant.api.get_xhquant_logger", lambda: FakeLogger())
    monkeypatch.setattr(workflow_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("ENABLE_HMINFERENCE_V2", raising=False)

    workflow = workflow_module.Qwen35Workflow.__new__(workflow_module.Qwen35Workflow)
    export_result = ExportResult(
        work_dir=str(tmp_path),
        config_file="effective.yaml",
        meta=SimpleNamespace(gears=[96, 1536]),
    )

    assert workflow.dump_golden(export_result, "cuda:0", {}) == str(meta_file)
    assert events == [
        (str(meta_file), "cuda:0", True, "1"),
        "run_all_gears",
    ]
    assert "ENABLE_HMINFERENCE_V2" not in os.environ


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
    assert calls[0]["disable_auto_offload"] is True
    assert calls[0]["auto_offload_max_memory"] is None
    assert calls[0]["resource_tight_mode"] is False


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
    assert calls[0]["disable_auto_offload"] is True
    assert calls[0]["auto_offload_max_memory"] is None
    assert calls[0]["resource_tight_mode"] is False


def test_qwen35_workflow_spec_decode_golden_forwards_resource_tight_mode(
    monkeypatch,
    tmp_path: Path,
):
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
        def info(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(
        hmonnx_validation,
        "spec_decode_generate",
        fake_spec_decode_generate,
    )
    workflow = Qwen35Workflow.__new__(Qwen35Workflow)
    workflow._dump_spec_decode_golden(
        str(meta_file),
        "cuda:0",
        [{"role": "user", "content": "hello"}],
        logger=FakeLogger(),
        resource_tight_mode=True,
    )

    assert calls[0]["resource_tight_mode"] is True


def test_qwen35_spec_golden_restricts_explicit_auto_offload_devices(
    monkeypatch,
    tmp_path: Path,
):
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

    monkeypatch.setattr(
        hmonnx_validation,
        "spec_decode_generate",
        fake_spec_decode_generate,
    )
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.workflow.torch.cuda.mem_get_info",
        lambda index: ((10 + index) * 1000, 20_000),
    )
    workflow = Qwen35Workflow.__new__(Qwen35Workflow)
    workflow._dump_spec_decode_golden(
        str(meta_file),
        "cuda:1",
        [{"role": "user", "content": "hello"}],
        logger=FakeLogger(),
        auto_offload=True,
        device_map=[1, "cuda:3"],
    )

    assert calls[0]["disable_auto_offload"] is False
    assert json.loads(calls[0]["auto_offload_max_memory"]) == {
        "1": 9900,
        "3": 11700,
    }


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
