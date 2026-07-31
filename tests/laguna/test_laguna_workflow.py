import argparse
import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from .tiny_laguna import build_tiny_laguna


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_CONFIG = REPO_ROOT / "configs_merak/workflows/xh2a/llm_models/laguna/s_2_1/laguna_s_2_1_xh2a_w4a8.yaml"
AUTOROUND_WORKFLOW_CONFIG = REPO_ROOT / (
    "configs_merak/workflows/xh2a/llm_models/laguna/s_2_1/"
    "laguna_s_2_1_autoround_expert_w4_rest_w8_g64_xh2a_w4a8.yaml"
)


def test_laguna_workflow_config_exports_float_model_with_native_w4a8_ptq() -> None:
    from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig

    config = WorkflowConfig.from_file(str(WORKFLOW_CONFIG))

    assert config.quant is None
    assert config.export["model"]["model_type"] == "LagunaForCausalLM"
    assert config.export["model"]["quant_scheme"]["quant_type"] == "w4a8h0_ssfp"
    assert config.export["model"]["quant_scheme"]["nodes"]["lm_head"]["quant_type"] == "w8a8h1_sefp"
    assert config.export["model"]["only_first_block"] is False
    assert config.export["model"]["max_layers"] is None
    assert config.export["model"]["max_pe_length"] == 1048576


def test_laguna_autoround_workflow_config_preserves_source_checkpoint_identity() -> None:
    from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig

    config = WorkflowConfig.from_file(str(AUTOROUND_WORKFLOW_CONFIG))

    assert config.quant is None
    assert config.export["model"]["model_name"] == "laguna_s_2_1_autoround_expert_w4_rest_w8_g64"
    assert config.export["model"]["quant_scheme"]["quant_type"] == "w4a8h0_ssfp"
    assert config.export["model"]["quant_scheme"]["nodes"]["lm_head"]["quant_type"] == "w8a8h1_sefp"


def test_laguna_rejects_context_too_small_for_sliding_window_cache() -> None:
    from xhmodel_merak.xh_llm.models.laguna.workflow import LagunaWorkflow

    with pytest.raises(ValueError, match=r"512 < 512 \+ 64 \(576\)"):
        LagunaWorkflow._validate_sliding_window_cache(
            context_max_length=512,
            prefill_chunk_length=64,
            sliding_window=512,
        )

    LagunaWorkflow._validate_sliding_window_cache(
        context_max_length=1024,
        prefill_chunk_length=64,
        sliding_window=512,
    )


def test_laguna_example_builds_run_local_export_overrides() -> None:
    from examples_merak.llm.laguna.laguna_workflow import _export_overrides

    args = argparse.Namespace(
        context_max_length=512,
        prefill_chunk_length=16,
        quant_type="w8a8_sefp",
        only_first_block=True,
        max_layers=None,
    )

    assert _export_overrides(args) == {
        "export.model.context_max_length": 512,
        "export.model.prefill_chunk_length": 16,
        "export.model.quant_scheme.quant_type": "w8a8_sefp",
        "export.model.quant_scheme.nodes.lm_head.quant_type": "w8a8_sefp",
        "export.model.only_first_block": True,
    }


def test_laguna_example_builds_two_layer_moe_smoke_override() -> None:
    from examples_merak.llm.laguna.laguna_workflow import _export_overrides

    args = argparse.Namespace(
        context_max_length=512,
        prefill_chunk_length=16,
        quant_type=None,
        only_first_block=False,
        max_layers=2,
    )

    assert _export_overrides(args) == {
        "export.model.context_max_length": 512,
        "export.model.prefill_chunk_length": 16,
        "export.model.max_layers": 2,
    }


def test_laguna_example_selects_requested_cuda_device(monkeypatch: pytest.MonkeyPatch) -> None:
    from examples_merak.llm.laguna.laguna_workflow import _configure_cuda_device

    selected = []
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)

    _configure_cuda_device("cpu")
    _configure_cuda_device("cuda:6")

    assert selected == [6]


def test_laguna_golden_normalizes_single_and_multi_device_maps() -> None:
    from xhmodel_merak.xh_llm.models.laguna.workflow import LagunaWorkflow

    assert LagunaWorkflow._normalize_device_map("cuda:0") == ["cuda:0"]
    assert LagunaWorkflow._normalize_device_map("cuda:0, cuda:1") == ["cuda:0", "cuda:1"]
    assert LagunaWorkflow._normalize_device_map(["cuda:0", "cuda:1"]) == ["cuda:0", "cuda:1"]
    with pytest.raises(ValueError, match="at least one device"):
        LagunaWorkflow._normalize_device_map(" , ")


def test_laguna_golden_enables_hmonnx_pipeline_for_multi_device_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xhmodel_merak.xh_llm.models.laguna.workflow import LagunaWorkflow

    monkeypatch.delenv("HMONNX_PIPELINE_DEVICES", raising=False)
    device_map = LagunaWorkflow._normalize_device_map("cuda:0,cuda:1,cuda:2,cuda:3")
    LagunaWorkflow._configure_hmonnx_pipeline(device_map)

    assert os.environ["HMONNX_PIPELINE_DEVICES"] == "cuda:0,cuda:1,cuda:2,cuda:3"


def test_laguna_example_overwrite_clears_export_before_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from examples_merak.llm.laguna import laguna_workflow
    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "partial.onnx").write_bytes(b"partial")

    class _Workflow:
        def quant(self, **kwargs):
            assert kwargs["output_dir"] == str(export_dir)
            assert export_dir.exists()
            return object()

        def export(self, **kwargs):
            assert kwargs["output_dir"] == str(export_dir)
            assert not export_dir.exists()
            return object()

    args = argparse.Namespace(
        model_dir="/tmp/model",
        config_path=str(WORKFLOW_CONFIG),
        export_output_dir=str(export_dir),
        device="cpu",
        context_max_length=None,
        prefill_chunk_length=None,
        quant_type=None,
        only_first_block=False,
        max_layers=None,
        overwrite=True,
        dump_golden=False,
        prompt="test",
        debug=False,
    )
    monkeypatch.setattr(laguna_workflow, "parse_args", lambda: args)
    monkeypatch.setattr(laguna_workflow, "_configure_cuda_device", lambda _device: None)
    monkeypatch.setattr(AutoLLMWorkflow, "from_config", lambda **_kwargs: _Workflow())

    laguna_workflow.main()


def test_laguna_model_is_discovered_and_uses_model_workflow(tmp_path: Path) -> None:
    from xhmodel_merak.xh_llm.configuration_auto import update_model_type_mapping
    from xhmodel_merak.xh_llm.models.laguna.workflow import LagunaWorkflow
    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

    update_model_type_mapping()
    model_dir = tmp_path / "Laguna-S-2.1"
    model_dir.mkdir()

    workflow = AutoLLMWorkflow.from_config(
        model_dir=str(model_dir),
        config_path=str(WORKFLOW_CONFIG),
    )

    assert type(workflow) is LagunaWorkflow
    quant_result = workflow.quant(
        output_dir=str(tmp_path / "quant"),
        device="cpu",
        config_overrides={"quant": None},
    )
    assert quant_result.skipped is True
    assert quant_result.raw_model_dir == str(model_dir.resolve())


def test_laguna_workflow_uses_optional_gptqmodel_only_when_explicitly_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xhmodel_merak.xh_llm.models.laguna import quant_adapter
    from xhmodel_merak.xh_llm.models.laguna.workflow import LagunaWorkflow
    from xhmodel_merak.xh_llm.workflows.result import QuantResult

    workflow = LagunaWorkflow.__new__(LagunaWorkflow)
    workflow.model_dir = "/tmp/laguna"
    workflow.workflow_config = type(
        "Config",
        (),
        {
            "with_overrides": lambda self, _overrides: self,
            "quant": {
                "algorithm": "gptqmodel",
                "bits": 4,
                "calibration": {"texts": ["calibration text"]},
            },
        },
    )()

    captured = {}

    def _quantize(**kwargs):
        captured.update(kwargs)
        return QuantResult(raw_model_dir=kwargs["model_dir"], quanted_model_dir="/tmp/quantized")

    monkeypatch.setattr(quant_adapter, "quantize_with_gptqmodel_api", _quantize)
    result = workflow.quant(str(tmp_path), "cuda:1")

    assert result.quanted_model_dir == "/tmp/quantized"
    assert captured["model_dir"] == "/tmp/laguna"
    assert captured["device"] == "cuda:1"
    assert captured["quant_cfg"]["algorithm"] == "gptqmodel"


def test_laguna_registers_optional_gptqmodel_model_tree() -> None:
    from gptqmodel.models import auto as gptq_auto

    from xhmodel_merak.xh_llm.models.laguna import register_laguna_gptqmodel

    previous = gptq_auto.MODEL_MAP.pop("laguna", None)
    was_supported = "laguna" in gptq_auto.SUPPORTED_MODELS
    if was_supported:
        gptq_auto.SUPPORTED_MODELS.remove("laguna")
    try:
        assert register_laguna_gptqmodel() is True
        registered = gptq_auto.MODEL_MAP["laguna"]
        assert registered.require_trust_remote_code is True
        assert registered.dynamic_expert_index == "num_experts"
        assert registered.pre_lm_head_norm_module == "model.norm"
    finally:
        gptq_auto.MODEL_MAP.pop("laguna", None)
        if previous is not None:
            gptq_auto.MODEL_MAP["laguna"] = previous
        if was_supported:
            if isinstance(gptq_auto.SUPPORTED_MODELS, set):
                gptq_auto.SUPPORTED_MODELS.add("laguna")
            else:
                gptq_auto.SUPPORTED_MODELS.append("laguna")


def test_laguna_autoround_load_defers_expert_fusion_until_after_dequantization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xhmodel_merak.xh_llm.models.laguna import laguna_model

    events = []
    native_model = type(
        "LagunaForCausalLM",
        (),
        {
            "config": type(
                "Config",
                (),
                {
                    "quantization_config": {"provider": "auto-round"},
                    "num_hidden_layers": 3,
                    "mlp_only_layers": [0],
                    "_name_or_path": "/tmp/autoround-laguna",
                },
            )(),
        },
    )()

    class _Loader:
        def __enter__(self):
            events.append("enter_split_loader")
            return object

        def __exit__(self, *_args):
            events.append("exit_split_loader")

    monkeypatch.setattr(laguna_model.AutoConfig, "from_pretrained", lambda *_args, **_kwargs: native_model.config)
    monkeypatch.setattr(laguna_model.XHLagunaModel, "_ensure_autoround_available", lambda *_args: events.append("ensure_autoround"))
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.laguna.float_checkpoint_compat.split_expert_checkpoint_loader",
        lambda *_args: _Loader(),
    )
    monkeypatch.setattr(
        laguna_model.TextLLMModel,
        "_load_hf_model",
        classmethod(lambda _cls, *_args, **_kwargs: events.append("load") or native_model),
    )
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.laguna.float_checkpoint_compat.fuse_split_experts",
        lambda *_args: events.append("premature_fuse") or 2,
    )

    loaded = laguna_model.XHLagunaModel._load_hf_model("/tmp/autoround-laguna")

    assert loaded is native_model
    assert events == ["ensure_autoround", "enter_split_loader", "load", "exit_split_loader"]


def test_laguna_autoround_dequantization_restores_fused_experts(monkeypatch: pytest.MonkeyPatch) -> None:
    from xhmodel_merak.xh_llm.models.laguna import laguna_model

    events = []
    native_model = type(
        "LagunaForCausalLM",
        (),
        {
            "config": type(
                "Config",
                (),
                {
                    "num_hidden_layers": 3,
                    "mlp_only_layers": [0],
                    "_name_or_path": "/tmp/autoround-laguna",
                },
            )(),
        },
    )()
    monkeypatch.setattr(
        laguna_model.TextLLMModel,
        "_dequantize_autoround_hf_model",
        classmethod(lambda _cls, model: events.append("dequantize") or model),
    )
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.laguna.gptqmodel_compat.restore_gptqmodel_moe_structure",
        lambda model, model_dir: events.append(("fuse", model_dir)) or 2,
    )

    restored = laguna_model.XHLagunaModel._dequantize_autoround_hf_model(native_model)

    assert restored is native_model
    assert events == ["dequantize", ("fuse", "/tmp/autoround-laguna")]


def test_laguna_workflow_validates_text_messages() -> None:
    from xhmodel_merak.xh_llm.models.laguna.workflow import LagunaWorkflow

    workflow = LagunaWorkflow.__new__(LagunaWorkflow)

    assert workflow.build_input_message("hello") == [{"role": "user", "content": "hello"}]
    assert workflow.build_input_message({"text": "hello"}) == [{"role": "user", "content": "hello"}]

    with pytest.raises(ValueError, match="non-empty text"):
        workflow.build_input_message({"text": ""})


def test_laguna_workflow_resolves_list_eos_for_generation() -> None:
    from xhmodel_merak.xh_llm.models.laguna.workflow import LagunaWorkflow

    tokenizer = type("Tokenizer", (), {"pad_token_id": None, "eos_token_id": [2, 24]})()
    assert LagunaWorkflow._resolve_pad_token_id(tokenizer) == 2


def test_laguna_rotary_cache_keeps_float32_frequency_precision() -> None:
    from types import SimpleNamespace

    from xhmodel_merak.xh_llm.models.laguna._model import (
        _compute_rotary_cache,
        _LagunaRotaryEmbedding,
    )

    inv_freq = torch.tensor([1.0, 0.037121, 0.001378], dtype=torch.float32)
    rotary = SimpleNamespace(
        inv_freq=inv_freq.clone(),
        attention_scaling=1.4852030263919618,
        max_seq_len_cached=8192,
    )
    expected_cos, expected_sin = _compute_rotary_cache(
        inv_freq,
        rotary.attention_scaling,
        rotary.max_seq_len_cached,
    )

    _LagunaRotaryEmbedding._set_dtype(rotary, torch.float16)

    assert rotary.inv_freq.dtype == torch.float32
    torch.testing.assert_close(rotary.cos_cached[0, 0, [0, 2047, 8191]], expected_cos[[0, 2047, 8191]].half())
    torch.testing.assert_close(rotary.sin_cached[0, 0, [0, 2047, 8191]], expected_sin[[0, 2047, 8191]].half())


def _build_tiny_xh_model(tmp_path: Path, max_layers: int | None = None):
    from xhmodel_merak.xh_llm.models.laguna.laguna_model import XHLagunaModel, XHLagunaModelConfig

    return XHLagunaModel(
        XHLagunaModelConfig(
            model_name="tiny_laguna",
            model_type="LagunaForCausalLM",
            hf_model=str(tmp_path),
            chip_arch="XH2a",
            batch_size=1,
            context_max_length=16,
            prefill_chunk_length=4,
            max_pe_length=32,
            use_cache=True,
            num_logits_to_keep=1,
            max_layers=max_layers,
            quant_scheme={
                "quant_type": "w8a8h1_sefp",
                "nodes": {"lm_head": {"quant_type": "w8a8h1_sefp"}},
                "ops": {},
            },
        )
    )


def test_laguna_dynamic_remote_classes_wrap_and_frontend(tmp_path: Path) -> None:
    model = _build_tiny_xh_model(tmp_path)
    model.to_wrap(build_tiny_laguna())

    wrapped = model.wrap_model
    assert wrapped.__class__.__name__ == "XHTrace_LagunaForCausalLM"
    assert type(wrapped.model.layers[1].mlp).__name__ == "XHTrace_LagunaSparseMoeBlock"
    assert wrapped.model.layers[0].self_attn.num_heads == 4
    assert wrapped.model.layers[1].self_attn.num_heads == 6
    assert model.pad_token_id == 0

    model.to_fronted()
    frontend = model.frontend_model
    unsupported = {
        (node.op, str(node.target))
        for node in frontend.graph.nodes
        if (node.op == "call_function" and "linear" in str(node.target))
        or (node.op == "call_method" and str(node.target) in {"float", "gather"})
    }

    assert unsupported == set()
    assert [node.name for node in frontend.graph.nodes if node.op == "placeholder"] == [
        "inputs_embeds",
        "past_seq_length",
        "current_input_length",
        "past_key_cache_0",
        "past_key_cache_1",
        "past_value_cache_0",
        "past_value_cache_1",
    ]


def test_tiny_laguna_initializes_all_expert_weights() -> None:
    model = build_tiny_laguna(num_hidden_layers=4)

    for layer in model.model.layers:
        if isinstance(layer.mlp, torch.nn.Module) and hasattr(layer.mlp, "experts"):
            assert torch.isfinite(layer.mlp.experts.gate_up_proj).all()
            assert torch.isfinite(layer.mlp.experts.down_proj).all()
            assert torch.count_nonzero(layer.mlp.experts.gate_up_proj) > 0
            assert torch.count_nonzero(layer.mlp.experts.down_proj) > 0


def test_laguna_float_checkpoint_experts_fuse_exactly() -> None:
    from xhmodel_merak.xh_llm.models.laguna.float_checkpoint_compat import (
        LagunaSplitExperts,
        fuse_split_experts,
    )

    hf_model = build_tiny_laguna()
    native_experts = hf_model.model.layers[1].mlp.experts
    native_cls = native_experts.__class__
    config = hf_model.config
    split = LagunaSplitExperts(config).to(dtype=native_experts.gate_up_proj.dtype)
    expected_gate_up = native_experts.gate_up_proj.detach().clone()
    expected_down = native_experts.down_proj.detach().clone()
    expected_correction_bias = torch.arange(
        config.num_experts,
        dtype=native_experts.gate_up_proj.dtype,
    )
    hf_model.model.layers[1].mlp.gate.e_score_correction_bias.data.copy_(expected_correction_bias)
    for index, expert in enumerate(split):
        expert.gate_proj.weight.data.copy_(expected_gate_up[index, : config.moe_intermediate_size])
        expert.up_proj.weight.data.copy_(expected_gate_up[index, config.moe_intermediate_size :])
        expert.down_proj.weight.data.copy_(expected_down[index])
    hf_model.model.layers[1].mlp.experts = split

    assert fuse_split_experts(hf_model, native_cls) == 1
    fused = hf_model.model.layers[1].mlp.experts
    assert type(fused) is native_cls
    torch.testing.assert_close(fused.gate_up_proj, expected_gate_up)
    torch.testing.assert_close(fused.down_proj, expected_down)
    torch.testing.assert_close(hf_model.model.layers[1].mlp.gate.e_score_correction_bias, expected_correction_bias)


def test_laguna_autoround_expert_fusion_preserves_quant_weights() -> None:
    from xhmodel_merak.xh_llm.models.laguna.float_checkpoint_compat import (
        LagunaSplitExperts,
        fuse_split_experts,
    )

    hf_model = build_tiny_laguna()
    native_experts = hf_model.model.layers[1].mlp.experts
    native_cls = native_experts.__class__
    config = hf_model.config
    split = LagunaSplitExperts(config).to(dtype=native_experts.gate_up_proj.dtype)
    expected_gate_up_quant = []
    expected_down_quant = []
    for index, expert in enumerate(split):
        gate_quant = torch.full_like(expert.gate_proj.weight, index + 1, dtype=torch.int8)
        up_quant = torch.full_like(expert.up_proj.weight, -(index + 1), dtype=torch.int8)
        down_quant = torch.full_like(expert.down_proj.weight, index + 2, dtype=torch.int8)
        expert.gate_proj.register_buffer("quant_weight", gate_quant, persistent=False)
        expert.up_proj.register_buffer("quant_weight", up_quant, persistent=False)
        expert.down_proj.register_buffer("quant_weight", down_quant, persistent=False)
        expected_gate_up_quant.append(torch.cat((gate_quant, up_quant), dim=0))
        expected_down_quant.append(down_quant)
    hf_model.model.layers[1].mlp.experts = split

    assert fuse_split_experts(hf_model, native_cls) == 1
    fused = hf_model.model.layers[1].mlp.experts
    torch.testing.assert_close(fused.gate_up_proj_quant_weight, torch.stack(expected_gate_up_quant))
    torch.testing.assert_close(fused.down_proj_quant_weight, torch.stack(expected_down_quant))


def test_laguna_max_layers_smoke_keeps_first_fused_moe_block(tmp_path: Path) -> None:
    model = _build_tiny_xh_model(tmp_path, max_layers=2)
    model.to_wrap(build_tiny_laguna(num_hidden_layers=4))

    assert model.kvcache_config.num_layers == 2

    model.to_fronted()
    moe_nodes = [node for node in model.frontend_model.graph.nodes if "moe" in f"{node.name} {node.target}".lower()]

    assert len(moe_nodes) == 1


def test_laguna_hf_compatible_disables_native_expert_dispatch() -> None:
    from xhmodel_merak.xh_llm.models.laguna import build_laguna_hf_compatible_model

    hf_model = build_tiny_laguna()
    input_embeddings = hf_model.model.embed_tokens

    class _StubXHModel:
        def get_input_embeddings(self):
            return input_embeddings

    compatible_model = build_laguna_hf_compatible_model(hf_model, _StubXHModel())

    assert not hasattr(compatible_model, "model")
    assert not hasattr(compatible_model, "lm_head")
    assert compatible_model.get_correct_experts_implementation("grouped_mm") == "grouped_mm"
    compatible_model.set_experts_implementation("grouped_mm")
    assert compatible_model.config.experts_implementation == "grouped_mm"
    assert compatible_model._grouped_mm_can_dispatch() is False


def test_laguna_hmonnx_tokenizer_trusts_packaged_remote_code(monkeypatch: pytest.MonkeyPatch) -> None:
    from xhmodel_merak.xh_llm.hmonnx import base_llm_hmonnx_model
    from xhmodel_merak.xh_llm.models.laguna import XHLagunaHMONNXModel

    captured = {}

    def _from_pretrained(model_dir, **kwargs):
        captured["model_dir"] = model_dir
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(base_llm_hmonnx_model.AutoTokenizer, "from_pretrained", _from_pretrained)
    model = object.__new__(XHLagunaHMONNXModel)
    object.__setattr__(model, "hf_model_dir", "/tmp/laguna/hf_config")

    model.get_tokenizer()

    assert captured == {
        "model_dir": "/tmp/laguna/hf_config",
        "trust_remote_code": True,
        "fix_mistral_regex": True,
    }


def test_moeblock_parser_preserves_shared_initializers_until_last_consumer() -> None:
    import numpy as np
    import onnx_graphsurgeon as gs

    from xhmodel_merak.xh_llm.models.laguna.moeblock_parser_compat import (
        install_moeblock_shared_initializer_compat,
    )
    from xhquant.xhonnxruntime.parsers.moeblock import MoeBlock

    install_moeblock_shared_initializer_compat()

    hidden = gs.Variable("hidden", dtype=np.float16, shape=(1, 1, 16))
    routing = gs.Variable("routing", dtype=np.float32, shape=(1, 1, 2))
    selected = gs.Variable("selected", dtype=np.int64, shape=(1, 1, 2))
    qweight = gs.Constant("shared_qweight", np.ones((4, 1, 1, 1), dtype=np.int8))
    scale = gs.Constant("shared_scale", np.ones((4, 1, 1, 1), dtype=np.float32))
    empty = gs.Variable("", dtype=None, shape=None)
    lut = gs.Constant("shared_lut", np.ones((2,), dtype=np.float32))
    inputs = [
        hidden,
        routing,
        selected,
        qweight,
        scale,
        empty,
        qweight,
        scale,
        empty,
        qweight,
        scale,
        empty,
        lut,
        lut,
        lut,
    ]
    attrs = {"topk_outside": 1, "k": 2, "mode": "sefp"}
    first = gs.Node("MoeBlock", name="moe_0", inputs=inputs, outputs=[gs.Variable("out_0")], attrs=attrs)
    second = gs.Node("MoeBlock", name="moe_1", inputs=inputs, outputs=[gs.Variable("out_1")], attrs=attrs)

    first_module = MoeBlock.from_onnx_node(first, context=None)
    second_module = MoeBlock.from_onnx_node(second, context=None)

    assert first_module.expert_gate_proj_scale is not None
    assert second_module.expert_gate_proj_scale is not None
    assert scale.values is None


def test_laguna_export_metadata_packages_remote_code(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    for filename in ("configuration_laguna.py", "modeling_laguna.py"):
        (model_dir / filename).write_text(f"# {filename}\n", encoding="utf-8")

    model = _build_tiny_xh_model(model_dir)
    model.to_wrap(build_tiny_laguna())
    output_dir = tmp_path / "export"
    meta = model.create_export_metadata(str(output_dir))

    assert meta.pad_token_id == 0
    assert (output_dir / "hf_config" / "configuration_laguna.py").is_file()
    assert (output_dir / "hf_config" / "modeling_laguna.py").is_file()
    tokenizer_config = json.loads((output_dir / "hf_config" / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert tokenizer_config["fix_mistral_regex"] is True


def test_laguna_fused_moe_matches_poolside_routing_math(tmp_path: Path) -> None:
    torch.manual_seed(7)
    hf_model = build_tiny_laguna()
    for parameter in hf_model.parameters():
        if parameter.requires_grad:
            torch.nn.init.uniform_(parameter, -0.1, 0.1)

    moe = hf_model.model.layers[1].mlp
    moe.gate.e_score_correction_bias.copy_(torch.tensor([0.2, -0.1, 0.05, -0.2], dtype=torch.float16))
    hidden_states = torch.randn(1, 3, 16, dtype=torch.float16)
    flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

    shared = moe.shared_expert(flat_hidden_states)
    routing_scores = torch.sigmoid(F.linear(flat_hidden_states, moe.gate.weight).float())
    _, selected_experts = torch.topk(
        routing_scores + moe.gate.e_score_correction_bias.float(),
        moe.gate.top_k,
        dim=-1,
    )
    routing_weights = routing_scores.gather(-1, selected_experts)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

    routed = torch.zeros_like(flat_hidden_states)
    for token_index in range(flat_hidden_states.shape[0]):
        for route_index in range(moe.gate.top_k):
            expert_index = selected_experts[token_index, route_index]
            gate, up = F.linear(
                flat_hidden_states[token_index],
                moe.experts.gate_up_proj[expert_index],
            ).chunk(2, dim=-1)
            expert_output = F.linear(F.silu(gate) * up, moe.experts.down_proj[expert_index])
            routed[token_index] += expert_output * routing_weights[token_index, route_index].half()
    expected = (routed * moe.routed_scaling_factor + shared).reshape_as(hidden_states)

    model = _build_tiny_xh_model(tmp_path)
    model.to_wrap(hf_model)
    actual = model.wrap_model.model.layers[1].mlp(hidden_states)

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize(("layer_index", "rotary_dim"), [(0, 2), (1, 4)])
def test_laguna_attention_matches_poolside_math(
    tmp_path: Path,
    layer_index: int,
    rotary_dim: int,
) -> None:
    torch.manual_seed(11 + layer_index)
    hf_model = build_tiny_laguna()
    for parameter in hf_model.parameters():
        if parameter.requires_grad:
            torch.nn.init.uniform_(parameter, -0.1, 0.1)

    attention = hf_model.model.layers[layer_index].self_attn
    hidden_states = torch.randn(1, 4, 16, dtype=torch.float16)
    positions = torch.arange(hidden_states.shape[1], dtype=torch.float32).view(1, 1, -1, 1)
    inv_freq = torch.arange(1, rotary_dim // 2 + 1, dtype=torch.float32).reciprocal().view(1, 1, 1, -1)
    angles = positions * inv_freq
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1).half()
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1).half()

    def rms_norm(states, norm):
        normalized = states.float() * torch.rsqrt(states.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        return norm.weight * normalized.to(states.dtype)

    def apply_rope(states):
        rotary, passthrough = states[..., :rotary_dim], states[..., rotary_dim:]
        first, second = rotary.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return torch.cat((rotary * cos + rotated * sin, passthrough), dim=-1)

    batch_size, sequence_length, _ = hidden_states.shape
    query = attention.q_proj(hidden_states).view(
        batch_size,
        sequence_length,
        attention.num_heads,
        attention.head_dim,
    )
    key = attention.k_proj(hidden_states).view(
        batch_size,
        sequence_length,
        2,
        attention.head_dim,
    )
    value = attention.v_proj(hidden_states).view(
        batch_size,
        sequence_length,
        2,
        attention.head_dim,
    )
    query = apply_rope(rms_norm(query, attention.q_norm).transpose(1, 2))
    key = apply_rope(rms_norm(key, attention.k_norm).transpose(1, 2))
    value = value.transpose(1, 2)

    groups = attention.num_heads // 2
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    weights = torch.matmul(query * (attention.head_dim**-0.5), key.transpose(2, 3))
    causal_mask = torch.triu(
        torch.full_like(weights, float("-inf")),
        diagonal=1,
    )
    weights = torch.softmax(weights.float() + causal_mask.float(), dim=-1).half()
    expected = torch.matmul(weights, value).transpose(1, 2)
    gate = F.softplus(attention.g_proj(hidden_states).float()).half().unsqueeze(-1)
    expected = (expected * gate).reshape(batch_size, sequence_length, -1)
    expected = attention.o_proj(expected)

    model = _build_tiny_xh_model(tmp_path)
    model.to_wrap(hf_model)
    wrapped_attention = model.wrap_model.model.layers[layer_index].self_attn
    wrapped_attention.use_cache = False
    actual, _, _ = wrapped_attention(
        hidden_states,
        (cos, sin),
        past_seq_length=torch.tensor([0], dtype=torch.int32),
        current_input_length=torch.tensor([sequence_length], dtype=torch.int32),
    )

    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="XH2a aligned PTQ requires CUDA")
def test_laguna_fused_moe_reaches_aligned_quant_graph(tmp_path: Path) -> None:
    model = _build_tiny_xh_model(tmp_path)
    model.to_wrap(build_tiny_laguna())
    model.to_quanted_aligned()

    module_types = {type(module).__name__ for module in model.quanted_model.modules()}
    assert "XH2aQuantQMoeBlock" in module_types
