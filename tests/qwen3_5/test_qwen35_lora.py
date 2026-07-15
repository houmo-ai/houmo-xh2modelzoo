from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from xhmodel_merak.xh_llm.lora_layer import LoRALinear, apply_lora_to_linear
from xhmodel_merak.xh_llm.models.qwen3_5.lora import (
    XHQwen3_5LoRAConfig,
    apply_lora_to_frontend,
    attach_lora_buffers,
    finalize_lora_metadata,
    inspect_lora_adapter,
)
from xhquant.frontend.torchfx import xh_fx


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 3, bias=False)

    def forward(self, x):
        return self.q_proj(x)


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()

    def forward(self, x):
        return self.self_attn(x)


class _LanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer()])

    def forward(self, x):
        return self.layers[0](x)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _LanguageModel()

    def forward(self, x):
        return self.language_model(x)


class _ConditionalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Model()

    def forward(self, x):
        return self.model(x)


def _write_adapter(
    adapter_dir: Path,
    *,
    module_path: str = "base_model.model.model.language_model.layers.0.self_attn.q_proj",
) -> tuple[torch.Tensor, torch.Tensor]:
    adapter_dir.mkdir()
    config = {
        "peft_type": "LORA",
        "r": 2,
        "lora_alpha": 4,
        "bias": "none",
        "fan_in_fan_out": False,
        "rank_pattern": {},
        "alpha_pattern": {},
        "modules_to_save": None,
        "use_dora": False,
        "use_qalora": False,
        "use_rslora": False,
    }
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    lora_a = torch.tensor([[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 2.0]])
    lora_b = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    save_file(
        {
            f"{module_path}.lora_A.weight": lora_a,
            f"{module_path}.lora_B.weight": lora_b,
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )
    return lora_a, lora_b


def test_lora_config_has_no_mode_and_empty_w_schema_inherits_main_scheme(tmp_path: Path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    config = XHQwen3_5LoRAConfig(path=[str(adapter_dir)], w_schema={})

    assert config.path == [str(adapter_dir)]
    assert config.w_schema is None
    with pytest.raises(ValueError, match="no longer accepts 'mode'"):
        XHQwen3_5LoRAConfig(path=[str(adapter_dir)], mode="keep_lora")


def test_lora_config_rejects_duplicate_output_directory_names(tmp_path: Path):
    with pytest.raises(ValueError, match="duplicate final path"):
        XHQwen3_5LoRAConfig(path=[str(tmp_path / "one" / "adapter"), str(tmp_path / "two" / "adapter")])


def test_visual_lora_tensor_is_rejected(tmp_path: Path):
    adapter_dir = tmp_path / "visual-adapter"
    _write_adapter(adapter_dir, module_path="base_model.model.model.visual.blocks.0.attn.q_proj")

    with pytest.raises(ValueError, match="does not support ViT/visual adapters"):
        inspect_lora_adapter(str(adapter_dir))

    from xhmodel_merak.xh_llm.models.qwen3_5.xh_qwen3_5_config import XHQwen3_5_VisualConfig

    with pytest.raises(ValueError, match="visual export does not support LoRA"):
        XHQwen3_5_VisualConfig(
            model_name="visual",
            max_size_w=448,
            max_size_h=448,
            lora={"path": [str(adapter_dir)]},
        )


def test_lora_without_w_schema_inherits_main_weight_and_activation_scheme(tmp_path: Path):
    adapter_dir = tmp_path / "adapter"
    _write_adapter(adapter_dir)
    adapter = inspect_lora_adapter(str(adapter_dir))
    model = _ConditionalModel()
    model.model.language_model.layers[0].self_attn.q_proj.register_buffer(
        "quant_weight",
        torch.ones(3, 4, dtype=torch.int8),
        persistent=False,
    )
    attach_lora_buffers(model, adapter)
    frontend = SimpleNamespace(
        prefill=xh_fx.symbolic_trace(model),
        decode=xh_fx.symbolic_trace(model),
    )

    apply_lora_to_frontend(frontend, adapter, None)

    lora_nodes = [
        node
        for node in frontend.prefill.graph.nodes
        if node.op == "call_module" and isinstance(frontend.prefill.get_submodule(str(node.target)), LoRALinear)
    ]
    assert len(lora_nodes) == 2
    assert all(node.meta["quant_config"] == {} for node in lora_nodes)

    from xhquant.api import FrontendGraph, to_quant_graph

    quanted = to_quant_graph(
        FrontendGraph.convert(frontend.prefill),
        "XH2a",
        {"quant_type": "w8a16h1_sefp"},
    )
    quanted_lora_modules = [
        module
        for name, module in quanted.named_modules()
        if name.endswith(("lora_A", "lora_B")) and hasattr(module, "w_cfg")
    ]
    assert len(quanted_lora_modules) == 2
    assert all(module.w_cfg.qspec.man_bit == 8 for module in quanted_lora_modules)
    assert all(module.i_cfg.qspec.man_bit == 16 for module in quanted_lora_modules)


def test_static_lora_rewrite_preserves_quant_weight_and_marks_only_weight_schema(tmp_path: Path):
    adapter_dir = tmp_path / "adapter"
    lora_a, lora_b = _write_adapter(adapter_dir)
    adapter = inspect_lora_adapter(str(adapter_dir))

    model = _ConditionalModel()
    base_linear = model.model.language_model.layers[0].self_attn.q_proj
    base_linear.register_buffer(
        "quant_weight",
        torch.arange(12, dtype=torch.int8).reshape(3, 4),
        persistent=False,
    )
    base_weight = base_linear.weight.detach().clone()
    assert attach_lora_buffers(model, adapter) == 1

    frontend = SimpleNamespace(
        prefill=xh_fx.symbolic_trace(model),
        decode=xh_fx.symbolic_trace(model),
    )
    apply_lora_to_frontend(
        frontend,
        adapter,
        {"bits": 4, "fp_mode": "ssfp"},
    )

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    expected = torch.nn.functional.linear(x, base_weight)
    expected = (
        expected
        + torch.nn.functional.linear(
            torch.nn.functional.linear(x, lora_a),
            lora_b,
        )
        * 2.0
    )
    torch.testing.assert_close(frontend.prefill(x), expected)
    torch.testing.assert_close(frontend.decode(x), expected)

    for graph_module in (frontend.prefill, frontend.decode):
        placeholders = [node.name for node in graph_module.graph.nodes if node.op == "placeholder"]
        assert "lora_mask" not in placeholders
        lora_nodes = [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_module" and isinstance(graph_module.get_submodule(str(node.target)), LoRALinear)
        ]
        assert len(lora_nodes) == 2
        assert all(node.meta["quant_config"] == {"w_schema": {"bits": 4, "fp_mode": "ssfp"}} for node in lora_nodes)
        scale_nodes = [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_module" and type(graph_module.get_submodule(str(node.target))).__name__ == "ScalarMul"
        ]
        assert len(scale_nodes) == 1
        scale_module = graph_module.get_submodule(str(scale_nodes[0].target))
        torch.testing.assert_close(scale_module.scalar, torch.tensor(2.0))
        assert not tuple(scale_module.named_parameters())

        base_modules = [
            module
            for _, module in graph_module.named_modules()
            if isinstance(module, nn.Linear) and not isinstance(module, LoRALinear) and hasattr(module, "quant_weight")
        ]
        assert len(base_modules) == 1
        torch.testing.assert_close(
            base_modules[0].quant_weight,
            torch.arange(12, dtype=torch.int8).reshape(3, 4),
        )

    prefill_lora = {name: module for name, module in frontend.prefill.named_modules() if isinstance(module, LoRALinear)}
    decode_lora = {name: module for name, module in frontend.decode.named_modules() if isinstance(module, LoRALinear)}
    assert prefill_lora.keys() == decode_lora.keys()
    assert all(prefill_lora[name] is decode_lora[name] for name in prefill_lora)

    # The FireRedASR-style add/mul branch and per-node w_schema must be
    # accepted by the real xhquant frontend-to-quant conversion.
    from xhquant.api import FrontendGraph, to_quant_graph

    quant_input = FrontendGraph.convert(frontend.prefill)
    quanted = to_quant_graph(
        quant_input,
        "XH2a",
        {"quant_type": "w8a16h1_sefp"},
    )
    quanted_lora_modules = [
        module
        for name, module in quanted.named_modules()
        if name.endswith(("lora_A", "lora_B")) and hasattr(module, "w_cfg")
    ]
    assert len(quanted_lora_modules) == 2
    assert all(module.w_cfg.qspec.man_bit == 4 for module in quanted_lora_modules)
    assert all(module.w_cfg.qspec.fp_mode == "ssfp" for module in quanted_lora_modules)
    assert all(module.i_cfg.qspec.man_bit == 16 for module in quanted_lora_modules)
    quanted_scales = [module for module in quanted.modules() if "ScalarMul" in type(module).__name__]
    assert len(quanted_scales) == 1
    torch.testing.assert_close(quanted_scales[0].scalar, torch.tensor(2.0))

    assert sum(isinstance(module, LoRALinear) for module in frontend.decode.modules()) == 2
    decode_quanted = to_quant_graph(
        FrontendGraph.convert(frontend.decode),
        "XH2a",
        {"quant_type": "w8a16h1_sefp"},
    )
    decode_lora_modules = [
        module
        for name, module in decode_quanted.named_modules()
        if name.endswith(("lora_A", "lora_B")) and hasattr(module, "w_cfg")
    ]
    assert len(decode_lora_modules) == 2
    assert all(module.w_cfg.qspec.man_bit == 4 for module in decode_lora_modules)


def test_lora_child_metadata_reuses_root_artifacts(tmp_path: Path):
    from xhmodel_merak.xh_llm.types import ExportData, VisualModelMeta, VLLMModelMeta

    adapter_dir = tmp_path / "source-adapter"
    _write_adapter(adapter_dir)
    adapter = inspect_lora_adapter(str(adapter_dir))

    root_dir = tmp_path / "hmquant-model"
    child_dir = root_dir / "lora" / adapter.name
    child_dir.mkdir(parents=True)
    (root_dir / "hf_config").mkdir()
    (root_dir / "hf_config" / "config.json").write_text("{}", encoding="utf-8")
    (root_dir / "quant_embedding.pt").write_bytes(b"embedding")
    (root_dir / "visual").mkdir()
    (root_dir / "visual" / "visual.onnx").write_bytes(b"visual")
    (root_dir / "visual" / "visual_external_data").write_bytes(b"weights")
    (root_dir / "visual" / "step_0").mkdir()
    (root_dir / "mtp_draft_prefill").mkdir()
    (root_dir / "mtp_draft_prefill" / "mtp.onnx").write_bytes(b"mtp")
    root_meta = VLLMModelMeta(
        hf_config="hf_config",
        quant_embedding="quant_embedding.pt",
        prefill_hmonnx="prefill/base.onnx",
        decode_hmonnx="decode/base.onnx",
    )
    root_meta.visual_config = VisualModelMeta()
    root_meta.visual_config.hmonnx = "visual/visual.onnx"
    root_meta.mtp_prefill_config = VisualModelMeta()
    root_meta.mtp_prefill_config.hmonnx = "mtp_draft_prefill/mtp.onnx"
    root_meta.spec_decode = {"mode": "mtp", "draft_prefill_onnx": "mtp_draft_prefill/mtp.onnx"}

    graph_meta = VLLMModelMeta(
        prefill_hmonnx="prefill/adapter.onnx",
        prefill_hmonnx_md5="prefill-md5",
        decode_hmonnx="decode/adapter.onnx",
        decode_hmonnx_md5="decode-md5",
    )
    exported_info = ExportData()
    exported_info.exported_dir = str(root_dir)
    exported_info.meta = root_meta
    adapter_export = ExportData()
    adapter_export.exported_dir = str(child_dir)
    adapter_export.meta = graph_meta

    lora_config = XHQwen3_5LoRAConfig(
        path=[str(adapter_dir)],
        w_schema={"bits": 4, "fp_mode": "ssfp"},
    )
    finalize_lora_metadata(root_meta, exported_info, [(adapter, adapter_export)], lora_config)

    child_meta = json.loads((child_dir / "golden_meta_info.json").read_text(encoding="utf-8"))
    assert child_meta["prefill_hmonnx"] == "prefill/adapter.onnx"
    assert child_meta["decode_hmonnx"] == "decode/adapter.onnx"
    assert child_meta["quant_embedding"] == "quant_embedding.pt"
    assert child_meta["hf_config"] == "hf_config"
    assert child_meta["visual_config"]["hmonnx"] == "visual/visual.onnx"
    assert child_meta["mtp_prefill_config"]["hmonnx"] == "mtp_draft_prefill/mtp.onnx"
    assert child_meta["spec_decode"]["draft_prefill_onnx"] == "mtp_draft_prefill/mtp.onnx"
    assert child_meta["active_lora"]["scale"] == 2.0
    assert (child_dir / "hf_config").is_dir()
    assert not (child_dir / "hf_config").is_symlink()
    assert not (child_dir / "hf_config" / "config.json").is_symlink()
    assert (child_dir / "quant_embedding.pt").is_symlink()
    assert (child_dir / "visual").is_dir()
    assert not (child_dir / "visual").is_symlink()
    assert (child_dir / "visual" / "visual.onnx").is_symlink()
    assert (child_dir / "visual" / "visual_external_data").is_symlink()
    assert not (child_dir / "visual" / "step_0").exists()
    assert (child_dir / "mtp_draft_prefill" / "mtp.onnx").is_symlink()
    assert root_meta.quant_embedding == "quant_embedding.pt"
    assert root_meta.lora_adapters[0]["meta_file"] == f"lora/{adapter.name}/golden_meta_info.json"

    from examples_merak.llm.qwen3_5.debug_scripts.validate_hm_release_layout import ensure_step_artifact_links

    child_visual_step = child_dir / "visual" / "step_0"
    child_visual_step.mkdir()
    ensure_step_artifact_links(child_dir)
    assert (child_visual_step / "visual.onnx").is_symlink()
    assert (child_visual_step / "visual_external_data").is_symlink()


def test_spec_runtime_path_preserves_lora_file_symlink(tmp_path: Path):
    from xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation import _resolve_path

    source = tmp_path / "visual" / "visual.onnx"
    source.parent.mkdir()
    source.write_bytes(b"visual")
    adapter_dir = tmp_path / "lora" / "adapter"
    local_model = adapter_dir / "visual" / "visual.onnx"
    local_model.parent.mkdir(parents=True)
    local_model.symlink_to(Path(os.path.relpath(source, local_model.parent)))

    runtime_path = _resolve_path(adapter_dir, "visual/visual.onnx")

    assert runtime_path == local_model
    assert runtime_path.is_symlink()
    assert runtime_path.resolve() == source.resolve()


def test_quantize_wrap_variant_clears_previous_decode_wrap_cfg(monkeypatch: pytest.MonkeyPatch):
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model import XHQwen3_5Model
    from xhmodel_merak.xh_llm.models.qwen3_5.xh_qwen3_5_config import XHQwen3_5ModelConfig

    config = XHQwen3_5ModelConfig(
        model_name="qwen3_5",
        spec_decode_mode="mtp",
        output_post_norm_hidden=True,
        num_logits_to_keep=1,
        prefill_chunk_length=256,
    )
    model = XHQwen3_5Model(config)
    model.wrap_cfg["verify_output_intermediates"] = True
    model.wrap_cfg["num_logits_to_keep"] = 0
    model.wrap_cfg["input_sequence_length"] = 5

    observed = {}
    frontend = SimpleNamespace()

    def _capture_frontend(_wrap_model):
        observed.update(
            verify_output_intermediates=model.wrap_cfg["verify_output_intermediates"],
            num_logits_to_keep=model.wrap_cfg["num_logits_to_keep"],
            input_sequence_length=model.wrap_cfg["input_sequence_length"],
        )
        return frontend

    class _FixedModel:
        def fixed(self):
            return self

    quanted = SimpleNamespace(prefill=_FixedModel(), decode=_FixedModel())
    monkeypatch.setattr(model, "_to_fronted", _capture_frontend)
    monkeypatch.setattr(model, "release_wraped_model", lambda: None)
    monkeypatch.setattr(model, "_to_quanted", lambda _frontend, _state: quanted)

    model._quantize_wrap_variant(nn.Linear(1, 1))

    assert observed == {
        "verify_output_intermediates": False,
        "num_logits_to_keep": 1,
        "input_sequence_length": 256,
    }


def test_shared_lora_helper_keeps_runtime_mask_mode_compatible():
    model = nn.Sequential(nn.Linear(4, 3, bias=False))
    model[0].register_buffer("weight_lora_a", torch.randn(2, 4), persistent=False)
    model[0].register_buffer("weight_lora_b", torch.randn(3, 2), persistent=False)
    frontend = xh_fx.symbolic_trace(model)
    inputs = []

    apply_lora_to_linear(frontend, inputs, lora_scale=4.0, runtime_mask=True)

    assert len(inputs) == 1
    assert any(node.op == "placeholder" and node.name == "lora_mask" for node in frontend.graph.nodes)
    assert not any(node.op == "call_function" for node in frontend.graph.nodes)
    output = frontend(torch.randn(1, 4), torch.tensor([1.0]))
    assert output.shape == (1, 3)
