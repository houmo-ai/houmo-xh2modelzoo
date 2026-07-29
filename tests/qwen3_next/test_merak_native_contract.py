from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def test_qwen3_next_merak_uses_transformers_native_identity():
    from xhmodel_merak.xh_llm.models.qwen3_next import (
        XHQwen3NextModel,
        XHQwen3NextModelConfig,
    )

    cfg = XHQwen3NextModelConfig(model_name="qwen3-next")

    assert XHQwen3NextModel.HF_MODEL_CLS.__name__ == "Qwen3NextForCausalLM"
    assert cfg.model_type == "Qwen3NextForCausalLM"
    assert XHQwen3NextModel.CONFIG_CLS is XHQwen3NextModelConfig
    assert XHQwen3NextModel.BUILD_HF_COMPATIBLE_FUNC.__name__ == ("build_qwen3_next_hf_compatible_model")
    assert not hasattr(cfg, "visual_config")


def test_qwen3_next_merak_config_keeps_hybrid_and_mtp_defaults():
    from xhmodel_merak.xh_llm.models.qwen3_next import XHQwen3NextModelConfig

    cfg = XHQwen3NextModelConfig(
        model_name="qwen3-next",
        hf_model="weights/qwen3-next",
        spec_decode_mode="mtp",
        mtp_config={},
    )

    assert cfg.split_conv_cache is True
    assert cfg.linear_chunk_size == 64
    assert cfg.spec_decode_mode == "mtp"
    assert cfg.mtp_config.model_type == "Qwen3NextMTP"
    assert cfg.mtp_config.hf_model == "weights/qwen3-next"


def test_qwen3_next_mtp_inherits_flash_attention_contract():
    from xhmodel_merak.xh_llm.models.qwen3_next import XHQwen3NextModelConfig
    from xhmodel_merak.xh_llm.models.qwen3_next._mtp_model_impl import (
        MTPGatedAttention,
    )
    from xhquant.nn import FlashAttention

    flash_attention = {
        "enable": True,
        "q_bits": 8,
        "k_bits": 8,
        "v_bits": 8,
        "s_bits": 8,
        "p_bits": 8,
    }
    cfg = XHQwen3NextModelConfig(
        model_name="qwen3-next",
        hf_model="weights/qwen3-next",
        flash_attention=flash_attention,
        spec_decode_mode="mtp",
        mtp_config={},
    )

    assert cfg.mtp_config.flash_attention == flash_attention
    attention = MTPGatedAttention(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        rotary_dim=2,
        rms_norm_eps=1e-6,
        input_sequence_length=1,
        rope_theta=10_000.0,
        max_pe_length=16,
        use_cache=True,
        flash_attention=cfg.mtp_config.flash_attention,
    )
    assert attention.use_flash_attention is True
    assert isinstance(attention.flash_attn, FlashAttention)


def test_qwen3_next_mtp_layer_discovery_requires_contiguous_indices():
    from xhmodel_merak.xh_llm.models.qwen3_next._mtp_model_impl import (
        infer_mtp_layer_indices,
    )

    assert infer_mtp_layer_indices(
        [
            "mtp.fc.weight",
            "mtp.layers.0.input_layernorm.weight",
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.1.input_layernorm.weight",
        ]
    ) == [0, 1]

    with pytest.raises(ValueError, match="contiguous"):
        infer_mtp_layer_indices(
            [
                "mtp.layers.0.input_layernorm.weight",
                "mtp.layers.2.input_layernorm.weight",
            ]
        )

    with pytest.raises(ValueError, match="No MTP decoder layers"):
        infer_mtp_layer_indices(["model.layers.0.input_layernorm.weight"])


def test_qwen3_next_mtp_checkpoint_schema_is_discovered_from_index(tmp_path: Path):
    from xhmodel_merak.xh_llm.models.qwen3_next._mtp_model_impl import (
        discover_mtp_checkpoint,
    )

    index = {
        "weight_map": {
            "mtp.fc.weight": "model-00001-of-00002.safetensors",
            "mtp.layers.0.input_layernorm.weight": "model-00002-of-00002.safetensors",
            "mtp.norm.weight": "model-00002-of-00002.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    schema = discover_mtp_checkpoint(tmp_path)

    assert schema.layer_indices == (0,)
    assert schema.tensor_names == tuple(sorted(index["weight_map"]))
    assert schema.shards == (
        tmp_path / "model-00001-of-00002.safetensors",
        tmp_path / "model-00002-of-00002.safetensors",
    )


def test_qwen3_next_text_processor_emits_hybrid_cache_signature():
    from xhmodel_merak.xh_llm.models.qwen3_next.data_preprocess import (
        Qwen3NextDataPreprocess,
    )
    from xhmodel_merak.xh_llm.types import CacheList

    embedding = torch.nn.Embedding(16, 8)
    processor = Qwen3NextDataPreprocess(
        token_embedding=embedding,
        input_sequence_length=4,
        past_key_caches=CacheList([torch.zeros(1, 1, 8, 4)]),
        past_value_caches=CacheList([torch.zeros(1, 1, 8, 4)]),
        past_conv_caches=CacheList(
            [
                CacheList(
                    [
                        torch.zeros(1, 4, 4),
                        torch.zeros(1, 4, 4),
                        torch.zeros(1, 4, 4),
                    ]
                )
            ]
        ),
        past_recurrent_states=CacheList([torch.zeros(1, 1, 4, 4)]),
    )

    values = processor({"input_ids": torch.tensor([[1, 2]]), "past_seq_length": 0})

    assert len(values) == 8
    inputs_embeds, past_seq, current_len, linear_mask = values[:4]
    assert inputs_embeds.shape == (1, 4, 8)
    assert past_seq.tolist() == [0]
    assert current_len.tolist() == [2]
    assert linear_mask.tolist() == [[1.0, 1.0, 0.0, 0.0]]
    assert len(values[6]) == 3


def test_qwen3_next_hf_adapter_chunks_prefill_and_advances_hybrid_cache_position():
    from xhmodel_merak.xh_llm.models.qwen3_next.qwen3_next_model import (
        _Qwen3NextHFCompatible,
    )

    calls = []
    page_contexts = []

    class FakeProcessor:
        def __call__(self, data):
            calls.append((data["past_seq_length"], data["inputs_embeds"].shape[1]))
            return (data["inputs_embeds"],)

    class FakeRuntime:
        enable_page_attention = True

        def get_input_sequence_length(self):
            return 2

        def get_data_preprocessor(self):
            return FakeProcessor()

        def get_num_logits_to_keep(self):
            return 1

        def prepare_page_attention_context(self, past_seq_length, current_input_length):
            page_contexts.append((past_seq_length, current_input_length))

        def forward(self, inputs_embeds):
            value = float(len(calls))
            return torch.full((1, 1, 8), value), (), ()

    adapter = object.__new__(_Qwen3NextHFCompatible)
    torch.nn.Module.__init__(adapter)
    adapter._llm_model = FakeRuntime()
    adapter._past_seq_length = 5

    output = adapter.forward(inputs_embeds=torch.zeros(1, 5, 4), use_cache=False)

    assert calls == [(5, 2), (7, 2), (9, 1)]
    assert page_contexts == calls
    assert output.logits.shape == (1, 1, 8)
    assert output.logits.unique().item() == 3
    assert adapter.set_experts_implementation("grouped_mm") is adapter


def test_qwen3_next_gdn_keeps_native_packed_projection():
    from xhmodel_merak.xh_llm.models.qwen3_5._hybrid_gated_delta_net import (
        HybridGatedDeltaNetMixin,
    )
    from xhmodel_merak.xh_llm.models.qwen3_next import _model

    next_gdn = _model._Qwen3NextGatedDeltaNet
    module = next_gdn.__new__(next_gdn)
    torch.nn.Module.__init__(module)
    module.num_k_heads = 2
    module.num_v_heads = 2
    module.head_k_dim = 2
    module.head_v_dim = 2
    module.in_proj_qkvz = torch.nn.Linear(8, 16, bias=False)
    module.in_proj_ba = torch.nn.Linear(8, 4, bias=False)

    query, key, value, z, b, a = module._project_qkvzba(torch.randn(1, 3, 8))

    assert issubclass(next_gdn, HybridGatedDeltaNetMixin)
    assert next_gdn.forward is HybridGatedDeltaNetMixin.forward
    assert next_gdn._setup is HybridGatedDeltaNetMixin._setup
    assert next_gdn._update_cfg is HybridGatedDeltaNetMixin._update_cfg
    assert "_project_qkvzba" in next_gdn.__dict__
    assert hasattr(module, "in_proj_qkvz") and hasattr(module, "in_proj_ba")
    assert [tensor.shape for tensor in (query, key, value, z, b, a)] == [
        torch.Size([1, 3, 2, 2]),
        torch.Size([1, 3, 2, 2]),
        torch.Size([1, 3, 2, 2]),
        torch.Size([1, 3, 2, 2]),
        torch.Size([1, 3, 2]),
        torch.Size([1, 3, 2]),
    ]


def test_qwen3_next_target_verification_exports_each_hybrid_cache_step():
    from xhmodel_merak.xh_llm.models.qwen3_next import XHQwen3NextModel

    class AttrDict(dict):
        __getattr__ = dict.__getitem__

    class FakeTarget:
        wrap_cfg = AttrDict(
            split_conv_cache=True,
            verify_output_intermediates=True,
            input_sequence_length=5,
            output_post_norm_hidden=True,
        )
        config = SimpleNamespace(split_conv_cache=True)
        _kvcache_mixin = SimpleNamespace(enable_page_attention=False, split_conv_cache=True)
        kvcache_config = SimpleNamespace(
            num_layers=1,
            linear_kv_cache_config=SimpleNamespace(num_layers=2),
        )

        def _sync_split_conv_cache_state(self):
            return None

        def _set_recurrent_state_output_contract(self):
            return False

    output_names = XHQwen3NextModel.get_export_cfg(FakeTarget())["output_names"]

    expected_conv = [
        f"conv_cache_out_{branch}_{layer}_{step}"
        for layer in range(2)
        for branch in ("q", "k", "v")
        for step in range(5)
    ]
    expected_recurrent = [f"recurrent_state_out_{layer}_{step}" for layer in range(2) for step in range(5)]
    assert output_names == ["logits", *expected_conv, *expected_recurrent, "post_norm_hidden"]


def test_qwen3_next_target_verification_advances_gdn_one_token_at_a_time(monkeypatch):
    from xhmodel_merak.xh_llm.models.qwen3_5 import _hybrid_gated_delta_net as shared
    from xhmodel_merak.xh_llm.models.qwen3_next import _model
    from xhquant.api import ConfigDict

    module = _model._Qwen3NextGatedDeltaNet.__new__(_model._Qwen3NextGatedDeltaNet)
    torch.nn.Module.__init__(module)
    module.hidden_size = 8
    module.key_dim = 4
    module.value_dim = 4
    module.conv_dim = 12
    module.conv_kernel_size = 4
    module.num_v_heads = 2
    module.num_k_heads = 2
    module.head_k_dim = 2
    module.head_v_dim = 2
    module.in_proj_qkvz = torch.nn.Linear(8, 16, bias=False)
    module.in_proj_ba = torch.nn.Linear(8, 4, bias=False)
    module.conv1d = torch.nn.Conv1d(12, 12, kernel_size=4, groups=12, padding=3, bias=False)
    module.out_proj = torch.nn.Linear(4, 8, bias=False)
    module.dt_bias = torch.nn.Parameter(torch.zeros(2))
    module.A_log = torch.nn.Parameter(torch.zeros(2))

    class IdentityGatedNorm(torch.nn.Module):
        def forward(self, hidden_states, gate):
            del gate
            return hidden_states

    module.norm = IdentityGatedNorm()
    module._setup(
        ConfigDict(
            use_cache=True,
            linear_attention_mode="recurrent",
            linear_chunk_size=8,
            input_sequence_length=5,
            batch_size=1,
            split_conv_cache=True,
            verify_output_intermediates=True,
            fuse_gdr_ops=False,
            fuse_gdr_block_recurrent_ops=False,
            use_manual_depthwise_conv1d=False,
        )
    )
    step_lengths = []

    def fake_recurrent_rule(query, key, value, **kwargs):
        del key, value
        step_lengths.append((query.shape[1], kwargs["sequence_length"]))
        output = torch.zeros(1, 1, 2, 2)
        return output, kwargs["initial_state"] + 1

    monkeypatch.setattr(shared, "torch_recurrent_gated_delta_rule", fake_recurrent_rule)
    _, conv_outputs, recurrent_outputs = module(
        torch.randn(1, 5, 8),
        (
            torch.zeros(1, 4, 4),
            torch.zeros(1, 4, 4),
            torch.zeros(1, 4, 4),
        ),
        torch.zeros(1, 2, 2, 2),
        torch.ones(1, 5),
        torch.tensor([5]),
    )

    assert step_lengths == [(1, 1)] * 5
    assert len(conv_outputs) == 15
    assert len(recurrent_outputs) == 5
    assert [state.flatten()[0].item() for state in recurrent_outputs] == [
        1,
        2,
        3,
        4,
        5,
    ]


def test_qwen3_next_target_verification_keeps_all_split_conv_snapshots():
    from xhmodel_merak.xh_llm.models.qwen3_5 import _hybrid_text_model
    from xhmodel_merak.xh_llm.models.qwen3_5._hybrid_text_model import (
        HybridTextModelMixin,
    )
    from xhmodel_merak.xh_llm.models.qwen3_next import _model

    next_text_model = _model._Qwen3NextModel
    shared_source = Path(_hybrid_text_model.__file__).read_text()

    assert issubclass(next_text_model, HybridTextModelMixin)
    assert next_text_model.forward is HybridTextModelMixin.forward
    assert "_flatten_split_conv_cache_outputs(conv_cache_out_list)" in shared_source
    assert "conv_cache_out_list.append(conv_cache_out[0])" not in shared_source


def test_qwen3_next_mtp_export_reuses_full_target_spec_decode_contract():
    from xhmodel_merak.xh_llm.models.qwen3_next import (
        XHQwen3NextModel,
        qwen3_next_model,
    )

    source = Path(qwen3_next_model.__file__).read_text()
    export_body = source.split("def export_hmonnx", 1)[1]

    assert "self._configure_spec_decode_target_export()" in export_body
    assert "build_qwen35_spec_decode_contract(" in export_body
    assert "meta_info.spec_decode_draft_head_weight_bits" in export_body

    class FakeTextGraph(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_logits_to_keep = 1
            self.output_post_norm_hidden = False

    target = SimpleNamespace(
        config=SimpleNamespace(spec_decode_mode="mtp", num_draft_tokens=4),
        wrap_cfg={},
        _quanted_model=SimpleNamespace(
            prefill=FakeTextGraph(),
            decode=FakeTextGraph(),
        ),
    )

    XHQwen3NextModel._configure_spec_decode_target_export(target)

    assert target._decode_input_sequence_length == 5
    assert target.wrap_cfg["num_logits_to_keep"] == 0
    assert target.wrap_cfg["output_post_norm_hidden"] is True
    assert target._quanted_model.prefill.num_logits_to_keep == 0
    assert target._quanted_model.prefill.output_post_norm_hidden is True
    assert target._quanted_model.decode.num_logits_to_keep == 0
    assert target._quanted_model.decode.output_post_norm_hidden is True


def test_qwen3_next_full_attention_exports_page_convertible_flash_op():
    from xhmodel_merak.xh_llm.models.qwen3_5._hybrid_gated_delta_net import (
        HybridGatedAttentionMixin,
    )
    from xhmodel_merak.xh_llm.models.qwen3_next import _model
    from xhquant.api import ConfigDict
    from xhquant.nn import FlashAttention

    next_attention = _model._Qwen3NextAttention
    module = next_attention.__new__(next_attention)
    torch.nn.Module.__init__(module)
    module.config = SimpleNamespace(
        num_key_value_heads=1,
        num_attention_heads=2,
        partial_rotary_factor=0.5,
    )
    module.head_dim = 4
    module.q_proj = torch.nn.Linear(8, 16, bias=False)
    module.k_proj = torch.nn.Linear(8, 4, bias=False)
    module.v_proj = torch.nn.Linear(8, 4, bias=False)
    module.o_proj = torch.nn.Linear(8, 8, bias=False)
    module.q_norm = torch.nn.Identity()
    module.k_norm = torch.nn.Identity()
    module._setup(
        ConfigDict(
            use_cache=False,
            enable_rope=False,
            flash_attention={
                "enable": True,
                "q_bits": 8,
                "k_bits": 8,
                "v_bits": 8,
                "s_bits": 8,
                "p_bits": 8,
            },
        )
    )

    assert issubclass(next_attention, HybridGatedAttentionMixin)
    assert next_attention.forward is HybridGatedAttentionMixin.forward
    assert next_attention._setup is HybridGatedAttentionMixin._setup
    assert isinstance(module.flash_attn, FlashAttention)

    calls = []

    class FakeFlashAttention(torch.nn.Module):
        def forward(self, query, key, value, **kwargs):
            calls.append((query.shape, key.shape, value.shape, kwargs))
            return torch.zeros_like(query).transpose(1, 2)

    module.flash_attn = FakeFlashAttention()
    past_seq_length = torch.tensor([7])
    current_input_length = torch.tensor([3])
    output = module(
        torch.randn(1, 3, 8),
        past_seq_length=past_seq_length,
        current_input_length=current_input_length,
    )

    assert output.shape == (1, 3, 8)
    assert len(calls) == 1
    assert calls[0][3]["past_seq_length"] is past_seq_length
    assert calls[0][3]["current_input_length"] is current_input_length


def test_qwen3_next_mtp_loads_native_explicit_schema(tmp_path: Path):
    from safetensors.torch import save_file

    from xhmodel_merak.xh_llm.models.qwen3_next._mtp_model_impl import (
        MTPModel,
        MTPSparseMoEBlock,
    )
    from xhquant.api import CacheTensor
    from xhquant.nn import RMSNorm

    config = {
        "architectures": ["Qwen3NextForCausalLM"],
        "model_type": "qwen3_next",
        "hidden_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "intermediate_size": 24,
        "moe_intermediate_size": 8,
        "shared_expert_intermediate_size": 8,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "vocab_size": 32,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "partial_rotary_factor": 0.5,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    reference = MTPModel(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        intermediate_size=24,
        rms_norm_eps=1e-6,
        vocab_size=32,
        input_sequence_length=2,
        rope_theta=1_000_000.0,
        partial_rotary_factor=0.5,
        max_pe_length=16,
    )
    reference.layer.mlp = MTPSparseMoEBlock(
        hidden_size=16,
        num_experts=2,
        top_k=1,
        expert_intermediate_size=8,
        shared_expert_intermediate_size=8,
    )
    state = reference.state_dict()
    tensors = {}
    rms_keys = {f"{name}.weight" for name, module in reference.named_modules() if isinstance(module, RMSNorm) and name}
    for name, tensor in state.items():
        if name.startswith("layer.mlp.moeblock.expert_"):
            continue
        if name == "lm_head.weight":
            tensors[name] = tensor
            continue
        source_name = "mtp." + ("layers.0." + name.removeprefix("layer.") if name.startswith("layer.") else name)
        tensors[source_name] = tensor - 1 if name in rms_keys else tensor
    tensors["mtp.layers.0.mlp.experts.gate_up_proj"] = torch.cat(
        [
            state["layer.mlp.moeblock.expert_gate_proj_weight"],
            state["layer.mlp.moeblock.expert_up_proj_weight"],
        ],
        dim=1,
    )
    tensors["mtp.layers.0.mlp.experts.down_proj"] = state["layer.mlp.moeblock.expert_down_proj_weight"]
    save_file(tensors, tmp_path / "model.safetensors")

    loaded = MTPModel.from_pretrained(
        tmp_path,
        dtype=torch.float32,
        input_sequence_length=2,
        max_pe_length=16,
    )
    logits, hidden = loaded(
        torch.randn(1, 2, 16),
        torch.randn(1, 2, 16),
        torch.tensor([0]),
        torch.tensor([2]),
        CacheTensor(torch.zeros(1, 1, 8, 8)),
        CacheTensor(torch.zeros(1, 1, 8, 8)),
    )

    assert logits.shape == (1, 2, 32)
    assert hidden.shape == (1, 2, 16)
