"""Regression tests for Qwen3.5 Merak export config defaults."""
from __future__ import annotations

import inspect

import pytest

from xhmodel_merak.xh_llm.base_model import XHBaseModel
from xhmodel_merak.xh_llm.types import KVCacheWithLinearConfig
from xhmodel_merak.xh_llm.models.qwen3_5.xh_qwen3_5_config import XHQwen3_5ModelConfig


def test_qwen3_5_config_uses_merak_export_defaults():
    cfg = XHQwen3_5ModelConfig(model_name="qwen3_5")

    assert cfg.split_conv_cache is True
    assert cfg.normalize_force_fp32 is False
    assert cfg.use_manual_depthwise_conv1d is False


def test_qwen3_5_model_forwards_manual_depthwise_flag_to_wrap_cfg():
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model import XHQwen3_5Model

    cfg = XHQwen3_5ModelConfig(model_name="qwen3_5")
    model = XHQwen3_5Model(cfg)

    assert model.wrap_cfg["split_conv_cache"] is True
    assert model.wrap_cfg["use_manual_depthwise_conv1d"] is False


def test_qwen3_5_spec_decode_restore_prefill_wrap_cfg():
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model import XHQwen3_5Model

    cfg = XHQwen3_5ModelConfig(
        model_name="qwen3_5",
        spec_decode_mode="mtp",
        output_post_norm_hidden=True,
        num_logits_to_keep=1,
        prefill_chunk_length=256,
    )
    model = XHQwen3_5Model(cfg)
    model.wrap_cfg["verify_output_intermediates"] = True
    model.wrap_cfg["num_logits_to_keep"] = 0
    model.wrap_cfg["input_sequence_length"] = 5
    model.wrap_cfg["output_post_norm_hidden"] = True

    model._restore_prefill_wrap_cfg()

    assert model.wrap_cfg["verify_output_intermediates"] is False
    assert model.wrap_cfg["num_logits_to_keep"] == 1
    assert model.wrap_cfg["input_sequence_length"] == 256
    assert model.wrap_cfg["output_post_norm_hidden"] is True


def test_qwen3_5_dflash_restore_prefill_wrap_cfg_without_post_norm_hidden():
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model import XHQwen3_5Model

    cfg = XHQwen3_5ModelConfig(
        model_name="qwen3_5",
        spec_decode_mode="dflash",
        num_logits_to_keep=1,
        output_hidden_state_indices=[1],
        prefill_chunk_length=256,
    )
    model = XHQwen3_5Model(cfg)
    model.wrap_cfg["verify_output_intermediates"] = True
    model.wrap_cfg["num_logits_to_keep"] = 0
    model.wrap_cfg["input_sequence_length"] = 10
    model.wrap_cfg["output_post_norm_hidden"] = True

    model._restore_prefill_wrap_cfg()

    assert model.wrap_cfg["verify_output_intermediates"] is False
    assert model.wrap_cfg["num_logits_to_keep"] == 1
    assert model.wrap_cfg["input_sequence_length"] == 256
    assert "output_post_norm_hidden" not in model.wrap_cfg


def test_qwen3_5_split_conv_cache_impl_imports_torch_nn():
    from xhmodel_merak.xh_llm.models.qwen3_5 import _llm_model_impl

    assert _llm_model_impl.nn.Linear is not None


def test_qwen3_5_split_conv_cache_helpers_round_trip_flat_inputs():
    from xhmodel_merak.xh_llm.models.qwen3_5 import _llm_model_impl
    from xhmodel_merak.xh_llm.models.qwen3_5_moe import _moe_model
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_hmonnx_inference import (
        _flatten_split_conv_cache_outputs,
        _regroup_flat_split_conv_cache,
    )

    flat = list(range(6))

    grouped = _llm_model_impl._regroup_flat_split_conv_cache(flat)
    assert grouped == [(0, 1, 2), (3, 4, 5)]
    assert _llm_model_impl._flatten_split_conv_cache_outputs(grouped) == flat

    hmonnx_grouped = _regroup_flat_split_conv_cache(flat)
    assert hmonnx_grouped == grouped
    assert _flatten_split_conv_cache_outputs(hmonnx_grouped) == flat

    moe_grouped = _moe_model._regroup_flat_split_conv_cache(flat)
    assert moe_grouped == grouped
    assert _moe_model._flatten_split_conv_cache_outputs(moe_grouped) == flat


def test_qwen3_5_split_conv_cache_helpers_flatten_mtp_composite_outputs():
    from xhmodel_merak.xh_llm.models.qwen3_5 import _llm_model_impl
    from xhmodel_merak.xh_llm.models.qwen3_5_moe import _moe_model
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_hmonnx_inference import (
        _flatten_split_conv_cache_outputs,
    )

    composite = [tuple(range(15))]
    expected = list(range(15))

    assert _llm_model_impl._flatten_split_conv_cache_outputs(composite) == expected
    assert _flatten_split_conv_cache_outputs(composite) == expected
    assert _moe_model._flatten_split_conv_cache_outputs(composite) == expected


def test_qwen3_5_split_conv_cache_helpers_reject_bad_flat_length():
    from xhmodel_merak.xh_llm.models.qwen3_5 import _llm_model_impl
    from xhmodel_merak.xh_llm.models.qwen3_5_moe import _moe_model
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_hmonnx_inference import (
        _regroup_flat_split_conv_cache,
    )

    with pytest.raises(RuntimeError, match="divisible by 3"):
        _llm_model_impl._regroup_flat_split_conv_cache([0, 1])
    with pytest.raises(RuntimeError, match="divisible by 3"):
        _regroup_flat_split_conv_cache([0, 1])
    with pytest.raises(RuntimeError, match="divisible by 3"):
        _moe_model._regroup_flat_split_conv_cache([0, 1])


def test_qwen3_5_split_conv_cache_helpers_do_not_bool_test_proxy_like_inputs():
    from xhmodel_merak.xh_llm.models.qwen3_5 import _llm_model_impl
    from xhmodel_merak.xh_llm.models.qwen3_5_moe import _moe_model
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_hmonnx_inference import (
        _regroup_flat_split_conv_cache,
    )
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model import (
        _regroup_flat_split_conv_cache as regroup_runtime_cache,
    )

    class ProxyLike:
        def __bool__(self):
            raise AssertionError("proxy-like inputs must not be used in Python truth tests")

    proxy_like = ProxyLike()
    assert _llm_model_impl._regroup_flat_split_conv_cache(proxy_like) is proxy_like
    assert _regroup_flat_split_conv_cache(proxy_like) is proxy_like
    assert regroup_runtime_cache(proxy_like) is proxy_like
    assert _moe_model._regroup_flat_split_conv_cache(proxy_like) is proxy_like



def test_qwen3_5_split_conv_cache_layer_indexing_uses_qkv_triples():
    from xhmodel_merak.xh_llm.models.qwen3_5 import _llm_model_impl
    from xhmodel_merak.xh_llm.models.qwen3_5_moe import _moe_model

    flat = ["q0", "k0", "v0", "q1", "k1", "v1"]
    nested = [("q0", "k0", "v0"), ("q1", "k1", "v1")]

    assert _llm_model_impl._get_linear_layer_conv_cache(flat, 1, True) == ("q1", "k1", "v1")
    assert _llm_model_impl._get_linear_layer_conv_cache(nested, 1, True) == ("q1", "k1", "v1")
    assert _llm_model_impl._get_linear_layer_conv_cache(["merged0", "merged1"], 1, False) == "merged1"
    assert _moe_model._get_linear_layer_conv_cache(flat, 1, True) == ("q1", "k1", "v1")
    assert _moe_model._get_linear_layer_conv_cache(nested, 1, True) == ("q1", "k1", "v1")

def test_qwen3_5_hmonnx_split_conv_cache_mixin_uses_flat_export_signature():
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_hmonnx_inference import (
        Qwen3_5HMONNXKVCacheMixin,
        _flatten_split_conv_cache_outputs,
    )

    cfg = KVCacheWithLinearConfig(
        linear_kv_cache_config={
            "conv_dim": 10,
            "conv_kernel_size": 4,
            "num_v_heads": 2,
            "head_k_dim": 2,
            "head_v_dim": 1,
            "num_layers": 2,
            "batch_size": 1,
        },
    )
    mixin = Qwen3_5HMONNXKVCacheMixin(cfg)
    mixin.split_conv_cache = True
    mixin.prepare_other_cache()

    assert len(mixin.past_conv_caches) == 2
    q_cache, k_cache, v_cache = mixin.past_conv_caches[0]
    assert tuple(q_cache.shape) == (1, 4, 4)
    assert tuple(k_cache.shape) == (1, 4, 4)
    assert tuple(v_cache.shape) == (1, 2, 4)
    assert len(_flatten_split_conv_cache_outputs(mixin.past_conv_caches)) == 6


def test_qwen3_5_export_cfg_honors_split_conv_cache_even_when_mixin_was_stale():
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model import XHQwen3_5Model

    model = XHQwen3_5Model(XHQwen3_5ModelConfig(model_name="qwen3_5"))
    model.kvcache_config.num_layers = 1
    model.kvcache_config.linear_kv_cache_config.num_layers = 2

    model._kvcache_mixin.split_conv_cache = False
    export_cfg = model.get_export_cfg()

    assert "past_conv_cache_q_0" in export_cfg["input_names"]
    assert "past_conv_cache_k_0" in export_cfg["input_names"]
    assert "past_conv_cache_v_0" in export_cfg["input_names"]
    assert "past_conv_cache_0" not in export_cfg["input_names"]


def test_qwen3_5_sync_rebuilds_legacy_merged_conv_caches_for_split_mode():
    import torch

    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model import XHQwen3_5Model
    from xhquant.core import CacheTensor

    model = XHQwen3_5Model(XHQwen3_5ModelConfig(model_name="qwen3_5"))
    linear_cfg = model.kvcache_config.linear_kv_cache_config
    linear_cfg.batch_size = 1
    linear_cfg.conv_dim = 14
    linear_cfg.conv_kernel_size = 4
    linear_cfg.num_v_heads = 2
    linear_cfg.head_k_dim = 2
    linear_cfg.head_v_dim = 1
    linear_cfg.num_layers = 1
    model._kvcache_mixin._linear_key_dim = 4
    model._kvcache_mixin._linear_value_dim = 6
    model._kvcache_mixin.past_conv_caches.append(
        CacheTensor(torch.zeros(1, 14, 4, dtype=torch.float16))
    )

    model._sync_split_conv_cache_state()

    assert len(model._kvcache_mixin.past_conv_caches) == 1
    q_cache, k_cache, v_cache = model._kvcache_mixin.past_conv_caches[0]
    assert tuple(q_cache.shape) == (1, 4, 4)
    assert tuple(k_cache.shape) == (1, 4, 4)
    assert tuple(v_cache.shape) == (1, 6, 4)
    assert [tuple(cache.shape) for cache in model.past_conv_caches] == [
        (1, 4, 4),
        (1, 4, 4),
        (1, 6, 4),
    ]



def test_qwen3_5_dense_text_model_setup_splits_child_linear_attn_modules():
    from xhmodel_merak.xh_llm.models.qwen3_5 import _llm_model_impl

    source = inspect.getsource(_llm_model_impl._Qwen3_5TextModel._setup)

    assert 'if self.split_conv_cache:' in source
    assert 'self.layer_types[idx_layer] != "linear_attention"' in source
    assert "linear_attn.split_conv_cache = True" in source
    assert 'not hasattr(linear_attn, "in_proj_q")' in source
    assert "linear_attn._setup(cfg)" in source

def test_qwen3_5_moe_text_model_regroups_flat_split_conv_cache_before_layer_indexing():
    from xhmodel_merak.xh_llm.models.qwen3_5_moe import _moe_model

    src = _moe_model._Qwen3_5MoeTextModel.forward.__code__
    names = set(src.co_names)

    assert "_regroup_flat_split_conv_cache" in names
    assert "_flatten_split_conv_cache_outputs" in names


def test_qwen3_5_hmonnx_forward_splits_qkv_conv_outputs_from_recurrent_tail(monkeypatch):
    import torch

    from xhmodel_merak.xh_llm.hmonnx.vision_llm_hmonnx_model import VisonLLMHMONNXModel
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_hmonnx_inference import (
        XHQwen3_5_HMONNXModel,
    )

    conv_outputs = [torch.full((1,), float(i)) for i in range(9)]
    recurrent_outputs = [torch.full((1,), 100.0 + i) for i in range(3)]

    def fake_forward(self, *args):
        return torch.tensor([42.0]), *conv_outputs, *recurrent_outputs

    monkeypatch.setattr(VisonLLMHMONNXModel, "forward", fake_forward)

    class DummyKVCacheMixin:
        split_conv_cache = True

        def __init__(self):
            self.past_conv_caches = [
                (torch.zeros(1), torch.zeros(1), torch.zeros(1)) for _ in range(3)
            ]
            self.past_recurrent_states = [torch.zeros(1) for _ in range(3)]

    model = XHQwen3_5_HMONNXModel.__new__(XHQwen3_5_HMONNXModel)
    model._kvcache_mixin = DummyKVCacheMixin()

    logits, conv_cache_out_list, recurrent_state_out_list = model.forward(torch.tensor([1]))

    assert logits.tolist() == [42.0]
    assert conv_cache_out_list == conv_outputs
    assert recurrent_state_out_list == recurrent_outputs
    for layer_idx, (past_q, past_k, past_v) in enumerate(model._kvcache_mixin.past_conv_caches):
        assert torch.equal(past_q, conv_outputs[layer_idx * 3])
        assert torch.equal(past_k, conv_outputs[layer_idx * 3 + 1])
        assert torch.equal(past_v, conv_outputs[layer_idx * 3 + 2])
    for past_state, recurrent_out in zip(
        model._kvcache_mixin.past_recurrent_states, recurrent_outputs, strict=True
    ):
        assert torch.equal(past_state, recurrent_out)



def test_qwen3_5_moe_text_model_forward_regroups_and_reflattens_split_conv_cache():
    import torch

    from xhmodel_merak.xh_llm.models.qwen3_5_moe import _moe_model

    class DummyRotaryEmb:
        def __init__(self):
            self.cos_cached = torch.ones(4, 2)
            self.sin_cached = torch.zeros(4, 2)
            self.time_mask = torch.ones(1, 1, 2)
            self.hight_mask = torch.zeros(1, 1, 2)
            self.width_mask = torch.zeros(1, 1, 2)

    class DummyLinearLayer:
        def __init__(self):
            self.seen_conv_cache = None

        def __call__(self, hidden_states, **kwargs):
            self.seen_conv_cache = kwargs["past_conv_cache"]
            assert isinstance(self.seen_conv_cache, tuple)
            assert self.seen_conv_cache == ("q0", "k0", "v0")
            return hidden_states, ("qo", "ko", "vo"), "state0"

    layer = DummyLinearLayer()
    model = _moe_model._Qwen3_5MoeTextModel.__new__(_moe_model._Qwen3_5MoeTextModel)
    model.rotary_emb = DummyRotaryEmb()
    model.split_conv_cache = True
    model.use_cache = False
    model.layer_types = ["linear_attention"]
    model.layers = [layer]
    model.max_layers = -1
    model.num_logits_to_keep = 0
    model.output_hidden_state_indices = None
    model.output_post_norm_hidden = False
    model.norm = lambda x: x

    hidden, conv_cache_out, recurrent_out = _moe_model._Qwen3_5MoeTextModel.forward(
        model,
        input_embeds=torch.ones(1, 1, 2),
        time_position_ids=torch.tensor([0]),
        hight_position_ids=torch.tensor([0]),
        width_position_ids=torch.tensor([0]),
        past_seq_length=torch.tensor([0]),
        current_input_length=torch.tensor([1]),
        linear_attn_mask=torch.ones(1, 1),
        past_conv_cache=["q0", "k0", "v0"],
        past_recurrent_state=["state_in"],
    )

    assert hidden.shape == (1, 1, 2)
    assert conv_cache_out == ["qo", "ko", "vo"]
    assert recurrent_out == ["state0"]
    assert layer.seen_conv_cache == ("q0", "k0", "v0")

def test_qwen3_5_moe_text_model_setup_splits_child_linear_attn_modules():
    from xhmodel_merak.xh_llm.models.qwen3_5_moe import _moe_model

    source = inspect.getsource(_moe_model._Qwen3_5MoeTextModel._setup)

    assert 'if self.split_conv_cache:' in source
    assert 'self.layer_types[idx_layer] != "linear_attention"' in source
    assert "linear_attn.split_conv_cache = True" in source
    assert 'not hasattr(linear_attn, "in_proj_q")' in source
    assert "linear_attn._setup(cfg)" in source


def test_qwen3_5_moe_gated_delta_net_sets_up_split_qkv_cache_path():
    from xhmodel_merak.xh_llm.models.qwen3_5_moe import _moe_model

    source = inspect.getsource(_moe_model._Qwen3_5MoeGatedDeltaNet)

    assert 'self.split_conv_cache = cfg.get("split_conv_cache", True)' in source
    assert "self.in_proj_q = nn.Linear" in source
    assert "self.conv1d_q = nn.Conv1d" in source
    assert "del self.in_proj_qkv" in source
    assert "del self.conv1d" in source


def test_quant_weight_directory_resolves_single_candidate(tmp_path):
    checkpoint = tmp_path / "only_weight.pth"
    checkpoint.write_bytes(b"checkpoint")

    assert XHBaseModel._resolve_quant_weight_archive(str(tmp_path)) == str(checkpoint)


def test_quant_weight_directory_prefers_common_filename(tmp_path):
    preferred = tmp_path / "quant_weight.pt"
    preferred.write_bytes(b"checkpoint")
    (tmp_path / "other_weight.pth").write_bytes(b"checkpoint")

    assert XHBaseModel._resolve_quant_weight_archive(str(tmp_path)) == str(preferred)


def test_quant_weight_directory_rejects_ambiguous_candidates(tmp_path):
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pth"
    first.write_bytes(b"checkpoint")
    second.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="multiple checkpoint candidates") as exc_info:
        XHBaseModel._resolve_quant_weight_archive(str(tmp_path))

    message = str(exc_info.value)
    assert str(first) in message
    assert str(second) in message


def test_quant_weight_directory_does_not_treat_hf_safetensors_dir_as_torch_checkpoint(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"checkpoint")

    with pytest.raises(FileNotFoundError, match="safetensors/GPTQ HF directories are not supported"):
        XHBaseModel._resolve_quant_weight_archive(str(tmp_path))


def test_get_hf_model_loads_float_model_then_external_quant_weight(monkeypatch):
    from types import SimpleNamespace

    calls = []

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.base_model.AutoConfig.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(quantization_config=None),
    )
    monkeypatch.setattr(XHBaseModel, "_load_hf_model", classmethod(lambda cls, hf_model_dir, **kwargs: calls.append(("hf", hf_model_dir, kwargs)) or object()))
    monkeypatch.setattr(
        XHBaseModel,
        "_load_quant_weight",
        classmethod(lambda cls, quant_weight, native_hf_model, strict=True: calls.append(("quant_weight", quant_weight)) or True),
    )

    XHBaseModel.get_hf_model("weights/Qwen3.5-9B", quant_weight="/tmp/quant_weight.pt", device_map="cpu")

    assert calls[0][0] == "hf"
    assert calls[0][1] == "weights/Qwen3.5-9B"
    assert calls[1] == ("quant_weight", "/tmp/quant_weight.pt")


def test_get_hf_model_loads_quantized_hf_repo_without_quant_weight(monkeypatch):
    from types import SimpleNamespace

    calls = []
    quant_cfg = {"quant_method": "gptq"}
    dummy_model = SimpleNamespace(config=SimpleNamespace(quantization_config=quant_cfg))

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.base_model.AutoConfig.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(quantization_config=quant_cfg),
    )
    monkeypatch.setattr(
        XHBaseModel,
        "_load_gptqmodel",
        classmethod(lambda cls, hf_model_dir, **kwargs: calls.append(("gptq", hf_model_dir, kwargs)) or dummy_model),
    )
    monkeypatch.setattr(
        XHBaseModel,
        "_dequantize_gptqmodel_hf_model",
        classmethod(lambda cls, native_hf_model: calls.append(("dequant",)) or native_hf_model),
    )
    monkeypatch.setattr(
        XHBaseModel,
        "_postprocess_gptqmodel_structure",
        classmethod(lambda cls, native_hf_model, **kwargs: calls.append(("postprocess", kwargs)) or native_hf_model),
    )
    monkeypatch.setattr(
        XHBaseModel,
        "_load_quant_weight",
        classmethod(lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("quant_weight loader must not run"))),
    )

    out = XHBaseModel.get_hf_model(
        "/data01/home/yujy/work/gptqmodel/output/Qwen3.5-9B-mode1-llm-only",
        quant_weight=None,
        device_map="cpu",
    )

    assert out is dummy_model
    assert calls[0][0] == "gptq"
    assert calls[0][1] == "/data01/home/yujy/work/gptqmodel/output/Qwen3.5-9B-mode1-llm-only"
    assert ("dequant",) in calls


def test_get_hf_model_rejects_quant_weight_for_quantized_hf_repo(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.base_model.AutoConfig.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(quantization_config={"quant_method": "gptq"}),
    )

    with pytest.raises(RuntimeError, match="already quantized"):
        XHBaseModel.get_hf_model(
            "/data01/home/yujy/work/gptqmodel/output/Qwen3.5-9B-mode1-llm-only",
            quant_weight="/tmp/quant_weight.pt",
        )
