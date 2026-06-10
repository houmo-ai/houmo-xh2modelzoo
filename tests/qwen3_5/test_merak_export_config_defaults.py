"""Regression tests for Qwen3.5 Merak export config defaults."""
from __future__ import annotations

import inspect
import importlib.util
from pathlib import Path

import pytest

from xhmodel_merak.xh_llm.base_model import XHBaseModel
from xhmodel_merak.xh_llm.types import KVCacheWithLinearConfig
from xhmodel_merak.xh_llm.models.qwen3_5.xh_qwen3_5_config import (
    XHQwen3_5_DFlashConfig,
    XHQwen3_5_MTPConfig,
    XHQwen3_5ModelConfig,
    build_spec_draft_quant_scheme,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MERAK_EXPORT_SCRIPT = REPO_ROOT / "examples_merak/llm/qwen3_5/qwen3_5_xh_export_hmonnx.py"


def _load_merak_export_script():
    spec = importlib.util.spec_from_file_location("qwen3_5_xh_export_hmonnx", MERAK_EXPORT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qwen3_5_config_uses_merak_export_defaults():
    cfg = XHQwen3_5ModelConfig(model_name="qwen3_5")

    assert cfg.split_conv_cache is True
    assert cfg.normalize_force_fp32 is False
    assert cfg.use_manual_depthwise_conv1d is False
    assert cfg.mtp_head_k is None
    assert cfg.reranked_repo_dir is None
    assert cfg.force_rerank is False


def test_qwen3_5_config_preserves_mtp_head_k_controls():
    cfg = XHQwen3_5ModelConfig(
        model_name="qwen3_5",
        hf_model="weights/qwen",
        mtp_head_k=81920,
        reranked_repo_dir="weights/qwen-reranked-K81920",
        force_rerank=True,
    )

    assert cfg.mtp_head_k == 81920
    assert cfg.reranked_repo_dir == "weights/qwen-reranked-K81920"
    assert cfg.force_rerank is True


@pytest.mark.parametrize("mode", ["mtp", "dflash"])
def test_qwen3_5_spec_decode_is_single_batch_only(mode):
    XHQwen3_5ModelConfig(model_name="qwen3_5", batch_size=1, spec_decode_mode=mode)

    with pytest.raises(ValueError, match="batch_size=1"):
        XHQwen3_5ModelConfig(model_name="qwen3_5", batch_size=2, spec_decode_mode=mode)


def test_merak_export_reranks_target_and_mtp_hf_model(monkeypatch):
    script = _load_merak_export_script()
    cfg = XHQwen3_5ModelConfig(
        model_name="qwen3_5",
        hf_model="weights/qwen",
        spec_decode_mode="mtp",
        mtp_head_k=81920,
        mtp_config=dict(hf_model="weights/qwen"),
    )
    calls = []

    def fake_prepare(*, original_model_dir, K, reranked_repo_dir, force_rerank):
        calls.append(
            dict(
                original_model_dir=original_model_dir,
                K=K,
                reranked_repo_dir=reranked_repo_dir,
                force_rerank=force_rerank,
            )
        )
        return "/abs/weights/qwen-reranked-K81920"

    monkeypatch.setattr(script, "prepare_reranked_repo", fake_prepare)

    script._apply_mtp_head_k_rerank(cfg, logger=None)

    assert calls == [
        dict(
            original_model_dir="weights/qwen",
            K=81920,
            reranked_repo_dir="weights/qwen-reranked-K81920",
            force_rerank=False,
        )
    ]
    assert cfg.hf_model == "/abs/weights/qwen-reranked-K81920"
    assert cfg.mtp_config.hf_model == "/abs/weights/qwen-reranked-K81920"


def test_qwen3_5_spec_draft_head_quant_defaults_to_w4():
    cfg = build_spec_draft_quant_scheme()

    assert cfg["quant_type"] == "w8a8h1_sefp"
    assert cfg["nodes_cfg"]["lm_head"]["w_schema"] == {
        "bits": 4,
        "fp_mode": "ssfp",
        "hidden_bit": False,
    }


def test_qwen3_5_spec_draft_head_quant_supports_w8_without_override():
    cfg = build_spec_draft_quant_scheme(8)

    assert cfg == {"quant_type": "w8a8h1_sefp"}


def test_qwen3_5_spec_draft_head_quant_rejects_unsupported_bits():
    with pytest.raises(ValueError, match="Expected 4 or 8"):
        build_spec_draft_quant_scheme(6)


@pytest.mark.parametrize("config_cls", [XHQwen3_5_MTPConfig, XHQwen3_5_DFlashConfig])
def test_qwen3_5_spec_draft_configs_default_lm_head_to_w4(config_cls):
    kwargs = dict(model_name="draft", hf_model="weights/draft")
    if config_cls is XHQwen3_5_DFlashConfig:
        kwargs["target_model_dir"] = "weights/target"

    cfg = config_cls(**kwargs)

    assert cfg.draft_head_weight_bits == 4
    assert cfg.quant_scheme["nodes_cfg"]["lm_head"]["w_schema"]["bits"] == 4


def test_qwen3_5_spec_draft_configs_keep_explicit_quant_scheme():
    cfg = XHQwen3_5_MTPConfig(
        model_name="draft",
        hf_model="weights/draft",
        draft_head_weight_bits=4,
        quant_scheme=dict(quant_type="w8a8h1_sefp"),
    )

    assert cfg.quant_scheme == {"quant_type": "w8a8h1_sefp"}


def test_qwen3_5_spec_draft_head_bits_propagate_from_main_config():
    cfg = XHQwen3_5ModelConfig(
        model_name="qwen3_5",
        hf_model="weights/target",
        spec_draft_head_weight_bits=8,
        mtp_config=dict(),
        dflash_config=dict(hf_model="weights/dflash"),
    )

    assert cfg.mtp_config.draft_head_weight_bits == 8
    assert cfg.mtp_config.quant_scheme == {"quant_type": "w8a8h1_sefp"}
    assert cfg.dflash_config.draft_head_weight_bits == 8
    assert cfg.dflash_config.quant_scheme == {"quant_type": "w8a8h1_sefp"}


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


def test_qwen3_5_merged_conv_cache_helper_flattens_spec_decode_steps_without_qkv_requirement():
    from xhmodel_merak.xh_llm.models.qwen3_5.split_conv_cache_utils import (
        _flatten_merged_conv_cache_outputs,
        _flatten_split_conv_cache_outputs,
    )

    merged_spec_outputs = [tuple(range(5)), tuple(range(5, 10))]

    assert _flatten_merged_conv_cache_outputs(merged_spec_outputs) == list(range(10))
    with pytest.raises(RuntimeError, match="divisible by 3"):
        _flatten_split_conv_cache_outputs(merged_spec_outputs)


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

def test_qwen3_5_hmonnx_decode_input_sequence_length_honors_spec_decode():
    from types import SimpleNamespace

    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_hmonnx_inference import (
        XHQwen3_5_HMONNXModel,
    )

    model = XHQwen3_5_HMONNXModel.__new__(XHQwen3_5_HMONNXModel)
    model.meta_info = SimpleNamespace(
        model_config=SimpleNamespace(prefill_chunk_length=256),
        spec_decode={"mode": "mtp", "num_draft_tokens": 4},
    )

    model._llm_prefill = True
    assert model.get_input_sequence_length() == 256

    model._llm_prefill = False
    assert model.get_input_sequence_length() == 5

    model.meta_info.spec_decode = SimpleNamespace(mode="dflash", num_draft_tokens=9)
    assert model.get_input_sequence_length() == 10

    model.meta_info.spec_decode = None
    assert model.get_input_sequence_length() == 1


def test_qwen3_5_moe_hmonnx_decode_input_sequence_length_honors_spec_decode():
    from types import SimpleNamespace

    from xhmodel_merak.xh_llm.models.qwen3_5_moe.qwen3_5_moe_hmonnx_inference import (
        XHQwen3_5MoeHMONNXModel,
    )

    model = XHQwen3_5MoeHMONNXModel.__new__(XHQwen3_5MoeHMONNXModel)
    model.meta_info = SimpleNamespace(
        model_config=SimpleNamespace(prefill_chunk_length=256),
        spec_decode={"mode": "mtp", "num_draft_tokens": 4},
    )

    model._llm_prefill = True
    assert model.get_input_sequence_length() == 256

    model._llm_prefill = False
    assert model.get_input_sequence_length() == 5

    model.meta_info.spec_decode = SimpleNamespace(mode="dflash", num_draft_tokens=9)
    assert model.get_input_sequence_length() == 10

    model.meta_info.spec_decode = None
    assert model.get_input_sequence_length() == 1


def test_text_llm_hf_compatible_decode_uses_model_declared_input_length():
    from xhmodel_merak.xh_llm import text_llm_hf_compatible

    source = inspect.getsource(text_llm_hf_compatible.TextLLMHFCompatible._sample_forward)

    assert "set_input_sequence_length(1)" not in source
    assert "self._llm_model.get_input_sequence_length()" in source


def test_qwen3_5_moe_text_model_regroups_flat_split_conv_cache_before_layer_indexing():
    from xhmodel_merak.xh_llm.models.qwen3_5_moe import _moe_model

    src = _moe_model._Qwen3_5MoeTextModel.forward.__code__
    names = set(src.co_names)

    assert "_regroup_flat_split_conv_cache" in names
    assert "_flatten_split_conv_cache_outputs" in names


def test_qwen3_5_text_models_use_merged_flatten_path_when_split_conv_cache_is_disabled():
    from xhmodel_merak.xh_llm.models.qwen3_5 import _llm_model_impl
    from xhmodel_merak.xh_llm.models.qwen3_5_moe import _moe_model

    dense_source = inspect.getsource(_llm_model_impl._Qwen3_5TextModel.forward)
    moe_source = inspect.getsource(_moe_model._Qwen3_5MoeTextModel.forward)

    assert "if split_conv_cache:" in dense_source
    assert "_flatten_merged_conv_cache_outputs(conv_cache_out_list)" in dense_source
    assert "if split_conv_cache:" in moe_source
    assert "_flatten_merged_conv_cache_outputs(conv_cache_out_list)" in moe_source


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



def test_qwen3_5_hmonnx_forward_uses_final_spec_decode_split_cache_step(monkeypatch):
    from types import SimpleNamespace

    import torch

    from xhmodel_merak.xh_llm.hmonnx.vision_llm_hmonnx_model import VisonLLMHMONNXModel
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_hmonnx_inference import (
        XHQwen3_5_HMONNXModel,
    )

    q0, q1 = torch.full((1,), 1.0), torch.full((1,), 2.0)
    k0, k1 = torch.full((1,), 3.0), torch.full((1,), 4.0)
    v0, v1 = torch.full((1,), 5.0), torch.full((1,), 6.0)
    r0, r1 = torch.full((1,), 7.0), torch.full((1,), 8.0)

    def fake_forward(self, *args):
        # verify_steps=2 split-cache export order: q steps, k steps, v steps, then recurrent steps.
        return torch.tensor([42.0]), q0, q1, k0, k1, v0, v1, r0, r1, torch.tensor([99.0])

    monkeypatch.setattr(VisonLLMHMONNXModel, "forward", fake_forward)

    class DummyKVCacheMixin:
        split_conv_cache = True

        def __init__(self):
            self.past_conv_caches = [(torch.zeros(1), torch.zeros(1), torch.zeros(1))]
            self.past_recurrent_states = [torch.zeros(1)]

    model = XHQwen3_5_HMONNXModel.__new__(XHQwen3_5_HMONNXModel)
    model._kvcache_mixin = DummyKVCacheMixin()
    model._llm_prefill = False
    model.meta_info = SimpleNamespace(
        spec_decode={"mode": "mtp", "num_draft_tokens": 1},
        model_config=SimpleNamespace(prefill_chunk_length=256),
    )

    logits, conv_cache_out_list, recurrent_state_out_list = model.forward(torch.tensor([1]))

    assert logits.tolist() == [42.0]
    assert conv_cache_out_list == [q1, k1, v1]
    assert recurrent_state_out_list == [r1]
    past_q, past_k, past_v = model._kvcache_mixin.past_conv_caches[0]
    assert torch.equal(past_q, q1)
    assert torch.equal(past_k, k1)
    assert torch.equal(past_v, v1)
    assert torch.equal(model._kvcache_mixin.past_recurrent_states[0], r1)


def test_qwen3_5_hmonnx_forward_uses_final_spec_decode_merged_cache_step(monkeypatch):
    from types import SimpleNamespace

    import torch

    from xhmodel_merak.xh_llm.hmonnx.vision_llm_hmonnx_model import VisonLLMHMONNXModel
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_hmonnx_inference import (
        XHQwen3_5_HMONNXModel,
    )

    c0, c1 = torch.full((1,), 1.0), torch.full((1,), 2.0)
    r0, r1 = torch.full((1,), 3.0), torch.full((1,), 4.0)

    def fake_forward(self, *args):
        return torch.tensor([42.0]), c0, c1, r0, r1

    monkeypatch.setattr(VisonLLMHMONNXModel, "forward", fake_forward)

    class DummyKVCacheMixin:
        split_conv_cache = False

        def __init__(self):
            self.past_conv_caches = [torch.zeros(1)]
            self.past_recurrent_states = [torch.zeros(1)]

    model = XHQwen3_5_HMONNXModel.__new__(XHQwen3_5_HMONNXModel)
    model._kvcache_mixin = DummyKVCacheMixin()
    model._llm_prefill = False
    model.meta_info = SimpleNamespace(
        spec_decode={"mode": "dflash", "num_draft_tokens": 1},
        model_config=SimpleNamespace(prefill_chunk_length=256),
    )

    logits, conv_cache_out_list, recurrent_state_out_list = model.forward(torch.tensor([1]))

    assert logits.tolist() == [42.0]
    assert conv_cache_out_list == [c1]
    assert recurrent_state_out_list == [r1]
    assert torch.equal(model._kvcache_mixin.past_conv_caches[0], c1)
    assert torch.equal(model._kvcache_mixin.past_recurrent_states[0], r1)


def test_qwen3_5_moe_hmonnx_forward_uses_final_spec_decode_split_cache_step(monkeypatch):
    from types import SimpleNamespace

    import torch

    from xhmodel_merak.xh_llm.hmonnx.vision_llm_hmonnx_model import VisonLLMHMONNXModel
    from xhmodel_merak.xh_llm.models.qwen3_5_moe.qwen3_5_moe_hmonnx_inference import (
        XHQwen3_5MoeHMONNXModel,
    )

    q0, q1 = torch.full((1,), 1.0), torch.full((1,), 2.0)
    k0, k1 = torch.full((1,), 3.0), torch.full((1,), 4.0)
    v0, v1 = torch.full((1,), 5.0), torch.full((1,), 6.0)
    r0, r1 = torch.full((1,), 7.0), torch.full((1,), 8.0)

    def fake_forward(self, *args):
        # verify_steps=2 split-cache export order: q steps, k steps, v steps, then recurrent steps.
        return torch.tensor([42.0]), q0, q1, k0, k1, v0, v1, r0, r1, torch.tensor([99.0])

    monkeypatch.setattr(VisonLLMHMONNXModel, "forward", fake_forward)

    class DummyKVCacheMixin:
        split_conv_cache = True

        def __init__(self):
            self.past_conv_caches = [(torch.zeros(1), torch.zeros(1), torch.zeros(1))]
            self.past_recurrent_states = [torch.zeros(1)]

    model = XHQwen3_5MoeHMONNXModel.__new__(XHQwen3_5MoeHMONNXModel)
    model._kvcache_mixin = DummyKVCacheMixin()
    model._llm_prefill = False
    model.meta_info = SimpleNamespace(
        spec_decode={"mode": "mtp", "num_draft_tokens": 1},
        model_config=SimpleNamespace(prefill_chunk_length=256),
    )

    logits, conv_cache_out_list, recurrent_state_out_list = model.forward(torch.tensor([1]))

    assert logits.tolist() == [42.0]
    assert conv_cache_out_list == [q1, k1, v1]
    assert recurrent_state_out_list == [r1]
    past_q, past_k, past_v = model._kvcache_mixin.past_conv_caches[0]
    assert torch.equal(past_q, q1)
    assert torch.equal(past_k, k1)
    assert torch.equal(past_v, v1)
    assert torch.equal(model._kvcache_mixin.past_recurrent_states[0], r1)


def test_qwen3_5_moe_hmonnx_forward_uses_final_spec_decode_merged_cache_step(monkeypatch):
    from types import SimpleNamespace

    import torch

    from xhmodel_merak.xh_llm.hmonnx.vision_llm_hmonnx_model import VisonLLMHMONNXModel
    from xhmodel_merak.xh_llm.models.qwen3_5_moe.qwen3_5_moe_hmonnx_inference import (
        XHQwen3_5MoeHMONNXModel,
    )

    c0, c1 = torch.full((1,), 1.0), torch.full((1,), 2.0)
    r0, r1 = torch.full((1,), 3.0), torch.full((1,), 4.0)

    def fake_forward(self, *args):
        return torch.tensor([42.0]), c0, c1, r0, r1

    monkeypatch.setattr(VisonLLMHMONNXModel, "forward", fake_forward)

    class DummyKVCacheMixin:
        split_conv_cache = False

        def __init__(self):
            self.past_conv_caches = [torch.zeros(1)]
            self.past_recurrent_states = [torch.zeros(1)]

    model = XHQwen3_5MoeHMONNXModel.__new__(XHQwen3_5MoeHMONNXModel)
    model._kvcache_mixin = DummyKVCacheMixin()
    model._llm_prefill = False
    model.meta_info = SimpleNamespace(
        spec_decode={"mode": "dflash", "num_draft_tokens": 1},
        model_config=SimpleNamespace(prefill_chunk_length=256),
    )

    logits, conv_cache_out_list, recurrent_state_out_list = model.forward(torch.tensor([1]))

    assert logits.tolist() == [42.0]
    assert conv_cache_out_list == [c1]
    assert recurrent_state_out_list == [r1]
    assert torch.equal(model._kvcache_mixin.past_conv_caches[0], c1)
    assert torch.equal(model._kvcache_mixin.past_recurrent_states[0], r1)


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


def test_qwen3_5_fused_recurrent_scan_avoids_proxy_shape_loop():
    from xhmodel_merak.xh_llm.models.qwen3_5 import _delta_rule

    source = inspect.getsource(_delta_rule.torch_recurrent_gated_delta_rule)
    fused_branch = source.split("if recurrent_scan_op is not None:", 1)[1].split("else:", 1)[0]

    assert "range(query.shape[2])" not in fused_branch
    assert "recurrent_scan_op(query, key, value, g, beta, last_recurrent_state)" in fused_branch


def test_qwen3_5_moe_gated_delta_net_wires_fused_gdr_ops():
    from xhmodel_merak.xh_llm.models.qwen3_5_moe import _moe_model

    source = inspect.getsource(_moe_model._Qwen3_5MoeGatedDeltaNet)

    assert 'self.fuse_gdr_ops = cfg.get("fuse_gdr_ops", False)' in source
    assert "self.block_tri_inverse_op = GDRBlockTriInverse" in source
    assert "self.chunk_scan_op = GDRChunkScan" in source
    assert "self.recurrent_scan_op = GDRRecurrentScan" in source
    assert "block_tri_inverse_op=self.block_tri_inverse_op" in source
    assert "chunk_scan_op=self.chunk_scan_op" in source
    assert "recurrent_scan_op=self.recurrent_scan_op" in source
    assert "self.chunk_scan_op.num_chunks = num_chunks" in source


def test_qwen3_5_moe_gated_delta_net_setup_creates_fused_gdr_ops():
    import torch
    import torch.nn as nn

    from xhquant.api import ConfigDict
    from xhmodel_merak.xh_llm.models.qwen3_5_moe import _moe_model

    module = _moe_model._Qwen3_5MoeGatedDeltaNet.__new__(
        _moe_model._Qwen3_5MoeGatedDeltaNet
    )
    nn.Module.__init__(module)
    module.hidden_size = 8
    module.key_dim = 4
    module.value_dim = 4
    module.conv_dim = 12
    module.conv_kernel_size = 4
    module.num_v_heads = 2
    module.num_k_heads = 2
    module.head_k_dim = 2
    module.head_v_dim = 2
    module.in_proj_qkv = nn.Linear(8, 12, bias=False)
    module.conv1d = nn.Conv1d(12, 12, kernel_size=4, groups=12, padding=3, bias=False)
    module.dt_bias = nn.Parameter(torch.zeros(2))
    module.A_log = nn.Parameter(torch.zeros(2))

    cfg = ConfigDict(
        dict(
            use_cache=True,
            linear_attention_mode="chunk",
            linear_chunk_size=8,
            input_sequence_length=16,
            batch_size=1,
            split_conv_cache=True,
            fuse_gdr_ops=True,
            use_manual_depthwise_conv1d=False,
        )
    )

    module._setup(cfg)

    assert module.fuse_gdr_ops is True
    assert module.block_tri_inverse_op is not None
    assert module.chunk_scan_op is not None
    assert module.recurrent_scan_op is not None
    assert module.chunk_scan_op.num_chunks == 2

    module._update_cfg(ConfigDict({**cfg, "input_sequence_length": 24}))
    assert module.chunk_scan_op.num_chunks == 3


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
