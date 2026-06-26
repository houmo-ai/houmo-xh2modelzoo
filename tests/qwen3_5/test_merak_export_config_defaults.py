"""Regression tests for Qwen3.5 Merak export config defaults."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn as nn

from xhmodel_merak.xh_llm import AutoLLMConfig
from xhmodel_merak.xh_llm.base_model import XHBaseModel
from xhmodel_merak.xh_llm.models.qwen3_5.xh_qwen3_5_config import (
    XHQwen3_5_DFlashConfig,
    XHQwen3_5_MTPConfig,
    XHQwen3_5ModelConfig,
    build_spec_draft_quant_scheme,
)
from xhmodel_merak.xh_llm.types import CacheList, KVCacheWithLinearConfig
from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig
from xhquant.api import Config


REPO_ROOT = Path(__file__).resolve().parents[2]
MERAK_EXPORT_SCRIPT = REPO_ROOT / "examples_merak/llm/qwen3_5/debug_scripts/qwen3_5_xh_export_hmonnx.py"
DFLASH_WORKFLOW_CONFIGS = [
    REPO_ROOT / "configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full_dflash.yaml",
    REPO_ROOT / "configs_merak/workflows/xh2a/llm_models/qwen3_5/27b/qwen3_6_27b_full_dflash.yaml",
    REPO_ROOT
    / "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_dflash.yaml",
]


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


@pytest.mark.parametrize("config_path", DFLASH_WORKFLOW_CONFIGS, ids=lambda p: p.stem)
def test_qwen3_5_dflash_workflow_resolves_target_model_dir_from_export_hf_model(config_path, tmp_path):
    workflow_config = WorkflowConfig.from_file(str(config_path))
    raw_dflash_cfg = workflow_config.export["model"]["dflash_config"]

    assert raw_dflash_cfg["target_model_dir"] is None

    override_hf_model = tmp_path / "override_qwen3_5_hf"
    override_hf_model.mkdir()
    export_cfg = workflow_config.build_export_dict()
    export_cfg["model"]["hf_model"] = str(override_hf_model)

    assert export_cfg["model"]["hf_model"] == str(override_hf_model)
    assert export_cfg["model"]["dflash_config"]["target_model_dir"] is None

    cfg = AutoLLMConfig.from_pretrained(Config(export_cfg).model)

    assert cfg.hf_model == str(override_hf_model)
    assert cfg.dflash_config.target_model_dir == str(override_hf_model)


@pytest.mark.parametrize("mode", ["mtp", "dflash"])
def test_qwen3_5_spec_decode_is_single_batch_only(mode):
    XHQwen3_5ModelConfig(model_name="qwen3_5", batch_size=1, spec_decode_mode=mode)

    with pytest.raises(ValueError, match="batch_size=1"):
        XHQwen3_5ModelConfig(model_name="qwen3_5", batch_size=2, spec_decode_mode=mode)


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
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_hmonnx_inference import (
        _flatten_split_conv_cache_outputs,
        _regroup_flat_split_conv_cache,
    )
    from xhmodel_merak.xh_llm.models.qwen3_5_moe import _moe_model

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
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_hmonnx_inference import (
        _flatten_split_conv_cache_outputs,
    )
    from xhmodel_merak.xh_llm.models.qwen3_5_moe import _moe_model

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
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_hmonnx_inference import (
        _regroup_flat_split_conv_cache,
    )
    from xhmodel_merak.xh_llm.models.qwen3_5_moe import _moe_model

    with pytest.raises(RuntimeError, match="divisible by 3"):
        _llm_model_impl._regroup_flat_split_conv_cache([0, 1])
    with pytest.raises(RuntimeError, match="divisible by 3"):
        _regroup_flat_split_conv_cache([0, 1])
    with pytest.raises(RuntimeError, match="divisible by 3"):
        _moe_model._regroup_flat_split_conv_cache([0, 1])


def test_qwen3_5_split_conv_cache_helpers_do_not_bool_test_proxy_like_inputs():
    from xhmodel_merak.xh_llm.models.qwen3_5 import _llm_model_impl
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_hmonnx_inference import (
        _regroup_flat_split_conv_cache,
    )
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model import (
        _regroup_flat_split_conv_cache as regroup_runtime_cache,
    )
    from xhmodel_merak.xh_llm.models.qwen3_5_moe import _moe_model

    class ProxyLike:
        def __bool__(self):
            raise AssertionError("proxy-like inputs must not be used in Python truth tests")

    proxy_like = ProxyLike()
    assert _llm_model_impl._regroup_flat_split_conv_cache(proxy_like) is proxy_like
    assert _regroup_flat_split_conv_cache(proxy_like) is proxy_like
    assert regroup_runtime_cache(proxy_like) is proxy_like
    assert _moe_model._regroup_flat_split_conv_cache(proxy_like) is proxy_like


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


def test_qwen3_5_dense_text_model_setup_splits_child_linear_attn_modules():
    from xhmodel_merak.xh_llm.models.qwen3_5 import _llm_model_impl

    source = inspect.getsource(_llm_model_impl._Qwen3_5TextModel._setup)

    assert "if self.split_conv_cache:" in source
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


def test_qwen3_5_moe_text_model_setup_splits_child_linear_attn_modules():
    from xhmodel_merak.xh_llm.models.qwen3_5_moe import _moe_model

    source = inspect.getsource(_moe_model._Qwen3_5MoeTextModel._setup)

    assert "if self.split_conv_cache:" in source
    assert 'self.layer_types[idx_layer] != "linear_attention"' in source
    assert "linear_attn.split_conv_cache = True" in source
    assert 'not hasattr(linear_attn, "in_proj_q")' in source
    assert "linear_attn._setup(cfg)" in source


def test_qwen3_5_wraped_post_reenforces_split_conv_cache_wrap_cfg():
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model import XHQwen3_5Model

    source = inspect.getsource(XHQwen3_5Model._wraped_post)

    assert "_enforce_split_conv_cache_wrap_cfg(llm_model, self.wrap_cfg)" in source


def test_qwen3_5_moe_wraped_post_reenforces_split_conv_cache_wrap_cfg():
    from xhmodel_merak.xh_llm.models.qwen3_5_moe.qwen3_5_moe_model import XHQwen3_5MoeModel

    source = inspect.getsource(XHQwen3_5MoeModel._wraped_post)

    assert "_enforce_split_conv_cache_wrap_cfg(language_model, self.wrap_cfg)" in source


def test_qwen3_5_moe_workflow_full_config_loads_expected_fields():
    config_path = (
        REPO_ROOT
        / "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b"
        / "qwen3_6_35b_a3b_full.yaml"
    )

    cfg = Config.fromfile(str(config_path))

    assert cfg.export.model.model_name == "qwen3_6_35b_a3b"
    assert "naming" not in cfg.export
    assert cfg.export.model.hf_model is None
    assert cfg.export.model.chip_arch == "XH2a"
    assert cfg.quant.algorithm == "gptqmodel"
    assert cfg.quant.artifact_format == "gptqmodel_hf"
    assert cfg.quant.group_size == 64
    assert cfg.quant.method == "autoround"
    assert cfg.quant.rotation is False
    assert cfg.quant.format == "auto_round:gptqmodel"
    assert cfg.export.model.quant_scheme.quant_type == "w8a8h1_sefp"
    assert cfg.export.model.quant_scheme.nodes.lm_head.quant_type == "w8a8h1_sefp"


def test_qwen3_5_moe_workflow_mtp_config_loads_expected_fields(monkeypatch):
    config_path = (
        REPO_ROOT
        / "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b"
        / "qwen3_6_35b_a3b_full_mtp.yaml"
    )

    workflow_cfg = WorkflowConfig.from_file(str(config_path))
    cfg = Config(workflow_cfg.build_export_dict())
    cfg.model.hf_model = "weights/Qwen3.6-35B-A3B"
    monkeypatch.setattr(Path, "exists", lambda self: True)
    model_cfg = AutoLLMConfig.from_pretrained(cfg.model)

    assert model_cfg.model_name == "qwen3_6_35b_a3b"
    assert model_cfg.chip_arch == "XH2a"
    assert model_cfg.spec_decode_mode == "mtp"
    assert model_cfg.quant_scheme.quant_type == "w8a8h1_sefp"
    assert model_cfg.mtp_config.hidden_size == 2048
    assert model_cfg.mtp_config.input_sequence_length == 1
    assert model_cfg.mtp_config.use_cache is True


def test_qwen3_5_moe_gated_delta_net_sets_up_split_qkv_cache_path():
    from xhmodel_merak.xh_llm.models.qwen3_5_moe import _moe_model

    source = inspect.getsource(_moe_model._Qwen3_5MoeGatedDeltaNet)

    assert 'self.split_conv_cache = cfg.get("split_conv_cache", True)' in source
    assert "self.in_proj_q = nn.Linear" in source
    assert "self.conv1d_q = nn.Conv1d" in source
    assert "del self.in_proj_qkv" in source
    assert "del self.conv1d" in source


def test_qwen3_5_moe_gated_delta_net_wires_fused_gdr_ops():
    from xhmodel_merak.xh_llm.models.qwen3_5_moe import _moe_model

    source = inspect.getsource(_moe_model._Qwen3_5MoeGatedDeltaNet)

    assert 'self.fuse_gdr_ops = cfg.get("fuse_gdr_ops", False)' in source
    assert 'self.fuse_gdr_block_recurrent_ops = cfg.get("fuse_gdr_block_recurrent_ops", False)' in source
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

    from xhmodel_merak.xh_llm.models.qwen3_5_moe import _moe_model
    from xhquant.api import ConfigDict

    module = _moe_model._Qwen3_5MoeGatedDeltaNet.__new__(_moe_model._Qwen3_5MoeGatedDeltaNet)
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
            fuse_gdr_block_recurrent_ops=False,
            use_manual_depthwise_conv1d=False,
        )
    )

    module._setup(cfg)

    assert module.fuse_gdr_ops is True
    assert module.fuse_gdr_block_recurrent_ops is False
    assert module.block_tri_inverse_op is None
    assert module.chunk_scan_op is not None
    assert module.recurrent_scan_op is None
    assert module.chunk_scan_op.num_chunks == 2

    module._update_cfg(ConfigDict({**cfg, "input_sequence_length": 24}))
    assert module.chunk_scan_op.num_chunks == 3

    module_all = _moe_model._Qwen3_5MoeGatedDeltaNet.__new__(_moe_model._Qwen3_5MoeGatedDeltaNet)
    nn.Module.__init__(module_all)
    module_all.hidden_size = module.hidden_size
    module_all.key_dim = module.key_dim
    module_all.value_dim = module.value_dim
    module_all.conv_dim = module.conv_dim
    module_all.conv_kernel_size = module.conv_kernel_size
    module_all.num_v_heads = module.num_v_heads
    module_all.num_k_heads = module.num_k_heads
    module_all.head_k_dim = module.head_k_dim
    module_all.head_v_dim = module.head_v_dim
    module_all.in_proj_qkv = nn.Linear(8, 12, bias=False)
    module_all.conv1d = nn.Conv1d(12, 12, kernel_size=4, groups=12, padding=3, bias=False)
    module_all.dt_bias = nn.Parameter(torch.zeros(2))
    module_all.A_log = nn.Parameter(torch.zeros(2))

    module_all._setup(ConfigDict({**cfg, "fuse_gdr_block_recurrent_ops": True}))

    assert module_all.fuse_gdr_ops is True
    assert module_all.fuse_gdr_block_recurrent_ops is True
    assert module_all.block_tri_inverse_op is not None
    assert module_all.chunk_scan_op is not None
    assert module_all.recurrent_scan_op is not None


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
    monkeypatch.setattr(
        XHBaseModel,
        "_load_hf_model",
        classmethod(lambda cls, hf_model_dir, **kwargs: calls.append(("hf", hf_model_dir, kwargs)) or object()),
    )
    monkeypatch.setattr(
        XHBaseModel,
        "_load_quant_weight",
        classmethod(
            lambda cls, quant_weight, native_hf_model, strict=True: calls.append(("quant_weight", quant_weight)) or True
        ),
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


def test_get_native_model_uses_cpu_device_map_when_auto_offload_disabled(monkeypatch):
    class DummyHFModel:
        pass

    class DummyModel(XHBaseModel):
        HF_MODEL_CLS = DummyHFModel

    model = DummyModel.__new__(DummyModel)
    model.config = type("Config", (), {"quant_weight": None, "enable_auto_offload": False})()
    model.hf_model_dir = "weights/Qwen3.5-9B"

    captured = {}

    monkeypatch.setattr(
        DummyModel,
        "get_hf_model",
        classmethod(lambda cls, hf_model_dir, quant_weight=None, **kwargs: captured.update(kwargs) or DummyHFModel()),
    )

    out = model.get_native_model()

    assert isinstance(out, DummyHFModel)
    assert captured["device_map"] == "cpu"


def test_get_native_model_uses_auto_device_map_when_auto_offload_enabled(monkeypatch):
    class DummyHFModel:
        pass

    class DummyModel(XHBaseModel):
        HF_MODEL_CLS = DummyHFModel

    model = DummyModel.__new__(DummyModel)
    model.config = type("Config", (), {"quant_weight": "/tmp/quant_weight.pt", "enable_auto_offload": True})()
    model.hf_model_dir = "weights/Qwen3.5-9B"

    captured = {}

    def fake_get_hf_model(cls, hf_model_dir, quant_weight=None, **kwargs):
        captured["hf_model_dir"] = hf_model_dir
        captured["quant_weight"] = quant_weight
        captured.update(kwargs)
        return DummyHFModel()

    monkeypatch.setattr(DummyModel, "get_hf_model", classmethod(fake_get_hf_model))

    out = model.get_native_model()

    assert isinstance(out, DummyHFModel)
    assert captured["hf_model_dir"] == "weights/Qwen3.5-9B"
    assert captured["quant_weight"] == "/tmp/quant_weight.pt"
    assert captured["device_map"] == "auto"


def test_cache_list_to_supports_dtype_only_conversion():
    cache_list = CacheList([torch.ones((1, 2), dtype=torch.float32)])

    returned = cache_list.to(torch.float16)

    assert returned is cache_list
    assert cache_list[0].dtype == torch.float16


def test_cache_list_to_supports_combined_device_and_dtype_conversion():
    cache_list = CacheList([torch.ones((1, 2), dtype=torch.float32)])

    cache_list.to(device="cpu", dtype=torch.float16)

    assert cache_list[0].device.type == "cpu"
    assert cache_list[0].dtype == torch.float16


def test_cache_list_to_supports_keyword_dtype_conversion():
    cache_list = CacheList([torch.ones((1, 2), dtype=torch.float32)])

    cache_list.to(dtype=torch.bfloat16)

    assert cache_list[0].dtype == torch.bfloat16


def test_dequantize_gptqmodel_hf_model_moves_modules_to_cpu_and_back(monkeypatch):
    class FakePackableQuantLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.move_calls = []
            self.fake_device = torch.device("cuda", 0)

        def to(self, *args, **kwargs):
            device = kwargs.get("device")
            if device is None and args:
                device = args[0]
            if device is not None:
                self.move_calls.append(torch.device(device))
            return self

    fake_gptqmodel_module = ModuleType("gptqmodel")
    fake_qlinear_module = ModuleType("gptqmodel.nn_modules.qlinear")
    fake_qlinear_module.PackableQuantLinear = FakePackableQuantLinear
    monkeypatch.setitem(sys.modules, "gptqmodel", fake_gptqmodel_module)
    monkeypatch.setitem(sys.modules, "gptqmodel.nn_modules.qlinear", fake_qlinear_module)
    monkeypatch.setattr("transformers.utils.is_gptqmodel_available", lambda: True)

    converted = []

    def fake_converter(module):
        converted.append(module)

    monkeypatch.setattr("xhmodel_merak.xh_llm.base_model.gptqmodel_torch_qlinear_converter", fake_converter)

    module = FakePackableQuantLinear()
    model = nn.Sequential(module)

    original_parameters = FakePackableQuantLinear.parameters
    original_buffers = FakePackableQuantLinear.buffers

    def fake_parameters(self, recurse=True):
        if self is module:
            return iter((torch.empty(1, device=module.fake_device),))
        return original_parameters(self, recurse=recurse)

    def fake_buffers(self, recurse=True):
        if self is module:
            return iter(())
        return original_buffers(self, recurse=recurse)

    monkeypatch.setattr(FakePackableQuantLinear, "parameters", fake_parameters)
    monkeypatch.setattr(FakePackableQuantLinear, "buffers", fake_buffers)

    out = XHBaseModel._dequantize_gptqmodel_hf_model(model)

    assert out is model
    assert converted == [module]
    assert module.move_calls == [torch.device("cpu"), torch.device("cuda", 0)]


def test_dequantize_gptqmodel_hf_model_keeps_cpu_modules_on_cpu(monkeypatch):
    class FakePackableQuantLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.move_calls = []

        def to(self, *args, **kwargs):
            device = kwargs.get("device")
            if device is None and args:
                device = args[0]
            if device is not None:
                self.move_calls.append(torch.device(device))
            return self

    fake_gptqmodel_module = ModuleType("gptqmodel")
    fake_qlinear_module = ModuleType("gptqmodel.nn_modules.qlinear")
    fake_qlinear_module.PackableQuantLinear = FakePackableQuantLinear
    monkeypatch.setitem(sys.modules, "gptqmodel", fake_gptqmodel_module)
    monkeypatch.setitem(sys.modules, "gptqmodel.nn_modules.qlinear", fake_qlinear_module)
    monkeypatch.setattr("transformers.utils.is_gptqmodel_available", lambda: True)

    converted = []
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.base_model.gptqmodel_torch_qlinear_converter",
        lambda module: converted.append(module),
    )

    module = FakePackableQuantLinear()
    model = nn.Sequential(module)

    out = XHBaseModel._dequantize_gptqmodel_hf_model(model)

    assert out is model
    assert converted == [module]
    assert module.move_calls == []


def test_dequantize_gptqmodel_hf_model_supports_empty_modules(monkeypatch):
    class FakePackableQuantLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.move_calls = []

        def to(self, *args, **kwargs):
            device = kwargs.get("device")
            if device is None and args:
                device = args[0]
            if device is not None:
                self.move_calls.append(torch.device(device))
            return self

    fake_gptqmodel_module = ModuleType("gptqmodel")
    fake_qlinear_module = ModuleType("gptqmodel.nn_modules.qlinear")
    fake_qlinear_module.PackableQuantLinear = FakePackableQuantLinear
    monkeypatch.setitem(sys.modules, "gptqmodel", fake_gptqmodel_module)
    monkeypatch.setitem(sys.modules, "gptqmodel.nn_modules.qlinear", fake_qlinear_module)
    monkeypatch.setattr("transformers.utils.is_gptqmodel_available", lambda: True)

    converted = []
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.base_model.gptqmodel_torch_qlinear_converter",
        lambda module: converted.append(module),
    )

    module = FakePackableQuantLinear()
    model = nn.Sequential(module)

    monkeypatch.setattr(FakePackableQuantLinear, "parameters", lambda self, recurse=True: iter(()))
    monkeypatch.setattr(FakePackableQuantLinear, "buffers", lambda self, recurse=True: iter(()))

    out = XHBaseModel._dequantize_gptqmodel_hf_model(model)

    assert out is model
    assert converted == [module]
    assert module.move_calls == []


def test_dequantize_gptqmodel_hf_model_processes_multiple_modules(monkeypatch):
    class FakePackableQuantLinear(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name
            self.weight = nn.Parameter(torch.ones(1))

    fake_gptqmodel_module = ModuleType("gptqmodel")
    fake_qlinear_module = ModuleType("gptqmodel.nn_modules.qlinear")
    fake_qlinear_module.PackableQuantLinear = FakePackableQuantLinear
    monkeypatch.setitem(sys.modules, "gptqmodel", fake_gptqmodel_module)
    monkeypatch.setitem(sys.modules, "gptqmodel.nn_modules.qlinear", fake_qlinear_module)
    monkeypatch.setattr("transformers.utils.is_gptqmodel_available", lambda: True)

    converted = []
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.base_model.gptqmodel_torch_qlinear_converter",
        lambda module: converted.append(module.name),
    )

    model = nn.Sequential(FakePackableQuantLinear("first"), FakePackableQuantLinear("second"))

    XHBaseModel._dequantize_gptqmodel_hf_model(model)

    assert converted == ["first", "second"]


def test_dequantize_gptqmodel_hf_model_rejects_multi_device_modules(monkeypatch):
    class FakePackableQuantLinear(nn.Module):
        pass

    fake_gptqmodel_module = ModuleType("gptqmodel")
    fake_qlinear_module = ModuleType("gptqmodel.nn_modules.qlinear")
    fake_qlinear_module.PackableQuantLinear = FakePackableQuantLinear
    monkeypatch.setitem(sys.modules, "gptqmodel", fake_gptqmodel_module)
    monkeypatch.setitem(sys.modules, "gptqmodel.nn_modules.qlinear", fake_qlinear_module)
    monkeypatch.setattr("transformers.utils.is_gptqmodel_available", lambda: True)
    monkeypatch.setattr("xhmodel_merak.xh_llm.base_model.gptqmodel_torch_qlinear_converter", lambda module: None)

    module = FakePackableQuantLinear()
    model = nn.Sequential(module)

    monkeypatch.setattr(
        FakePackableQuantLinear,
        "parameters",
        lambda self, recurse=True: iter((torch.empty(1, device="cpu"), torch.empty(1, device="meta"))),
    )
    monkeypatch.setattr(FakePackableQuantLinear, "buffers", lambda self, recurse=True: iter(()))

    with pytest.raises(NotImplementedError, match="single device"):
        XHBaseModel._dequantize_gptqmodel_hf_model(model)
