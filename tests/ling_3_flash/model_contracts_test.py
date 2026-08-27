from __future__ import annotations

import operator
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import xhmodel_merak.xh_llm.models.ling_3_flash._ling_3_flash_big_export as ling_big_export
import xhmodel_merak.xh_llm.models.ling_3_flash._model as ling_wrappers
import xhmodel_merak.xh_llm.models.ling_3_flash.ling_3_flash_model as ling_model
from xhmodel_merak.xh_llm.kv_cache_mixin import KVCacheContextManager
from xhmodel_merak.xh_llm.models.ling_3_flash._model import (
    _LingMLAAttention,
    _LingSparseMoeBlock,
    _LingTextModel,
)
from xhmodel_merak.xh_llm.models.ling_3_flash.ling_3_flash_hmonnx_inference import (
    XHLing3FlashHMONNXModel,
)
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_hmonnx_inference import (
    Qwen3_5HMONNXKVCacheMixin,
)
from xhmodel_merak.xh_llm.models.qwen3_5.split_conv_cache_utils import (
    _select_linear_attn_conv_cache,
)
from xhmodel_merak.xh_llm.types import KVCacheWithLinearConfig
from xhquant.core import CacheTensor


class _Cfg(dict):
    __getattr__ = dict.__getitem__


def _build_elementwise_mla(
    *,
    flash_attention: bool = False,
) -> tuple[_LingMLAAttention, torch.Tensor]:
    torch.manual_seed(20260811)
    module = object.__new__(_LingMLAAttention)
    nn.Module.__init__(module)
    module.config = SimpleNamespace(hidden_size=12)
    module.hidden_size = 12
    module.num_heads = 2
    module.qk_nope_head_dim = 3
    module.v_head_dim = 4
    module.kv_lora_rank = 5
    module.qk_rope_head_dim = 2
    module.q_lora_rank = None
    module.scaling = 0.25
    module.q_proj = nn.Linear(12, 2 * (3 + 2), bias=False)
    module.kv_b_proj = nn.Linear(5, 2 * (3 + 4), bias=False)
    module.dense = nn.Linear(2 * 4, 12, bias=False)
    module.g_proj = nn.Linear(12, 2 * 4, bias=False)
    module.gated_attention_proj_granularity_type = "element_wise"
    value_weight = (
        module.kv_b_proj.weight.detach().clone().view(2, 3 + 4, 5)[:, 3:]
    )
    module._setup(
        _Cfg(
            use_cache=False,
            batch_size=1,
            input_sequence_length=3,
            enable_rope=False,
            flash_attention={"enable": flash_attention},
        )
    )
    return module, value_weight


class _LingLayerStub(nn.Module):
    def __init__(self, attention_layer_type: str):
        super().__init__()
        self.attention_layer_type = attention_layer_type


class _LingRotaryStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("cos_cached", torch.ones(1, 1, 4, 2))
        self.register_buffer("sin_cached", torch.zeros(1, 1, 4, 2))


def _build_ling_text_model(*, enable_layer_tag=None):
    module = object.__new__(_LingTextModel)
    nn.Module.__init__(module)
    module.layers = nn.ModuleList(
        [_LingLayerStub("linear_attention"), _LingLayerStub("attention")]
    )
    module.rotary_emb = _LingRotaryStub()
    cfg = _Cfg(
        batch_size=1,
        input_sequence_length=1,
        use_cache=True,
        num_logits_to_keep=1,
        max_layers=None,
    )
    if enable_layer_tag is not None:
        cfg["enable_layer_tag"] = enable_layer_tag
    module._setup(cfg)
    return module


def test_ling_text_model_emits_explicit_page_attention_layer_tags_by_default():
    module = _build_ling_text_model()

    assert module.enable_layer_tag is True
    assert [tag.tag_name for tag in module.tags] == ["layer_0", "layer_1"]
    assert [tag.tag_type for tag in module.tags] == ["LLM", "LLM"]


def test_ling_text_model_allows_layer_tags_to_be_explicitly_disabled():
    module = _build_ling_text_model(enable_layer_tag=False)

    assert module.enable_layer_tag is False
    assert not hasattr(module, "tags")


def test_elementwise_mla_reconstructs_value_before_gate():
    module, value_weight = _build_elementwise_mla()
    latent_context = torch.randn(1, 3, module.num_heads, module.kv_lora_rank)
    hidden_states = torch.randn(1, 3, module.hidden_size)

    projected = module.value_up_conv(
        latent_context.permute(0, 2, 3, 1).reshape(
            1,
            module.num_heads * module.kv_lora_rank,
            3,
        )
    ).view(1, module.num_heads, module.v_head_dim, 3).permute(0, 3, 1, 2)
    gate = torch.sigmoid(module.g_proj(hidden_states)).view(
        1,
        3,
        module.num_heads,
        module.v_head_dim,
    )
    actual = module.dense((projected * gate).reshape(1, 3, -1))

    reference_value = torch.einsum(
        "bshc,hvc->bshv",
        latent_context,
        value_weight,
    )
    expected = module.dense((reference_value * gate).reshape(1, 3, -1))

    torch.testing.assert_close(actual, expected)
    assert module.value_up_conv.groups == module.num_heads
    assert not hasattr(module, "kv_b_proj")


def test_flash_mla_uses_the_same_absorbed_latent_representation_as_explicit_attention():
    module, _ = _build_elementwise_mla(flash_attention=True)

    assert module.flash_attn.num_heads == module.num_heads
    assert module.flash_attn.num_kv_heads == 1
    assert module.q_absorbed_proj.out_features == module.num_heads * module.kv_lora_rank
    assert module.q_rope_proj.out_features == module.num_heads * module.qk_rope_head_dim
    assert module.value_up_conv.in_channels == module.num_heads * module.kv_lora_rank
    assert not hasattr(module, "flash_head_dim")
    assert not hasattr(module, "flash_qk_padding")
    assert not hasattr(module, "flash_v_padding")
    assert not hasattr(module, "q_flash_proj")
    assert not hasattr(module, "kv_flash_proj")
    assert hasattr(module, "dense")
    assert not hasattr(module, "absorbed_dense")


def test_split_conv_cache_accepts_nested_and_flat_runtime_abi():
    caches = [torch.randn(1, 4, 3) for _ in range(6)]
    nested = [tuple(caches[:3]), tuple(caches[3:])]

    flat_selected = _select_linear_attn_conv_cache(caches, 1, True)
    nested_selected = _select_linear_attn_conv_cache(nested, 1, True)
    assert all(actual is expected for actual, expected in zip(flat_selected, caches[3:], strict=True))
    assert nested_selected is nested[1]


def test_hmonnx_runtime_allocates_asymmetric_absorbed_mla_page_cache():
    runtime = XHLing3FlashHMONNXModel.__new__(XHLing3FlashHMONNXModel)
    runtime.enable_page_attention = True
    runtime.kvcache_config = KVCacheWithLinearConfig(
        num_layers=1,
        kv_cache_shape=[[1, 1, 256, 576], [1, 1, 256, 512]],
        linear_kv_cache_config={
            "conv_dim": 8,
            "conv_kernel_size": 4,
            "num_v_heads": 1,
            "head_k_dim": 4,
            "head_v_dim": 4,
            "num_layers": 0,
            "batch_size": 1,
        },
    )
    runtime._device = torch.device("cpu")
    runtime._page_attention_block_size = 64
    runtime._paged_kv_caches = None
    runtime.get_input_sequence_length = lambda: 4
    captured = {}
    runtime.set_page_attention_context = lambda *args: captured.setdefault("args", args)

    runtime.prepare_page_attention_context(past_seq_length=5, current_input_length=2)

    cache = runtime._paged_kv_caches[0]
    assert cache.head_size == 576
    assert cache.head_size_v == 512
    assert tuple(cache.k_qdata.shape) == (4, 1, 9, 64, 64)
    assert tuple(cache.v_qdata.shape) == (4, 1, 1, 64, 512)
    assert captured["args"][1].tolist() == [0, 1, 2, 3]
    assert captured["args"][2].tolist() == [5, 6, -1, -1]


def test_hybrid_cache_scope_manages_gdr_state_with_asymmetric_kv_cache():
    config = KVCacheWithLinearConfig(
        num_layers=1,
        kv_cache_shape=[[1, 1, 8, 5], [1, 1, 8, 3]],
        linear_kv_cache_config={
            "conv_dim": 14,
            "conv_kernel_size": 4,
            "num_v_heads": 2,
            "head_k_dim": 3,
            "head_v_dim": 1,
            "num_layers": 1,
            "batch_size": 1,
        },
    )
    cache = Qwen3_5HMONNXKVCacheMixin(config)
    cache.split_conv_cache = True
    runtime = SimpleNamespace(get_kvcache_mixin=lambda: cache)

    with KVCacheContextManager(runtime):
        assert tuple(cache.past_key_caches[0].shape) == (1, 1, 8, 5)
        assert tuple(cache.past_value_caches[0].shape) == (1, 1, 8, 3)
        assert cache.past_key_caches[0] is not cache.past_value_caches[0]
        assert [tuple(item.shape) for item in cache.past_conv_caches[0]] == [
            (1, 6, 4),
            (1, 6, 4),
            (1, 2, 4),
        ]
        recurrent = cache.past_recurrent_states[0]
        assert isinstance(recurrent, CacheTensor)
        recurrent.fill_(1)

    assert not cache.past_key_caches
    assert not cache.past_value_caches
    assert not cache.past_conv_caches
    assert not cache.past_recurrent_states

    with KVCacheContextManager(runtime):
        assert torch.count_nonzero(cache.past_recurrent_states[0]) == 0


@pytest.mark.parametrize(
    "source_dtype",
    [torch.float16, torch.bfloat16, torch.float32],
)
def test_moe_group_mask_stays_finite_in_fp16_export_dtype(source_dtype):
    module = object.__new__(_LingSparseMoeBlock)
    nn.Module.__init__(module)
    module.register_buffer(
        "router_mask_floor",
        torch.tensor(-32768.0, dtype=source_dtype),
        persistent=False,
    )
    choice = torch.tensor([[[0.9, 0.8, 0.7, 0.6]]], dtype=source_dtype)
    expert_mask = torch.tensor(
        [[[1.0, 1.0, 0.0, 0.0]]],
        dtype=source_dtype,
    )

    masked = module._mask_excluded_experts(choice, expert_mask)
    exported = masked.to(torch.float16)

    assert torch.isfinite(masked).all()
    assert torch.isfinite(exported).all()
    assert torch.equal(masked[..., :2], choice[..., :2])
    assert torch.equal(
        masked[..., 2:],
        module.router_mask_floor.expand_as(masked[..., 2:]),
    )
    assert torch.topk(masked, 2, dim=-1).indices.tolist() == [[[0, 1]]]


def test_moe_group_mask_exports_tensor_minus_one_not_reflected_subtraction():
    class MaskModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("router_mask_floor", torch.tensor(-32768.0))

        def forward(self, choice, expert_mask):
            return _LingSparseMoeBlock._mask_excluded_experts(
                self,
                choice,
                expert_mask,
            )

    graph = torch.fx.symbolic_trace(MaskModule()).graph
    subtraction = [
        node
        for node in graph.nodes
        if node.op == "call_function" and node.target is operator.sub
    ]

    assert len(subtraction) == 1
    assert isinstance(subtraction[0].args[0], torch.fx.Node)
    assert subtraction[0].args[1] == 1.0


def _ling_group_limited_topk(choice: torch.Tensor) -> torch.Tensor:
    grouped = choice.view(-1, 8, 64)
    group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
    group_indices = group_scores.topk(4, dim=-1).indices
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_indices, 1)
    expert_mask = group_mask.unsqueeze(-1).expand_as(grouped).reshape_as(choice)
    return choice.masked_fill(~expert_mask.bool(), -32768.0).topk(8, dim=-1).indices


def test_moe_group_scatter_updates_match_topk_indices_shape():
    group_scores = torch.randn(256, 8, dtype=torch.float16)

    selected_group_scores, group_indices = torch.topk(group_scores, 4, dim=-1)
    group_updates = selected_group_scores * 0 + 1

    assert group_indices.shape == group_updates.shape == (256, 4)
    assert group_updates.dtype == group_scores.dtype
    group_mask = torch.zeros_like(group_scores).scatter(
        1,
        group_indices,
        group_updates,
    )
    assert torch.equal(group_mask.sum(dim=-1), torch.full((256,), 4.0, dtype=torch.float16))


def test_moe_selection_score_shift_is_fp32_equivalent():
    module = object.__new__(_LingSparseMoeBlock)
    nn.Module.__init__(module)
    torch.manual_seed(20260812)
    module.register_buffer("expert_bias", torch.randn(512) * 0.02, persistent=False)
    logits = torch.randn(7, 512)

    shifted = module._selection_scores(logits)
    reference = torch.sigmoid(logits) + module.expert_bias - 1.0

    torch.testing.assert_close(shifted, reference, rtol=1e-6, atol=1e-7)
    assert torch.equal(
        _ling_group_limited_topk(shifted),
        _ling_group_limited_topk(reference),
    )


def test_moe_shifted_selection_scores_preserve_fp16_topk_precision():
    module = object.__new__(_LingSparseMoeBlock)
    nn.Module.__init__(module)
    torch.manual_seed(0)
    logits = torch.randn(256, 512) * 0.5 + 3.0
    expert_bias = (torch.randn(512) * 0.02).to(torch.float16)
    module.register_buffer("expert_bias", expert_bias, persistent=False)

    reference = _ling_group_limited_topk(
        torch.sigmoid(logits) + expert_bias.float()
    ).sort(dim=-1).values
    naive = _ling_group_limited_topk(
        torch.sigmoid(logits.half()) + expert_bias
    ).sort(dim=-1).values
    shifted = _ling_group_limited_topk(
        module._selection_scores(logits.half())
    ).sort(dim=-1).values

    naive_exact = (naive == reference).all(dim=-1).sum()
    shifted_exact = (shifted == reference).all(dim=-1).sum()
    assert shifted_exact >= 250
    assert shifted_exact >= naive_exact + 20


def test_gptq_postprocess_restores_ling_short_convolution_class():
    class ShortConvolution(nn.Conv1d):
        pass

    original = ShortConvolution(2, 2, 1)
    safe_cls = type(
        "GPTQModelShortConvolution",
        (nn.Module,),
        {"_gptqmodel_ling_original_class": ShortConvolution},
    )
    original.__class__ = safe_cls
    holder = nn.Module()
    holder.short_conv = original

    result = ling_model.XHLing3FlashModel._postprocess_gptqmodel_structure(holder)

    assert result is holder
    assert type(holder.short_conv) is ShortConvolution
    assert isinstance(holder.short_conv, nn.Conv1d)
    assert list(holder.state_dict()) == ["short_conv.weight", "short_conv.bias"]


def test_gptq_lifecycle_trims_mtp_layer_before_generation_and_export():
    language_model = nn.Module()
    language_model.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])
    language_model.config = SimpleNamespace(num_hidden_layers=2)
    language_model.num_nextn_predict_layers = 1
    holder = nn.Module()
    holder.model = language_model
    holder.num_nextn_predict_layers = 1

    result = ling_model.XHLing3FlashModel._trim_mtp_layers(holder)

    assert result is holder
    assert len(holder.model.layers) == 2
    assert holder.model.num_nextn_predict_layers == 0
    assert holder.num_nextn_predict_layers == 0


def test_gptq_lifecycle_mtp_trim_is_idempotent_for_causal_checkpoint():
    language_model = nn.Module()
    original_layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(2)])
    language_model.layers = original_layers
    language_model.config = SimpleNamespace(num_hidden_layers=2)
    holder = nn.Module()
    holder.model = language_model

    result = ling_model.XHLing3FlashModel._trim_mtp_layers(holder)

    assert result is holder
    assert holder.model.layers is original_layers


def test_low_memory_worker_refreshes_exact_remote_code_classes(monkeypatch):
    model_after_load = nn.Module()
    model_after_quant = nn.Module()
    registered = []
    monkeypatch.setattr(
        ling_big_export,
        "patch_ling_remote_code_compatibility",
        lambda: None,
    )
    monkeypatch.setattr(
        ling_wrappers,
        "register_wrap_modules",
        registered.append,
    )

    ling_big_export.Ling3FlashBigHFModel.initialize_process_worker_after_model_load(
        model_after_load
    )
    ling_big_export.Ling3FlashBigHFModel.initialize_process_worker_after_quantized_preprocess(
        model_after_quant
    )

    assert registered == [model_after_load, model_after_quant]


def test_export_metadata_copies_remote_code_and_preserves_pad_id(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source"
    hf_config = tmp_path / "export" / "hf_config"
    source.mkdir()
    hf_config.mkdir(parents=True)
    (source / "configuration_bailing_moe_v3.py").write_text(
        "class Config: pass\n",
        encoding="utf-8",
    )
    (source / "modeling_bailing_moe_v3.py").write_text(
        "class Model: pass\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        ling_model,
        "load_ling_config",
        lambda _: SimpleNamespace(pad_token_id=156892, eos_token_id=156895),
    )
    model = object.__new__(ling_model.XHLing3FlashModel)
    model.config = SimpleNamespace(hf_model=str(source))
    metadata = SimpleNamespace(hf_config="hf_config", pad_token_id=None)

    result = model._extra_export_metadata(str(hf_config.parent), metadata)

    assert result.pad_token_id == 156892
    assert (hf_config / "configuration_bailing_moe_v3.py").is_file()
    assert (hf_config / "modeling_bailing_moe_v3.py").is_file()
