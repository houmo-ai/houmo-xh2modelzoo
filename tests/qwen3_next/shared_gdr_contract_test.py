from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
NEXT_MODEL_IMPL = REPO_ROOT / "xhmodel_merak/xh_llm/models/qwen3_next/_model.py"
NEXT_MTP_WORKFLOW = (
    REPO_ROOT / "configs_merak/workflows/xh2a/llm_models/qwen3_next/80b_a3b" / "qwen3_next_80b_a3b_full_mtp_gptq.yaml"
)


def _new_small_next_gdn():
    """Build the smallest packed-projection Next GDN accepted by ``_setup``."""
    from xhmodel_merak.xh_llm.models.qwen3_next import _model

    module = _model._Qwen3NextGatedDeltaNet.__new__(_model._Qwen3NextGatedDeltaNet)
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
    module.in_proj_qkvz = nn.Linear(8, 16, bias=False)
    module.in_proj_ba = nn.Linear(8, 4, bias=False)
    module.conv1d = nn.Conv1d(
        12,
        12,
        kernel_size=4,
        groups=12,
        padding=3,
        bias=False,
    )
    module.out_proj = nn.Linear(4, 8, bias=False)
    module.dt_bias = nn.Parameter(torch.zeros(2))
    module.A_log = nn.Parameter(torch.zeros(2))

    class IdentityGatedNorm(nn.Module):
        def forward(self, hidden_states, gate):
            del gate
            return hidden_states

    module.norm = IdentityGatedNorm()
    return module


def _gdn_cfg(**overrides):
    from xhquant.api import ConfigDict

    values = dict(
        use_cache=True,
        linear_attention_mode="chunk",
        linear_chunk_size=8,
        input_sequence_length=16,
        batch_size=1,
        split_conv_cache=True,
        normalize_force_fp32=False,
        use_manual_depthwise_conv1d=False,
        fuse_gdr_ops=True,
        fuse_gdr_block_recurrent_ops=True,
        suppress_recurrent_state_outputs=True,
    )
    values.update(overrides)
    return ConfigDict(values)


def _forward_small_gdn(module, *, sequence_length: int):
    hidden_states = torch.randn(1, sequence_length, 8)
    conv_caches = (
        torch.zeros(1, 4, 4),
        torch.zeros(1, 4, 4),
        torch.zeros(1, 4, 4),
    )
    recurrent_state = torch.zeros(1, 2, 2, 2)
    return module(
        hidden_states,
        conv_caches,
        recurrent_state,
        torch.ones(1, sequence_length),
        torch.tensor([sequence_length]),
    )


def test_qwen3_next_uses_the_canonical_hybrid_gdn_instead_of_local_delta_rule_copies():
    tree = ast.parse(NEXT_MODEL_IMPL.read_text(), filename=str(NEXT_MODEL_IMPL))
    forbidden = {
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
    }
    locally_defined = {
        node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert not (forbidden & locally_defined), (
        "Qwen3-Next must not fork the canonical Qwen3.5 delta-rule functions; "
        f"local copies remain: {sorted(forbidden & locally_defined)}"
    )

    from xhmodel_merak.xh_llm.models.qwen3_5._hybrid_gated_delta_net import (
        HybridGatedDeltaNetMixin,
    )
    from xhmodel_merak.xh_llm.models.qwen3_next import _model

    next_gdn = _model._Qwen3NextGatedDeltaNet
    assert issubclass(next_gdn, HybridGatedDeltaNetMixin)
    assert next_gdn.forward is HybridGatedDeltaNetMixin.forward
    assert next_gdn._setup is HybridGatedDeltaNetMixin._setup
    assert next_gdn._update_cfg is HybridGatedDeltaNetMixin._update_cfg


def test_dense_moe_and_next_text_models_share_the_canonical_cache_loop():
    from xhmodel_merak.xh_llm.models.qwen3_5._hybrid_text_model import (
        HybridTextModelMixin,
    )
    from xhmodel_merak.xh_llm.models.qwen3_5._llm_model_impl import (
        _Qwen3_5TextModel,
    )
    from xhmodel_merak.xh_llm.models.qwen3_5_moe._moe_model import (
        _Qwen3_5MoeTextModel,
    )
    from xhmodel_merak.xh_llm.models.qwen3_next._model import _Qwen3NextModel

    for model_cls in (
        _Qwen3_5TextModel,
        _Qwen3_5MoeTextModel,
        _Qwen3NextModel,
    ):
        assert issubclass(model_cls, HybridTextModelMixin)
        assert model_cls.forward is HybridTextModelMixin.forward
        assert model_cls._setup is HybridTextModelMixin._setup


def test_hybrid_hmonnx_args_are_recursively_flattened_before_dtype_normalization():
    from xhmodel_merak.xh_llm.models.qwen3_5.hybrid_cache_runtime import (
        normalize_hybrid_hmonnx_args,
    )

    int64_input = torch.tensor([1], dtype=torch.int64)
    float_input = torch.tensor([2.0], dtype=torch.float16)
    nested_int64_cache = torch.tensor([3], dtype=torch.int64)

    normalized = normalize_hybrid_hmonnx_args((int64_input, (float_input, (nested_int64_cache,))))

    assert len(normalized) == 3
    assert normalized[0].dtype is torch.int32
    assert normalized[1] is float_input
    assert normalized[2].dtype is torch.int32


def test_qwen3_next_model_config_propagates_both_fuse_flags_into_wrap_cfg():
    from xhmodel_merak.xh_llm.models.qwen3_next import (
        XHQwen3NextModel,
        XHQwen3NextModelConfig,
    )

    config = XHQwen3NextModelConfig(
        model_name="qwen3-next-test",
        fuse_gdr_ops=True,
        fuse_gdr_block_recurrent_ops=True,
    )
    model = XHQwen3NextModel(config)

    assert config.fuse_gdr_ops is True
    assert config.fuse_gdr_block_recurrent_ops is True
    assert model.wrap_cfg.fuse_gdr_ops is True
    assert model.wrap_cfg.fuse_gdr_block_recurrent_ops is True


def test_qwen3_next_gdn_setup_and_update_create_all_requested_fused_ops():
    from xhmodel_merak.xh_llm.models.qwen3_5._gdr_ops import (
        GDRBlockTriInverse,
        GDRChunkScan,
        GDRRecurrentScan,
    )

    module = _new_small_next_gdn()
    module._setup(_gdn_cfg())

    assert module.fuse_gdr_ops is True
    assert module.fuse_gdr_block_recurrent_ops is True
    assert isinstance(module.block_tri_inverse_op, GDRBlockTriInverse)
    assert isinstance(module.chunk_scan_op, GDRChunkScan)
    assert isinstance(module.recurrent_scan_op, GDRRecurrentScan)
    assert module.chunk_scan_op.num_chunks == 2

    module._update_cfg(_gdn_cfg(input_sequence_length=24))

    assert module.chunk_scan_op.num_chunks == 3


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("linear_chunk_size", 16),
        ("split_conv_cache", False),
        ("fuse_gdr_ops", False),
        ("fuse_gdr_block_recurrent_ops", False),
        ("use_manual_depthwise_conv1d", True),
    ],
)
def test_gdn_update_rejects_setup_time_topology_changes(field, changed_value):
    module = _new_small_next_gdn()
    module._setup(_gdn_cfg())
    update_cfg = _gdn_cfg(input_sequence_length=24)
    update_cfg[field] = changed_value

    with pytest.raises(ValueError, match=rf"{field} is setup-time immutable"):
        module._update_cfg(update_cfg)


def test_qwen3_next_chunk_and_recurrent_paths_pass_real_fused_ops_to_shared_delta_rules(
    monkeypatch,
):
    from xhmodel_merak.xh_llm.models.qwen3_5 import _hybrid_gated_delta_net as shared

    module = _new_small_next_gdn()
    module._setup(_gdn_cfg())
    captured = {}

    def fake_chunk_rule(query, key, value, **kwargs):
        del key, value
        captured["chunk"] = kwargs
        output = torch.zeros(query.shape[0], query.shape[1], module.num_v_heads, module.head_v_dim)
        return output, kwargs["initial_state"]

    monkeypatch.setattr(shared, "torch_chunk_gated_delta_rule", fake_chunk_rule)
    _, _, recurrent_output = _forward_small_gdn(module, sequence_length=16)

    assert captured["chunk"]["block_tri_inverse_op"] is module.block_tri_inverse_op
    assert captured["chunk"]["chunk_scan_op"] is module.chunk_scan_op
    assert captured["chunk"]["chunk_eye_matrix"] is not None
    assert captured["chunk"]["chunk_eye_8_batched"] is not None
    assert captured["chunk"]["cumsum_matrix"] is not None
    assert recurrent_output is None

    module._update_cfg(
        _gdn_cfg(
            linear_attention_mode="recurrent",
            input_sequence_length=1,
            suppress_recurrent_state_outputs=False,
        )
    )

    def fake_recurrent_rule(query, key, value, **kwargs):
        del key, value
        captured["recurrent"] = kwargs
        output = torch.zeros(query.shape[0], query.shape[1], module.num_v_heads, module.head_v_dim)
        return output, kwargs["initial_state"] + 1

    monkeypatch.setattr(shared, "torch_recurrent_gated_delta_rule", fake_recurrent_rule)
    _, _, recurrent_output = _forward_small_gdn(module, sequence_length=1)

    assert captured["recurrent"]["recurrent_scan_op"] is module.recurrent_scan_op
    assert recurrent_output is not None
    assert torch.count_nonzero(recurrent_output).item() == recurrent_output.numel()


def test_qwen3_next_unfused_chunk_prefill_exports_explicit_recurrent_state(monkeypatch):
    from xhmodel_merak.xh_llm.models.qwen3_5 import _hybrid_gated_delta_net as shared

    module = _new_small_next_gdn()
    module._setup(
        _gdn_cfg(
            fuse_gdr_ops=False,
            fuse_gdr_block_recurrent_ops=False,
            suppress_recurrent_state_outputs=False,
        )
    )
    captured = {}

    def fake_chunk_rule(query, key, value, **kwargs):
        del key, value
        captured.update(kwargs)
        output = torch.zeros(query.shape[0], query.shape[1], module.num_v_heads, module.head_v_dim)
        return output, kwargs["initial_state"] + 2

    monkeypatch.setattr(shared, "torch_chunk_gated_delta_rule", fake_chunk_rule)
    _, _, recurrent_output = _forward_small_gdn(module, sequence_length=16)

    assert module.block_tri_inverse_op is None
    assert module.chunk_scan_op is None
    assert module.recurrent_scan_op is None
    assert captured["block_tri_inverse_op"] is None
    assert captured["chunk_scan_op"] is None
    assert recurrent_output is not None
    assert torch.equal(recurrent_output, torch.full_like(recurrent_output, 2))


def test_canonical_fused_chunk_scan_mutates_cache_tensor_state_in_place():
    from xhmodel_merak.xh_llm.models.qwen3_5._delta_rule import (
        torch_chunk_gated_delta_rule,
    )
    from xhquant.core import CacheTensor

    class InPlaceChunkScan:
        state_is_cache = True

        def __init__(self):
            self.received_state = None

        def __call__(
            self,
            query,
            key,
            value,
            k_cumdecay,
            decay_mask,
            mask_incl,
            g,
            initial_state,
        ):
            del query, key, k_cumdecay, decay_mask, mask_incl, g
            self.received_state = initial_state
            initial_state[:] = torch.full_like(initial_state.data, 7)
            return torch.zeros_like(value)

    batch = heads = 1
    sequence_length = chunk_size = 8
    k_dim = v_dim = 2
    query = torch.randn(batch, sequence_length, heads, k_dim)
    key = torch.randn_like(query)
    value = torch.randn(batch, sequence_length, heads, v_dim)
    g = torch.zeros(batch, sequence_length, heads)
    beta = torch.ones_like(g)
    mask_qkv = torch.ones(batch, sequence_length, 1, 1)
    recurrent_state = CacheTensor(torch.zeros(batch, heads, k_dim, v_dim))
    scan = InPlaceChunkScan()

    _, returned_state = torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        mask_qkv=mask_qkv,
        initial_state=recurrent_state,
        output_final_state=True,
        chunk_size=chunk_size,
        chunk_mask_incl=torch.tril(torch.ones(chunk_size, chunk_size)),
        chunk_mask_strict=torch.tril(torch.ones(chunk_size, chunk_size), diagonal=-1),
        chunk_eye_matrix=torch.eye(chunk_size).unsqueeze(0),
        chunk_eye_8_batched=torch.eye(8).unsqueeze(0),
        cumsum_matrix=torch.triu(torch.ones(chunk_size, chunk_size)),
        input_sequence_length=sequence_length,
        num_heads=heads,
        k_head_dim=k_dim,
        v_head_dim=v_dim,
        batch_size=batch,
        scale=torch.tensor(k_dim**-0.5),
        chunk_row_masks=torch.ones(chunk_size, chunk_size, chunk_size),
        chunk_scan_op=scan,
    )

    assert scan.received_state is recurrent_state
    assert returned_state is recurrent_state
    assert torch.equal(recurrent_state.data, torch.full_like(recurrent_state.data, 7))


@pytest.mark.parametrize(
    ("split_conv_cache", "suppress_recurrent", "expected_input_count", "expected_output_count"),
    [
        (True, False, 18, 13),
        (False, False, 12, 7),
        (True, True, 18, 10),
        (False, True, 12, 4),
    ],
)
def test_four_block_f1_l3_export_cache_io_counts(
    split_conv_cache,
    suppress_recurrent,
    expected_input_count,
    expected_output_count,
):
    from xhmodel_merak.xh_llm.models.qwen3_next import XHQwen3NextModel

    class AttrDict(dict):
        __getattr__ = dict.__getitem__

    class FakeTarget:
        wrap_cfg = AttrDict(
            split_conv_cache=split_conv_cache,
            verify_output_intermediates=False,
            input_sequence_length=256,
            output_post_norm_hidden=False,
        )
        config = SimpleNamespace(split_conv_cache=split_conv_cache)
        _kvcache_mixin = SimpleNamespace(
            enable_page_attention=False,
            split_conv_cache=split_conv_cache,
        )
        kvcache_config = SimpleNamespace(
            num_layers=1,
            linear_kv_cache_config=SimpleNamespace(num_layers=3),
        )

        def _sync_split_conv_cache_state(self):
            return None

        def _set_recurrent_state_output_contract(self):
            return suppress_recurrent

    export_cfg = XHQwen3NextModel.get_export_cfg(FakeTarget())

    assert len(export_cfg["input_names"]) == expected_input_count
    assert len(export_cfg["output_names"]) == expected_output_count
    assert sum(name.startswith("past_key_cache_") for name in export_cfg["input_names"]) == 1
    assert sum(name.startswith("past_conv_cache_") for name in export_cfg["input_names"]) == (
        9 if split_conv_cache else 3
    )
    assert sum(name.startswith("past_recurrent_state_") for name in export_cfg["input_names"]) == 3


def test_qwen3_next_hmonnx_mtp_verify_commits_step4_cache_snapshots(monkeypatch):
    from xhmodel_merak.xh_llm.hmonnx import TextLLMHMONNXModel
    from xhmodel_merak.xh_llm.models.qwen3_next.qwen3_next_hmonnx_inference import (
        XHQwen3NextHMONNXModel,
    )

    model = object.__new__(XHQwen3NextHMONNXModel)
    model._llm_prefill = False
    model.meta_info = SimpleNamespace(
        spec_decode={"mode": "mtp", "num_draft_tokens": 4},
        model_config=SimpleNamespace(fuse_gdr_ops=False),
    )
    model.decode_model = SimpleNamespace()
    model._kvcache_mixin = SimpleNamespace(
        split_conv_cache=True,
        past_conv_caches=[
            (torch.zeros(1), torch.zeros(1), torch.zeros(1)),
        ],
        past_recurrent_states=[torch.zeros(1)],
    )

    logits = torch.tensor([[-1.0]])
    # Export order for L=1 and verify_steps=5 is q[0:5], k[0:5], v[0:5], recurrent[0:5].
    linear_outputs = [torch.tensor([float(value)]) for value in range(5)]
    linear_outputs += [torch.tensor([float(10 + value)]) for value in range(5)]
    linear_outputs += [torch.tensor([float(20 + value)]) for value in range(5)]
    linear_outputs += [torch.tensor([float(30 + value)]) for value in range(5)]
    monkeypatch.setattr(
        TextLLMHMONNXModel,
        "forward",
        lambda self, *args: (logits, *linear_outputs),
    )

    _, conv_outputs, recurrent_outputs = XHQwen3NextHMONNXModel.forward(model)

    q_cache, k_cache, v_cache = model.past_conv_caches[0]
    recurrent_cache = model.past_recurrent_states[0]
    assert q_cache.item() == 4
    assert k_cache.item() == 14
    assert v_cache.item() == 24
    assert recurrent_cache.item() == 34
    assert [item.item() for item in conv_outputs] == [4, 14, 24]
    assert [item.item() for item in recurrent_outputs] == [34]


def test_qwen3_next_mtp_workflow_keeps_shared_gdr_configuration_parity():
    from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig

    assert NEXT_MTP_WORKFLOW.is_file(), (
        "Missing Qwen3-Next workflow parity artifact: "
        f"{NEXT_MTP_WORKFLOW.relative_to(REPO_ROOT)}. The Qwen3-Next export must "
        "have a structured MTP+GPTQ workflow rather than relying on ad-hoc CLI defaults."
    )
    workflow = WorkflowConfig.from_file(str(NEXT_MTP_WORKFLOW))
    quant = workflow.quant
    model = workflow.export["model"]

    assert quant is not None
    assert quant["algorithm"] == "gptqmodel"
    assert quant["method"] == "gptq"
    assert quant["rotation"] is False
    assert model["model_type"] == "Qwen3NextForCausalLM"
    assert model["linear_chunk_size"] == 64
    assert model["split_conv_cache"] is True
    assert model["normalize_force_fp32"] is False
    assert model["use_manual_depthwise_conv1d"] is False
    assert model["fuse_gdr_ops"] is False
    assert model["fuse_gdr_block_recurrent_ops"] is False
    assert model["spec_decode_mode"] == "mtp"
    assert model["num_draft_tokens"] == 4
    assert model["output_post_norm_hidden"] is True
    assert model["mtp_config"]["input_sequence_length"] == 1
    assert model["visual_config"] is None
    # These Qwen3.5 rotary/long-context controls are not consumed by the
    # Qwen3-Next target graph and must not masquerade as cross-family knobs.
    assert "linear_attention_mode" not in model
    assert "support_long_context_over_fp16_limit" not in model
    assert "max_pe_length" not in model


def test_qwen35_moe_and_next_leaves_share_non_gdr_optimization_profiles():
    from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig

    config_root = REPO_ROOT / "configs_merak/workflows/xh2a/llm_models"
    next_root = config_root / "qwen3_next/80b_a3b"
    q35_root = config_root / "qwen3_5_moe/35b_a3b"
    q35_122_root = config_root / "qwen3_5_moe/122b_a10b"
    pairs = [
        (q35_root / "qwen3_6_35b_a3b_full.yaml", next_root / "qwen3_next_80b_a3b_full.yaml"),
        (q35_root / "qwen3_6_35b_a3b_full_gptq.yaml", next_root / "qwen3_next_80b_a3b_full_gptq.yaml"),
        (
            q35_root / "qwen3_6_35b_a3b_full_fa_all16.yaml",
            next_root / "qwen3_next_80b_a3b_full_fa_all16.yaml",
        ),
        (
            q35_root / "qwen3_6_35b_a3b_full_mtp_gptq.yaml",
            next_root / "qwen3_next_80b_a3b_full_mtp_gptq.yaml",
        ),
        (q35_122_root / "qwen3_5_122b_a10b_full.yaml", next_root / "qwen3_next_80b_a3b_full.yaml"),
        (
            q35_122_root / "qwen3_5_122b_a10b_full_gptq.yaml",
            next_root / "qwen3_next_80b_a3b_full_gptq.yaml",
        ),
    ]
    optimization_fields = {
        "context_max_length",
        "prefill_chunk_length",
        "use_cache",
        "num_logits_to_keep",
        "linear_chunk_size",
        "split_conv_cache",
        "normalize_force_fp32",
        "use_manual_depthwise_conv1d",
        "quant_scheme",
        "only_first_block",
    }

    for q35_path, next_path in pairs:
        q35_workflow = WorkflowConfig.from_file(str(q35_path))
        next_workflow = WorkflowConfig.from_file(str(next_path))
        q35_model = q35_workflow.export["model"]
        next_model = next_workflow.export["model"]
        assert {key: q35_model[key] for key in optimization_fields} == {
            key: next_model[key] for key in optimization_fields
        }
        assert q35_model["fuse_gdr_ops"] is True
        assert q35_model["fuse_gdr_block_recurrent_ops"] is True
        if q35_path.name.endswith("full_fa_all16.yaml"):
            assert q35_model["flash_attention"] == next_model["flash_attention"]
            assert q35_workflow.quant is not None
            assert next_workflow.quant is not None
            assert q35_workflow.quant["method"] == "autoround"
            assert next_workflow.quant["method"] == "autoround"


def test_qwen3_next_gptq_profile_contains_only_recipe_consumed_keys():
    from xhmodel_merak.xh_llm.models.qwen3_next.workflow import Qwen3NextWorkflow
    from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig

    workflow = WorkflowConfig.from_file(str(NEXT_MTP_WORKFLOW))
    assert workflow.quant is not None
    Qwen3NextWorkflow._validate_qwen3_next_gptq_config(workflow.quant)

    forbidden = {
        "preset",
        "hessian_mse",
        "seed",
        "quant_nontext_module",
        "offload_to_disk",
        "check_quant_vision_demo",
        "check_rotation_ppl",
        "expert_down_bits",
        "routing",
    }
    resolved_keys = set(workflow.quant)
    for value in workflow.quant.values():
        if isinstance(value, dict):
            resolved_keys.update(value)
    assert not (forbidden & resolved_keys)

    invalid = dict(workflow.quant)
    invalid["moe"] = {**invalid["moe"], "expert_down_bits": 5}
    with pytest.raises(ValueError, match="not consumed"):
        Qwen3NextWorkflow._validate_qwen3_next_gptq_config(invalid)


def test_runtime_requires_explicit_chunk_scan_cache_mutation_metadata():
    from xhmodel_merak.xh_llm.models.qwen3_5.hybrid_cache_runtime import (
        model_config_prefill_recurrent_state_uses_cache,
    )

    assert model_config_prefill_recurrent_state_uses_cache(SimpleNamespace(fuse_gdr_ops=True)) is False
    assert (
        model_config_prefill_recurrent_state_uses_cache(
            SimpleNamespace(fuse_gdr_ops=True, prefill_recurrent_state_uses_cache=True)
        )
        is True
    )
    assert (
        model_config_prefill_recurrent_state_uses_cache(
            SimpleNamespace(fuse_gdr_ops=False, prefill_recurrent_state_uses_cache=False)
        )
        is False
    )
    assert (
        model_config_prefill_recurrent_state_uses_cache(
            {"fuse_gdr_ops": True, "prefill_recurrent_state_uses_cache": True}
        )
        is True
    )


def test_legacy_fused_prefill_without_state_is_cache_metadata_keeps_recurrent_output():
    from xhmodel_merak.xh_llm.models.qwen3_5.hybrid_cache_runtime import (
        commit_hybrid_cache_outputs,
    )

    recurrent_cache = torch.zeros(1)
    runtime = SimpleNamespace(
        meta_info=SimpleNamespace(
            model_config=SimpleNamespace(fuse_gdr_ops=True),
            spec_decode=None,
        ),
        _kvcache_mixin=SimpleNamespace(
            split_conv_cache=False,
            past_conv_caches=[torch.zeros(1)],
            past_recurrent_states=[recurrent_cache],
        ),
        prefill_model=SimpleNamespace(_onnx_graph=None, hmonnx_session=None),
        is_prefill=lambda: True,
    )

    _, _, recurrent = commit_hybrid_cache_outputs(
        runtime,
        (torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([7.0])),
        model_label="legacy-fused-prefill",
    )

    assert [item.item() for item in recurrent] == [7.0]
    assert recurrent_cache.item() == 7.0


def test_runtime_skips_callable_graph_output_and_uses_session_output_names():
    from xhmodel_merak.xh_llm.models.qwen3_5.hybrid_cache_runtime import (
        _graph_output_names,
    )

    session = SimpleNamespace(
        get_output_names=lambda: ["logits", "conv_cache_out_0", "recurrent_state_out_0"],
        onnx_graph=None,
        graph=None,
        graph_module=None,
    )
    active_model = SimpleNamespace(
        _onnx_graph=SimpleNamespace(
            graph=SimpleNamespace(output=lambda result: result),
        ),
        hmonnx_session=session,
    )
    runtime = SimpleNamespace(
        prefill_model=active_model,
        is_prefill=lambda: True,
    )

    assert _graph_output_names(runtime, 3) == [
        "logits",
        "conv_cache_out_0",
        "recurrent_state_out_0",
    ]


def test_runtime_prefers_session_output_api_without_probing_dynamic_ops():
    from xhmodel_merak.xh_llm.models.qwen3_5.hybrid_cache_runtime import (
        _graph_output_names,
    )

    class DynamicOpSession:
        def get_output_names(self):
            return ["logits", "conv_cache_out_0", "recurrent_state_out_0"]

        def __getattr__(self, name):
            raise AssertionError(f"dynamic HMONNX op namespace was probed: {name}")

    active_model = SimpleNamespace(
        _onnx_graph=None,
        hmonnx_session=DynamicOpSession(),
    )
    runtime = SimpleNamespace(
        prefill_model=active_model,
        is_prefill=lambda: True,
    )

    assert _graph_output_names(runtime, 3) == [
        "logits",
        "conv_cache_out_0",
        "recurrent_state_out_0",
    ]
