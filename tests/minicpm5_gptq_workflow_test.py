import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_CONFIG = REPO_ROOT / (
    "configs_merak/workflows/xh2a/llm_models/minicpm5/15b_a2_5b/minicpm5_15b_a2_5b_xh2a_w4a8_autoround.yaml"
)


class _Tokenizer:
    def __call__(self, text):
        del text
        return {"input_ids": list(range(20)), "attention_mask": [1] * 20}


def test_minicpm5_gptq_config_uses_wikitext_and_w4a8_export() -> None:
    from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig

    config = WorkflowConfig.from_file(str(WORKFLOW_CONFIG))

    assert config.quant["algorithm"] == "gptqmodel"
    assert config.quant["method"] == "autoround"
    assert config.quant["bits"] == 4
    assert config.quant["group_size"] == 64
    assert config.quant["calibration"] == {
        "jsonl": "path-to-minicpm5-calibration-jsonl",
        "text_key": "text",
        "nsamples": 8,
        "seqlen": 256,
    }
    assert config.quant["moe"]["routing"] == "bypass"
    assert config.quant["moe"]["routing_batch_size"] == 48
    assert "expert_down_bits" not in config.quant["moe"]
    model = config.export["model"]
    assert model["prefill_chunk_length"] == 256
    assert model["quant_scheme"]["quant_type"] == "w4a8h0_ssfp"
    assert model["quant_scheme"]["nodes"]["lm_head"]["quant_type"] == "w8a8h1_sefp"


def test_minicpm5_calibration_builder_renders_jsonl_text_format() -> None:
    from examples_merak.llm.minicpm5.build_calibration import _render_mmlu_prompt

    prompt = _render_mmlu_prompt(
        {
            "subject": "abstract_algebra",
            "question": "Find the identity element.",
            "choices": ["0", "1", "2", "3"],
            "answer": 1,
        }
    )

    assert prompt == (
        "The following is a multiple choice question about abstract algebra.\n\n"
        "Find the identity element.\nA. 0\nB. 1\nC. 2\nD. 3\nAnswer: B"
    )


def test_minicpm5_calibration_dataset_load_error_is_actionable(monkeypatch) -> None:
    import pytest

    import examples_merak.llm.minicpm5.build_calibration as calibration

    def _load_dataset(*args, **kwargs):
        del args, kwargs
        raise OSError("network unavailable")

    monkeypatch.setattr(calibration, "load_dataset", _load_dataset)
    with pytest.raises(RuntimeError, match="Check network access or provide a cache_dir"):
        calibration._load_calibration_dataset(
            dataset_name="wikitext",
            dataset_config="wikitext-2-raw-v1",
            split="train",
            cache_dir=None,
            required_fields={"text"},
        )


def test_minicpm5_calibration_dataset_schema_error_lists_missing_fields(monkeypatch) -> None:
    import pytest

    import examples_merak.llm.minicpm5.build_calibration as calibration

    class _Dataset:
        column_names = ["question", "choices"]

    monkeypatch.setattr(calibration, "load_dataset", lambda *args, **kwargs: _Dataset())
    with pytest.raises(ValueError, match="missing required fields: answer, subject"):
        calibration._load_calibration_dataset(
            dataset_name="cais/mmlu",
            dataset_config="all",
            split="test",
            cache_dir=None,
            required_fields={"subject", "question", "choices", "answer"},
        )


def test_minicpm5_calibration_dataset_rejects_empty_result(monkeypatch) -> None:
    import pytest

    import examples_merak.llm.minicpm5.build_calibration as calibration

    class _Dataset:
        column_names = ["text"]

        def __len__(self):
            return 0

    monkeypatch.setattr(calibration, "load_dataset", lambda *args, **kwargs: _Dataset())
    with pytest.raises(ValueError, match="split 'train' is empty"):
        calibration._load_calibration_dataset(
            dataset_name="wikitext",
            dataset_config="wikitext-2-raw-v1",
            split="train",
            cache_dir=None,
            required_fields={"text"},
        )


def test_minicpm5_calibration_dataset_retries_without_verification_mode(monkeypatch) -> None:
    import examples_merak.llm.minicpm5.build_calibration as calibration

    class _Dataset:
        column_names = ["text"]

        def __len__(self):
            return 1

    calls = []

    def _load_dataset(*args, **kwargs):
        calls.append(kwargs)
        if "verification_mode" in kwargs:
            raise TypeError("unexpected keyword argument 'verification_mode'")
        return _Dataset()

    monkeypatch.setattr(calibration, "load_dataset", _load_dataset)
    dataset = calibration._load_calibration_dataset(
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        split="train",
        cache_dir=None,
        required_fields={"text"},
    )

    assert isinstance(dataset, _Dataset)
    assert calls == [
        {"split": "train", "cache_dir": None, "verification_mode": "no_checks"},
        {"split": "train", "cache_dir": None},
    ]


def test_minicpm5_calibration_dataset_does_not_retry_unrelated_type_error(monkeypatch) -> None:
    import pytest

    import examples_merak.llm.minicpm5.build_calibration as calibration

    calls = []

    def _load_dataset(*args, **kwargs):
        calls.append(kwargs)
        raise TypeError("verification_mode value is invalid")

    monkeypatch.setattr(calibration, "load_dataset", _load_dataset)
    with pytest.raises(RuntimeError, match="Failed to load wikitext"):
        calibration._load_calibration_dataset(
            dataset_name="wikitext",
            dataset_config="wikitext-2-raw-v1",
            split="train",
            cache_dir=None,
            required_fields={"text"},
        )
    assert len(calls) == 1


def test_minicpm5_gptq_calibration_rejects_placeholder_path() -> None:
    import pytest

    from xhmodel_merak.xh_llm.models.minicpm5.quant_adapter import (
        MiniCPM5QuantConfigError,
        _resolve_calibration_jsonl_path,
    )

    with pytest.raises(MiniCPM5QuantConfigError, match="placeholder path"):
        _resolve_calibration_jsonl_path({"jsonl": "path-to-minicpm5-calibration-jsonl"})


def test_minicpm5_formal_configs_have_single_w8a8_definition() -> None:
    from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig

    config_dir = WORKFLOW_CONFIG.parent
    assert sorted(path.name for path in config_dir.glob("*.yaml")) == [
        "minicpm5_15b_a2_5b_xh2a_w4a8_autoround.yaml",
        "minicpm5_15b_a2_5b_xh2a_w8a8.yaml",
    ]

    w8a8 = WorkflowConfig.from_file(str(config_dir / "minicpm5_15b_a2_5b_xh2a_w8a8.yaml"))
    assert w8a8.export["model"]["quant_scheme"] == {"quant_type": "w8a8h1_sefp", "ops": {}}


def test_minicpm5_wikitext_calibration_uses_contiguous_fixed_length_blocks() -> None:
    from gptqmodel.recipes.minicpm5 import tokenize_minicpm5_calibration

    samples = tokenize_minicpm5_calibration(
        texts=["first", "", "second"],
        tokenizer=_Tokenizer(),
        nsamples=3,
        seqlen=6,
    )

    assert samples == [
        {"input_ids": list(range(0, 6)), "attention_mask": [1] * 6},
        {"input_ids": list(range(6, 12)), "attention_mask": [1] * 6},
        {"input_ids": list(range(12, 18)), "attention_mask": [1] * 6},
    ]


def test_minicpm5_gptq_calibration_accepts_jsonl(tmp_path) -> None:
    import json

    from xhmodel_merak.xh_llm.models.minicpm5.quant_adapter import (
        _resolve_calibration_jsonl_path,
    )

    path = tmp_path / "calibration.jsonl"
    path.write_text(json.dumps({"text": "first"}) + "\n" + json.dumps({"text": "second"}) + "\n")

    assert _resolve_calibration_jsonl_path({"jsonl": str(path)}) == str(path)


def test_minicpm5_moe_router_and_output_match_native_fp16_and_bf16() -> None:
    from types import SimpleNamespace

    import torch
    from torch import nn

    from xhmodel_merak.xh_llm.models.minicpm5._model import _MiniCPM5MoE

    class _Expert(nn.Module):
        def __init__(self, dtype: torch.dtype):
            super().__init__()
            self.gate_proj = nn.Linear(4, 3, bias=False, dtype=dtype)
            self.up_proj = nn.Linear(4, 3, bias=False, dtype=dtype)
            self.down_proj = nn.Linear(3, 4, bias=False, dtype=dtype)

        def forward(self, hidden_states):
            return self.down_proj(torch.nn.functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))

    class _Gate(nn.Module):
        def __init__(self, dtype: torch.dtype):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(2, 4, dtype=dtype))
            self.register_buffer("e_score_correction_bias", torch.tensor([0.1, -0.2], dtype=torch.float32))
            self.top_k = 1
            self.n_group = 1
            self.topk_group = 1
            self.norm_topk_prob = True
            self.routed_scaling_factor = 3.66

    for dtype in (torch.float16, torch.bfloat16):
        torch.manual_seed(7)
        experts = nn.ModuleList([_Expert(dtype), _Expert(dtype)])
        shared_expert = _Expert(dtype)
        gate = _Gate(dtype)
        hidden_states = torch.randn(1, 3, 4, dtype=dtype)

        flat_hidden_states = hidden_states.reshape(-1, 4)
        expected_router_logits = torch.nn.functional.linear(flat_hidden_states.float(), gate.weight.float())
        expected_scores = expected_router_logits.sigmoid()
        expected_selected = torch.topk(
            expected_scores + gate.e_score_correction_bias.unsqueeze(0),
            k=1,
            dim=-1,
            sorted=False,
        ).indices
        expected_routed = torch.zeros_like(flat_hidden_states, dtype=expected_scores.dtype)
        for token_index, expert_index in enumerate(expected_selected[:, 0]):
            expected_routed[token_index] = (
                experts[int(expert_index)](flat_hidden_states[token_index]).float() * gate.routed_scaling_factor
            )
        expected_output = expected_routed.to(dtype).view_as(hidden_states) + shared_expert(hidden_states)

        wrapped = object.__new__(_MiniCPM5MoE)
        nn.Module.__init__(wrapped)
        wrapped.experts = experts
        wrapped.shared_experts = shared_expert
        wrapped.gate = gate
        wrapped.config = SimpleNamespace(
            num_experts_per_tok=1,
            n_group=1,
            topk_group=1,
            norm_topk_prob=True,
            routed_scaling_factor=3.66,
        )
        wrapped._setup()

        actual_router_logits = wrapped.router(wrapped.router_input_cast(flat_hidden_states))
        actual_scores = actual_router_logits.sigmoid()
        actual_selected = wrapped.router_topk(
            actual_scores + wrapped.e_score_correction_bias.unsqueeze(0)
        )[1]
        actual_output = wrapped(hidden_states)

        torch.testing.assert_close(actual_router_logits, expected_router_logits)
        torch.testing.assert_close(actual_selected, expected_selected)
        torch.testing.assert_close(actual_output.float(), expected_output.float(), rtol=2e-2, atol=2e-2)


def test_minicpm5_moe_fusion_preserves_expert_quant_weights() -> None:
    from types import SimpleNamespace

    import torch
    from torch import nn

    from xhmodel_merak.xh_llm.models.minicpm5._model import _MiniCPM5MoE

    class _Expert(nn.Module):
        def __init__(self, offset: int):
            super().__init__()
            for projection_index, projection_name in enumerate(("gate_proj", "up_proj", "down_proj")):
                linear = nn.Linear(3, 2, bias=False, dtype=torch.float16)
                value_offset = offset + projection_index * 10
                with torch.no_grad():
                    linear.weight.copy_(torch.arange(6, dtype=torch.float16).view(2, 3) + value_offset)
                linear.register_buffer(
                    "quant_weight",
                    torch.tensor([[-8 + offset, -1, 0], [1, 6, 7 - offset]], dtype=torch.int8)
                    + projection_index,
                )
                setattr(self, projection_name, linear)

    class _Gate(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2, 3, dtype=torch.float16))
            self.register_buffer("e_score_correction_bias", torch.zeros(2))
            self.top_k = 1
            self.n_group = 1
            self.topk_group = 1
            self.norm_topk_prob = True
            self.routed_scaling_factor = 1.0

    experts = nn.ModuleList([_Expert(0), _Expert(1)])
    expected_quant_weights = {
        projection_name: torch.stack(
            [getattr(expert, projection_name).quant_weight.clone() for expert in experts], dim=0
        )
        for projection_name in ("gate_proj", "up_proj", "down_proj")
    }
    wrapped = object.__new__(_MiniCPM5MoE)
    nn.Module.__init__(wrapped)
    wrapped.experts = experts
    wrapped.shared_experts = nn.Identity()
    wrapped.gate = _Gate()
    wrapped.config = SimpleNamespace(
        num_experts_per_tok=1,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
    )
    wrapped._setup()

    for projection_name in ("gate_proj", "up_proj", "down_proj"):
        packed_weight = getattr(wrapped.moeblock, f"expert_{projection_name}_weight")
        packed_quant_weight = getattr(wrapped.moeblock, f"expert_{projection_name}_quant_weight")
        assert tuple(packed_weight.shape) == (2, 2, 3)
        assert tuple(packed_quant_weight.shape) == (2, 2, 3)
        assert packed_quant_weight.dtype == torch.int8
        torch.testing.assert_close(packed_quant_weight, expected_quant_weights[projection_name])
    assert not hasattr(wrapped, "experts")


def test_minicpm5_moe_fusion_rejects_partial_expert_quant_weights() -> None:
    import pytest
    import torch
    from torch import nn

    from xhmodel_merak.xh_llm.models.minicpm5._model import _pack_expert_projection

    class _Expert(nn.Module):
        def __init__(self, quantized: bool):
            super().__init__()
            self.gate_proj = nn.Linear(3, 2, bias=False)
            if quantized:
                self.gate_proj.register_buffer("quant_weight", torch.ones(2, 3, dtype=torch.int8))

    experts = nn.ModuleList([_Expert(True), _Expert(False)])
    with pytest.raises(RuntimeError, match="must exist for every expert"):
        _pack_expert_projection(experts, "gate_proj")


def test_minicpm5_moe_gptq_options_follow_repository_moe_precedent() -> None:
    from gptqmodel.recipes.minicpm5 import build_minicpm5_dynamic

    from xhmodel_merak.xh_llm.models.minicpm5.quant_adapter import (
        _normalize_moe_routing,
    )

    moe = {
        "attn_bits": 8,
        "dense_mlp_bits": 8,
        "shared_expert_bits": 8,
        "expert_bits": 4,
        "routing": "bypass",
        "routing_batch_size": 48,
    }

    dynamic = build_minicpm5_dynamic(
        self_attn_bits=8,
        dense_mlp_bits=8,
        shared_expert_bits=8,
        expert_bits=4,
        group_size=64,
        sym=True,
    )

    assert dynamic[r"model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$"] == {
        "bits": 8,
        "group_size": 64,
        "sym": True,
    }
    assert dynamic[r"model\.layers\.\d+\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"] == {
        "bits": 4,
        "group_size": 64,
        "sym": True,
    }
    assert dynamic[r"model\.layers\.\d+\.mlp\.shared_experts\.(gate_proj|up_proj|down_proj)$"] == {
        "bits": 8,
        "group_size": 64,
        "sym": True,
    }
    assert dynamic[r"model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)$"] == {
        "bits": 8,
        "group_size": 64,
        "sym": True,
    }
    assert _normalize_moe_routing(moe) == ("bypass", 48, None)


def test_minicpm5_moe_routing_rejects_unknown_mode_with_typed_error() -> None:
    import pytest

    from xhmodel_merak.xh_llm.models.minicpm5.quant_adapter import (
        MiniCPM5QuantConfigError,
        _normalize_moe_routing,
    )

    with pytest.raises(MiniCPM5QuantConfigError, match="routing must be one of"):
        _normalize_moe_routing({"routing": "unknown"})


def test_minicpm5_gptq_rejects_non_64_group_size(tmp_path) -> None:
    import pytest

    from xhmodel_merak.xh_llm.models.minicpm5 import quant_adapter

    with pytest.raises(quant_adapter.MiniCPM5QuantConfigError, match="group_size must be 64"):
        quant_adapter.quantize_with_gptqmodel_api(
            model_dir="model",
            output_dir=str(tmp_path),
            device="cpu",
            quant_cfg={
                "bits": 4,
                "group_size": 128,
                "calibration": {"jsonl": "calibration.jsonl"},
            },
        )


def test_minicpm5_workflow_dispatches_explicit_gptq_config(tmp_path, monkeypatch) -> None:
    from xhmodel_merak.xh_llm.models.minicpm5 import quant_adapter
    from xhmodel_merak.xh_llm.models.minicpm5.workflow import MiniCPM5Workflow
    from xhmodel_merak.xh_llm.workflows.result import QuantResult

    workflow = MiniCPM5Workflow.__new__(MiniCPM5Workflow)
    workflow.model_dir = "/tmp/minicpm5"
    workflow.workflow_config = type(
        "Config",
        (),
        {
            "with_overrides": lambda self, _overrides: self,
            "quant": {"algorithm": "gptqmodel", "bits": 4},
        },
    )()
    captured = {}

    def _quantize(**kwargs):
        captured.update(kwargs)
        return QuantResult(raw_model_dir=kwargs["model_dir"], quanted_model_dir="/tmp/minicpm5-gptq")

    monkeypatch.setattr(quant_adapter, "quantize_with_gptqmodel_api", _quantize)

    result = workflow.quant(str(tmp_path), "cuda:0")

    assert result.quanted_model_dir == "/tmp/minicpm5-gptq"
    assert captured == {
        "model_dir": "/tmp/minicpm5",
        "output_dir": str(tmp_path),
        "device": "cuda:0",
        "quant_cfg": workflow.workflow_config.quant,
    }


def test_minicpm5_example_has_separate_quant_and_export_output_dirs(monkeypatch) -> None:
    from examples_merak.llm.minicpm5 import minicpm5_workflow

    monkeypatch.setattr(sys, "argv", ["minicpm5_workflow.py", "--model-dir", "/tmp/minicpm5"])
    args = minicpm5_workflow.parse_args()

    assert args.model_dir == "/tmp/minicpm5"
    assert args.quant_output_dir == "work_dirs/minicpm5_15b_a2_5b_quant"
    assert args.export_output_dir == "work_dirs/minicpm5_15b_a2_5b_export"
    assert not args.export_from_quanted_model


def test_minicpm5_example_can_export_an_existing_gptq_checkpoint(monkeypatch) -> None:
    from examples_merak.llm.minicpm5 import minicpm5_workflow

    monkeypatch.setattr(
        sys,
        "argv",
        ["minicpm5_workflow.py", "--model-dir", "/tmp/minicpm5", "--export-from-quanted-model"],
    )

    assert minicpm5_workflow.parse_args().export_from_quanted_model
