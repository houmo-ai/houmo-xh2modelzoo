from __future__ import annotations

import inspect
import json
import sys
import types
from pathlib import Path

import onnx
import pytest
import torch
import yaml
from onnx import TensorProto, helper
from torch import nn

from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_mtp_model import (
    Gemma4AssistantBackbone,
    Gemma4AssistantDraftModule,
    Gemma4AssistantSelfAttention,
)
from xhmodel_merak.xh_llm.models.gemma4_series.mtp_contract import (
    MTP_DRAFT_INPUT_NAMES,
    MTP_SHARED_KV_INPUT_NAMES,
    normalize_readonly_attention_lowering,
)
from xhmodel_merak.xh_llm.models.gemma4_series.mtp_workflow import (
    resolve_context_length,
    update_manifest_with_draft,
    validate_mtp_model_inputs,
)


def _attention(layer_type: str, sliding_window: int = 4):
    attention = object.__new__(Gemma4AssistantSelfAttention)
    attention.layer_type = layer_type
    attention.config = types.SimpleNamespace(sliding_window=sliding_window)
    return attention


def test_readonly_full_attention_reads_exact_target_prefix() -> None:
    valid_length, ranges = _attention("full_attention")._query_kv_range_abs(
        torch.tensor([7], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
    )

    assert valid_length.tolist() == [8]
    assert ranges.tolist() == [[[0, 8]]]


def test_readonly_sliding_attention_reads_target_window_only() -> None:
    valid_length, ranges = _attention(
        "sliding_attention",
        sliding_window=4,
    )._query_kv_range_abs(
        torch.tensor([7], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
    )

    assert valid_length.tolist() == [8]
    assert ranges.tolist() == [[[4, 8]]]


def test_readonly_sliding_attention_clamps_short_context_start() -> None:
    _, ranges = _attention(
        "sliding_attention",
        sliding_window=16,
    )._query_kv_range_abs(
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
    )

    assert ranges.tolist() == [[[0, 3]]]


def test_readonly_draft_keeps_next_token_rope_position() -> None:
    backbone = object.__new__(Gemma4AssistantBackbone)
    nn.Module.__init__(backbone)
    backbone.use_readonly_page_attention = True
    backbone.layers = nn.ModuleList()
    backbone.norm = nn.Identity()
    seen: list[tuple[str, int]] = []

    def capture_position(self, position, layer_type):
        del self
        seen.append((layer_type, int(position.item())))
        value = torch.empty(1)
        return value, value

    backbone._get_position_embeddings = types.MethodType(
        capture_position,
        backbone,
    )
    hidden = torch.empty(1, 1, 8)
    cache = torch.empty(1)
    backbone(
        hidden,
        torch.tensor([11], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.empty(1),
        cache,
        cache,
        cache,
        cache,
    )

    assert seen == [
        ("full_attention", 12),
        ("sliding_attention", 12),
    ]


def test_readonly_draft_manifest_declares_runtime_v2_contract(
    tmp_path: Path,
) -> None:
    meta_path = tmp_path / "golden_meta_info.json"
    draft_path = tmp_path / "mtp/draft.onnx"
    draft_path.parent.mkdir()
    draft_path.write_bytes(b"graph")
    meta_path.write_text(
        json.dumps(
            {
                "model_type": "Gemma4UnifiedForConditionalGeneration",
                "model_config": {
                    "num_draft_tokens": 4,
                    "sliding_window": 1024,
                    "mtp_config": {
                        "assistant_layer_pattern": [
                            "sliding_attention",
                            "sliding_attention",
                            "sliding_attention",
                            "full_attention",
                        ]
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    update_manifest_with_draft(
        meta_path,
        draft_path,
        lm_head_quant_type="w4a8h0_ssfp",
        shared_sliding_len=1344,
        shared_full_len=2048,
        readonly_page_attention=True,
    )

    spec = json.loads(meta_path.read_text(encoding="utf-8"))["spec_decode"]
    assert spec["runtime_contract_version"] == 2
    assert spec["num_draft_tokens"] == 4
    assert spec["draft"] == {
        "abi": "gemma4_mtp_readonly_page_attention_v1",
        "decode_hmonnx": "mtp/draft.onnx",
        "standalone_decode_hmonnx": "mtp/draft.onnx",
        "cache_mutation": "read_only",
        "cache_binding": "target_attention_type_owner",
        "constant_draft_positions": True,
        "position_semantics": "target_kv_valid_length_minus_one",
        "layer_attention_types": [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        "minimum_sliding_cache_slack": 4,
    }
    assert spec["standalone_draft_decode_onnx"] == "mtp/draft.onnx"


def test_causal_readonly_draft_manifest_declares_v2_page_attention_abi(
    tmp_path: Path,
) -> None:
    meta_path = tmp_path / "golden_meta_info.json"
    draft_path = tmp_path / "mtp/draft.onnx"
    draft_path.parent.mkdir()
    draft_path.write_bytes(b"graph")
    meta_path.write_text(
        json.dumps(
            {
                "model_type": "Gemma4UnifiedForConditionalGeneration",
                "model_config": {
                    "num_draft_tokens": 4,
                    "sliding_window": 1024,
                    "mtp_config": {
                        "assistant_layer_pattern": [
                            "sliding_attention",
                            "sliding_attention",
                            "sliding_attention",
                            "full_attention",
                        ]
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    update_manifest_with_draft(
        meta_path,
        draft_path,
        lm_head_quant_type="w4a8h0_ssfp",
        shared_sliding_len=1344,
        shared_full_len=2048,
        readonly_page_attention=True,
        readonly_attention_lowering="causal",
    )

    draft = json.loads(meta_path.read_text(encoding="utf-8"))["spec_decode"]["draft"]
    assert draft["abi"] == "gemma4_mtp_readonly_page_attention_v2"
    assert draft["attention_lowering"] == "causal"
    assert draft["cache_mutation"] == "read_only"
    assert draft["constant_draft_positions"] is True


def test_readonly_attention_lowering_defaults_and_rejects_invalid_values() -> None:
    assert normalize_readonly_attention_lowering() == "exact_range"
    assert normalize_readonly_attention_lowering(None) == "exact_range"
    assert normalize_readonly_attention_lowering(" causal ") == "causal"
    for invalid in ("none", "unsupported"):
        with pytest.raises(ValueError, match="exact_range.*causal"):
            normalize_readonly_attention_lowering(invalid)


@pytest.mark.parametrize(
    "shared_kv_inputs",
    [
        None,
        ["shared_key_cache_sliding"],
        [
            "shared_key_cache_sliding",
            "shared_value_cache_sliding",
            "shared_key_cache_full",
            "legacy_value_cache_full",
        ],
    ],
)
def test_mtp_config_rejects_missing_or_wrong_shared_kv_inputs(
    tmp_path: Path,
    shared_kv_inputs: list[str] | None,
) -> None:
    mtp_config = {
        "assistant_hf_model": str(tmp_path / "assistant"),
        "target_hf_model": str(tmp_path / "target"),
    }
    if shared_kv_inputs is not None:
        mtp_config["shared_kv_inputs"] = shared_kv_inputs

    with pytest.raises(
        ValueError,
        match="shared_kv_inputs.*Duplicate and legacy KV names",
    ):
        validate_mtp_model_inputs({"mtp_config": mtp_config})


def test_assistant_contract_rejects_zero_layers(tmp_path: Path) -> None:
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_mtp_model import (
        validate_assistant_target_contract,
    )

    assistant_dir = tmp_path / "assistant"
    target_dir = tmp_path / "target"
    assistant_dir.mkdir()
    target_dir.mkdir()
    assistant_dir.joinpath("config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4_assistant",
                "backbone_hidden_size": 8,
                "text_config": {
                    "hidden_size": 4,
                    "num_hidden_layers": 0,
                    "num_kv_shared_layers": 0,
                    "layer_types": [],
                    "num_attention_heads": 1,
                    "num_key_value_heads": 1,
                    "head_dim": 4,
                },
            }
        ),
        encoding="utf-8",
    )
    target_dir.joinpath("config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4",
                "text_config": {
                    "hidden_size": 8,
                    "layer_types": ["full_attention"],
                    "num_key_value_heads": 1,
                    "head_dim": 4,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="num_hidden_layers must be positive"):
        validate_assistant_target_contract(assistant_dir, target_dir)


def test_readonly_manifest_separates_deploy_and_standalone_graphs(
    tmp_path: Path,
) -> None:
    from examples_merak.llm.gemma4_series.mtp_hmonnx_inference import (
        PRESETS,
        _resolve_draft_onnx,
    )

    meta_path = tmp_path / "golden_meta_info.json"
    deploy_path = tmp_path / "mtp/draft_readonly_page_attention.onnx"
    standalone_path = tmp_path / "mtp/draft.onnx"
    deploy_path.parent.mkdir()
    deploy_path.write_bytes(b"deploy")
    standalone_path.write_bytes(b"standalone")
    meta_path.write_text(
        json.dumps(
            {
                "model_config": {"num_draft_tokens": 4},
                "spec_decode": {"mode": "mtp"},
            }
        ),
        encoding="utf-8",
    )

    update_manifest_with_draft(
        meta_path,
        deploy_path,
        standalone_draft_onnx=standalone_path,
        lm_head_quant_type="w4a8h0_ssfp",
        shared_sliding_len=1344,
        shared_full_len=2048,
        readonly_page_attention=True,
    )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    preset = PRESETS["12b-unified"]
    assert (
        _resolve_draft_onnx(
            preset,
            tmp_path,
            None,
            meta_path=meta_path,
            meta_dict=meta,
        )
        == deploy_path
    )
    assert (
        _resolve_draft_onnx(
            preset,
            tmp_path,
            None,
            meta_path=meta_path,
            meta_dict=meta,
            prefer_standalone=True,
        )
        == standalone_path
    )


def test_standalone_session_rejects_page_graph_before_hmonnx_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from examples_merak.llm.gemma4_series.mtp_hmonnx_inference import (
        AssistantDraftSession,
    )

    graph_path = tmp_path / "readonly.onnx"
    graph = helper.make_graph(
        [],
        "readonly",
        [
            helper.make_tensor_value_info(
                "input_1",
                TensorProto.FLOAT16,
                [1, 1, 8],
            ),
            helper.make_tensor_value_info(
                "valid_length",
                TensorProto.INT32,
                [1],
            ),
            helper.make_tensor_value_info(
                "current_length",
                TensorProto.INT32,
                [1],
            ),
        ],
        [],
    )
    onnx.save(helper.make_model(graph), graph_path)

    constructed = False

    def fail_if_constructed(_path):
        nonlocal constructed
        constructed = True
        raise AssertionError("HMONNX must not load an unbound PageAttention graph")

    fake_runtime = types.ModuleType("xhquant.xhonnxruntime")
    fake_runtime.HMONNXGrapInference = fail_if_constructed
    monkeypatch.setitem(sys.modules, "xhquant.xhonnxruntime", fake_runtime)

    with pytest.raises(RuntimeError, match="shared-tensor draft graph"):
        AssistantDraftSession(graph_path, torch.device("cuda"))

    assert constructed is False


def test_standalone_session_keeps_legacy_non_flash_attention_abi(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The example still executes the original mask + shared-KV draft graph."""

    from examples_merak.llm.gemma4_series.mtp_hmonnx_inference import (
        LEGACY_DRAFT_INPUTS,
        AssistantDraftSession,
    )

    graph_path = tmp_path / "legacy.onnx"
    graph_inputs = [
        helper.make_tensor_value_info(
            name,
            TensorProto.INT32
            if name in {"valid_length", "current_length"}
            else TensorProto.FLOAT16,
            (
                [1]
                if name in {"valid_length", "current_length"}
                else [1, 1, 8]
                if name == "input_1"
                else [1, 1, 1, 4]
                if name == "sliding_attention_mask"
                else [1, 1, 4, 2]
            ),
        )
        for name in LEGACY_DRAFT_INPUTS
    ]
    onnx.save(helper.make_model(helper.make_graph([], "legacy", graph_inputs, [])), graph_path)

    calls: list[dict[str, torch.Tensor]] = []

    class FakeRuntime:
        def __init__(self, _path: str):
            self.exec_device = None

        def get_input_names(self):
            return list(LEGACY_DRAFT_INPUTS)

        def get_output_names(self):
            return ["logits", "assistant_hidden_state"]

        def get_input(self, name: str):
            shapes = {
                "input_1": (1, 1, 8),
                "valid_length": (1,),
                "current_length": (1,),
                "sliding_attention_mask": (1, 1, 1, 4),
                "shared_key_cache_sliding": (1, 1, 4, 2),
                "shared_value_cache_sliding": (1, 1, 4, 2),
                "shared_key_cache_full": (1, 1, 4, 2),
                "shared_value_cache_full": (1, 1, 4, 2),
            }
            dtype = (
                torch.int32
                if name in {"valid_length", "current_length"}
                else torch.float16
            )
            return types.SimpleNamespace(shape=shapes[name], dtype=dtype)

        def run(self, feed):
            calls.append(feed)
            return (
                torch.ones(1, 1, 4, dtype=torch.float16),
                torch.ones(1, 1, 4, dtype=torch.float16),
            )

    fake_runtime = types.ModuleType("xhquant.xhonnxruntime")
    fake_runtime.HMONNXGrapInference = FakeRuntime
    monkeypatch.setitem(sys.modules, "xhquant.xhonnxruntime", fake_runtime)

    session = AssistantDraftSession(graph_path, torch.device("cpu"))
    cache = torch.zeros(1, 1, 4, 2, dtype=torch.float16)
    logits, hidden = session(
        inputs_embeds=torch.zeros(1, 1, 8, dtype=torch.float16),
        past_seq_length=torch.tensor([3], dtype=torch.int32),
        current_length=torch.tensor([1], dtype=torch.int32),
        sliding_attention_mask=torch.zeros(1, 1, 1, 4, dtype=torch.float16),
        shared_key_cache_sliding=cache,
        shared_value_cache_sliding=cache,
        shared_key_cache_full=cache,
        shared_value_cache_full=cache,
    )

    assert len(calls) == 1
    assert list(calls[0]) == LEGACY_DRAFT_INPUTS
    assert logits.shape == hidden.shape == (1, 1, 4)


@pytest.mark.parametrize(
    "relative_path",
    [
        "12b_unified/gemma4_12b_unified_full_mtp_page_attention.yaml",
        "e2b/gemma4_e2b_full_mtp_page_attention.yaml",
        "e4b/gemma4_e4b_full_mtp_page_attention.yaml",
        "31b/gemma4_31b_full_mtp_page_attention.yaml",
        "26b_a4b/gemma4_26b_a4b_full_mtp_page_attention.yaml",
    ],
)
def test_gemma4_readonly_mtp_configs_select_causal_contract_v2(
    relative_path: str,
) -> None:
    from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig

    repo_root = Path(__file__).resolve().parents[2]
    config = WorkflowConfig.from_file(
        repo_root / "configs_merak/workflows/xh2a/llm_models/gemma4_series/" / relative_path
    )
    model = config.export["model"]

    # YAML selects the implementation. The typed model config derives ABI v2,
    # so non-FlashAttention YAMLs do not have to carry an unrelated version.
    assert "attention_contract_version" not in model
    assert model["flash_attention"]["enable"] is True
    assert model["spec_decode_mode"] == "mtp"
    assert model["num_draft_tokens"] == 4
    assert model["mtp_config"]["readonly_attention_lowering"] == "causal"
    assert tuple(model["mtp_config"]["shared_kv_inputs"]) == (
        MTP_SHARED_KV_INPUT_NAMES
    )

    # Keep YAML, BaseModel export names, and the source forward ABI tied to one
    # contract.  The deploy lowering later replaces these four source tensors
    # with target-owned PageAttention cache bindings; the retained standalone
    # graph continues to expose them verbatim.
    forward_inputs = tuple(
        name
        for name in inspect.signature(
            Gemma4AssistantDraftModule.forward
        ).parameters
        if name != "self"
    )
    assert forward_inputs == MTP_DRAFT_INPUT_NAMES


def test_gemma4_mtp_context_is_owned_by_the_target_model_config() -> None:
    """All presets expose one context override, including legacy non-FA MTP."""

    repo_root = Path(__file__).resolve().parents[2]
    config_root = (
        repo_root
        / "configs_merak/workflows/xh2a/llm_models/gemma4_series"
    )
    mtp_paths = sorted(config_root.glob("*/*mtp*.yaml"))
    assert mtp_paths
    for path in mtp_paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        model = payload["export"]["model"]
        assert "context_max_length" in model, path
        assert "context_max_length" not in model.get("mtp_config", {}), path

    # Old external manifests may retain the duplicated nested field, but a
    # run-local target override must win.
    assert (
        resolve_context_length(
            {"context_max_length": 8192},
            {"context_max_length": 262144},
        )
        == 8192
    )
