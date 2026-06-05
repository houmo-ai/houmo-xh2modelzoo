from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest
import torch
import torch.nn as nn
import onnx
from onnx import TensorProto, helper

from xh_model_zoo.xh_llm.models.qwen3_5_moe import inference as moe_inference
from xh_model_zoo.xh_llm.models.qwen3_5_moe.qwen3_5_moe_convert_config import (
    Qwen3_5MoeConvertConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MOE_CONVERTER = REPO_ROOT / "xh_model_zoo/xh_llm/models/qwen3_5_moe/qwen3_5_moe_converter.py"



def _load_function_from_source(path: Path, name: str):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(module)
            namespace = {"torch": torch, "nn": nn, "List": List, "Path": Path, "onnx": onnx}
            exec(compile(module, filename=str(path), mode="exec"), namespace)
            return namespace[name]
    raise AssertionError(f"function {name!r} not found in {path}")


def test_qwen3_5_moe_split_cache_dims_prefer_actual_split_modules():
    dim_resolver = _load_function_from_source(MOE_CONVERTER, "_linear_split_conv_dims")

    class Conv:
        def __init__(self, in_channels: int):
            self.in_channels = in_channels

    class LinearAttn:
        key_dim = 4096
        value_dim = 4096
        head_k_dim = 64
        head_v_dim = 128
        num_v_heads = 32
        conv1d_q = Conv(2048)
        conv1d_k = Conv(2048)
        conv1d_v = Conv(4096)

    assert dim_resolver(LinearAttn()) == (2048, 2048, 4096)

    class PreSplitLinearAttn:
        key_dim = 4096
        value_dim = 4096
        head_k_dim = 64
        head_v_dim = 128
        num_v_heads = 32

    assert dim_resolver(PreSplitLinearAttn()) == (2048, 2048, 4096)


def test_qwen3_5_moe_config_preserves_split_conv_cache_modes():
    default_cfg = Qwen3_5MoeConvertConfig()
    legacy_cfg = Qwen3_5MoeConvertConfig(split_conv_cache=False)
    split_cfg = Qwen3_5MoeConvertConfig(split_conv_cache=True)

    assert default_cfg.split_conv_cache is True
    assert legacy_cfg.split_conv_cache is False
    assert split_cfg.split_conv_cache is True
    assert default_cfg.normalize_force_fp32 is False
    assert default_cfg.use_manual_depthwise_conv1d is False
    assert getattr(default_cfg, "fuse_gdr_ops", False) is False


def test_qwen3_5_moe_flatten_cache_outputs_splits_logits_but_keeps_pre_split_cache_outputs():
    flatten = _load_function_from_source(MOE_CONVERTER, "_flatten_cache_outputs")

    class Wrapped(nn.Module):
        cfg = {"batch_size": 2}

        def _qwen3_5_moe_original_forward(self):
            return (
                torch.tensor([[1.0], [2.0]]),
                [torch.tensor([[10.0]]), torch.tensor([[11.0]])],
                [torch.tensor([[20.0]]), torch.tensor([[21.0]])],
            )

    outputs = flatten(Wrapped())

    assert len(outputs) == 6
    assert [float(t.item()) for t in outputs] == [1.0, 2.0, 10.0, 11.0, 20.0, 21.0]


def test_qwen3_5_moe_flatten_cache_outputs_accepts_logits_already_split_by_wrapper():
    flatten = _load_function_from_source(MOE_CONVERTER, "_flatten_cache_outputs")

    class Wrapped(nn.Module):
        cfg = {"batch_size": 2}

        def _qwen3_5_moe_original_forward(self):
            return (
                (torch.tensor([[1.0]]), torch.tensor([[2.0]])),
                [torch.tensor([[10.0]]), torch.tensor([[11.0]])],
                [torch.tensor([[20.0]]), torch.tensor([[21.0]])],
            )

    outputs = flatten(Wrapped())

    assert len(outputs) == 6
    assert [float(t.item()) for t in outputs] == [1.0, 2.0, 10.0, 11.0, 20.0, 21.0]


def test_qwen3_5_moe_patch_hmonnx_standard_add_ops_moves_default_add_to_xh2a(tmp_path):
    patch_adds = _load_function_from_source(MOE_CONVERTER, "_patch_hmonnx_standard_add_ops")
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1])
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "y"], ["z"], name="node_add")],
        "test_graph",
        [x, y],
        [z],
    )
    model_path = tmp_path / "model.onnx"
    onnx.save(helper.make_model(graph), str(model_path))

    assert patch_adds(model_path) == 1
    patched = onnx.load(str(model_path), load_external_data=False)
    assert patched.graph.node[0].domain == "ai.houmo.xh2a"
    assert patch_adds(model_path) == 0


def test_qwen3_5_moe_load_meta_artifacts_accepts_release_golden_meta_fallbacks(
    monkeypatch, tmp_path
):
    meta_file = tmp_path / "golden_meta_info.json"
    (tmp_path / "prefill").mkdir()
    (tmp_path / "decode").mkdir()
    (tmp_path / "hf_config").mkdir()
    (tmp_path / "quant_embedding.pt").write_bytes(b"placeholder")
    (tmp_path / "prefill/model.onnx").write_bytes(b"prefill")
    (tmp_path / "decode/model.onnx").write_bytes(b"decode")
    meta_file.write_text(
        """
        {
          "prefill_onnx": "prefill/model.onnx",
          "decode_onnx": "decode/model.onnx",
          "kv_cache": {"shape": [1, 2, 512, 8]}
        }
        """,
        encoding="utf-8",
    )

    tokenizer = SimpleNamespace(pad_token_id=None, eos_token_id=151643)
    token_embedding = nn.Embedding(16, 8)
    loaded = {}

    def fake_from_pretrained(path):
        loaded["hf_config"] = Path(path)
        return tokenizer

    def fake_load_token_embedding(path):
        loaded["token_embedding"] = Path(path)
        return token_embedding

    monkeypatch.setattr(moe_inference.AutoTokenizer, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(moe_inference, "_load_token_embedding", fake_load_token_embedding)

    artifacts = moe_inference._load_meta_artifacts(str(meta_file))

    assert artifacts["prefill_onnx"] == (tmp_path / "prefill/model.onnx").resolve()
    assert artifacts["decode_onnx"] == (tmp_path / "decode/model.onnx").resolve()
    assert loaded["hf_config"] == (tmp_path / "hf_config").resolve()
    assert loaded["token_embedding"] == (tmp_path / "quant_embedding.pt").resolve()
    assert artifacts["tokenizer"] is tokenizer
    assert artifacts["token_embedding"] is token_embedding
    assert artifacts["pad_token_id"] == 151643
    assert artifacts["max_context_tokens"] == 512


def test_qwen3_5_moe_load_meta_artifacts_merges_structural_sidecar_for_golden_meta(
    monkeypatch, tmp_path
):
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    (release_dir / "prefill").mkdir()
    (release_dir / "decode").mkdir()
    (release_dir / "hf_config").mkdir()
    (release_dir / "quant_embedding.pt").write_bytes(b"placeholder")
    (release_dir / "prefill/model.onnx").write_bytes(b"prefill")
    (release_dir / "decode/model.onnx").write_bytes(b"decode")

    (tmp_path / "meta.json").write_text(
        """
        {
          "max_context_tokens": 512,
          "pad_token_id": 0,
          "wrap_cfg": {"batch_size": 1, "input_sequence_length": 128},
          "kv_cache": {"shape": [1, 2, 512, 256], "num_decoder_layers": 10},
          "linear_cache": {
            "layer_indices": [0],
            "layers": [{"conv_shape": [1, 8192, 4], "recurrent_shape": [1, 32, 128, 128]}]
          },
          "token_embedding_file": "token_embedding.pt"
        }
        """,
        encoding="utf-8",
    )
    meta_file = release_dir / "golden_meta_info.json"
    meta_file.write_text(
        """
        {
          "prefill_onnx": "prefill/model.onnx",
          "decode_onnx": "decode/model.onnx"
        }
        """,
        encoding="utf-8",
    )

    monkeypatch.setattr(
        moe_inference.AutoTokenizer,
        "from_pretrained",
        lambda path: SimpleNamespace(pad_token_id=0, eos_token_id=0),
    )
    monkeypatch.setattr(
        moe_inference,
        "_load_token_embedding",
        lambda path: nn.Embedding(16, 8),
    )

    artifacts = moe_inference._load_meta_artifacts(str(meta_file))

    assert artifacts["max_context_tokens"] == 512
    assert artifacts["meta_info"]["wrap_cfg"]["input_sequence_length"] == 128
    assert artifacts["meta_info"]["kv_cache"]["num_decoder_layers"] == 10
    assert artifacts["meta_info"]["linear_cache"]["layers"][0]["conv_shape"] == [1, 8192, 4]
    assert artifacts["prefill_onnx"] == (release_dir / "prefill/model.onnx").resolve()
    assert artifacts["decode_onnx"] == (release_dir / "decode/model.onnx").resolve()


@pytest.mark.parametrize(
    ("linear_layer_meta", "expected_conv_cache_count"),
    [
        ({"conv_shape": [1, 6, 4], "recurrent_shape": [1, 2, 8]}, 1),
        (
            {
                "conv_shapes": [[1, 2, 4], [1, 2, 4], [1, 2, 4]],
                "recurrent_shape": [1, 2, 8],
            },
            3,
        ),
    ],
)
def test_qwen3_5_moe_resource_tight_init_uses_meta_sequence_length_for_both_cache_modes(
    monkeypatch, tmp_path, linear_layer_meta, expected_conv_cache_count
):
    meta_file = tmp_path / "meta.json"
    meta_file.write_text("{}")
    token_embedding = nn.Embedding(16, 8)

    monkeypatch.setattr(
        moe_inference,
        "_load_meta_artifacts",
        lambda model_config_file: {
            "meta_file": Path(model_config_file),
            "model_dir": tmp_path,
            "meta_info": {
                "wrap_cfg": {"batch_size": 1, "input_sequence_length": 128},
                "kv_cache": {"shape": [1, 2, 4, 8], "num_decoder_layers": 1},
                "linear_cache": {
                    "layer_indices": [0],
                    "layers": [linear_layer_meta],
                },
            },
            "prefill_onnx": tmp_path / "prefill.onnx",
            "decode_onnx": tmp_path / "decode.onnx",
            "tokenizer": SimpleNamespace(pad_token_id=0, eos_token_id=0),
            "token_embedding": token_embedding,
            "pad_token_id": 0,
            "max_context_tokens": 8,
        },
    )

    engine = moe_inference.Qwen3_5MoeInference(
        str(meta_file), device="cpu", execution_device="cpu", resource_tight_mode=True
    )

    assert engine.prefill_session is None
    assert engine.decode_session is None
    assert engine._prefill_inputs_info is None
    assert engine.prefill_input_sequence_length == 128
    assert engine.input_sequence_length == 128
    assert len(engine.past_conv_caches) == expected_conv_cache_count
    assert len(engine.past_recurrent_states) == 1


@pytest.mark.parametrize(
    ("linear_layer_meta", "expected_conv_cache_count"),
    [
        (
            {
                "conv_shapes": [[1, 6, 4], [1, 6, 4]],
                "recurrent_shapes": [[1, 2, 8], [1, 2, 8]],
                "per_batch": True,
            },
            2,
        ),
        (
            {
                "conv_shapes": [
                    [1, 2, 4],
                    [1, 2, 4],
                    [1, 2, 4],
                    [1, 2, 4],
                    [1, 2, 4],
                    [1, 2, 4],
                ],
                "recurrent_shapes": [[1, 2, 8], [1, 2, 8]],
                "per_batch": True,
            },
            6,
        ),
    ],
)
def test_qwen3_5_moe_resource_tight_init_uses_expanded_batch_cache_meta_for_both_cache_modes(
    monkeypatch, tmp_path, linear_layer_meta, expected_conv_cache_count
):
    meta_file = tmp_path / "meta.json"
    meta_file.write_text("{}")
    token_embedding = nn.Embedding(16, 8)

    monkeypatch.setattr(
        moe_inference,
        "_load_meta_artifacts",
        lambda model_config_file: {
            "meta_file": Path(model_config_file),
            "model_dir": tmp_path,
            "meta_info": {
                "wrap_cfg": {"batch_size": 2, "input_sequence_length": 128},
                "kv_cache": {"shape": [1, 2, 4, 8], "num_decoder_layers": 1},
                "linear_cache": {
                    "layer_indices": [0],
                    "layers": [linear_layer_meta],
                },
            },
            "prefill_onnx": tmp_path / "prefill.onnx",
            "decode_onnx": tmp_path / "decode.onnx",
            "tokenizer": SimpleNamespace(pad_token_id=0, eos_token_id=0),
            "token_embedding": token_embedding,
            "pad_token_id": 0,
            "max_context_tokens": 8,
        },
    )

    engine = moe_inference.Qwen3_5MoeInference(
        str(meta_file), device="cpu", execution_device="cpu", resource_tight_mode=True
    )

    assert len(engine.past_key_caches) == 2
    assert len(engine.past_value_caches) == 2
    assert len(engine.past_conv_caches) == expected_conv_cache_count
    assert len(engine.past_recurrent_states) == 2


@pytest.mark.parametrize(
    ("input_names", "output_names", "expected_cache_ids"),
    [
        (
            ["past_conv_cache_0", "past_conv_cache_1", "past_conv_cache_2", "past_conv_cache_3"],
            ["conv_cache_out_0", "conv_cache_out_1", "conv_cache_out_2", "conv_cache_out_3"],
            [0, 1, 2, 3],
        ),
        (
            [
                "past_conv_cache_q_0",
                "past_conv_cache_k_0",
                "past_conv_cache_v_0",
                "past_conv_cache_q_1",
            ],
            [
                "conv_cache_out_q_0",
                "conv_cache_out_k_0",
                "conv_cache_out_v_0",
                "conv_cache_out_q_1",
            ],
            [0, 1, 2, 3],
        ),
        (
            ["past_conv_cache_0_batch_0", "past_conv_cache_0_batch_1"],
            ["conv_cache_out_0_batch_0", "conv_cache_out_0_batch_1"],
            [0, 1],
        ),
        (
            [
                "past_conv_cache_q_0_batch_0",
                "past_conv_cache_q_0_batch_1",
                "past_conv_cache_k_0_batch_0",
                "past_conv_cache_k_0_batch_1",
                "past_conv_cache_v_0_batch_0",
                "past_conv_cache_v_0_batch_1",
            ],
            [
                "conv_cache_out_q_0_batch_0",
                "conv_cache_out_q_0_batch_1",
                "conv_cache_out_k_0_batch_0",
                "conv_cache_out_k_0_batch_1",
                "conv_cache_out_v_0_batch_0",
                "conv_cache_out_v_0_batch_1",
            ],
            [0, 1, 2, 3, 4, 5],
        ),
    ],
)
def test_qwen3_5_moe_forward_accepts_and_updates_legacy_and_qkv_conv_cache_names(
    input_names, output_names, expected_cache_ids
):
    class FakeSession:
        def get_input_names(self):
            return list(input_names)

    engine = moe_inference.Qwen3_5MoeInference.__new__(moe_inference.Qwen3_5MoeInference)
    nn.Module.__init__(engine)
    engine._phase_prefill = True
    engine.batch_size = 2 if any("_batch_" in name for name in input_names) else 1
    engine.prefill_session = FakeSession()
    engine.decode_session = None
    engine._prefill_inputs_name = "inputs_embeds"
    engine._prefill_past_seq_name = "past_seq_length"
    engine._prefill_current_seq_name = "current_input_length"
    engine._prefill_mask_name = "linear_attn_mask"
    engine._device = torch.device("cpu")

    captured = {}
    output_tensors = {
        name: torch.full((1,), float(idx + 10)) for idx, name in enumerate(output_names)
    }

    def fake_run_hmonnx(session, feed):
        captured.update(feed)
        return (torch.ones(1, 1, 2),), {"logits": torch.ones(1, 1, 2), **output_tensors}

    engine._run_hmonnx = fake_run_hmonnx

    conv_caches = [torch.full((1,), float(idx)) for idx in range(6)]
    original_caches = list(conv_caches)

    logits = moe_inference.Qwen3_5MoeInference._forward(
        engine,
        inputs_embeds=torch.zeros(1, 1, 2),
        time_position_ids=torch.zeros(1, 1, dtype=torch.int32),
        hight_position_ids=torch.zeros(1, 1, dtype=torch.int32),
        width_position_ids=torch.zeros(1, 1, dtype=torch.int32),
        past_seq_length=torch.tensor([0], dtype=torch.int32),
        current_input_length=torch.tensor([1], dtype=torch.int32),
        linear_attn_mask=torch.ones(1, 1),
        past_key_caches=[],
        past_value_caches=[],
        past_conv_caches=conv_caches,
        past_recurrent_states=[],
    )

    assert tuple(logits.shape) == (1, 1, 2)
    assert [captured[name] for name in input_names] == [
        original_caches[idx] for idx in expected_cache_ids
    ]
    for list_idx, out_name in enumerate(output_names):
        assert torch.equal(conv_caches[list_idx], output_tensors[out_name])


def test_qwen3_5_moe_forward_slices_split_false_batch_inputs_and_updates_recurrent_cache():
    input_names = [
        "inputs_embeds_batch_0",
        "inputs_embeds_batch_1",
        "time_position_ids_batch_0",
        "time_position_ids_batch_1",
        "hight_position_ids_batch_0",
        "hight_position_ids_batch_1",
        "width_position_ids_batch_0",
        "width_position_ids_batch_1",
        "past_seq_length_batch_0",
        "past_seq_length_batch_1",
        "current_input_length_batch_0",
        "current_input_length_batch_1",
        "linear_attn_mask_batch_0",
        "linear_attn_mask_batch_1",
        "past_conv_cache_0_batch_0",
        "past_conv_cache_0_batch_1",
        "past_recurrent_state_0_batch_0",
        "past_recurrent_state_0_batch_1",
    ]

    class FakeSession:
        def get_input_names(self):
            return list(input_names)

    engine = moe_inference.Qwen3_5MoeInference.__new__(moe_inference.Qwen3_5MoeInference)
    nn.Module.__init__(engine)
    engine._phase_prefill = True
    engine.batch_size = 2
    engine.prefill_session = FakeSession()
    engine.decode_session = None
    engine._prefill_inputs_name = "inputs_embeds_batch_0"
    engine._prefill_past_seq_name = "past_seq_length_batch_0"
    engine._prefill_current_seq_name = "current_input_length_batch_0"
    engine._prefill_mask_name = "linear_attn_mask_batch_0"
    engine._device = torch.device("cpu")

    conv_outputs = {
        "conv_cache_out_0_batch_0": torch.full((1, 2, 3), 10.0),
        "conv_cache_out_0_batch_1": torch.full((1, 2, 3), 11.0),
    }
    recurrent_outputs = {
        "recurrent_state_out_0_batch_0": torch.full((1, 2, 4, 5), 20.0),
        "recurrent_state_out_0_batch_1": torch.full((1, 2, 4, 5), 21.0),
    }
    captured = {}

    def fake_run_hmonnx(session, feed):
        captured.update(feed)
        return (), {
            "logits_batch_0": torch.full((1, 1, 3), 1.0),
            "logits_batch_1": torch.full((1, 1, 3), 2.0),
            **conv_outputs,
            **recurrent_outputs,
        }

    engine._run_hmonnx = fake_run_hmonnx

    conv_caches = [
        torch.zeros(1, 2, 3),
        torch.ones(1, 2, 3),
    ]
    recurrent_caches = [
        torch.zeros(1, 2, 4, 5),
        torch.ones(1, 2, 4, 5),
    ]

    logits = moe_inference.Qwen3_5MoeInference._forward(
        engine,
        inputs_embeds=torch.arange(12, dtype=torch.float32).reshape(2, 3, 2),
        time_position_ids=torch.arange(6, dtype=torch.int32).reshape(2, 3),
        hight_position_ids=torch.arange(6, dtype=torch.int32).reshape(2, 3) + 10,
        width_position_ids=torch.arange(6, dtype=torch.int32).reshape(2, 3) + 20,
        past_seq_length=torch.tensor([0, 2], dtype=torch.int32),
        current_input_length=torch.tensor([3, 1], dtype=torch.int32),
        linear_attn_mask=torch.ones(2, 3),
        past_key_caches=[],
        past_value_caches=[],
        past_conv_caches=conv_caches,
        past_recurrent_states=recurrent_caches,
    )

    assert tuple(logits.shape) == (2, 1, 3)
    for name in input_names:
        assert captured[name].shape[0] == 1, name
    assert torch.equal(captured["inputs_embeds_batch_1"], torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)[1:2])
    assert torch.equal(captured["past_seq_length_batch_1"], torch.tensor([2], dtype=torch.int32))
    assert torch.equal(conv_caches[0], conv_outputs["conv_cache_out_0_batch_0"])
    assert torch.equal(conv_caches[1], conv_outputs["conv_cache_out_0_batch_1"])
    assert torch.equal(recurrent_caches[0], recurrent_outputs["recurrent_state_out_0_batch_0"])
    assert torch.equal(recurrent_caches[1], recurrent_outputs["recurrent_state_out_0_batch_1"])
