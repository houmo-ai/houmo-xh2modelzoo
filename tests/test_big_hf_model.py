"""Regression tests for big-model graph placeholder helpers."""

from __future__ import annotations

import torch
import torch.nn as nn
from safetensors.torch import save_file

from xhmodel_merak.xh_llm.big_hf_model_helper import (
    BigHFModelExportHelper,
    WeightMapping,
    _FlattenedInputAdapter,
    _merge_call_args_kwargs_as_args,
    _placeholder_input_fake_tensors,
)


class FakeDecoderLayer(nn.Module):
    def forward(
        self,
        hidden_states,
        past_seq_length=None,
        current_input_length=None,
        position_embeddings=None,
        linear_attn_mask=None,
        past_k_cache=None,
        past_v_cache=None,
        past_conv_cache=None,
        past_recurrent_state=None,
    ):
        return hidden_states


class FakeNode:
    def __init__(self, value):
        self.meta = {"val": value}


class TinyMetaDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 3, device="meta")
        self.register_buffer("scale", torch.empty(3, device="meta"))


class RecordingDecoderLayer(nn.Module):
    def forward(self, *args):
        return args


def test_merge_call_args_kwargs_as_args_uses_forward_parameter_order():
    module = FakeDecoderLayer()

    merged_args = _merge_call_args_kwargs_as_args(
        module,
        args=("hidden_states",),
        kwargs={
            "past_v_cache": "past_v",
            "linear_attn_mask": "linear_mask",
            "position_embeddings": ("cos", "sin"),
            "past_seq_length": None,
            "past_conv_cache": ["conv_q", "conv_k", "conv_v"],
            "past_recurrent_state": "recurrent",
        },
    )

    assert merged_args == (
        "hidden_states",
        None,
        None,
        ("cos", "sin"),
        "linear_mask",
        None,
        "past_v",
        ["conv_q", "conv_k", "conv_v"],
        "recurrent",
    )


def test_placeholder_input_fake_tensors_preserves_positional_tensor_order():
    hidden_states = torch.empty(1, 2, 3)
    cos = torch.empty(1, 1, 2, 3)
    sin = torch.empty(1, 1, 2, 3)
    current_input_length = torch.empty(1, dtype=torch.int32)

    fake_tensors = _placeholder_input_fake_tensors(
        (
            FakeNode(hidden_states),
            FakeNode(None),
            FakeNode(current_input_length),
            FakeNode((cos, sin)),
            "non_tensor_constant",
        )
    )

    assert fake_tensors == [hidden_states, current_input_length, cos, sin]


def test_flattened_input_adapter_rebuilds_nested_decoder_layer_args():
    hidden_states = torch.empty(1, 2, 3)
    cos = torch.empty(1, 1, 2, 3)
    sin = torch.empty(1, 1, 2, 3)
    linear_attn_mask = torch.empty(1, 2)
    conv_q = torch.empty(1, 4)
    conv_k = torch.empty(1, 4)
    conv_v = torch.empty(1, 4)

    args_template = (
        FakeNode(hidden_states),
        FakeNode(None),
        FakeNode(None),
        FakeNode((cos, sin)),
        FakeNode(linear_attn_mask),
        FakeNode(None),
        FakeNode(None),
        FakeNode([conv_q, conv_k, conv_v]),
    )
    fake_tensors = _placeholder_input_fake_tensors(args_template)
    adapter = _FlattenedInputAdapter(RecordingDecoderLayer(), args_template)

    assert fake_tensors == [hidden_states, cos, sin, linear_attn_mask, [conv_q, conv_k, conv_v]]

    rebuilt_args = adapter(*fake_tensors)

    assert rebuilt_args[0] is hidden_states
    assert rebuilt_args[1] is None
    assert rebuilt_args[2] is None
    assert rebuilt_args[3] == (cos, sin)
    assert rebuilt_args[4] is linear_attn_mask
    assert rebuilt_args[5] is None
    assert rebuilt_args[6] is None
    assert rebuilt_args[7] is fake_tensors[4]


def test_load_module_from_safetensor_materializes_meta_params_and_buffers(tmp_path):
    safetensor_path = tmp_path / "model.safetensors"
    tensors = {
        "model.layers.0.proj.weight": torch.arange(6, dtype=torch.float32).reshape(3, 2),
        "model.layers.0.proj.bias": torch.arange(3, dtype=torch.float32),
        "model.layers.0.scale": torch.ones(3, dtype=torch.float32),
    }
    save_file(tensors, safetensor_path)

    big_model = BigHFModelExportHelper.__new__(BigHFModelExportHelper)
    big_model._weight_mapping = WeightMapping()
    big_model._weight_mapping.weight_map = {name: str(safetensor_path) for name in tensors}
    layer = TinyMetaDecoderLayer()

    big_model._load_module_from_safetensor(layer, "model.layers.0")

    assert layer.proj.weight.device.type == "cpu"
    assert layer.proj.bias.device.type == "cpu"
    assert layer.scale.device.type == "cpu"
    torch.testing.assert_close(layer.proj.weight, tensors["model.layers.0.proj.weight"])
    torch.testing.assert_close(layer.proj.bias, tensors["model.layers.0.proj.bias"])
    torch.testing.assert_close(layer.scale, tensors["model.layers.0.scale"])
