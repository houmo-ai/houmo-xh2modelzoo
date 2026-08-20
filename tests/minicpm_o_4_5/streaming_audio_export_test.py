from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import onnx
import pytest
import torch
import torch.nn as nn


def test_streaming_input_capture_uses_processor_emitted_frame_shapes() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.export_audio import (
        capture_streaming_audio_inputs,
    )

    class Processor:
        def __init__(self) -> None:
            self.calls: list[tuple[int, bool]] = []
            self.chunk_index = 0

        def set_streaming_mode(self, **kwargs) -> None:
            assert kwargs == {
                "mode": "exact",
                "chunk_ms": 1000,
                "first_chunk_ms": 1035,
                "cnn_redundancy_ms": 20,
                "enable_sliding_window": False,
            }

        def reset_streaming(self) -> None:
            self.chunk_index = 0

        def get_streaming_chunk_size(self) -> int:
            return (16560, 16000)[self.chunk_index]

        def process_audio_streaming(self, audio, *, return_batch_feature: bool):
            assert return_batch_feature is True
            frames = (107, 103)[self.chunk_index]
            self.calls.append((len(audio), return_batch_feature))
            self.chunk_index += 1
            return SimpleNamespace(
                audio_features=torch.zeros((1, 80, frames), dtype=torch.float32),
                audio_feature_lens=[torch.tensor([frames])],
            )

    processor = Processor()

    prefill, decode = capture_streaming_audio_inputs(
        processor,
        {
            "first_chunk_ms": 1035,
            "chunk_ms": 1000,
            "cnn_redundancy_ms": 20,
            "sample_rate": 16000,
            "prefix_overlap_first": 0,
            "prefix_overlap_later": 2,
            "suffix_overlap": 2,
            "cache_capacity": 1500,
        },
    )

    assert processor.calls == [(16560, True), (16000, True)]
    assert prefill["input_features"].shape == (1, 80, 107)
    assert decode["input_features"].shape == (1, 80, 103)
    assert int(prefill["valid_mel_length"]) == 107
    assert int(decode["valid_mel_length"]) == 103
    assert int(prefill["current_input_length"]) == 53
    assert int(decode["current_input_length"]) == 50
    assert prefill["attention_mask"].shape == (1, 1, 53, 1500)
    assert decode["attention_mask"].shape == (1, 1, 50, 1500)
    assert torch.all(prefill["attention_mask"][..., :53] == 0)
    assert torch.isneginf(prefill["attention_mask"][..., 53:]).all()
    assert set(prefill) == {
        "input_features",
        "valid_mel_length",
        "past_seq_length",
        "current_input_length",
        "attention_mask",
    }
    assert set(decode) == {
        "input_features",
        "valid_mel_length",
        "past_seq_length",
        "current_input_length",
        "attention_mask",
    }


def test_streaming_input_capture_rejects_non_16khz_sample_rate() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.export_audio import (
        StreamingAudioConfigError,
        capture_streaming_audio_inputs,
    )

    with pytest.raises(StreamingAudioConfigError, match="sample_rate=8000"):
        capture_streaming_audio_inputs(SimpleNamespace(), {"sample_rate": 8000})


def test_audio_export_records_offline_and_streaming_graph_contracts(tmp_path, monkeypatch) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import export_audio

    class Processor:
        def __init__(self) -> None:
            self.chunk_index = 0

        def set_streaming_mode(self, **kwargs) -> None:
            del kwargs

        def reset_streaming(self) -> None:
            self.chunk_index = 0

        def get_streaming_chunk_size(self) -> int:
            return (16560, 16000)[self.chunk_index]

        def process_audio_streaming(self, audio, *, return_batch_feature: bool):
            del audio, return_batch_feature
            frames = (107, 103)[self.chunk_index]
            self.chunk_index += 1
            return SimpleNamespace(
                audio_features=torch.zeros((1, 80, frames), dtype=torch.float32),
                audio_feature_lens=[torch.tensor([frames])],
            )

    class Model:
        def __init__(self) -> None:
            self.host = SimpleNamespace(
                processor=Processor(),
                apm=SimpleNamespace(
                    config=SimpleNamespace(
                        encoder_layers=24,
                        encoder_attention_heads=16,
                        d_model=1024,
                        max_source_positions=1500,
                    )
                ),
                audio_encoder_layer=-1,
            )
            self.streaming_frames: list[int] = []
            self.past_key_caches = [torch.zeros((1, 16, 1500, 64)) for _ in range(24)]
            self.past_value_caches = [torch.zeros((1, 16, 1500, 64)) for _ in range(24)]

        def get_hf_model(self, device_map: str):
            assert device_map == "cpu"
            return self.host

        def init_wrap_model(
            self,
            host,
            *,
            streaming: bool = False,
            prefix_extra_frames: int = 0,
            suffix_extra_frames: int = 0,
            input_frame_capacity: int | None = None,
        ) -> None:
            assert host is self.host
            if streaming:
                self.streaming_frames.extend([prefix_extra_frames, suffix_extra_frames, input_frame_capacity])

    models = [Model(), Model(), Model(), Model(), Model(), Model()]
    built_models: list[Model] = []

    def build_model(*_args):
        model = models.pop(0)
        built_models.append(model)
        return model

    monkeypatch.setattr(export_audio, "build_component_model", build_model)
    exported: list[str] = []

    def fake_quantize(model, data, output_dir: Path, prefix: str, device: str) -> str:
        del data, device
        expected_model_indexes = (2, 3, 4, 5)
        # exported already contains the ONNX-direct main graph entry.
        assert model is built_models[expected_model_indexes[len(exported) - 1]]
        exported.append(prefix)
        return str(output_dir / f"{prefix}.onnx")

    monkeypatch.setattr(export_audio, "quantize_and_export", fake_quantize)

    def fake_main_onnx_direct(
        *, host, component_cfg, target_device, output_dir, device, model_name="minicpm_o_4_5"
    ) -> str:
        del host, component_cfg, target_device, device, model_name
        exported.append("minicpm_o_4_5_audio_offline_XH2a_w8a8_sefp")
        return str(output_dir / "minicpm_o_4_5_audio_offline_XH2a_w8a8_sefp.onnx")

    monkeypatch.setattr(export_audio, "_export_audio_main_onnx_direct", fake_main_onnx_direct)
    component = {
        "model_type": "MiniCPMO45AudioModel",
        "quant_type": "w8a8_sefp",
        "static_batch_size": 4,
        "max_audio_frames": 3000,
        "input_names": ["input_features", "audio_attention_mask"],
        "output_names": ["audio_embeddings"],
        "streaming": {
            "enabled": True,
            "first_chunk_ms": 1035,
            "chunk_ms": 1000,
            "cnn_redundancy_ms": 20,
            "sample_rate": 16000,
            "prefix_overlap_first": 0,
            "prefix_overlap_later": 2,
            "suffix_overlap": 2,
            "cache_capacity": 1500,
            "session_frames": 100,
        },
    }

    result = export_audio.export_minicpm_o_4_5_audio(
        work_dir=tmp_path,
        model_dir="/models/MiniCPM-o-4_5",
        component_cfg=component,
        target_device="XH2a",
        device="cpu",
    )

    assert exported == [
        "minicpm_o_4_5_audio_offline_XH2a_w8a8_sefp",
        "minicpm_o_4_5_audio_stream_prefill_XH2a_w8a8_sefp",
        "minicpm_o_4_5_audio_stream_decode_XH2a_w8a8_sefp",
        "minicpm_o_4_5_audio_session_prefill_XH2a_w8a8_sefp",
        "minicpm_o_4_5_audio_session_decode_XH2a_w8a8_sefp",
    ]
    assert built_models[2].streaming_frames == [0, 2, 107]
    assert built_models[3].streaming_frames == [2, 2, 103]
    assert built_models[4].streaming_frames == [0, 0, 100]
    assert built_models[5].streaming_frames == [0, 0, 100]
    assert set(result["graphs"]) == {
        "main",
        "stream_prefill",
        "stream_decode",
        "session_prefill",
        "session_decode",
    }
    assert result["stream_prefill_frames"] == 107
    assert result["stream_decode_frames"] == 103
    assert result["session_prefill_frames"] == 100
    assert result["session_decode_frames"] == 100
    assert result["num_hidden_layers"] == 24
    assert result["kv_cache_shape"] == [1, 16, 1500, 64]
    assert result["cache_capacity"] == 1500
    assert result["audio_encoder_layer"] == -1
    assert result["selected_hidden_state_index"] == -1
    assert result["prefix_overlap_first"] == 0
    assert result["prefix_overlap_later"] == 2
    assert result["suffix_overlap"] == 2
    assert result["pool_step"] == 5
    expected_inputs = [
        "input_features",
        "valid_mel_length",
        "past_seq_length",
        "current_input_length",
        "attention_mask",
    ] + [
        *(f"past_k_cache_{index}" for index in range(24)),
        *(f"past_v_cache_{index}" for index in range(24)),
    ]
    expected_outputs = ["audio_hidden_states"] + [
        *(f"present_k_cache_{index}" for index in range(24)),
        *(f"present_v_cache_{index}" for index in range(24)),
    ]
    assert result["graph_contracts"]["stream_prefill"] == {
        "input_names": expected_inputs,
        "output_names": expected_outputs,
        "prefix_overlap": 0,
        "suffix_overlap": 2,
        "current_length_rule": "ceil(valid_mel_length / 2) - ceil(prefix_overlap / 2) - ceil(suffix_overlap / 2)",
        "cache_capacity": 1500,
        "kv_cache_shape": [1, 16, 1500, 64],
        "selected_hidden_state_index": -1,
        "projection_in_graph": True,
        "pooling_in_graph": True,
    }
    assert result["graph_contracts"]["stream_decode"]["prefix_overlap"] == 2
    assert result["graph_contracts"]["session_prefill"]["prefix_overlap"] == 0
    assert result["graph_contracts"]["session_prefill"]["suffix_overlap"] == 0
    assert result["graph_contracts"]["session_decode"]["prefix_overlap"] == 0
    assert result["graph_contracts"]["session_decode"]["suffix_overlap"] == 0
    for role in ("session_prefill", "session_decode"):
        assert result["graph_contracts"][role] == {
            "input_names": expected_inputs,
            "output_names": expected_outputs,
            "prefix_overlap": 0,
            "suffix_overlap": 0,
            "current_length_rule": "ceil(valid_mel_length / 2)",
            "cache_capacity": 1500,
            "kv_cache_shape": [1, 16, 1500, 64],
            "selected_hidden_state_index": -1,
            "projection_in_graph": True,
            "pooling_in_graph": True,
            "output_valid_region": "audio_hidden_states[:, :(current_input_length - pool_step) // pool_step + 1, :]",
        }


def test_production_streaming_attention_onnx_cache_outputs_depend_on_past_and_projection(tmp_path) -> None:
    from transformers.models.whisper.configuration_whisper import WhisperConfig
    from transformers.models.whisper.modeling_whisper import WhisperAttention

    from xhmodel_merak.xh_llm.register import XHLLM_TRACEABLE_MODULES
    from xhmodel_merak.xh_llm.wrap_model import convert_module
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5._audio_model_impl import _WhisperAttention
    from xhquant.api import ConfigDict

    class ProductionStreamingAttentionGraph(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            config = WhisperConfig(d_model=8, encoder_attention_heads=2, encoder_layers=1, encoder_ffn_dim=16)
            self.attention = WhisperAttention(8, 2, layer_idx=0, config=config)
            self.attention = convert_module(self.attention, ConfigDict(), XHLLM_TRACEABLE_MODULES)
            self.attention.half()

        def forward(self, hidden_states, past_seq_length, current_input_length, past_k_cache, past_v_cache):
            output = self.attention(
                hidden_states=hidden_states,
                attention_mask=torch.zeros((1, 1, 2, 4), dtype=hidden_states.dtype),
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                past_k_cache=past_k_cache,
                past_v_cache=past_v_cache,
            )
            return output[0], output[-2], output[-1]

    model = ProductionStreamingAttentionGraph().eval()
    assert isinstance(model.attention, _WhisperAttention)
    inputs = (
        torch.ones((1, 2, 8), dtype=torch.float16),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        torch.zeros((1, 2, 4, 4), dtype=torch.float16),
        torch.zeros((1, 2, 4, 4), dtype=torch.float16),
    )
    graph_path = tmp_path / "production_streaming_attention.onnx"
    input_names = ["input_features", "past_seq_length", "current_input_length", "past_k_cache_0", "past_v_cache_0"]
    output_names = ["audio_hidden_states", "present_k_cache_0", "present_v_cache_0"]

    torch.onnx.export(
        model,
        inputs,
        graph_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
    )
    graph = onnx.load(graph_path).graph

    assert [value.name for value in graph.input] == input_names
    assert [value.name for value in graph.output] == output_names
    producer = {output: node for node in graph.node for output in node.output}

    def dependencies(output_name: str) -> set[str]:
        pending = [output_name]
        visited: set[str] = set()
        graph_inputs = {value.name for value in graph.input}
        found_inputs: set[str] = set()
        while pending:
            value = pending.pop()
            if value in visited:
                continue
            visited.add(value)
            if value in graph_inputs:
                found_inputs.add(value)
                continue
            node = producer.get(value)
            if node is not None:
                pending.extend(node.input)
        return found_inputs

    assert dependencies("present_k_cache_0") >= {"past_k_cache_0", "input_features"}
    assert dependencies("present_v_cache_0") >= {"past_v_cache_0", "input_features"}


def test_streaming_wrapper_builds_query_sized_attention_mask() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5._audio_streaming import build_streaming_attention_mask

    mask = build_streaming_attention_mask(
        past_seq_length=torch.tensor([3], dtype=torch.int32),
        current_input_length=torch.tensor([4], dtype=torch.int32),
        query_capacity=4,
        cache_capacity=8,
        reference=torch.zeros((), dtype=torch.float32),
    )

    assert mask.shape == (1, 1, 4, 8)
    assert torch.all(mask[:, :, :, :7] == 0)
    assert torch.isneginf(mask[:, :, :, 7:]).all()


def test_streaming_export_adapter_exposes_flat_cache_inputs() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5._audio_streaming import (
        StreamingAudioExportAdapter,
    )

    class Encoder(nn.Module):
        def forward_streaming(
            self,
            input_features,
            valid_mel_length,
            past_seq_length,
            current_input_length,
            attention_mask,
            past_key_caches,
            past_value_caches,
        ):
            del valid_mel_length, past_seq_length, current_input_length, attention_mask
            return (
                input_features,
                *past_key_caches,
                *past_value_caches,
            )

    adapter = StreamingAudioExportAdapter(Encoder(), num_hidden_layers=2)
    key_caches = [torch.zeros((1, 2, 8, 4)) for _ in range(2)]
    value_caches = [torch.ones((1, 2, 8, 4)) for _ in range(2)]

    outputs = adapter(
        torch.ones((1, 80, 7)),
        torch.tensor([7], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([3], dtype=torch.int32),
        torch.zeros((1, 1, 3, 8)),
        *key_caches,
        *value_caches,
    )

    assert len(outputs) == 5
    assert outputs[1] is key_caches[0]
    assert outputs[3] is value_caches[0]


def test_streaming_export_adapter_has_named_cache_placeholders() -> None:
    import inspect

    from xhmodel_merak.xh_llm.models.minicpm_o_4_5._audio_streaming import (
        StreamingAudioExportAdapter,
    )

    adapter = StreamingAudioExportAdapter(nn.Identity(), num_hidden_layers=2)
    parameters = list(inspect.signature(adapter.forward).parameters)

    assert parameters == [
        "input_features",
        "valid_mel_length",
        "past_seq_length",
        "current_input_length",
        "attention_mask",
        "past_k_cache_0",
        "past_k_cache_1",
        "past_v_cache_0",
        "past_v_cache_1",
    ]


def test_streaming_encoder_uses_static_position_capacity_for_fx_trace() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5._audio_streaming import (
        streaming_position_ids,
    )

    positions = streaming_position_ids(
        output_capacity=52,
        past_seq_length=torch.tensor([7], dtype=torch.int32),
        reference=torch.zeros((), dtype=torch.float16),
    )

    assert positions.shape == (52,)
    assert positions[:3].tolist() == [7, 8, 9]
