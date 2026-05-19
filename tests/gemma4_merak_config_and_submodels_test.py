from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from xhmodel_merak.xh_llm.types import CacheList


MODEL_DIR = Path("/data01/datasets/gemma-4-E4B")


def test_gemma4_model_config_builds_visual_and_audio_subconfigs():
    from xhmodel_merak.xh_llm.models.gemma4e.xh_gemma4_config import (
        XHGemma4AudioConfig,
        XHGemma4ModelConfig,
        XHGemma4VisualConfig,
    )

    config = XHGemma4ModelConfig(
        model_name="gemma4_e",
        model_type="Gemma4ForConditionalGeneration",
        hf_model=str(MODEL_DIR),
        visual_config={},
        audio_config={},
    )

    assert isinstance(config.visual_config, XHGemma4VisualConfig)
    assert isinstance(config.audio_config, XHGemma4AudioConfig)
    assert config.visual_config.model_name == "gemma4_e_visual"
    assert config.audio_config.model_name == "gemma4_e_audio"
    assert config.visual_config.hf_model == str(MODEL_DIR)
    assert config.audio_config.hf_model == str(MODEL_DIR)
    assert config.visual_config.patch_size == 16
    assert config.visual_config.image_seq_length == 280
    assert config.visual_config.pooling_kernel_size == 3
    assert config.visual_config.export_mode == "full"
    assert config.audio_config.sampling_rate == 16000
    assert config.audio_config.feature_size == 128


def test_gemma4_model_config_supports_compact_visual_export_via_auto_llm_config():
    from xhmodel_merak.xh_llm import AutoLLMConfig
    from xhmodel_merak.xh_llm.models.gemma4e.xh_gemma4_config import XHGemma4ModelConfig

    config = AutoLLMConfig.from_pretrained(
        {
            "model_name": "gemma4_e",
            "model_type": "Gemma4ForConditionalGeneration",
            "hf_model": str(MODEL_DIR),
            "visual_config": {
                "export_mode": "compact",
            },
            "audio_config": {},
        }
    )

    assert isinstance(config, XHGemma4ModelConfig)
    assert config.visual_config.export_mode == "compact"
    assert config.visual_config.max_size_w == 448
    assert config.visual_config.max_size_h == 448
    assert config.visual_config.image_seq_length == 256
    assert config.visual_config.pooling_kernel_size == 1


def test_gemma4_vision_submodel_contracts():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_vision_model import XHGemma4VisionModel
    from xhmodel_merak.xh_llm.models.gemma4e.xh_gemma4_config import XHGemma4VisualConfig

    config = XHGemma4VisualConfig(model_name="gemma4_visual", hf_model=str(MODEL_DIR))
    model = XHGemma4VisionModel(config)

    dummy_inputs = model.get_dummy_inputs()
    assert set(dummy_inputs) == {"pixel_values", "image_position_ids"}
    assert dummy_inputs["pixel_values"].ndim == 3
    assert dummy_inputs["pixel_values"].shape[1] == 2304
    assert dummy_inputs["image_position_ids"].shape[1] == 2304
    assert dummy_inputs["image_position_ids"].shape[-1] == 2
    processed_inputs = model.get_data_preprocessor()(dummy_inputs)
    assert len(processed_inputs) == 2
    assert processed_inputs[0].shape == dummy_inputs["pixel_values"].shape
    assert torch.equal(processed_inputs[1], dummy_inputs["image_position_ids"])

    export_cfg = model.get_export_cfg()
    assert export_cfg["input_names"] == ["pixel_values", "image_position_ids"]
    assert export_cfg["output_names"] == ["image_embeds", "image_embeds_mask"]

    meta = model.create_export_metadata("work_dirs/gemma4_visual_meta")
    assert meta.image_size_h == config.max_size_h
    assert meta.image_size_w == config.max_size_w
    assert meta.patch_size == config.patch_size
    assert meta.image_seq_length == config.image_seq_length
    assert meta.onnx is None
    assert meta.output_scale == pytest.approx(768 ** -0.25)


def test_gemma4_compact_vision_submodel_contracts():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_vision_model import XHGemma4VisionModel
    from xhmodel_merak.xh_llm.models.gemma4e.xh_gemma4_config import XHGemma4VisualConfig

    config = XHGemma4VisualConfig(model_name="gemma4_visual", hf_model=str(MODEL_DIR), export_mode="compact")
    model = XHGemma4VisionModel(config)

    dummy_inputs = model.get_dummy_inputs()
    assert set(dummy_inputs) == {"image"}
    assert dummy_inputs["image"].shape[1] == 256
    processed_inputs = model.get_data_preprocessor()(dummy_inputs)
    assert len(processed_inputs) == 1
    assert processed_inputs[0].shape == dummy_inputs["image"].shape

    export_cfg = model.get_export_cfg()
    assert export_cfg["input_names"] == ["pixel_values"]
    assert export_cfg["output_names"] == ["image_embeds"]

    meta = model.create_export_metadata("work_dirs/gemma4_visual_meta")
    assert meta.image_size_h == 448
    assert meta.image_size_w == 448
    assert meta.patch_size == config.patch_size
    assert meta.image_seq_length == 256
    assert meta.export_mode == "compact"
    assert meta.num_image_tokens == 256
    assert meta.onnx is None


def test_gemma4_llm_prefill_dummy_inputs_include_visual_embeds_for_calibration():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_llm_model import XHGemma4Model
    from xhmodel_merak.xh_llm.models.gemma4e.xh_gemma4_config import XHGemma4ModelConfig

    config = XHGemma4ModelConfig(
        model_name="gemma4_e",
        model_type="Gemma4ForConditionalGeneration",
        hf_model=str(MODEL_DIR),
        prefill_chunk_length=512,
        visual_config={},
        audio_config=None,
    )
    model = XHGemma4Model(config)
    model.config.image_token_id = 258880
    model.config.audio_token_id = 258881
    model.config.video_token_id = 258884

    dummy_inputs = model.get_prefill_dummy_inputs()

    assert "image_embeds" in dummy_inputs
    assert "mm_token_type_ids" in dummy_inputs
    image_token_count = int((dummy_inputs["input_ids"] == model.config.image_token_id).sum().item())
    assert image_token_count > 0
    assert dummy_inputs["image_embeds"].shape == (image_token_count, 2560)
    assert int((dummy_inputs["mm_token_type_ids"] == 1).sum().item()) == image_token_count
    assert dummy_inputs["input_ids"].shape[1] <= config.prefill_chunk_length


def test_gemma4_visual_export_hmonnx_preserves_plain_onnx_sidecar(monkeypatch, tmp_path):
    from xhmodel_merak.xh_llm.base_vision_model import BaseVisionModel
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_vision_model import XHGemma4VisionModel
    from xhmodel_merak.xh_llm.models.gemma4e.xh_gemma4_config import XHGemma4VisualConfig

    config = XHGemma4VisualConfig(model_name="gemma4_visual", hf_model=str(MODEL_DIR), export_mode="compact")
    model = XHGemma4VisionModel(config)

    def _fake_export(self, output_dir: str):
        self.config.work_dir = str(output_dir)
        output_path = Path(output_dir) / "gemma4_visual_hm.onnx"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        onnx_dir = Path(output_dir) / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        (onnx_dir / "gemma4_visual.onnx").write_bytes(b"fake-onnx")
        (onnx_dir / "gemma4_visual.onnx.data").write_bytes(b"fake-external-data")
        output_path.touch()
        return str(output_path)

    monkeypatch.setattr(BaseVisionModel, "_export_hmonnx", _fake_export)

    export_dir = tmp_path / "final_visual"
    meta = model.export_hmonnx(str(export_dir))

    assert meta.hmonnx == str(export_dir / "gemma4_visual_hm.onnx")
    assert meta.onnx == str(export_dir / "onnx" / "gemma4_visual.onnx")
    assert (export_dir / "onnx" / "gemma4_visual.onnx").exists()
    assert (export_dir / "onnx" / "gemma4_visual.onnx.data").exists()


def test_gemma4_visual_plain_onnx_sidecar_reuses_cached_onnx_after_quantized(tmp_path):
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_vision_model import XHGemma4VisionModel
    from xhmodel_merak.xh_llm.models.gemma4e.xh_gemma4_config import XHGemma4VisualConfig
    from xhmodel_merak.xh_llm.types import LLMModelState

    config = XHGemma4VisualConfig(model_name="gemma4_visual", hf_model=str(MODEL_DIR))
    model = XHGemma4VisionModel(config)
    model.work_dir = str(tmp_path / "cached_visual")
    cached_onnx_dir = Path(model.work_dir) / "onnx"
    cached_onnx_dir.mkdir(parents=True)
    (cached_onnx_dir / "gemma4_visual.onnx").write_bytes(b"cached-onnx")
    (cached_onnx_dir / "gemma4_visual_external_data").write_bytes(b"cached-external-data")

    model._wrap_model = None
    model._state = LLMModelState.QUANTED_ALIGNED

    export_dir = tmp_path / "final_visual"
    onnx_path = model._export_plain_onnx_sidecar(str(export_dir))

    assert onnx_path == export_dir / "onnx" / "gemma4_visual.onnx"
    assert onnx_path.read_bytes() == b"cached-onnx"
    assert (export_dir / "onnx" / "gemma4_visual_external_data").read_bytes() == b"cached-external-data"


def test_gemma4_visual_export_hmonnx_reuses_preexport_cached_sidecar_after_work_dir_changes(
    monkeypatch, tmp_path
):
    from xhmodel_merak.xh_llm.base_vision_model import BaseVisionModel
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_vision_model import XHGemma4VisionModel
    from xhmodel_merak.xh_llm.models.gemma4e.xh_gemma4_config import XHGemma4VisualConfig
    from xhmodel_merak.xh_llm.types import LLMModelState

    config = XHGemma4VisualConfig(model_name="gemma4_visual", hf_model=str(MODEL_DIR))
    model = XHGemma4VisionModel(config)
    model.work_dir = str(tmp_path / "frontend_cache")
    cached_onnx_dir = Path(model.work_dir) / "onnx"
    cached_onnx_dir.mkdir(parents=True)
    (cached_onnx_dir / "gemma4_visual.onnx").write_bytes(b"cached-onnx")
    (cached_onnx_dir / "gemma4_visual_external_data").write_bytes(b"cached-external-data")

    def _fake_export(self, output_dir: str):
        self.config.work_dir = str(output_dir)
        self._state = LLMModelState.QUANTED_ALIGNED
        output_path = Path(output_dir) / "gemma4_visual_hm.onnx"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch()
        return str(output_path)

    monkeypatch.setattr(BaseVisionModel, "_export_hmonnx", _fake_export)

    export_dir = tmp_path / "final_visual"
    meta = model.export_hmonnx(str(export_dir))

    assert meta.onnx == str(export_dir / "onnx" / "gemma4_visual.onnx")
    assert (export_dir / "onnx" / "gemma4_visual.onnx").read_bytes() == b"cached-onnx"
    assert (export_dir / "onnx" / "gemma4_visual_external_data").read_bytes() == b"cached-external-data"


def test_gemma4_audio_submodel_contracts():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_audio_model import XHGemma4AudioModel
    from xhmodel_merak.xh_llm.models.gemma4e.xh_gemma4_config import XHGemma4AudioConfig

    config = XHGemma4AudioConfig(
        model_name="gemma4_audio",
        hf_model=str(MODEL_DIR),
        input_feature_length=400,
    )
    model = XHGemma4AudioModel(config)

    dummy_inputs = model.get_dummy_inputs()
    assert set(dummy_inputs) == {"input_features", "input_features_mask"}
    assert dummy_inputs["input_features"].ndim == 3
    assert dummy_inputs["input_features_mask"].ndim == 2
    assert dummy_inputs["input_features"].shape[1] == config.input_feature_length
    assert dummy_inputs["input_features_mask"].shape[1] == config.input_feature_length
    processed_inputs = model.get_data_preprocessor()(dummy_inputs)
    assert len(processed_inputs) == 2
    assert processed_inputs[0].shape == dummy_inputs["input_features"].shape
    assert torch.equal(processed_inputs[1], dummy_inputs["input_features_mask"])

    export_cfg = model.get_export_cfg()
    assert export_cfg["input_names"] == ["input_features", "input_features_mask"]
    assert export_cfg["output_names"] == ["audio_embeds", "audio_embeds_mask"]

    meta = model.create_export_metadata("work_dirs/gemma4_audio_meta")
    assert meta.sampling_rate == config.sampling_rate
    assert meta.feature_size == config.feature_size
    assert meta.input_feature_length == config.input_feature_length
    assert meta.onnx is None


def test_gemma4_audio_export_hmonnx_preserves_plain_onnx_sidecar(monkeypatch, tmp_path):
    from xhmodel_merak.xh_llm.base_vision_model import BaseVisionModel
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_audio_model import XHGemma4AudioModel
    from xhmodel_merak.xh_llm.models.gemma4e.xh_gemma4_config import XHGemma4AudioConfig

    config = XHGemma4AudioConfig(
        model_name="gemma4_audio",
        hf_model=str(MODEL_DIR),
        input_feature_length=400,
    )
    model = XHGemma4AudioModel(config)

    def _fake_export(self, output_dir: str):
        self.config.work_dir = str(output_dir)
        output_path = Path(output_dir) / "gemma4_audio_hm.onnx"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        onnx_dir = Path(output_dir) / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        (onnx_dir / "gemma4_audio.onnx").write_bytes(b"fake-onnx")
        (onnx_dir / "gemma4_audio_external_data").write_bytes(b"fake-external-data")
        output_path.touch()
        return str(output_path)

    monkeypatch.setattr(BaseVisionModel, "_export_hmonnx", _fake_export)

    export_dir = tmp_path / "final_audio"
    meta = model.export_hmonnx(str(export_dir))

    assert meta.hmonnx == str(export_dir / "gemma4_audio_hm.onnx")
    assert meta.onnx == str(export_dir / "onnx" / "gemma4_audio.onnx")
    assert (export_dir / "onnx" / "gemma4_audio.onnx").exists()
    assert (export_dir / "onnx" / "gemma4_audio_external_data").exists()


def test_gemma4_audio_export_bridge_preserves_output_mask():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_audio_model import _Gemma4AudioExportBridge

    hidden_states = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    output_mask = torch.tensor([[True, False, True]])

    class _FakeAudioTower:
        def __call__(self, input_features, attention_mask):
            return hidden_states.clone(), output_mask.clone()

    class _FakeEmbedAudio:
        def __call__(self, inputs_embeds):
            return (inputs_embeds + 1.0,)

    hf_model = SimpleNamespace(
        model=SimpleNamespace(audio_tower=_FakeAudioTower(), embed_audio=_FakeEmbedAudio()),
    )
    bridge = _Gemma4AudioExportBridge(hf_model)

    audio_embeds, audio_embeds_mask = bridge(torch.randn(1, 4, 8), torch.ones(1, 4, dtype=torch.long))

    assert torch.equal(audio_embeds, hidden_states + 1.0)
    assert torch.equal(audio_embeds_mask, output_mask)


def test_gemma4_custom_vision_pool_matches_hf_pooler():
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionPooler

    from xhmodel_merak.xh_llm.models.gemma4e._vision_model_impl import _Gemma4VisionModel

    hidden_states = torch.randn(1, 8, 8)
    pixel_position_ids = torch.tensor(
        [
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1],
                [2, 0],
                [3, 0],
                [2, 1],
                [-1, -1],
            ]
        ],
        dtype=torch.long,
    )
    padding_positions = (pixel_position_ids == -1).all(dim=-1)
    masked_hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)
    pooler = Gemma4VisionPooler(Gemma4VisionConfig(hidden_size=8))

    expected_output, expected_mask = pooler._avg_pool_by_positions(masked_hidden_states, pixel_position_ids, length=2)
    actual_output, actual_mask = _Gemma4VisionModel._avg_pool_by_positions(
        masked_hidden_states,
        pixel_position_ids,
        output_length=2,
        pooling_kernel_size=2,
    )

    assert torch.allclose(actual_output, expected_output)
    assert torch.equal(actual_mask, expected_mask)


def test_gemma4_visual_hmonnx_casts_int64_inputs_to_int32(monkeypatch):
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_hmonnx_inference import VisualHMONNXModel

    captured = {}

    def _fake_set_session_env():
        captured["env"] = True

    def _fake_forward(*args):
        captured["args"] = args
        return args

    model = object.__new__(VisualHMONNXModel)
    model.hmonnx_session = SimpleNamespace(
        _set_session_env=_fake_set_session_env,
        _session=SimpleNamespace(forward=_fake_forward, node_modules=[]),
    )
    pixel_values = torch.randn(1, 4, 4)
    image_position_ids = torch.zeros((1, 1, 2), dtype=torch.int64)

    VisualHMONNXModel.forward(model, pixel_values, image_position_ids)

    assert captured["env"] is True
    assert captured["args"][0].dtype == pixel_values.dtype
    assert captured["args"][1].dtype == torch.int32


def test_gemma4_visual_hmonnx_applies_output_scale():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_hmonnx_inference import VisualHMONNXModel

    def _fake_set_session_env():
        return None

    def _fake_forward(*args):
        del args
        return (torch.ones(1, 2, 4), torch.tensor([[True, False]]))

    model = object.__new__(VisualHMONNXModel)
    model.export_mode = "full"
    model.output_scale = 0.25
    model.hmonnx_session = SimpleNamespace(
        _set_session_env=_fake_set_session_env,
        _session=SimpleNamespace(forward=_fake_forward, node_modules=[]),
    )

    outputs = VisualHMONNXModel.forward(model, torch.randn(1, 2, 4), torch.zeros((1, 2, 2), dtype=torch.int64))

    assert torch.allclose(outputs[0], torch.full((1, 2, 4), 0.25))
    assert torch.equal(outputs[1], torch.tensor([[True, False]]))


def test_gemma4_full_vision_attention_mask_keeps_padded_queries_finite():
    from xhmodel_merak.xh_llm.models.gemma4e._vision_model_impl import _Gemma4VisionModel

    valid_positions = torch.tensor([[True, True, False, False]])

    mask = _Gemma4VisionModel._build_bidirectional_attention_mask(valid_positions)

    assert mask.shape == (1, 1, 4, 4)
    assert torch.equal(mask[0, 0, 0], valid_positions[0])
    assert torch.equal(mask[0, 0, 1], valid_positions[0])
    assert torch.equal(mask[0, 0, 2], valid_positions[0])
    assert torch.equal(mask[0, 0, 3], valid_positions[0])
    assert mask[0, 0].any(dim=-1).all()


def test_gemma4_full_vision_wrap_replaces_pow_based_rmsnorm():
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm
    import xhquant.nn.modules as xhnn

    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_vision_model import XHGemma4VisionModel
    from xhmodel_merak.xh_llm.models.gemma4e.xh_gemma4_config import XHGemma4VisualConfig

    model = XHGemma4VisionModel(XHGemma4VisualConfig(model_name="gemma4_visual", hf_model=str(MODEL_DIR)))
    hf_model = model.get_native_model()
    wrapped = model.init_wrap_model(hf_model)

    assert not any(isinstance(module, Gemma4RMSNorm) for module in wrapped.vision_tower.modules())
    assert any(isinstance(module, xhnn.RMSNorm) for module in wrapped.vision_tower.modules())


def test_gemma4_visual_hmonnx_compact_mode_ignores_position_ids():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_hmonnx_inference import VisualHMONNXModel

    captured = {}

    def _fake_set_session_env():
        return None

    def _fake_forward(*args):
        captured["args"] = args
        return args

    model = object.__new__(VisualHMONNXModel)
    model.export_mode = "compact"
    model.hmonnx_session = SimpleNamespace(
        _set_session_env=_fake_set_session_env,
        _session=SimpleNamespace(forward=_fake_forward, node_modules=[]),
    )

    VisualHMONNXModel.forward(
        model,
        torch.randn(1, 256, 768),
        torch.zeros((1, 256, 2), dtype=torch.int64),
    )

    assert len(captured["args"]) == 1


def test_gemma4_visual_hmonnx_runs_reducesum_in_fast_mode():
    from xhquant.common import PrecisionMode

    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_hmonnx_inference import VisualHMONNXModel

    reduce_module = SimpleNamespace(op_type="ReduceSum", precision_mode=PrecisionMode.ALIGNED)
    one_hot_module = SimpleNamespace(op_type="OneHot", precision_mode=PrecisionMode.FAST)
    matmul_module = SimpleNamespace(op_type="MatMul", precision_mode=PrecisionMode.ALIGNED)
    graph_session = SimpleNamespace(node_modules=[reduce_module, one_hot_module, matmul_module])

    patched = VisualHMONNXModel._patch_visual_session_precision(graph_session)

    assert patched == 1
    assert reduce_module.precision_mode == PrecisionMode.FAST
    assert one_hot_module.precision_mode == PrecisionMode.ALIGNED
    assert matmul_module.precision_mode == PrecisionMode.ALIGNED


def test_gemma4_audio_runtime_prefers_plain_onnx_artifact(monkeypatch, tmp_path):
    from xhmodel_merak.xh_llm.models.gemma4e import gemma4_hmonnx_inference as inference_mod

    created = {}
    audio_dir = tmp_path / "audio"
    hmonnx_path = audio_dir / "gemma4_audio_hm.onnx"
    onnx_dir = audio_dir / "onnx"
    onnx_path = onnx_dir / "gemma4_audio.onnx"
    audio_dir.mkdir(parents=True)
    onnx_dir.mkdir(parents=True)
    hmonnx_path.touch()
    onnx_path.touch()

    class _FakeAudioHMONNXModel:
        def __init__(self, path: str):
            created["hmonnx"] = path

    class _FakeAudioONNXModel:
        def __init__(self, path: str):
            created["onnx"] = path

    monkeypatch.setattr(inference_mod, "AudioHMONNXModel", _FakeAudioHMONNXModel)
    monkeypatch.setattr(inference_mod, "AudioONNXModel", _FakeAudioONNXModel)

    runtime = inference_mod.XHGemma4_HMONNXModel._build_audio_runtime(
        SimpleNamespace(hmonnx=str(hmonnx_path), onnx=str(onnx_path)),
    )

    assert isinstance(runtime, _FakeAudioONNXModel)
    assert created["onnx"] == str(onnx_path)


def test_gemma4_visual_runtime_prefers_plain_onnx_artifact(monkeypatch, tmp_path):
    from xhmodel_merak.xh_llm.models.gemma4e import gemma4_hmonnx_inference as inference_mod

    created = {}
    visual_dir = tmp_path / "visual"
    hmonnx_path = visual_dir / "gemma4_visual_hm.onnx"
    onnx_dir = visual_dir / "onnx"
    onnx_path = onnx_dir / "gemma4_visual.onnx"
    visual_dir.mkdir(parents=True)
    onnx_dir.mkdir(parents=True)
    hmonnx_path.touch()
    onnx_path.touch()

    class _FakeVisualHMONNXModel:
        def __init__(self, path: str, *, export_mode: str = "full", output_scale: float = 1.0):
            created["hmonnx"] = (path, export_mode, output_scale)

    class _FakeVisualONNXModel:
        def __init__(self, path: str):
            created["onnx"] = path

    monkeypatch.setattr(inference_mod, "VisualHMONNXModel", _FakeVisualHMONNXModel)
    monkeypatch.setattr(inference_mod, "VisualONNXModel", _FakeVisualONNXModel)

    runtime = inference_mod.XHGemma4_HMONNXModel._build_visual_runtime(
        SimpleNamespace(hmonnx=str(hmonnx_path), onnx=str(onnx_path), export_mode="compact"),
    )

    assert isinstance(runtime, _FakeVisualONNXModel)
    assert created["onnx"] == str(onnx_path)


def test_gemma4_visual_runtime_prefers_hmonnx_for_full_export(monkeypatch, tmp_path):
    from xhmodel_merak.xh_llm.models.gemma4e import gemma4_hmonnx_inference as inference_mod

    created = {}
    visual_dir = tmp_path / "visual"
    hmonnx_path = visual_dir / "gemma4_visual_hm.onnx"
    onnx_dir = visual_dir / "onnx"
    onnx_path = onnx_dir / "gemma4_visual.onnx"
    visual_dir.mkdir(parents=True)
    onnx_dir.mkdir(parents=True)
    hmonnx_path.touch()
    onnx_path.touch()

    class _FakeVisualHMONNXModel:
        def __init__(self, path: str, *, export_mode: str = "full", output_scale: float = 1.0):
            created["hmonnx"] = (path, export_mode, output_scale)

    class _FakeVisualONNXModel:
        def __init__(self, path: str):
            created["onnx"] = path

    monkeypatch.setattr(inference_mod, "VisualHMONNXModel", _FakeVisualHMONNXModel)
    monkeypatch.setattr(inference_mod, "VisualONNXModel", _FakeVisualONNXModel)

    runtime = inference_mod.XHGemma4_HMONNXModel._build_visual_runtime(
        SimpleNamespace(hmonnx=str(hmonnx_path), onnx=str(onnx_path), export_mode="full", output_scale=0.25),
    )

    assert isinstance(runtime, _FakeVisualHMONNXModel)
    assert created["hmonnx"] == (str(hmonnx_path), "full", 0.25)
    assert "onnx" not in created


def test_gemma4_audio_runtime_requires_existing_hmonnx_artifact():
    from xhmodel_merak.xh_llm.models.gemma4e import gemma4_hmonnx_inference as inference_mod

    with pytest.raises(FileNotFoundError):
        inference_mod.XHGemma4_HMONNXModel._build_audio_runtime(SimpleNamespace(hmonnx="/tmp/missing_audio_hm.onnx"))


def test_gemma4_runtime_disables_kvcache_fast_mode_on_text_sessions():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_hmonnx_inference import XHGemma4_HMONNXModel

    cache_a = SimpleNamespace(op_type="KVcache", fast_mode=True)
    cache_b = SimpleNamespace(op_type="KVcache", fast_mode=True)
    other = SimpleNamespace(op_type="MatMul", fast_mode=True)
    fake_model = SimpleNamespace(
        hmonnx_session=SimpleNamespace(
            _session=SimpleNamespace(node_modules=[cache_a, other, cache_b]),
            initialize=lambda: None,
        )
    )

    patched = XHGemma4_HMONNXModel._disable_kvcache_fast_mode(fake_model)

    assert patched == 2
    assert cache_a.fast_mode is False
    assert cache_b.fast_mode is False
    assert other.fast_mode is True


def test_gemma4_runtime_aligns_text_cache_modes_during_prefill_and_decode(monkeypatch):
    from xhmodel_merak.xh_llm.hmonnx.base_llm_hmonnx_model import BaseLLMHMONNXModel
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_hmonnx_inference import XHGemma4_HMONNXModel

    def _fake_set_prefill(self):
        self._llm_prefill = True

    def _fake_set_decode(self):
        self._llm_prefill = False

    monkeypatch.setattr(BaseLLMHMONNXModel, "set_prefill", _fake_set_prefill)
    monkeypatch.setattr(BaseLLMHMONNXModel, "set_decode", _fake_set_decode)

    runtime = object.__new__(XHGemma4_HMONNXModel)
    runtime.prefill_model = object()
    runtime.decode_model = object()
    runtime._aligned_text_cache_modes = set()

    calls = []

    def _fake_align(self, mode, hmonnx_model):
        calls.append((mode, hmonnx_model))
        self._aligned_text_cache_modes.add(mode)

    monkeypatch.setattr(XHGemma4_HMONNXModel, "_ensure_text_cache_alignment", _fake_align)

    XHGemma4_HMONNXModel.set_prefill(runtime)
    XHGemma4_HMONNXModel.set_decode(runtime)

    assert calls == [("prefill", runtime.prefill_model), ("decode", runtime.decode_model)]


def test_gemma4_custom_patch_position_embeddings_match_hf_embedder():
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionPatchEmbedder

    from xhmodel_merak.xh_llm.models.gemma4e._vision_model_impl import _build_patch_position_embeddings

    patch_embedder = Gemma4VisionPatchEmbedder(Gemma4VisionConfig(hidden_size=8, patch_size=2, position_embedding_size=8))
    pixel_position_ids = torch.tensor(
        [
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, -1],
            ]
        ],
        dtype=torch.long,
    )
    padding_positions = (pixel_position_ids == -1).all(dim=-1)

    expected = patch_embedder._position_embeddings(pixel_position_ids, padding_positions)
    actual = _build_patch_position_embeddings(
        patch_embedder.position_embedding_table,
        pixel_position_ids,
        padding_positions,
    )

    assert torch.allclose(actual, expected)


def test_gemma4_custom_vision_forward_returns_pooler_mask():
    from xhmodel_merak.xh_llm.models.gemma4e._vision_model_impl import _Gemma4VisionModel

    pooled_hidden_states = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    pooler_mask = torch.tensor([[True, True, False, True]])

    class _FakePatchEmbedder:
        def __call__(self, pixel_values, pixel_position_ids, padding_positions):
            return pooled_hidden_states.clone()

    class _FakeEncoder:
        def __call__(self, inputs_embeds, attention_mask, pixel_position_ids):
            return SimpleNamespace(last_hidden_state=inputs_embeds)

    class _FakePooler:
        def __call__(self, hidden_states, pixel_position_ids, padding_positions, output_length):
            assert output_length == 4
            return pooled_hidden_states.clone(), pooler_mask

    fake_model = SimpleNamespace(
        config=SimpleNamespace(pooling_kernel_size=2, standardize=False),
        patch_embedder=_FakePatchEmbedder(),
        encoder=_FakeEncoder(),
        pooler=_FakePooler(),
        _build_bidirectional_attention_mask=_Gemma4VisionModel._build_bidirectional_attention_mask,
        _pool_hidden_states=lambda hidden_states, pixel_position_ids, padding_positions, output_length: (
            pooled_hidden_states.clone(),
            pooler_mask,
        ),
    )

    pixel_values = torch.randn(1, 16, 8)
    pixel_position_ids = torch.tensor([[[0, 0], [1, 0], [2, 0], [-1, -1]]], dtype=torch.long)

    output, output_mask = _Gemma4VisionModel.forward(fake_model, pixel_values, pixel_position_ids)

    assert torch.equal(output, pooled_hidden_states)
    assert torch.equal(output_mask, pooler_mask)
