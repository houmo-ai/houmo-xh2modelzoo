from __future__ import annotations

import dataclasses
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image


MODEL_DIR = Path("/data01/datasets/gemma-4-12B-it")


pytestmark = pytest.mark.skipif(
    not MODEL_DIR.exists(),
    reason=f"Gemma4 12B Unified fixture is unavailable: {MODEL_DIR}",
)


@pytest.fixture(scope="module")
def unified_processor():
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_processor import (
        XHGemma4Processor,
    )

    return XHGemma4Processor.from_pretrained(str(MODEL_DIR))


def test_gemma4_12b_resolves_encoder_free_variant():
    from xhmodel_merak.xh_llm.models.gemma4_series.variants import (
        resolve_gemma4_series_variant,
    )

    config = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    variant = resolve_gemma4_series_variant(config)

    assert variant.name == "12b_unified"
    assert variant.frontend_kind == "encoder_free"
    assert variant.hf_architecture == "Gemma4UnifiedForConditionalGeneration"
    assert variant.topology == "dense"
    assert variant.has_audio is True
    assert variant.has_per_layer_input is False
    assert variant.has_shared_kv_layers is False
    assert variant.bidirectional_vision_attention is True


def test_gemma4_12b_modality_contract_comes_from_checkpoint_config():
    from xhmodel_merak.xh_llm.models.gemma4_series.modality_contract import (
        Gemma4SeriesModalityContract,
    )

    contract = Gemma4SeriesModalityContract.from_pretrained(MODEL_DIR)

    assert contract.frontend_kind == "encoder_free"
    assert contract.image_soft_tokens == 280
    assert contract.video_soft_tokens_per_frame == 70
    assert contract.audio_soft_tokens == 750
    assert contract.vision_patch_dim == 6912
    assert contract.audio_feature_dim == 640
    assert contract.position_capacity == 1120
    assert contract.sampling_rate == 16000


def test_gemma4_12b_processor_uses_fixed_encoder_free_contract(unified_processor):
    processor = unified_processor
    assert type(processor).__name__ == "XHGemma4UnifiedProcessor"

    image_inputs = processor(
        text=[processor.image_token],
        images=[[Image.new("RGB", (640, 480), color="white")]],
        return_tensors="pt",
    )
    valid_image_tokens = int((~(image_inputs["image_position_ids"] == -1).all(dim=-1)).sum())

    assert tuple(image_inputs["pixel_values"].shape) == (1, 280, 6912)
    assert tuple(image_inputs["image_position_ids"].shape) == (1, 280, 2)
    assert int(image_inputs["image_soft_token_count"].item()) == valid_image_tokens == 266
    assert int((image_inputs["input_ids"] == processor.image_token_id).sum()) == valid_image_tokens
    assert "pooling_matrix" not in image_inputs
    assert "visual_attention_mask" not in image_inputs

    audio_inputs = processor(
        text=[processor.audio_token],
        audio=[np.zeros(641, dtype=np.float32)],
        sampling_rate=16000,
        return_tensors="pt",
    )
    assert tuple(audio_inputs["input_features"].shape) == (1, 750, 640)
    assert tuple(audio_inputs["input_features_mask"].shape) == (1, 750)
    assert int(audio_inputs["input_features_mask"].sum()) == 2
    assert int((audio_inputs["input_ids"] == processor.audio_token_id).sum()) == 2
    assert "audio_attention_mask" not in audio_inputs


def test_gemma4_12b_processor_rejects_audio_over_checkpoint_limit(unified_processor):
    processor = unified_processor

    with pytest.raises(ValueError, match="exceeds the checkpoint audio limit"):
        processor(
            text=[processor.audio_token],
            audio=[np.zeros(480001, dtype=np.float32)],
            sampling_rate=16000,
            return_tensors="pt",
        )


@pytest.mark.parametrize(
    ("num_samples", "expected_tokens"),
    [(1, 1), (640, 1), (641, 2), (1280, 2), (1281, 3), (480000, 750)],
)
def test_gemma4_12b_audio_token_boundaries(unified_processor, num_samples, expected_tokens):
    processor = unified_processor
    inputs = processor(
        text=[processor.audio_token],
        audio=[np.zeros(num_samples, dtype=np.float32)],
        sampling_rate=16000,
        return_tensors="pt",
    )

    assert tuple(inputs["input_features"].shape) == (1, 750, 640)
    assert int(inputs["input_features_mask"].sum()) == expected_tokens
    assert int((inputs["input_ids"] == processor.audio_token_id).sum()) == expected_tokens


def test_gemma4_12b_presampled_video_requires_metadata_and_uses_70_tokens_per_frame(unified_processor):
    processor = unified_processor
    frames = [Image.new("RGB", (320, 240), color="white") for _ in range(4)]
    with pytest.raises(ValueError, match="pre-sampled video frames require video_metadata"):
        processor(
            text=[processor.video_token],
            videos=[frames],
            do_sample_frames=False,
            return_tensors="pt",
        )

    inputs = processor(
        text=[processor.video_token],
        videos=[frames],
        video_metadata=[
            {
                "fps": 2.0,
                "duration": 2.0,
                "total_num_frames": 4,
                "frames_indices": [0, 1, 2, 3],
                "video_backend": "synthetic",
            }
        ],
        do_sample_frames=False,
        return_tensors="pt",
    )
    valid = (~(inputs["video_position_ids"] == -1).all(dim=-1)).sum()

    assert tuple(inputs["pixel_values_videos"].shape) == (1, 4, 70, 6912)
    assert tuple(inputs["video_position_ids"].shape) == (1, 4, 70, 2)
    assert int(inputs["video_soft_token_count"].sum()) == int(valid)
    assert int((inputs["input_ids"] == processor.video_token_id).sum()) == int(valid)


def test_gemma4_12b_chat_template_loads_local_wav_without_torchcodec(unified_processor, tmp_path):
    image_path = tmp_path / "image.png"
    wav_path = tmp_path / "audio.wav"
    Image.new("RGB", (640, 480), color="white").save(image_path)
    pcm = np.zeros(1600, dtype=np.int16)
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(pcm.tobytes())

    processor = unified_processor
    inputs = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "audio", "audio": str(wav_path)},
                    {"type": "text", "text": "describe both"},
                ],
            }
        ],
        add_generation_prompt=True,
    )

    assert int((inputs["input_ids"] == processor.image_token_id).sum()) == int(inputs["image_soft_token_count"].sum())
    assert int((inputs["input_ids"] == processor.audio_token_id).sum()) == int(inputs["audio_soft_token_count"].sum())
    assert tuple(inputs["input_features"].shape) == (1, 750, 640)


def test_gemma4_12b_public_architecture_routes_to_series_model():
    from xhmodel_merak.xh_llm.builder import get_model_class
    from xhmodel_merak.xh_llm.configuration_auto import MODEL_TYPE_MAPPING_MODULES

    model_type = "Gemma4UnifiedForConditionalGeneration"

    assert MODEL_TYPE_MAPPING_MODULES[model_type] == "gemma4_series"
    model_cls = get_model_class(
        {
            "chip_arch": "XH2a",
            "model_type": model_type,
            "model_name": "gemma4_12b_unified_test",
        }
    )
    assert model_cls.__name__ == "XHGemma4UnifiedModel"
    assert model_cls.transformers_min_version == "5.13.0"


def test_gemma4_12b_model_config_builds_encoder_free_submodels():
    from xhmodel_merak.xh_llm.models.gemma4_series.xh_gemma4_series_config import (
        XHGemma4SeriesModelConfig,
    )

    config = XHGemma4SeriesModelConfig(
        model_name="gemma4_12b_unified_test",
        model_type="Gemma4UnifiedForConditionalGeneration",
        hf_model=str(MODEL_DIR),
        visual_config={},
    )

    assert config.variant == "12b_unified"
    assert config.frontend_kind == "encoder_free"
    assert type(config.visual_config).__name__ == "XHGemma4UnifiedVisualConfig"
    assert type(config.video_visual_config).__name__ == "XHGemma4UnifiedVisualConfig"
    assert type(config.audio_config).__name__ == "XHGemma4UnifiedAudioConfig"
    assert config.visual_config.image_seq_length == 280
    assert config.visual_config.input_dim == 6912
    assert config.video_visual_config.image_seq_length == 70
    assert config.audio_config.input_feature_length == 750
    assert config.audio_config.feature_size == 640


def test_gemma4_12b_model_config_round_trips_encoder_free_submodels():
    from xhmodel_merak.xh_llm.models.gemma4_series.xh_gemma4_series_config import (
        XHGemma4SeriesModelConfig,
    )

    config = XHGemma4SeriesModelConfig(
        model_name="gemma4_12b_unified_round_trip",
        model_type="Gemma4UnifiedForConditionalGeneration",
        hf_model=str(MODEL_DIR),
        visual_config={},
    )

    restored = XHGemma4SeriesModelConfig.from_dict(config.to_dict())

    assert restored.visual_config.input_modality == "image"
    assert restored.visual_config.image_seq_length == 280
    assert restored.video_visual_config.input_modality == "video"
    assert restored.video_visual_config.image_seq_length == 70
    assert restored.audio_config.input_feature_length == 750
    assert restored.audio_config.feature_size == 640


def test_gemma4_12b_auto_model_builds_all_encoder_free_frontends():
    from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel

    raw_config = {
        "model_name": "gemma4_12b_unified_auto",
        "model_type": "Gemma4UnifiedForConditionalGeneration",
        "hf_model": str(MODEL_DIR),
        "context_max_length": 2048,
        "prefill_chunk_length": 320,
        "visual_config": {},
    }

    config = AutoLLMConfig.from_pretrained(raw_config)
    model = AutoLLMModel.from_pretrained(config=config)

    assert type(model).__name__ == "XHGemma4UnifiedModel"
    assert type(model.visual).__name__ == "XHGemma4UnifiedVisionModel"
    assert type(model.video_visual).__name__ == "XHGemma4UnifiedVisionModel"
    assert type(model.audio).__name__ == "XHGemma4UnifiedAudioModel"


def test_gemma4_12b_workflow_demo_reports_audio_support():
    from examples_merak.llm.gemma4_series.gemma4_workflow_demo import (
        PRESETS,
        _audio_support_status,
    )

    preset = dataclasses.replace(PRESETS["12b-unified"], hf_model_dir=str(MODEL_DIR))

    assert _audio_support_status(preset) == "supported"


def test_gemma4_12b_workflow_routes_gptqmodel_recipe(monkeypatch, tmp_path):
    from xhmodel_merak.xh_llm.models.gemma4_series.workflow import Gemma4SeriesWorkflow
    from xhmodel_merak.xh_llm.workflows.result import QuantResult

    config_path = (
        Path("configs_merak/workflows/xh2a/llm_models/gemma4_series/12b_unified")
        / "gemma4_12b_unified_full.yaml"
    )
    workflow = Gemma4SeriesWorkflow.from_config(str(MODEL_DIR), str(config_path))
    calls = []

    def fake_quant(**kwargs):
        calls.append(kwargs)
        return QuantResult(raw_model_dir=str(MODEL_DIR), quanted_model_dir=str(tmp_path / "quant"))

    monkeypatch.setattr(workflow, "_quant_gptqmodel_recipe", fake_quant)
    result = workflow.quant(output_dir=str(tmp_path), device="cuda:0")

    assert result.quanted_model_dir == str(tmp_path / "quant")
    assert calls[0]["quant_cfg"]["algorithm"] == "gptqmodel"
    assert calls[0]["quant_cfg"]["method"] == "gptq"


def test_gemma4_12b_workflow_routes_autoround_mode1(monkeypatch, tmp_path):
    from xhmodel_merak.xh_llm.models.gemma4_series.workflow import Gemma4SeriesWorkflow
    from xhmodel_merak.xh_llm.workflows.result import QuantResult

    config_path = (
        Path("configs_merak/workflows/xh2a/llm_models/gemma4_series/12b_unified")
        / "gemma4_12b_unified_full.yaml"
    )
    workflow = Gemma4SeriesWorkflow.from_config(str(MODEL_DIR), str(config_path))
    calls = []

    def fake_quant(**kwargs):
        calls.append(kwargs)
        return QuantResult(raw_model_dir=str(MODEL_DIR), quanted_model_dir=str(tmp_path / "quant"))

    monkeypatch.setattr(workflow, "_quant_autoround_mode1", fake_quant)
    result = workflow.quant(
        output_dir=str(tmp_path),
        device="cuda:0",
        config_overrides={
            "quant": {
                "algorithm": "gptqmodel",
                "method": "autoround",
                "preset": "mode1",
                "rotation": None,
                "artifact_format": "gptqmodel_hf",
                "output_format": "gptqmodel_hf",
                "bits": 4,
                "group_size": 64,
            }
        },
    )

    assert result.quanted_model_dir == str(tmp_path / "quant")
    assert calls[0]["quant_cfg"]["method"] == "autoround"


def test_gemma4_12b_existing_quant_export_preserves_w4_metadata(tmp_path):
    from examples_merak.llm.gemma4_series import gemma4_series_quant_export

    quant_dir = tmp_path / "quant"
    quant_dir.mkdir()
    (quant_dir / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "provider": "auto-round",
                    "bits": 4,
                    "group_size": 64,
                    "sym": True,
                }
            }
        ),
        encoding="utf-8",
    )
    args = gemma4_series_quant_export.build_parser().parse_args(
        [
            "--hf-model-dir",
            str(MODEL_DIR),
            "--config",
            "configs_merak/workflows/xh2a/llm_models/gemma4_series/12b_unified/"
            "gemma4_12b_unified_full.yaml",
            "--export-output-dir",
            str(tmp_path / "export"),
            "--existing-hf-model-dir",
            str(quant_dir),
        ]
    )

    quant_config = gemma4_series_quant_export._build_quant_overrides(args)["quant"]

    assert quant_config["algorithm"] == "existing_hf"
    assert quant_config["method"] == "autoround"
    assert quant_config["bits"] == 4
    assert quant_config["group_size"] == 64
    assert quant_config["sym"] is True


def test_gemma4_12b_frontend_graphs_are_mask_free_and_token_independent(monkeypatch):
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_unified_audio_model import (
        Gemma4UnifiedAudioAdapter,
    )
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_unified_vision_model import (
        Gemma4UnifiedVisionAdapter,
        _Gemma4UnifiedVisualProcessor,
    )

    class VisionEmbedder(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_ln1 = nn.LayerNorm(4)
            self.patch_dense = nn.Linear(4, 4)
            self.patch_ln2 = nn.LayerNorm(4)
            self.pos_embedding = nn.Parameter(torch.zeros(4, 2, 4))
            self.pos_norm = nn.LayerNorm(4)
            self.multimodal_embedder = SimpleNamespace(
                embedding_pre_projection_norm=SimpleNamespace(eps=1e-6),
                embedding_projection=nn.Linear(4, 4),
            )

    class AudioEmbedder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding_pre_projection_norm = SimpleNamespace(eps=1e-6)
            self.embedding_projection = nn.Linear(4, 4)

    clamp_dtypes = []
    original_clamp = torch.clamp

    def recording_clamp(input_tensor, *args, **kwargs):
        clamp_dtypes.append(input_tensor.dtype)
        return original_clamp(input_tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "clamp", recording_clamp)
    vision = Gemma4UnifiedVisionAdapter(VisionEmbedder())
    audio = Gemma4UnifiedAudioAdapter(AudioEmbedder())
    pixels = torch.randn(1, 2, 4)
    positions = torch.tensor([[[0, 0], [0, 0]]], dtype=torch.int32)
    features = torch.randn(1, 2, 4)

    vision_output = vision(pixels, positions)
    audio_output = audio(features)

    assert clamp_dtypes == []
    assert torch.allclose(vision_output[:, :1], vision(pixels[:, :1], positions[:, :1]))
    assert torch.allclose(audio_output[:, :1], audio(features[:, :1]))
    positions, valid = _Gemma4UnifiedVisualProcessor.normalize_position_inputs(
        torch.tensor([[[0, 0], [-1, -1]]], dtype=torch.int64)
    )
    assert positions.tolist() == [[[0, 0], [0, 0]]]
    assert positions.dtype == torch.int32
    assert valid.tolist() == [[[True], [False]]]
    assert valid.dtype == torch.bool


def test_gemma4_12b_frontend_export_abi_does_not_expose_filter_masks():
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_unified_audio_model import (
        XHGemma4UnifiedAudioModel,
    )
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_unified_vision_model import (
        XHGemma4UnifiedVisionModel,
    )

    assert XHGemma4UnifiedVisionModel.get_export_cfg(None)["input_names"] == [
        "pixel_values",
        "position_ids",
    ]
    assert XHGemma4UnifiedAudioModel.get_export_cfg(None)["input_names"] == [
        "input_features",
    ]


def test_gemma4_12b_runtime_preserves_padding_sentinel_for_encoder_free_frontend():
    from xhmodel_merak.xh_llm.models.gemma4_series.llm_text import _Gemma4HFCompatible

    position_ids = torch.tensor([[[0, 0], [-1, -1]]], dtype=torch.int64)

    encoder_free = _Gemma4HFCompatible._prepare_visual_position_ids(
        position_ids, encoder_free=True
    )
    tower = _Gemma4HFCompatible._prepare_visual_position_ids(
        position_ids, encoder_free=False
    )

    assert encoder_free.tolist() == [[[0, 0], [-1, -1]]]
    assert tower.tolist() == [[[0, 0], [0, 0]]]


def test_gemma4_12b_default_golden_video_has_explicit_metadata(unified_processor):
    from xhmodel_merak.xh_llm.models.gemma4_series.workflow import Gemma4SeriesWorkflow

    messages = Gemma4SeriesWorkflow._messages_for_golden_modality(
        [{"role": "user", "content": "describe the input"}],
        "video",
    )
    video_item = messages[0]["content"][0]

    assert video_item["video_metadata"]["fps"] == 2.0
    assert video_item["video_metadata"]["total_num_frames"] == 16
    inputs = unified_processor.apply_chat_template(messages)
    assert tuple(inputs["pixel_values_videos"].shape[:3]) == (1, 16, 70)
    assert 1024 < inputs["input_ids"].shape[1] <= 2048


def test_gemma4_12b_export_plan_rejects_soft_token_override():
    from xhmodel_merak.xh_llm.models.gemma4_series.export_plan import (
        build_gemma4_series_export_plan,
    )

    with pytest.raises(ValueError, match="checkpoint-owned"):
        build_gemma4_series_export_plan(
            hf_model_dir=str(MODEL_DIR),
            export_model_cfg={
                "context_max_length": 2048,
                "prefill_chunk_length": 320,
                "quant_scheme": {"quant_type": "w8a8h1_sefp"},
                "visual_config": {"image_seq_length": 1120},
                "video_visual_config": {"image_seq_length": 70},
            },
        )


def test_gemma4_12b_processor_rejects_nested_soft_token_override(unified_processor):
    with pytest.raises(ValueError, match="checkpoint-owned"):
        unified_processor(
            text=[unified_processor.image_token],
            images=[[Image.new("RGB", (640, 480), color="white")]],
            images_kwargs={"max_soft_tokens": 1120},
            return_tensors="pt",
        )


def test_gemma4_12b_export_plan_accepts_empty_frontend_mappings():
    from xhmodel_merak.xh_llm.models.gemma4_series.export_plan import (
        build_gemma4_series_export_plan,
    )

    plan = build_gemma4_series_export_plan(
        hf_model_dir=str(MODEL_DIR),
        export_model_cfg={
            "context_max_length": 2048,
            "prefill_chunk_length": 320,
            "quant_scheme": {"quant_type": "w8a8h1_sefp"},
            "visual_config": {},
            "video_visual_config": {},
        },
    )
    plan.validate_fixed_contract()

    assert plan.export_image_visual is True
    assert plan.export_video_visual is True
    assert plan.image_visual_seq_length == 280
    assert plan.video_visual_seq_length == 70


def test_gemma4_12b_audio_tokens_remain_causal_in_vision_bidi_model():
    from xhmodel_merak.xh_llm.models.gemma4_series.data_preprocess import (
        Gemma4DataPreprocess,
    )

    preprocess = Gemma4DataPreprocess(
        token_embedding=nn.Embedding(16, 4),
        input_sequence_length=4,
        context_length=8,
        pad_token_id=0,
        audio_token_id=9,
        sliding_window=4,
        bidirectional_vision_attention=True,
        emit_full_attention_mask=False,
    )
    outputs = preprocess(
        {
            "input_ids": torch.tensor([[1, 9, 9]], dtype=torch.long),
            "past_seq_length": 0,
            "mm_token_type_ids": torch.tensor([[0, 3, 3]], dtype=torch.long),
            "audio_embeds": torch.zeros((2, 4), dtype=torch.float32),
        }
    )
    sliding_mask = outputs[3]

    assert sliding_mask[0, 0, 1, 2].item() == torch.finfo(torch.float16).min


def test_gemma4_12b_cache_layout_has_40_sliding_and_8_full_caches():
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_llm_model import (
        _build_gemma4_layer_cache_layout,
    )

    text_config = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))["text_config"]
    layers = []
    for layer_type in text_config["layer_types"]:
        head_dim = text_config["head_dim"] if layer_type == "sliding_attention" else text_config["global_head_dim"]
        kv_heads = (
            text_config["num_key_value_heads"]
            if layer_type == "sliding_attention"
            else text_config["num_global_key_value_heads"]
        )
        attention = SimpleNamespace(
            head_dim=head_dim,
            k_proj=SimpleNamespace(out_features=kv_heads * head_dim),
            is_kv_shared_layer=False,
        )
        layers.append(SimpleNamespace(self_attn=attention))

    shapes, cache_types, owners, indices = _build_gemma4_layer_cache_layout(
        layers=layers,
        layer_types=text_config["layer_types"],
        context_max_length=2048,
        sliding_window=text_config["sliding_window"],
        input_seq_len=320,
        sliding_kv_cache_input_mode="slice_window",
    )

    assert cache_types.count("sliding_attention") == 40
    assert cache_types.count("full_attention") == 8
    assert {tuple(shape) for shape, kind in zip(shapes, cache_types, strict=True) if kind == "sliding_attention"} == {
        (1, 8, 1344, 256)
    }
    assert {tuple(shape) for shape, kind in zip(shapes, cache_types, strict=True) if kind == "full_attention"} == {
        (1, 1, 2048, 512)
    }
    assert owners == indices == list(range(48))


def test_gemma4_unified_text_classes_are_registered_for_lowering():
    from transformers.models.gemma4_unified.modeling_gemma4_unified import (
        Gemma4UnifiedForConditionalGeneration,
        Gemma4UnifiedRMSNorm,
        Gemma4UnifiedTextAttention,
        Gemma4UnifiedTextDecoderLayer,
        Gemma4UnifiedTextModel,
        Gemma4UnifiedTextRotaryEmbedding,
    )

    from xhmodel_merak.xh_llm.models.gemma4_series import _llm_model_impl  # noqa: F401
    from xhmodel_merak.xh_llm.register import XHLLM_TRACEABLE_MODULES

    for module_type in (
        Gemma4UnifiedForConditionalGeneration,
        Gemma4UnifiedRMSNorm,
        Gemma4UnifiedTextAttention,
        Gemma4UnifiedTextDecoderLayer,
        Gemma4UnifiedTextModel,
        Gemma4UnifiedTextRotaryEmbedding,
    ):
        assert module_type in XHLLM_TRACEABLE_MODULES


@pytest.mark.parametrize(
    ("layer_type", "head_dim_attr"),
    [
        ("sliding_attention", "head_dim"),
        ("full_attention", "global_head_dim"),
    ],
)
def test_gemma4_12b_mtp_rope_matches_unified_hf(layer_type, head_dim_attr):
    from transformers import AutoConfig
    from transformers.models.gemma4_unified.modeling_gemma4_unified import (
        Gemma4UnifiedTextRotaryEmbedding,
    )

    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_mtp_model import (
        _compute_rotary_cache,
    )

    assistant_dir = Path("/data01/datasets/gemma-4-12B-it-assistant")
    assistant_config = AutoConfig.from_pretrained(assistant_dir)
    text_config = assistant_config.text_config
    rotary = Gemma4UnifiedTextRotaryEmbedding(text_config)
    positions = torch.tensor([[7]], dtype=torch.long)
    hf_cos, hf_sin = rotary(
        torch.zeros(1, 1, getattr(text_config, head_dim_attr)),
        positions,
        layer_type=layer_type,
    )
    cache_cos, cache_sin = _compute_rotary_cache(
        getattr(rotary, f"{layer_type}_inv_freq"),
        getattr(rotary, f"{layer_type}_attention_scaling"),
        16,
        partial_rotary_factor=float(
            text_config.rope_parameters[layer_type].get("partial_rotary_factor", 1.0)
        ),
    )

    torch.testing.assert_close(cache_cos[7], hf_cos[0, 0])
    torch.testing.assert_close(cache_sin[7], hf_sin[0, 0])


def test_gemma4_12b_mtp_uses_unified_transformers_text_types():
    from transformers.models.gemma4_unified.configuration_gemma4_unified import (
        Gemma4UnifiedTextConfig,
    )
    from transformers.models.gemma4_unified.modeling_gemma4_unified import (
        Gemma4UnifiedRMSNorm,
        Gemma4UnifiedTextModel,
    )

    from xhmodel_merak.xh_llm.models.gemma4_series import gemma4_series_mtp_model

    config_cls, model_cls, norm_cls = gemma4_series_mtp_model._resolve_assistant_text_types(
        {"model_type": "gemma4_unified_assistant"}
    )

    assert config_cls is Gemma4UnifiedTextConfig
    assert model_cls is Gemma4UnifiedTextModel
    assert norm_cls is Gemma4UnifiedRMSNorm


def test_gemma4_12b_mtp_workflow_config_matches_assistant_checkpoint(tmp_path):
    from xhmodel_merak.xh_llm.models.gemma4_series.workflow import (
        list_recommended_mtp_configs,
    )
    from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig

    config_path = Path(
        "configs_merak/workflows/xh2a/llm_models/gemma4_series/12b_unified/"
        "gemma4_12b_unified_full_mtp.yaml"
    )
    workflow_config = WorkflowConfig.from_file(config_path)
    model_config = workflow_config.export["model"]
    mtp_config = model_config["mtp_config"]
    assistant = json.loads(
        Path("/data01/datasets/gemma-4-12B-it-assistant/config.json").read_text(
            encoding="utf-8"
        )
    )
    text_config = assistant["text_config"]

    assert list_recommended_mtp_configs()["12b-unified"] == str(config_path)
    assert model_config["model_type"] == "Gemma4UnifiedForConditionalGeneration"
    assert model_config["spec_decode_mode"] == "mtp"
    assert model_config["context_max_length"] == 2048
    assert mtp_config["context_max_length"] == 2048
    assert mtp_config["assistant_hf_model"] is None
    assert mtp_config["target_hf_model"] is None
    from xhmodel_merak.xh_llm.models.gemma4_series.mtp_workflow import (
        validate_mtp_model_inputs,
    )

    with pytest.raises(ValueError) as helper_error:
        validate_mtp_model_inputs(model_config)
    assert "export.model.mtp_config.assistant_hf_model" in str(
        helper_error.value
    )
    assert "export.model.mtp_config.target_hf_model" in str(helper_error.value)
    assert "--mtp-assistant-model-dir" in str(helper_error.value)
    from xhmodel_merak.xh_llm.models.gemma4_series.workflow import (
        Gemma4SeriesWorkflow,
    )

    direct_workflow = Gemma4SeriesWorkflow(
        model_dir=str(MODEL_DIR),
        config_path=str(config_path),
    )
    normalized_context = direct_workflow._normalize_export_overrides(
        {"export.model.context_max_length": 4096}
    )
    assert normalized_context == {
        "export.model.context_max_length": 4096,
        "export.model.mtp_config.context_max_length": 4096,
    }
    with pytest.raises(ValueError) as workflow_error:
        direct_workflow._validate_export_model(None)
    assert "export.model.mtp_config.assistant_hf_model" in str(
        workflow_error.value
    )
    assert "export.model.mtp_config.target_hf_model" in str(
        workflow_error.value
    )
    assert "--mtp-assistant-model-dir" in str(workflow_error.value)
    injected = workflow_config.with_overrides(
        {
            "export.model.mtp_config.assistant_hf_model": (
                "weights/gemma-4-12B-it-assistant"
            ),
            "export.model.mtp_config.target_hf_model": "weights/gemma-4-12B-it",
        }
    )
    injected_mtp = injected.export["model"]["mtp_config"]
    assert (
        injected_mtp["assistant_hf_model"]
        == "weights/gemma-4-12B-it-assistant"
    )
    assert injected_mtp["target_hf_model"] == "weights/gemma-4-12B-it"
    validated_paths = validate_mtp_model_inputs(injected.export["model"])
    assert validated_paths["assistant"].name == "gemma-4-12B-it-assistant"
    assert validated_paths["target"].name == "gemma-4-12B-it"
    assert mtp_config["shared_kv_inputs"] == [
        "shared_key_cache_sliding",
        "shared_value_cache_sliding",
        "shared_key_cache_full",
        "shared_value_cache_full",
    ]
    duplicate_kv_model_config = json.loads(json.dumps(injected.export["model"]))
    duplicate_kv_model_config["mtp_config"]["shared_kv_inputs"][1] = (
        "shared_key_cache_sliding"
    )
    with pytest.raises(ValueError, match="Duplicate and legacy KV names"):
        validate_mtp_model_inputs(duplicate_kv_model_config)
    missing_base_workflow = Gemma4SeriesWorkflow(
        model_dir=str(tmp_path / "missing-base"),
        config_path=str(config_path),
    )
    with pytest.raises(FileNotFoundError, match="base.*does not exist"):
        missing_base_workflow._validate_export_model(
            {
                "export.model.mtp_config.assistant_hf_model": (
                    "weights/gemma-4-12B-it-assistant"
                ),
                "export.model.mtp_config.target_hf_model": (
                    "weights/gemma-4-12B-it"
                ),
            }
        )
    assert mtp_config["assistant_num_hidden_layers"] == text_config["num_hidden_layers"]
    assert mtp_config["assistant_layer_pattern"] == text_config["layer_types"]
    assert mtp_config["assistant_hidden_size"] == text_config["hidden_size"]
    assert mtp_config["assistant_num_attention_heads"] == text_config["num_attention_heads"]
    assert mtp_config["assistant_num_key_value_heads"] == text_config["num_key_value_heads"]
    assert (
        mtp_config["assistant_num_global_key_value_heads"]
        == text_config["num_global_key_value_heads"]
    )
    assert mtp_config["head_dim"] == text_config["head_dim"]
    assert mtp_config["use_ordered_embeddings"] is assistant["use_ordered_embeddings"]


def test_gemma4_12b_mtp_assistant_target_contract_is_validated():
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_mtp_model import (
        validate_assistant_target_contract,
    )

    contract = validate_assistant_target_contract(
        "/data01/datasets/gemma-4-12B-it-assistant",
        MODEL_DIR,
    )

    assert contract["assistant_model_type"] == "gemma4_unified_assistant"
    assert contract["target_model_type"] == "gemma4_unified"
    assert contract["backbone_hidden_size"] == 3840
    assert contract["assistant_hidden_size"] == 1024
    assert contract["shared_cache_geometry"] == {
        "sliding_attention": {"num_key_value_heads": 8, "head_dim": 256},
        "full_attention": {"num_key_value_heads": 1, "head_dim": 512},
    }


def test_gemma4_12b_mtp_model_paths_are_validated(tmp_path):
    from xhmodel_merak.xh_llm.models.gemma4_series.mtp_workflow import (
        require_model_dir,
    )

    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="assistant.*does not exist"):
        require_model_dir(str(missing), role="assistant")

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(FileNotFoundError, match="target.*missing config.json"):
        require_model_dir(str(incomplete), role="target")

    complete = tmp_path / "complete"
    complete.mkdir()
    (complete / "config.json").write_text("{}", encoding="utf-8")
    assert require_model_dir(str(complete), role="target") == complete.resolve()


def test_gemma4_12b_workflow_configs_keep_common_export_contract():
    from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig

    config_root = Path(
        "configs_merak/workflows/xh2a/llm_models/gemma4_series/12b_unified"
    )
    expected = {
        "model_type": "Gemma4UnifiedForConditionalGeneration",
        "context_max_length": 2048,
        "prefill_chunk_length": 320,
        "sliding_kv_cache_input_mode": "slice_window",
    }
    for config_name in (
        "gemma4_12b_unified_full.yaml",
        "gemma4_12b_unified_full_mtp.yaml",
        "gemma4_12b_unified_autoround.yaml",
    ):
        model_cfg = WorkflowConfig.from_file(config_root / config_name).export[
            "model"
        ]
        assert {key: model_cfg[key] for key in expected} == expected
        assert model_cfg["quant_scheme"]["quant_type"] == "w8a8h1_sefp"
        assert (
            model_cfg["visual_config"]["quant_scheme"]["quant_type"]
            == "w8a8h1_sefp"
        )
        assert (
            model_cfg["video_visual_config"]["quant_scheme"]["quant_type"]
            == "w8a8h1_sefp"
        )
        assert (
            model_cfg["audio_config"]["quant_scheme"]["quant_type"]
            == "w8a8h1_sefp"
        )
