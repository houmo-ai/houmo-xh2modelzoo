# Copyright 2025 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import yaml


def _write_checkpoint_config(
    root: Path,
    architecture: str = "HunYuanVLForConditionalGeneration",
    generation_eos_token_id: int | list[int] = 120020,
) -> None:
    config = {
        "architectures": [architecture],
        "image_token_id": 120120,
        "text_config": {
            "bos_token_id": 120000,
            "eos_token_id": 120007,
            "pad_token_id": 120002,
            "max_position_embeddings": 131072,
        },
        "vision_config": {
            "patch_size": 16,
            "temporal_patch_size": 1,
            "spatial_patch_size": 1,
            "spatial_merge_size": 2,
            "hidden_size": 1152,
            "text_hidden_size": 1024,
            "num_hidden_layers": 27,
            "cat_extra_token": 1,
        },
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "generation_config.json").write_text(
        json.dumps({"eos_token_id": generation_eos_token_id}),
        encoding="utf-8",
    )


def test_hunyuan_ocr_config_derives_native_architecture_contract(tmp_path: Path) -> None:
    from xhmodel_merak.xh_llm.models.hunyuan_ocr import (
        XHHunYuanOCRModelConfig,
        XHHunYuanOCRVisualConfig,
    )

    _write_checkpoint_config(tmp_path)
    config = XHHunYuanOCRModelConfig(model_name="hunyuan_ocr", hf_model=str(tmp_path))

    assert config.model_type == "HunYuanVLForConditionalGeneration"
    assert config.context_max_length == 131072
    assert config.image_token_id == 120120
    assert config.tokenizer_eos_token_id == 120007
    assert config.generation_eos_token_id == 120020
    assert isinstance(config.visual_config, XHHunYuanOCRVisualConfig)
    assert config.visual_config.model_type == "HunYuanVLForConditionalGeneration_visual"
    assert config.visual_config.model_name == "hunyuan_ocr_visual"
    assert config.visual_config.patch_size == 16
    assert config.visual_config.spatial_merge_size == 2
    assert config.visual_config.cat_extra_token is True
    assert config.visual_config.image_size_w == 896
    assert config.visual_config.image_size_h == 1152
    assert config.visual_config.grid_h == 72
    assert config.visual_config.grid_w == 56
    assert config.visual_config.num_patches == 4032
    assert config.visual_config.image_token_count == 1046

    restored = XHHunYuanOCRModelConfig.from_dict(config.to_dict())
    assert restored.to_dict() == config.to_dict()


def test_hunyuan_ocr_config_preserves_multiple_generation_eos_tokens(tmp_path: Path) -> None:
    from xhmodel_merak.xh_llm.models.hunyuan_ocr import XHHunYuanOCRModelConfig

    _write_checkpoint_config(tmp_path, generation_eos_token_id=[120007, 120020, 120007])

    config = XHHunYuanOCRModelConfig(model_name="hunyuan_ocr", hf_model=str(tmp_path))

    assert config.generation_eos_token_id == [120007, 120020]
    restored = XHHunYuanOCRModelConfig.from_dict(config.to_dict())
    assert restored.generation_eos_token_id == [120007, 120020]


def test_hunyuan_ocr_visual_config_rejects_an_unsupported_fixed_bucket() -> None:
    from xhmodel_merak.xh_llm.models.hunyuan_ocr import XHHunYuanOCRVisualConfig

    with pytest.raises(ValueError, match="visual bucket is not approved"):
        XHHunYuanOCRVisualConfig(
            model_name="unsupported_visual",
            image_size_w=1024,
            image_size_h=1024,
        )


def test_hunyuan_ocr_visual_config_accepts_manifest_approved_bucket() -> None:
    from xhmodel_merak.xh_llm.models.hunyuan_ocr import XHHunYuanOCRVisualConfig

    config = XHHunYuanOCRVisualConfig(
        model_name="approved_visual",
        image_size_w=1280,
        image_size_h=960,
        approved_bucket_sizes=[[1280, 960], [928, 1312]],
    )

    assert config.image_size_w == 1280
    assert config.image_size_h == 960
    assert config.grid_h == 60
    assert config.grid_w == 80
    assert config.num_patches == 4800
    assert config.image_token_count == 1232


def test_hunyuan_ocr_config_rejects_a_different_checkpoint_architecture(tmp_path: Path) -> None:
    from xhmodel_merak.xh_llm.models.hunyuan_ocr import XHHunYuanOCRModelConfig

    _write_checkpoint_config(tmp_path, architecture="DifferentForConditionalGeneration")

    with pytest.raises(ValueError, match="architectures"):
        XHHunYuanOCRModelConfig(model_name="wrong", hf_model=str(tmp_path))


def test_hunyuan_ocr_scanner_finds_main_and_visual_once() -> None:
    from xhmodel_merak.xh_llm.scan_model_types import (
        get_support_all_model_types,
        get_support_master_model_types,
        parse_register_llm_models,
    )

    package = Path("xhmodel_merak/xh_llm/models/hunyuan_ocr").resolve()
    declarations = []
    for python_file in package.glob("*.py"):
        declarations.extend(parse_register_llm_models(python_file))

    by_type = {row["model_type"]: row for row in declarations}
    assert set(by_type) == {
        "HunYuanOCR_DFlash_Draft",
        "HunYuanVLForConditionalGeneration",
        "HunYuanVLForConditionalGeneration_visual",
    }
    assert all(row["force"] is False for row in declarations)
    assert by_type["HunYuanVLForConditionalGeneration"]["master"] is True
    assert by_type["HunYuanOCR_DFlash_Draft"]["master"] is False
    assert by_type["HunYuanVLForConditionalGeneration_visual"]["master"] is False
    assert get_support_all_model_types()["HunYuanOCR_DFlash_Draft"] == "hunyuan_ocr"
    assert get_support_all_model_types()["HunYuanVLForConditionalGeneration"] == "hunyuan_ocr"
    assert get_support_all_model_types()["HunYuanVLForConditionalGeneration_visual"] == "hunyuan_ocr"
    assert "HunYuanVLForConditionalGeneration" in get_support_master_model_types()
    assert "HunYuanOCR_DFlash_Draft" not in get_support_master_model_types()
    assert "HunYuanVLForConditionalGeneration_visual" not in get_support_master_model_types()


def test_hunyuan_ocr_registration_and_auto_entrypoints(tmp_path: Path) -> None:
    from transformers import AutoModelForImageTextToText
    from transformers.models.hunyuan_vl.modeling_hunyuan_vl import (
        HunYuanVLForConditionalGeneration,
    )

    from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel
    from xhmodel_merak.xh_llm.builder import get_model_class
    from xhmodel_merak.xh_llm.models.hunyuan_ocr import (
        XHHunYuanOCRModel,
        XHHunYuanOCRModelConfig,
        XHHunYuanOCRVisualModel,
    )

    _write_checkpoint_config(tmp_path)
    base = {
        "model_name": "hunyuan_ocr",
        "model_type": "HunYuanVLForConditionalGeneration",
        "hf_model": str(tmp_path),
    }
    config = AutoLLMConfig.from_pretrained(base)
    model = AutoLLMModel.from_pretrained(config)

    assert isinstance(config, XHHunYuanOCRModelConfig)
    assert isinstance(model, XHHunYuanOCRModel)
    assert isinstance(model.visual, XHHunYuanOCRVisualModel)
    assert get_model_class(base) is XHHunYuanOCRModel
    assert model.HF_MODEL_CLS is HunYuanVLForConditionalGeneration
    assert model.HF_AUTO_MODEL_CLS is AutoModelForImageTextToText
    assert model.HMONNXINFERENCE_CLS is None
    assert model.transformers_min_version == "5.13.0"
    assert model.transformers_max_version == "5.13.0"

    visual_config = AutoLLMConfig.from_pretrained(config.visual_config.to_dict())
    visual_model = AutoLLMModel.from_pretrained(visual_config)
    assert isinstance(visual_model, XHHunYuanOCRVisualModel)
    assert get_model_class(config.visual_config) is XHHunYuanOCRVisualModel

    with patch.object(visual_model.HF_AUTO_MODEL_CLS, "from_pretrained") as load_checkpoint:
        for method in (
            visual_model.get_native_model,
            visual_model.get_empty_native_model,
            visual_model.get_compatible_native_model,
        ):
            with pytest.raises(RuntimeError, match="load it through XHHunYuanOCRModel"):
                method()
        load_checkpoint.assert_not_called()


def test_hunyuan_ocr_yaml_selects_model_specific_workflow(tmp_path: Path) -> None:
    from xhmodel_merak.xh_llm.models.hunyuan_ocr import HunyuanOCRWorkflow
    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

    model_dir = tmp_path / "checkpoint"
    model_dir.mkdir()
    _write_checkpoint_config(model_dir)
    config_path = tmp_path / "hunyuan_ocr.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "quant": None,
                "export": {
                    "model": {
                        "chip_arch": "XH2a",
                        "model_type": "HunYuanVLForConditionalGeneration",
                        "hf_model": None,
                        "model_name": "hunyuan_ocr_base",
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    workflow = AutoLLMWorkflow.from_config(str(model_dir), str(config_path))

    assert isinstance(workflow, HunyuanOCRWorkflow)
    assert workflow.expected_model_config_cls_name == "XHHunYuanOCRModelConfig"
    assert workflow.expected_model_cls_name == "XHHunYuanOCRModel"
    assert workflow.build_input_messages({"image": "document.png", "prompt": "Extract text"}) == [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "document.png"},
                {"type": "text", "text": "Extract text"},
            ],
        }
    ]


def test_hf_compatible_inputs_switch_from_multimodal_prefill_to_text_decode() -> None:
    from xhmodel_merak.xh_llm.models.hunyuan_ocr.hunyuan_ocr_llm_model import (
        _HunyuanOCRHFCompatible,
    )

    runtime = object.__new__(_HunyuanOCRHFCompatible)
    object.__setattr__(runtime, "_past_seq_length", 0)
    input_ids = torch.tensor([[11, 12, 13]])
    pixel_values = torch.ones(4, 3)
    image_grid_thw = torch.tensor([[1, 2, 2]])
    token_types = torch.tensor([[0, 1, 1]])

    prefill = runtime.prepare_inputs_for_generation(
        input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        mm_token_type_ids=token_types,
    )

    assert prefill["input_ids"] is input_ids
    assert prefill["pixel_values"] is pixel_values
    assert prefill["image_grid_thw"] is image_grid_thw
    assert prefill["mm_token_type_ids"] is token_types

    object.__setattr__(runtime, "_past_seq_length", 3)
    decode = runtime.prepare_inputs_for_generation(
        input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        mm_token_type_ids=token_types,
    )

    assert decode["input_ids"].tolist() == [[13]]
    assert "pixel_values" not in decode
    assert "image_grid_thw" not in decode
    assert "mm_token_type_ids" not in decode


def test_hf_compatible_forward_delegates_semantic_generation_step() -> None:
    from xhmodel_merak.xh_llm.models.hunyuan_ocr.hunyuan_ocr_llm_model import (
        _HunyuanOCRHFCompatible,
    )

    runtime = object.__new__(_HunyuanOCRHFCompatible)
    object.__setattr__(runtime, "_past_seq_length", 0)
    calls = []

    class FakeLLMRuntime:
        @staticmethod
        def run_generation_step(data):
            calls.append(data)
            return torch.zeros((1, 3, 16))

        @staticmethod
        def get_data_preprocessor():
            raise AssertionError("HF-compatible forward bypassed semantic generation step")

    object.__setattr__(runtime, "_llm_model", FakeLLMRuntime())
    cache = object()

    output = runtime.forward(input_ids=torch.tensor([[1, 2, 3]]), past_key_values=cache)

    assert output.logits.shape == (1, 3, 16)
    assert output.past_key_values is cache
    assert len(calls) == 1
    assert calls[0]["past_seq_length"] == 0
    torch.testing.assert_close(calls[0]["input_ids"], torch.tensor([[1, 2, 3]]))


def test_text_preprocessor_pads_prefill_and_advances_decode_positions() -> None:
    from xhmodel_merak.xh_llm.models.hunyuan_ocr.data_preprocess import HunyuanOCRTextDataPreprocess
    from xhmodel_merak.xh_llm.types import CacheList

    key_caches = CacheList([torch.zeros(1)])
    value_caches = CacheList([torch.zeros(1)])
    processor = HunyuanOCRTextDataPreprocess(
        token_embedding=torch.nn.Embedding(32, 8, padding_idx=0),
        input_sequence_length=5,
        context_max_length=8,
        past_key_caches=key_caches,
        past_value_caches=value_caches,
        pad_token_id=0,
    )
    positions = torch.tensor(
        [
            [[4, 5, 6]],
            [[0, 1, 2]],
            [[0, 0, 1]],
            [[0, 0, 0]],
        ]
    )

    prefill = processor(
        {
            "input_ids": torch.tensor([[7, 8, 9]]),
            "position_ids": positions,
            "past_seq_length": 0,
        }
    )

    assert prefill[0].shape == (1, 5, 8)
    for axis, expected in zip(prefill[1:5], positions, strict=True):
        torch.testing.assert_close(axis[:, :3], expected)
    assert prefill[5].tolist() == [0]
    assert prefill[6].tolist() == [3]
    assert prefill[7] is key_caches
    assert prefill[8] is value_caches

    decode = processor({"input_ids": torch.tensor([[10]]), "past_seq_length": 3})
    assert all(axis[0, 0].item() == 3 for axis in decode[1:5])
    assert decode[5].tolist() == [3]
    assert decode[6].tolist() == [1]

    with pytest.raises(ValueError, match="exceeds context_max_length"):
        processor({"input_ids": torch.tensor([[10, 11]]), "past_seq_length": 7})


def test_hunyuan_ocr_builder_dynamically_imports_package(tmp_path: Path) -> None:
    _write_checkpoint_config(tmp_path)
    code = f"""
from xhmodel_merak.xh_llm.builder import get_model_class
model_cls = get_model_class({{
    'model_name': 'hunyuan_ocr',
    'model_type': 'HunYuanVLForConditionalGeneration',
    'hf_model': {str(tmp_path)!r},
}})
print(model_cls.__module__, model_cls.__name__)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "xhmodel_merak.xh_llm.models.hunyuan_ocr" in result.stdout
    assert "XHHunYuanOCRModel" in result.stdout


def test_hunyuan_ocr_unknown_model_type_fails() -> None:
    from xhmodel_merak.xh_llm.builder import get_model_class

    with pytest.raises(ValueError, match="Unsupported model_type"):
        get_model_class(
            {
                "model_name": "unknown",
                "model_type": "NotAHunYuanModel",
                "hf_model": "unused",
            }
        )
