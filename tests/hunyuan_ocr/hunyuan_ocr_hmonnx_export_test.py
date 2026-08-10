# Copyright 2025 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

from xhmodel_merak.xh_llm.types import LLMModelState


TEXT_EXPORT_SCRIPT = Path("examples_merak/llm/hunyuan_ocr/export_text_hmonnx.py")
BUNDLE_EXPORT_SCRIPT = Path("examples_merak/llm/hunyuan_ocr/export_hmonnx.py")
WORKFLOW_CONFIG = Path(
    "configs_merak/workflows/xh2a/llm_models/hunyuan_ocr/base/"
    "hunyuan_ocr_base_xh2a_w16a16.yaml"
)
DFLASH_WORKFLOW_CONFIG = Path(
    "configs_merak/workflows/xh2a/llm_models/hunyuan_ocr/base/"
    "hunyuan_ocr_base_xh2a_w16a16_dflash.yaml"
)


def _load_script(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_text_artifacts(export_dir: Path) -> None:
    for relative_path in (
        "hf_config/config.json",
        "quant_embedding.pt",
        "prefill/model.onnx",
        "prefill/model_external_data/weights.bin",
        "decode/model.onnx",
        "decode/model_external_data/weights.bin",
    ):
        path = export_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake")


def test_hunyuan_ocr_text_metadata_is_relocatable_and_text_only(tmp_path: Path) -> None:
    from xhmodel_merak.xh_llm.models.hunyuan_ocr import HunyuanOCRTextExportMeta

    export_dir = tmp_path / "first" / "text"
    _write_text_artifacts(export_dir)
    metadata = HunyuanOCRTextExportMeta(
        model_config={"model_name": "hunyuan_ocr", "model_type": "HunYuanVLForConditionalGeneration"},
        hf_config="hf_config",
        quant_embedding="quant_embedding.pt",
        prefill_hmonnx="prefill/model.onnx",
        decode_hmonnx="decode/model.onnx",
        max_sequence_length=131072,
        prefill_chunk_length=256,
        output_names=["logits"],
    )

    meta_path = Path(metadata.save(export_dir / "text_meta_info.json"))
    serialized = json.loads(meta_path.read_text(encoding="utf-8"))

    assert serialized["schema_version"] == 1
    assert serialized["meta"]["class_name"] == "HunyuanOCRTextExportMeta"
    assert serialized["cache_update_mode"] == "in_place"
    assert serialized["output_names"] == ["logits"]
    assert "visual_config" not in serialized["model_config"]

    relocated_dir = tmp_path / "release" / "text"
    shutil.copytree(export_dir, relocated_dir)
    loaded = HunyuanOCRTextExportMeta.from_file(relocated_dir / "text_meta_info.json")

    assert loaded.prefill_hmonnx == str(relocated_dir / "prefill/model.onnx")
    assert loaded.decode_hmonnx == str(relocated_dir / "decode/model.onnx")
    assert loaded.quant_embedding == str(relocated_dir / "quant_embedding.pt")


def test_hunyuan_ocr_visual_metadata_is_relocatable(tmp_path: Path) -> None:
    from xhmodel_merak.xh_llm.models.hunyuan_ocr import HunyuanOCRVisualMeta

    root = tmp_path / "visual"
    graph = root / "model.onnx"
    external_data = root / "model_external_data"
    graph.parent.mkdir(parents=True)
    graph.write_bytes(b"fake")
    external_data.mkdir()
    (external_data / "weights.bin").write_bytes(b"fake")
    metadata = HunyuanOCRVisualMeta(
        hmonnx="model.onnx",
        external_data="model_external_data",
        image_size_w=896,
        image_size_h=1152,
    )

    meta_path = Path(metadata.save(root / "visual_meta_info.json"))
    loaded = HunyuanOCRVisualMeta.from_file(meta_path)

    assert loaded.hmonnx == str(graph)
    assert loaded.external_data == str(external_data)
    assert loaded.input_shape == [4032, 768]
    assert loaded.output_shape == [1, 1046, 1024]
    assert loaded.image_grid_thw == [1, 72, 56]
    assert loaded.image_token_count == 1046


def test_hunyuan_ocr_bundle_visual_bucket_artifacts_are_relocatable(tmp_path: Path) -> None:
    from xhmodel_merak.xh_llm.models.hunyuan_ocr import HunyuanOCRModelMeta

    root = tmp_path / "bundle"
    metadata_path = root / "golden_meta_info.json"
    for relative_path in (
        "hf_config/config.json",
        "quant_embedding.pt",
        "prefill/model.onnx",
        "decode/model.onnx",
        "visual/default/model.onnx",
        "visual/default/model_external_data/weights.bin",
        "visual/routed/model.onnx",
        "visual/routed/model_external_data/weights.bin",
    ):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake")
    metadata_path.write_text(
        json.dumps(
            {
                "meta": {"class_name": "HunyuanOCRModelMeta"},
                "model_config": {"model_type": "HunYuanVLForConditionalGeneration"},
                "hf_config": "hf_config",
                "quant_embedding": "quant_embedding.pt",
                "prefill_hmonnx": "prefill/model.onnx",
                "decode_hmonnx": "decode/model.onnx",
                "visual_config": {
                    "hmonnx": "visual/default/model.onnx",
                    "external_data": "visual/default/model_external_data",
                },
                "visual_buckets": {
                    "bucket_routed": {
                        "hmonnx": "visual/routed/model.onnx",
                        "external_data": "visual/routed/model_external_data",
                        "image_grid_thw": [1, 82, 58],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = HunyuanOCRModelMeta.from_file(metadata_path)

    assert loaded.visual_config.hmonnx == str(root / "visual/default/model.onnx")
    assert loaded.visual_config.external_data == str(root / "visual/default/model_external_data")
    assert loaded.visual_buckets["bucket_routed"].hmonnx == str(root / "visual/routed/model.onnx")
    assert loaded.visual_buckets["bucket_routed"].external_data == str(
        root / "visual/routed/model_external_data"
    )


def test_export_text_hmonnx_does_not_export_visual(monkeypatch, tmp_path: Path) -> None:
    import torch

    from xhmodel_merak.xh_llm.models.hunyuan_ocr import (
        HunyuanOCRTextExportMeta,
        XHHunYuanOCRModel,
        XHHunYuanOCRModelConfig,
    )

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["HunYuanVLForConditionalGeneration"],
                "text_config": {"max_position_embeddings": 131072},
                "vision_config": {"cat_extra_token": True},
            }
        ),
        encoding="utf-8",
    )
    model = XHHunYuanOCRModel(
        XHHunYuanOCRModelConfig(model_name="hunyuan_ocr", hf_model=str(checkpoint), context_max_length=131072)
    )
    model._state = LLMModelState.QUANTED_ALIGNED
    model.visual._state = LLMModelState.WRAP
    model.embed_tokens = torch.nn.Embedding(16, 1024)
    model.pad_token_id = 0
    model.kvcache_config.num_layers = 24
    model.kvcache_config.kv_cache_shape = [1, 8, 131072, 128]
    calls: list[str] = []

    class FakeQuantedModel:
        def fixed(self):
            calls.append("text.fixed")

    def fake_export(exported_info):
        assert "visual" not in model._models
        assert isinstance(exported_info.meta, HunyuanOCRTextExportMeta)
        calls.append("text._export_hmonnx")
        root = Path(exported_info.exported_dir)
        _write_text_artifacts(root)
        exported_info.meta.prefill_hmonnx = "prefill/model.onnx"
        exported_info.meta.decode_hmonnx = "decode/model.onnx"
        return exported_info

    model._quanted_model = FakeQuantedModel()
    monkeypatch.setattr(model, "_export_hmonnx", fake_export)
    monkeypatch.setattr(model.visual, "export_hmonnx", lambda _: calls.append("visual.export_hmonnx"))

    metadata = model.export_text_hmonnx(str(tmp_path / "export"))

    assert isinstance(metadata, HunyuanOCRTextExportMeta)
    assert calls == ["text.fixed", "text._export_hmonnx"]
    assert model._models["visual"] is model.visual
    assert (tmp_path / "export/text/text_meta_info.json").is_file()
    assert not list((tmp_path / "export").rglob("golden_meta_info.json"))


def test_full_export_quantizes_text_without_recursing_into_visual(monkeypatch, tmp_path: Path) -> None:
    import torch

    from xhmodel_merak.xh_llm.models.hunyuan_ocr import (
        HunyuanOCRModelMeta,
        HunyuanOCRVisualMeta,
        XHHunYuanOCRModel,
        XHHunYuanOCRModelConfig,
    )

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _write_checkpoint_config = {
        "architectures": ["HunYuanVLForConditionalGeneration"],
        "text_config": {"max_position_embeddings": 131072},
        "vision_config": {"cat_extra_token": True},
    }
    (checkpoint / "config.json").write_text(json.dumps(_write_checkpoint_config), encoding="utf-8")
    manifest = {
        "status": "approved",
        "buckets": [
            {
                "id": "bucket_1280x960",
                "width": 1280,
                "height": 960,
            }
        ],
    }
    model = XHHunYuanOCRModel(
        XHHunYuanOCRModelConfig(
            model_name="hunyuan_ocr",
            hf_model=str(checkpoint),
            context_max_length=131072,
        )
    )
    model.config.resolution_bucket_contract = manifest
    model._state = LLMModelState.WRAP
    model.embed_tokens = torch.nn.Embedding(16, 1024)
    model.pad_token_id = 0
    model.kvcache_config.num_layers = 1
    model.kvcache_config.kv_cache_shape = [1, 1, 16, 1]
    calls = []

    def fake_to_quanted_aligned():
        assert "visual" not in model._models
        model._state = LLMModelState.QUANTED_ALIGNED
        model._quanted_model = type(
            "FakeSwitcher",
            (),
            {
                "prefill": type("FakeGraph", (), {"fixed": lambda self: None})(),
                "decode": type("FakeGraph", (), {"fixed": lambda self: None})(),
                "fixed": lambda self: None,
            },
        )()
        calls.append("text.quantized")

    def fake_export_text(exported_info):
        assert "visual" not in model._models
        root = Path(exported_info.exported_dir)
        _write_text_artifacts(root)
        exported_info.meta.prefill_hmonnx = "prefill/model.onnx"
        exported_info.meta.decode_hmonnx = "decode/model.onnx"
        calls.append("text.exported")
        return exported_info

    bundle_root = tmp_path / "bundle"
    visual_meta = HunyuanOCRVisualMeta(
        hmonnx=str(bundle_root / "visual/bucket_1280x960/model.onnx"),
        external_data=str(bundle_root / "visual/bucket_1280x960/model_external_data"),
        image_size_w=1280,
        image_size_h=960,
        input_shape=[4800, 768],
        output_shape=[1, 1232, 1024],
        image_grid_thw=[1, 60, 80],
    )
    Path(visual_meta.hmonnx).parent.mkdir(parents=True)
    Path(visual_meta.hmonnx).write_bytes(b"fake")
    Path(visual_meta.external_data).mkdir()
    monkeypatch.setattr(model, "to_quanted_aligned", fake_to_quanted_aligned)
    monkeypatch.setattr(model, "_export_hmonnx", fake_export_text)
    monkeypatch.setattr(model, "_export_additional_visual_bucket", lambda *_args: visual_meta)

    metadata = model.export_hmonnx(str(bundle_root))

    assert isinstance(metadata, HunyuanOCRModelMeta)
    assert calls == ["text.quantized", "text.exported"]
    assert metadata.visual_config.bucket_id == "bucket_1280x960"
    assert metadata.visual_buckets["bucket_1280x960"].hmonnx.startswith("visual/")
    assert (bundle_root / "golden_meta_info.json").is_file()
    assert model._models["visual"] is model.visual


def test_export_cli_loads_text_only_config_from_workflow_yaml(tmp_path: Path) -> None:
    module = _load_script(TEXT_EXPORT_SCRIPT, "hunyuan_ocr_text_export_testmod")

    values = module.load_model_config(WORKFLOW_CONFIG, tmp_path / "checkpoint")

    assert values["model_type"] == "HunYuanVLForConditionalGeneration"
    assert values["context_max_length"] == 131072
    assert values["prefill_chunk_length"] == 256
    assert values["max_pe_length"] == 131072
    assert values["num_logits_to_keep"] == 1
    assert values["quant_scheme"]["quant_type"] == "w16a16h0_sefp"
    assert "resolution_bucket_manifest" not in values
    assert "visual_config" not in values


def test_full_export_cli_loads_bundle_config_from_workflow_yaml(tmp_path: Path) -> None:
    module = _load_script(BUNDLE_EXPORT_SCRIPT, "hunyuan_ocr_bundle_export_testmod")

    values = module.load_model_config(WORKFLOW_CONFIG, tmp_path / "checkpoint")

    assert values["model_type"] == "HunYuanVLForConditionalGeneration"
    assert values["context_max_length"] == 131072
    assert values["prefill_chunk_length"] == 256
    assert values["resolution_bucket_manifest"] == (
        "examples_merak/llm/hunyuan_ocr/assets/hunyuan_ocr_resolution_buckets.json"
    )
    assert values["quant_scheme"]["quant_type"] == "w16a16h0_sefp"
    assert values["visual_config"]["quant_scheme"]["quant_type"] == "w16a16h0_sefp"


def test_dflash_export_cli_relocates_target_and_draft_checkpoints(tmp_path: Path) -> None:
    module = _load_script(BUNDLE_EXPORT_SCRIPT, "hunyuan_ocr_dflash_bundle_export_testmod")
    checkpoint = tmp_path / "checkpoint"

    values = module.load_model_config(DFLASH_WORKFLOW_CONFIG, checkpoint)

    assert values["hf_model"] == str(checkpoint)
    assert values["dflash_config"]["hf_model"] == str(checkpoint / "dflash")
