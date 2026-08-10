# Copyright 2025 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image


W8_CONFIG = Path("configs_merak/workflows/xh2a/llm_models/hunyuan_ocr/base/hunyuan_ocr_base_xh2a_w8a8.yaml")
W16_CONFIG = Path("configs_merak/workflows/xh2a/llm_models/hunyuan_ocr/base/hunyuan_ocr_base_xh2a_w16a16.yaml")
DFLASH_CONFIG = Path(
    "configs_merak/workflows/xh2a/llm_models/hunyuan_ocr/base/"
    "hunyuan_ocr_base_xh2a_w16a16_dflash.yaml"
)
BUCKET_MANIFEST = Path("examples_merak/llm/hunyuan_ocr/assets/hunyuan_ocr_resolution_buckets.json")
TRAJECTORY_MANIFEST = Path("examples_merak/llm/hunyuan_ocr/assets/hunyuan_ocr_calibration_trajectories.json")


def _conversation(*images: Image.Image) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                *({"type": "image", "image": image} for image in images),
                {"type": "text", "text": "OCR"},
            ],
        }
    ]


class _FakeProcessor:
    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(self, conversation, chat_template=None, **kwargs):
        images = [
            item["image"]
            for message in conversation
            for item in message.get("content", [])
            if item.get("type") == "image"
        ]
        self.calls.append({"image_sizes": [image.size for image in images], "kwargs": kwargs})
        return {"image_grid_thw": torch.tensor([[1, image.height // 16, image.width // 16] for image in images])}


def test_hunyuan_ocr_workflow_configs_reference_approved_assets() -> None:
    from examples_merak.llm.hunyuan_ocr.hunyuan_ocr_resolution_buckets import load_resolution_bucket_manifest
    from xhmodel_merak.xh_llm.workflows import WorkflowConfig

    w16_model = WorkflowConfig.from_file(str(W16_CONFIG)).export["model"]
    w8_model = WorkflowConfig.from_file(str(W8_CONFIG)).export["model"]
    dflash_model = WorkflowConfig.from_file(str(DFLASH_CONFIG)).export["model"]

    assert w16_model["resolution_bucket_manifest"] == str(BUCKET_MANIFEST)
    assert w16_model["smoke_only"] is True
    assert w8_model["quant_scheme"]["quant_type"] == "w8a8h1_sefp"
    assert w8_model["smoke_only"] is False
    assert w8_model["calibration"]["reference_trajectory_manifest"] == str(TRAJECTORY_MANIFEST)
    assert len(w8_model["calibration"]["images"]) == 5
    assert all(Path(image).is_file() for image in w8_model["calibration"]["images"])
    assert dflash_model["spec_decode_mode"] == "dflash"
    assert dflash_model["num_draft_tokens"] == 15
    assert dflash_model["dflash_config"]["hf_model"] == "data/models/HunyuanOCR/dflash"
    assert dflash_model["resolution_bucket_manifest"] == str(BUCKET_MANIFEST)
    manifest = load_resolution_bucket_manifest(w8_model["resolution_bucket_manifest"], require_approved=True)
    assert manifest["status"] == "approved"


def test_reference_trajectories_validate_manifest_schema_and_image_hashes(tmp_path: Path) -> None:
    from xhmodel_merak.xh_llm.models.hunyuan_ocr.calibration import (
        build_calibration_messages,
        load_calibration_samples,
        load_reference_trajectories,
    )

    image = tmp_path / "page.png"
    image.write_bytes(b"page")
    manifest = tmp_path / "reference_trajectories.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "trajectories": [
                    {
                        "request_index": 0,
                        "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                        "token_ids": [17, 23],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    prompt, images = load_calibration_samples({"prompt": " Extract text ", "images": [str(image)]})
    trajectories = load_reference_trajectories(
        {"reference_trajectory_manifest": str(manifest)},
        request_count=1,
        image_paths=images,
    )

    assert prompt == "Extract text"
    assert trajectories == ((0, (17, 23)),)
    assert build_calibration_messages(images[0], prompt)[0]["content"][0]["image"] == str(image.resolve())

    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "trajectories": [{"request_index": 0, "image_sha256": "0" * 64, "token_ids": [17]}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="image SHA256 mismatch"):
        load_reference_trajectories(
            {"reference_trajectory_manifest": str(manifest)},
            request_count=1,
            image_paths=images,
        )


def test_resolution_bucket_manifest_routes_expected_document_shapes() -> None:
    from examples_merak.llm.hunyuan_ocr.hunyuan_ocr_resolution_buckets import (
        load_resolution_bucket_manifest,
        route_image,
    )

    manifest = load_resolution_bucket_manifest(BUCKET_MANIFEST, require_approved=True)

    horizontal = route_image(1600, 900, manifest)
    vertical = route_image(900, 1600, manifest)
    unsupported = route_image(4096, 4096, manifest)

    assert horizontal["accepted"] is True
    assert horizontal["bucket_id"] == "bucket_1440x896"
    assert vertical["accepted"] is True
    assert vertical["bucket_id"] == "bucket_672x1408"
    assert unsupported["accepted"] is False
    assert unsupported["reason"] == "no_feasible_bucket"


def test_multi_bucket_processor_letterboxes_and_preserves_prompt_order() -> None:
    from examples_merak.llm.hunyuan_ocr.hunyuan_ocr_resolution_buckets import load_resolution_bucket_manifest
    from xhmodel_merak.xh_llm.models.hunyuan_ocr import HunyuanOCRMultiBucketProcessor

    processor = _FakeProcessor()
    adapter = HunyuanOCRMultiBucketProcessor(processor, load_resolution_bucket_manifest(BUCKET_MANIFEST))

    result = adapter.apply_chat_template(
        _conversation(Image.new("RGB", (1600, 900)), Image.new("RGB", (900, 1600))),
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )

    assert result["image_grid_thw"].tolist() == [[1, 56, 90], [1, 88, 42]]
    assert [route["image_index"] for route in adapter.last_routes] == [0, 1]
    assert [route["bucket_id"] for route in adapter.last_routes] == ["bucket_1440x896", "bucket_672x1408"]
    assert processor.calls == [
        {
            "image_sizes": [(1440, 896), (672, 1408)],
            "kwargs": {"tokenize": True, "return_tensors": "pt", "return_dict": True, "do_resize": False},
        }
    ]


def test_formal_w8_workflow_validates_reference_trajectories_before_export(monkeypatch, tmp_path: Path) -> None:
    from xhmodel_merak.xh_llm.models.hunyuan_ocr import HunyuanOCRWorkflow
    from xhmodel_merak.xh_llm.workflows.base import BaseLLMWorkflow
    from xhmodel_merak.xh_llm.workflows.result import QuantResult

    image = tmp_path / "page.png"
    image.write_bytes(b"page")
    config_path = tmp_path / "w8.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "quant": None,
                "export": {
                    "model": {
                        "chip_arch": "XH2a",
                        "model_type": "HunYuanVLForConditionalGeneration",
                        "model_name": "hunyuan_ocr_w8a8",
                        "quant_scheme": {"quant_type": "w8a8h1_sefp", "ops": {}},
                        "smoke_only": False,
                        "calibration": {"prompt": "Extract text", "images": [str(image)]},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(BaseLLMWorkflow, "export", lambda self, **kwargs: calls.append(kwargs))

    workflow = HunyuanOCRWorkflow(str(tmp_path / "checkpoint"), str(config_path))

    with pytest.raises(ValueError, match="reference_trajectories"):
        workflow.export(
            quant_result=QuantResult(raw_model_dir=workflow.model_dir, skipped=True),
            output_dir=str(tmp_path / "export"),
            device="cpu",
        )
    assert calls == []
