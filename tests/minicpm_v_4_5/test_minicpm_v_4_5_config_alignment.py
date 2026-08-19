# Copyright 2026 HOUMO AI
#
# File: test_minicpm_v_4_5_config_alignment.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for MiniCPM-V-4.5 config alignment with repository VLM conventions.

The workflow config must accept the qwen2_vl-style ``export.model.visual_config``
nested structure (``max_size_w`` / ``max_size_h`` / ``patch_capacity`` /
``quant_scheme``) while the export layout (``vision_1x`` / ``vision_6x`` /
``prefill`` / ``decode``) stays unchanged. These tests exercise only the
config-parse seam, no model weights.
"""

from __future__ import annotations

import pytest


def _load_workflow_module():
    from xhmodel_merak.xh_llm.models.minicpm_v_4_5 import workflow as wf

    return wf


def test_workflow_reads_visual_config_from_model_config():
    """The workflow export must consume export.model.visual_config for vision params."""
    wf = _load_workflow_module()
    assert hasattr(wf, "MiniCPMV45Workflow")
    assert hasattr(wf.MiniCPMV45Workflow, "visual_config_from_model")


def test_visual_config_from_model_uses_explicit_patch_capacity():
    wf = _load_workflow_module()
    result = wf.MiniCPMV45Workflow.visual_config_from_model(
        {
            "visual_config": {
                "max_size_w": 448,
                "max_size_h": 448,
                "max_size_t": 6,
                "patch_capacity": 1600,
                "quant_scheme": {"quant_type": "w8a8h1_sefp"},
            }
        }
    )
    assert result["patch_capacity"] == 1600
    assert result["group_capacity"] == 6
    assert result["quant_type"] == "w8a8h1_sefp"


def test_visual_config_from_model_legacy_resolution_fallback():
    wf = _load_workflow_module()
    result = wf.MiniCPMV45Workflow.visual_config_from_model({"visual_config": {"max_size_w": 448}})
    assert result["patch_capacity"] == 1024
    assert result["group_capacity"] == 6
    assert result["quant_type"] == "w8a8h1_sefp"


def test_visual_config_preserves_complete_quant_scheme():
    wf = _load_workflow_module()
    result = wf.MiniCPMV45Workflow.visual_config_from_model(
        {
            "visual_config": {
                "patch_capacity": 1600,
                "quant_scheme": {
                    "quant_type": "w8a8h1_sefp",
                    "ops": {"MatMul": {"act_scheme": {"bits": 16}}},
                },
            }
        }
    )
    assert result["quant_scheme"]["ops"]["MatMul"]["act_scheme"]["bits"] == 16


def test_visual_config_from_model_none_when_absent():
    wf = _load_workflow_module()
    assert wf.MiniCPMV45Workflow.visual_config_from_model({}) is None


def test_visual_config_from_model_rejects_non_square_slice():
    wf = _load_workflow_module()
    with pytest.raises(ValueError, match="square"):
        wf.MiniCPMV45Workflow.visual_config_from_model({"visual_config": {"max_size_w": 448, "max_size_h": 336}})


def test_visual_config_from_model_rejects_resolution_not_divisible_by_patch():
    wf = _load_workflow_module()
    with pytest.raises(ValueError, match="divisible"):
        wf.MiniCPMV45Workflow.visual_config_from_model({"visual_config": {"max_size_w": 450}})


def test_runtime_normalize_meta_from_golden_vllm_meta():
    from xhmodel_merak.xh_llm.models.minicpm_v_4_5.inference import MiniCPMV45HMONNXRuntime

    raw = {
        "create_time": "2026-08-18 00:00:00",
        "model_config": {
            "hf_model": "/data03/nfs_shared/llm_models/openbmb/MiniCPM-V-4_5",
            "context_max_length": 8192,
            "prefill_chunk_length": 256,
        },
        "prefill_hmonnx": "prefill/x_prefill_with_act.onnx",
        "decode_hmonnx": "decode/x_decode_with_act.onnx",
        "quant_embedding": "quant_embedding.pt",
        "hf_config": "hf_config",
        "visual_config": {"patch_capacity": 1600, "hmonnx": "vision_1x/x_visual.onnx"},
        "visual_video_config": {"patch_capacity": 1600, "group_capacity": 6, "hmonnx": "vision_6x/x_visual.onnx"},
    }
    meta = MiniCPMV45HMONNXRuntime._normalize_meta(raw)
    assert meta["model_type"] == "MiniCPM-V-4.5"
    assert meta["hf_model"].endswith("MiniCPM-V-4_5")
    assert meta["vision"]["hmonnx"] == "vision_1x/x_visual.onnx"
    assert meta["vision_video"]["group_capacity"] == 6
    assert meta["llm"]["prefill_hmonnx"].startswith("prefill/")
    assert meta["llm"]["context_max_length"] == 8192


def test_runtime_normalize_meta_accepts_legacy_layout():
    from xhmodel_merak.xh_llm.models.minicpm_v_4_5.inference import MiniCPMV45HMONNXRuntime

    legacy = {
        "model_type": "MiniCPM-V-4.5",
        "hf_model": "/data03/nfs_shared/llm_models/openbmb/MiniCPM-V-4_5",
        "vision": {"hmonnx": "vision_1x/x.onnx"},
        "llm": {"metadata": "golden_meta_info.json"},
    }
    meta = MiniCPMV45HMONNXRuntime._normalize_meta(legacy)
    assert meta is legacy


def test_runtime_normalize_meta_requires_model_config():
    from xhmodel_merak.xh_llm.models.minicpm_v_4_5.inference import MiniCPMV45HMONNXRuntime

    with pytest.raises(ValueError, match="model_config"):
        MiniCPMV45HMONNXRuntime._normalize_meta({"prefill_hmonnx": "x"})


def test_build_messages_ignores_empty_video_list_for_image_requests():
    from PIL import Image

    from xhmodel_merak.xh_llm.models.minicpm_v_4_5.inference import _build_messages

    messages, media, temporal_ids = _build_messages(
        {
            "images": [Image.new("RGB", (16, 16))],
            "videos": [],
            "text": "描述图片",
        }
    )
    assert len(media) == 1
    assert temporal_ids is None
    assert "(<image>./</image>)" in messages[0]["content"]


def test_w8a8_config_declares_llm_quant_type():
    import yaml

    with open(
        "configs_merak/workflows/xh2a/llm_models/minicpm_v_4_5/8b/minicpm_v_4_5_xh2a_w8a8.yaml",
        encoding="utf-8",
    ) as f:
        cfg = yaml.safe_load(f)
    assert cfg["export"]["model"]["quant_scheme"]["quant_type"] == "w8a8h1_sefp"
    assert "quant_type" not in cfg["export"]["components"]["llm"]


def test_variant_configs_declare_llm_quant_type():
    import yaml

    cases = {
        "minicpm_v_4_5_xh2a_w4a8_gptq.yaml": "w4a8h0_ssfp",
    }
    for name, expected in cases.items():
        with open(
            f"configs_merak/workflows/xh2a/llm_models/minicpm_v_4_5/8b/{name}",
            encoding="utf-8",
        ) as f:
            cfg = yaml.safe_load(f)
        assert cfg["export"]["model"]["quant_scheme"]["quant_type"] == expected
        assert "quant_type" not in cfg["export"]["components"]["llm"]


def test_configs_declare_full_quant_scheme_and_only_first_block():
    """All tracked configs must carry lm_head node / MatMul op overrides (aligned
    with qwen2_vl) and only_first_block=false for full-model export."""
    import yaml

    names = [
        "minicpm_v_4_5_xh2a_w8a8.yaml",
        "minicpm_v_4_5_xh2a_w4a8_gptq.yaml",
    ]
    for name in names:
        with open(
            f"configs_merak/workflows/xh2a/llm_models/minicpm_v_4_5/8b/{name}",
            encoding="utf-8",
        ) as f:
            cfg = yaml.safe_load(f)
        model = cfg["export"]["model"]
        mq = model["quant_scheme"]
        assert mq["nodes"]["lm_head"]["quant_type"]
        assert mq["ops"]["MatMul"]["act_scheme"]["bits"] == 16
        assert mq["ops"]["MatMul"]["act_schema_2"]["bits"] == 16
        vq = model["visual_config"]["quant_scheme"]
        assert model["visual_config"]["patch_capacity"] == 1600
        assert vq["quant_type"] == "w8a8h1_sefp"
        assert vq["ops"]["MatMul"]["act_scheme"]["bits"] == 16
        assert model["only_first_block"] is False


def test_only_two_release_configs_are_present():
    from pathlib import Path

    config_dir = Path("configs_merak/workflows/xh2a/llm_models/minicpm_v_4_5/8b")
    assert {path.name for path in config_dir.glob("*.yaml")} == {
        "minicpm_v_4_5_xh2a_w8a8.yaml",
        "minicpm_v_4_5_xh2a_w4a8_gptq.yaml",
    }


def test_w4a8_config_uses_reproducible_qwenvl_gptq64_calibration():
    import yaml

    with open(
        "configs_merak/workflows/xh2a/llm_models/minicpm_v_4_5/8b/minicpm_v_4_5_xh2a_w4a8_gptq.yaml",
        encoding="utf-8",
    ) as f:
        cfg = yaml.safe_load(f)
    quant = cfg["quant"]
    assert quant["bits"] == 4
    assert quant["group_size"] == 64
    assert quant["nsamples"] == 64
    assert quant["sampling_seed"] == 0
    assert quant["seqlen"] == 2048
    assert quant["calibration_jsonl"].endswith("minicpm_v_4_5_qwen_mixed_295.jsonl")


def test_calibration_sampling_is_deterministic(tmp_path):
    import json

    from xhmodel_merak.xh_llm.models.minicpm_v_4_5.quant_llm import _load_calibration_samples

    path = tmp_path / "calibration.jsonl"
    path.write_text(
        "".join(json.dumps({"text": f"sample-{index}"}) + "\n" for index in range(10)),
        encoding="utf-8",
    )
    first = _load_calibration_samples(str(path), nsamples=4, sampling_seed=0)
    second = _load_calibration_samples(str(path), nsamples=4, sampling_seed=0)
    assert first == second == ["sample-2", "sample-4", "sample-5", "sample-7"]


def test_workflow_llm_quant_scheme_preserves_nodes_and_ops():
    """The LLM quant_scheme passed to export must keep the yaml nodes/ops
    instead of collapsing to a bare quant_type."""
    # The model-level scheme is passed through unchanged; no component-level
    # quant_type shadow is merged at export time.
    import yaml

    with open(
        "configs_merak/workflows/xh2a/llm_models/minicpm_v_4_5/8b/minicpm_v_4_5_xh2a_w8a8.yaml",
        encoding="utf-8",
    ) as f:
        cfg = yaml.safe_load(f)
    model_quant_scheme = cfg["export"]["model"]["quant_scheme"]
    assert model_quant_scheme["nodes"]["lm_head"]["quant_type"] == "w8a8h1_sefp"
    assert model_quant_scheme["ops"]["MatMul"]["act_scheme"]["bits"] == 16
    assert model_quant_scheme["quant_type"] == "w8a8h1_sefp"


def test_component_quant_type_shadow_is_rejected():
    wf = _load_workflow_module()
    with pytest.raises(ValueError, match="component.*quant_type"):
        wf._reject_component_quant_types({"llm": {"enabled": True, "quant_type": "w8a8h1_sefp"}})


def test_quant_scheme_builder_preserves_visual_ops_and_nodes():
    from xhmodel_merak.xh_llm.models.minicpm_v_4_5.vision_export import _build_quant_scheme

    class FakeQuantScheme:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    scheme = _build_quant_scheme(
        FakeQuantScheme,
        "XH2a",
        {
            "quant_type": "w8a8h1_sefp",
            "nodes": {"resampler": {"quant_type": "w8a8h1_sefp"}},
            "ops": {"MatMul": {"act_scheme": {"bits": 16}}},
        },
    )
    assert scheme.kwargs["nodes"]["resampler"]["quant_type"] == "w8a8h1_sefp"
    assert scheme.kwargs["ops"]["MatMul"]["act_scheme"]["bits"] == 16
