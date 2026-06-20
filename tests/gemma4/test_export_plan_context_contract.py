from __future__ import annotations

import pytest

from xhmodel_merak.xh_llm.models.gemma4_series.export_plan import Gemma4SeriesExportPlan
from xhmodel_merak.xh_llm.models.gemma4_series.variants import Gemma4SeriesVariantSpec
from xhmodel_merak.xh_llm.models.gemma4_series.workflow import Gemma4SeriesWorkflow


def _plan(context_max_length: int) -> Gemma4SeriesExportPlan:
    return Gemma4SeriesExportPlan(
        variant=Gemma4SeriesVariantSpec(
            name="gemma4-e2b",
            topology="dense",
            has_image=True,
            has_video=True,
            has_audio=True,
            has_per_layer_input=True,
            has_shared_kv_layers=True,
            attention_k_eq_v=False,
            visual_hidden_size=1152,
            audio_feature_size=128,
            sliding_window=512,
            local_attention_window_size=512,
            global_attention_window_size=8192,
        ),
        context_max_length=context_max_length,
        input_sequence_length=256,
        quant_type="w8a8h1_sefp",
        export_image_visual=True,
        export_video_visual=True,
        export_audio=True,
        export_per_layer_input=True,
        image_visual_seq_length=280,
        image_visual_max_patches=2520,
        video_visual_seq_length=70,
        video_visual_max_patches=630,
    )


@pytest.mark.parametrize("context_max_length", [2048, 8192])
def test_gemma4_export_contract_allows_full_context_lengths(context_max_length: int):
    _plan(context_max_length).validate_fixed_contract()


def test_gemma4_export_contract_rejects_unapproved_context_length():
    with pytest.raises(ValueError, match="context_max_length.*2048.*8192"):
        _plan(4096).validate_fixed_contract()


def test_gemma4_workflow_model_name_tracks_context_override():
    workflow = Gemma4SeriesWorkflow(
        hf_model_dir="/tmp/gemma4-e2b",
        config_path="configs_merak/workflows/xh2a/llm_models/gemma4_series/e2b/gemma4_e2b_full.yaml",
    )

    overrides = workflow._with_context_aware_model_name_override(
        {"export.model.context_max_length": 8192}
    )
    model_cfg = workflow.workflow_config.with_overrides(overrides).export["model"]

    assert model_cfg["context_max_length"] == 8192
    assert model_cfg["prefill_chunk_length"] == 256
    assert model_cfg["model_name"].endswith("_256_8k")


def test_gemma4_workflow_model_name_respects_explicit_override():
    workflow = Gemma4SeriesWorkflow(
        hf_model_dir="/tmp/gemma4-e2b",
        config_path="configs_merak/workflows/xh2a/llm_models/gemma4_series/e2b/gemma4_e2b_full.yaml",
    )

    overrides = workflow._with_context_aware_model_name_override(
        {
            "export.model.context_max_length": 8192,
            "export.model.model_name": "custom_gemma4_name",
        }
    )
    model_cfg = workflow.workflow_config.with_overrides(overrides).export["model"]

    assert model_cfg["context_max_length"] == 8192
    assert model_cfg["model_name"] == "custom_gemma4_name"
