from __future__ import annotations

import pytest

from xhmodel_merak.xh_llm.models.gemma4_series.export_plan import Gemma4SeriesExportPlan
from xhmodel_merak.xh_llm.models.gemma4_series.variants import Gemma4SeriesVariantSpec


def _plan(context_max_length: int, input_sequence_length: int = 320) -> Gemma4SeriesExportPlan:
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
        input_sequence_length=input_sequence_length,
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


@pytest.mark.parametrize("context_max_length", [320, 2048, 4096, 8192, 13107])
def test_gemma4_export_contract_allows_any_context_not_smaller_than_prefill(
    context_max_length: int,
):
    _plan(context_max_length).validate_fixed_contract()


@pytest.mark.parametrize("input_sequence_length", [0, 256, 279])
def test_gemma4_export_contract_rejects_prefill_shorter_than_visual_atomic_range(
    input_sequence_length: int,
):
    with pytest.raises(ValueError, match="prefill/input length >= 280"):
        _plan(2048, input_sequence_length=input_sequence_length).validate_fixed_contract()


def test_gemma4_export_contract_rejects_context_shorter_than_prefill():
    with pytest.raises(ValueError, match="context_max_length must be >= prefill/input length"):
        _plan(319).validate_fixed_contract()
