"""data_preprocess (scatter) contract tests for Unlimited-OCR (CPU, no weights)."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from xhmodel_merak.xh_llm.models.unlimited_ocr.data_preprocess import UnlimitedOCRDataPreprocess
from xhmodel_merak.xh_llm.types import CacheList

HIDDEN = 8


def _make_processor(input_sequence_length=280):
    embed = nn.Embedding(128816, HIDDEN)
    return UnlimitedOCRDataPreprocess(
        token_embedding=embed,
        input_sequence_length=input_sequence_length,
        past_key_caches=CacheList([]),
        past_value_caches=CacheList([]),
    )


def test_image_token_count_is_273():
    proc = _make_processor()
    assert proc.image_grid_size == 16
    assert proc.image_token_count == 273


def test_scatter_replaces_image_positions():
    proc = _make_processor()
    input_ids = torch.tensor([[1] + [proc.image_token_id] * 273 + [2]])
    image_embeds = torch.ones(273, HIDDEN) * 7.0
    out = proc({"input_ids": input_ids, "image_embeds": image_embeds, "past_seq_length": 0})
    inputs_embeds = out[0]
    # image token slots (index 1..273) must carry the scattered value
    assert torch.allclose(inputs_embeds[0, 1], torch.full((HIDDEN,), 7.0))
    # current_input_length reflects the real (pre-padding) sequence length
    assert int(out[2][0]) == 275


def test_token_feature_mismatch_raises():
    proc = _make_processor()
    input_ids = torch.tensor([[1] + [proc.image_token_id] * 273 + [2]])
    image_embeds = torch.ones(272, HIDDEN)  # one short
    with pytest.raises(ValueError):
        proc({"input_ids": input_ids, "image_embeds": image_embeds, "past_seq_length": 0})


def test_decode_step_rejects_image_inputs():
    proc = _make_processor(input_sequence_length=1)
    input_ids = torch.tensor([[proc.image_token_id]])
    image_embeds = torch.ones(1, HIDDEN)
    with pytest.raises(ValueError):
        proc({"input_ids": input_ids, "image_embeds": image_embeds, "past_seq_length": 1})


def test_decode_step_text_only_position():
    proc = _make_processor(input_sequence_length=1)
    out = proc({"inputs_embeds": torch.ones(1, 1, HIDDEN), "past_seq_length": 275})
    assert int(out[2][0]) == 1
    assert int(proc.last_position_ids[0]) == 275


def test_decode_past_position_can_extend_beyond_sliding_window():
    proc = _make_processor(input_sequence_length=1)
    protected_prefix_len = 278
    past_seq_length = protected_prefix_len + 129
    out = proc({"inputs_embeds": torch.ones(1, 1, HIDDEN), "past_seq_length": past_seq_length})
    assert int(out[1][0]) == past_seq_length
    assert int(out[2][0]) == 1
    assert int(proc.last_position_ids[0]) == past_seq_length
    assert int(out[1][0]) != 128
    assert int(proc.last_position_ids[0]) != past_seq_length % 128


def test_wrap_attention_uses_full_kv_not_absolute_sliding_window():
    source = Path("xhmodel_merak/xh_llm/models/unlimited_ocr/_llm_model_impl.py").read_text(encoding="utf-8")
    assert "MaskedSoftmax(dim=-1, attention_max_length=-1)" in source
    assert source.count("LLMCacheV2(axis=cache_axis, attention_max_length=-1)") == 2


def test_crop_mode_build_helpers_point_to_processor():
    embed = nn.Embedding(128816, HIDDEN)
    proc = UnlimitedOCRDataPreprocess(
        token_embedding=embed,
        input_sequence_length=280,
        past_key_caches=CacheList([]),
        past_value_caches=CacheList([]),
        crop_mode=True,
    )
    with pytest.raises(NotImplementedError):
        proc.build_image_token_ids(1)
