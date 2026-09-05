"""CPU-only contracts for MiniCPM-V-4.6's shared text-input preprocessing.

Call the real export/runtime factories without loading checkpoints or HMONNX
sessions. Vision resizing belongs to MiniCPM's processor/vision tower; this
interface scatters already-computed features and uses sequential positions.
"""

from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(params=("export", "runtime", "runtime-page"))
def processor_and_owner(request):
    from xhmodel_merak.xh_llm.models.minicpm_v_4_6.inference import MiniCPMV46TextHMONNXModel
    from xhmodel_merak.xh_llm.models.minicpm_v_4_6.text_model import MiniCPMV46TextModel
    from xhmodel_merak.xh_llm.types import CacheList

    embedding = torch.nn.Embedding(64, 4, dtype=torch.float16)
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(256).reshape(64, 4))
    config = SimpleNamespace(
        image_token_id=31,
        video_token_id=32,
        vision_start_token_id=33,
        vision_end_token_id=34,
    )
    owner = SimpleNamespace(
        embed_tokens=embedding,
        wrap_cfg=SimpleNamespace(input_sequence_length=8),
        get_input_sequence_length=lambda: 8,
        config=config,
        meta_info=SimpleNamespace(model_config=config),
        enable_page_attention=request.param == "runtime-page",
        past_key_caches=CacheList([torch.ones(1, 2, 8, 4)]),
        past_value_caches=CacheList([torch.full((1, 2, 8, 4), 2.0)]),
        past_conv_caches=CacheList([torch.full((1, 2, 3), float(i)) for i in range(3)]),
        past_recurrent_states=CacheList([torch.full((1, 2, 4, 4), 3.0)]),
    )
    factory = (
        MiniCPMV46TextModel._get_data_preprocessor
        if request.param == "export"
        else MiniCPMV46TextHMONNXModel._get_data_preprocessor
    )
    return factory(owner), owner


def _assert_positions_and_lengths(outputs, *, capacity, length, past):
    for positions in outputs[1:4]:
        torch.testing.assert_close(positions, torch.arange(past, past + capacity, dtype=torch.int64))
    torch.testing.assert_close(outputs[4], torch.tensor([past], dtype=torch.int32))
    torch.testing.assert_close(outputs[5], torch.tensor([length], dtype=torch.int32))
    mask = torch.tensor([[1] * length + [0] * (capacity - length)], dtype=torch.float16)
    torch.testing.assert_close(outputs[6], mask)
    assert outputs[0].device.type == "cpu"


def test_minicpm46_factories_preserve_text_preprocess_configuration(processor_and_owner):
    processor, owner = processor_and_owner
    assert processor.embed_tokens is owner.embed_tokens
    assert processor.input_sequence_length == 8
    assert processor.enable_page_attention is owner.enable_page_attention
    assert processor.image_token_id == owner.config.image_token_id
    assert processor.video_token_id == owner.config.video_token_id
    assert processor.vision_start_token_id == owner.config.vision_start_token_id
    assert processor.vision_end_token_id == owner.config.vision_end_token_id
    assert processor.patch_size == 14
    assert processor.spatial_merge_size == 2


def test_minicpm46_text_prefill_pads_embeddings_not_actual_length(processor_and_owner):
    processor, owner = processor_and_owner
    outputs = processor({"input_ids": torch.tensor([[1, 2, 3]]), "past_seq_length": 0})
    expected = owner.embed_tokens(torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0]]))
    torch.testing.assert_close(outputs[0], expected, rtol=0, atol=0)
    _assert_positions_and_lengths(outputs, capacity=8, length=3, past=0)


@pytest.mark.parametrize("past", (0, 8), ids=("prefill", "continuation"))
def test_minicpm46_visual_features_use_sequential_text_positions(processor_and_owner, past):
    processor, owner = processor_and_owner
    if past:
        processor({"input_ids": torch.tensor([[1] * 8]), "past_seq_length": 0})
    features = torch.tensor([[101, 102, 103, 104], [201, 202, 203, 204]], dtype=torch.float32)
    outputs = processor({
        "input_ids": torch.tensor([[1, owner.config.image_token_id, owner.config.image_token_id, 2, 3]]),
        "image_embeds": features,
        "image_grid_thw": None,
        "past_seq_length": past,
    })
    expected = owner.embed_tokens(torch.tensor([[1, 31, 31, 2, 3, 0, 0, 0]])).detach().clone()
    expected[:, 1:3] = features.to(expected.dtype)
    torch.testing.assert_close(outputs[0], expected, rtol=0, atol=0)
    _assert_positions_and_lengths(outputs, capacity=8, length=5, past=past)


def test_minicpm46_decode_continues_from_unpadded_prompt_length(processor_and_owner):
    processor, owner = processor_and_owner
    processor({"input_ids": torch.tensor([[1, 2, 3]]), "past_seq_length": 0})
    processor.input_sequence_length = 1
    outputs = processor({"input_ids": torch.tensor([[4]]), "past_seq_length": 3})
    torch.testing.assert_close(outputs[0], owner.embed_tokens(torch.tensor([[4]])), rtol=0, atol=0)
    _assert_positions_and_lengths(outputs, capacity=1, length=1, past=3)


def test_minicpm46_precomputed_embeddings_keep_padding_contract(processor_and_owner):
    processor, owner = processor_and_owner
    embeds = torch.full((1, 3, 4), 42.0, dtype=torch.float16)
    outputs = processor({"inputs_embeds": embeds, "past_seq_length": 0})
    padding = owner.embed_tokens(torch.zeros((1, 5), dtype=torch.long))
    torch.testing.assert_close(outputs[0], torch.cat((embeds, padding), dim=1), rtol=0, atol=0)
    _assert_positions_and_lengths(outputs, capacity=8, length=3, past=0)


@pytest.mark.parametrize("feature_count", (1, 3), ids=("missing-feature", "extra-feature"))
def test_minicpm46_rejects_visual_feature_count_mismatch(processor_and_owner, feature_count):
    processor, owner = processor_and_owner
    with pytest.raises(ValueError, match="Image features and image tokens do not match"):
        processor({
            "input_ids": torch.tensor([[1, owner.config.image_token_id, owner.config.image_token_id]]),
            "image_embeds": torch.ones(feature_count, 4),
            "past_seq_length": 0,
        })


def test_minicpm46_rejects_over_capacity_text_chunk(processor_and_owner):
    processor, _ = processor_and_owner
    with pytest.raises(AssertionError, match="Input sequence length is too long"):
        processor({"input_ids": torch.ones((1, 9), dtype=torch.long), "past_seq_length": 0})


@pytest.mark.parametrize("grouped", (False, True), ids=("flat-conv-cache", "grouped-conv-cache"))
def test_minicpm46_preserves_cache_order_and_page_attention_abi(processor_and_owner, grouped):
    processor, owner = processor_and_owner
    flat_conv = list(owner.past_conv_caches)
    if grouped:
        processor.past_conv_caches[:] = [tuple(flat_conv)]
    outputs = processor({"input_ids": torch.tensor([[1, 2]]), "past_seq_length": 0})
    if owner.enable_page_attention:
        assert len(outputs) == 9
        conv, recurrent = outputs[7:]
    else:
        assert len(outputs) == 11
        assert outputs[7] is owner.past_key_caches
        assert outputs[8] is owner.past_value_caches
        conv, recurrent = outputs[9:]
    assert len(conv) == len(flat_conv)
    assert all(actual is expected for actual, expected in zip(conv, flat_conv, strict=True))
    assert recurrent is owner.past_recurrent_states
