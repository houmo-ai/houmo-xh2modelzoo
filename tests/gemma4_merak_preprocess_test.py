import torch
from torch import nn

from xhmodel_merak.xh_llm.types import CacheList


def test_gemma4_preprocess_injects_multimodal_features():
    from xhmodel_merak.xh_llm.models.gemma4e.data_preprocess import (
        Gemma4DataPreprocess,
        Gemma4InputProcessorConfig,
    )

    embed_tokens = nn.Embedding(32, 4)
    with torch.no_grad():
        embed_tokens.weight.copy_(torch.arange(32 * 4, dtype=torch.float32).reshape(32, 4))

    class RecordingPerLayerInputBuilder(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_ids = None
            self.inputs_embeds = None

        def forward(self, input_ids, inputs_embeds):
            self.input_ids = input_ids.detach().clone()
            self.inputs_embeds = inputs_embeds.detach().clone()
            return inputs_embeds.unsqueeze(1)

    per_layer_input_builder = RecordingPerLayerInputBuilder()
    config = Gemma4InputProcessorConfig(
        embed_tokens=embed_tokens,
        input_sequence_length=6,
        context_max_length=8,
        sliding_window=3,
        image_token_id=9,
        audio_token_id=10,
        video_token_id=11,
        past_key_caches=CacheList([torch.zeros((1, 2, 8, 8), dtype=torch.float16)]),
        past_value_caches=CacheList([torch.zeros((1, 2, 8, 8), dtype=torch.float16)]),
        per_layer_input_builder=per_layer_input_builder,
        pad_token_id=0,
    )
    preprocess = Gemma4DataPreprocess(config)

    image_embed = torch.tensor([[100.0, 101.0, 102.0, 103.0]])
    audio_embed = torch.tensor([[200.0, 201.0, 202.0, 203.0]])
    outputs = preprocess(
        {
            "input_ids": torch.tensor([[1, 9, 2, 10]], dtype=torch.long),
            "image_embeds": image_embed,
            "audio_embeds": audio_embed,
            "past_seq_length": 2,
        }
    )

    (
        per_layer_inputs,
        inputs_embeds,
        position_ids,
        past_seq_length,
        current_input_length,
        local_attention_mask,
        global_attention_mask,
        past_key_caches,
        past_value_caches,
    ) = outputs

    assert per_layer_inputs.shape == (1, 1, 6, 4)
    assert inputs_embeds.shape == (1, 6, 4)
    assert position_ids.shape == (1, 6)
    assert past_seq_length.item() == 2
    assert current_input_length.item() == 4
    assert local_attention_mask.shape == (1, 1, 6, 8)
    assert global_attention_mask.shape == (1, 1, 6, 8)
    assert past_key_caches is config.past_key_caches
    assert past_value_caches is config.past_value_caches

    assert torch.equal(per_layer_input_builder.input_ids[0, :4], torch.tensor([1, 0, 2, 0]))
    assert torch.allclose(inputs_embeds[0, 1], image_embed[0])
    assert torch.allclose(inputs_embeds[0, 3], audio_embed[0])
    assert torch.allclose(per_layer_input_builder.inputs_embeds[0, 1], image_embed[0])
    assert torch.allclose(per_layer_input_builder.inputs_embeds[0, 3], audio_embed[0])
    assert torch.allclose(per_layer_inputs[0, 0, 1], image_embed[0])
    assert torch.allclose(per_layer_inputs[0, 0, 3], audio_embed[0])

    assert torch.equal(position_ids[0, :4], torch.tensor([2, 3, 4, 5], dtype=position_ids.dtype))

    assert len(outputs) == 9


def test_gemma4_preprocess_output_contract_includes_attention_masks_by_default():
    from xhmodel_merak.xh_llm.models.gemma4e.data_preprocess import (
        Gemma4DataPreprocess,
        Gemma4InputProcessorConfig,
    )

    embed_tokens = nn.Embedding(32, 4)

    class PassthroughPerLayerInputBuilder(nn.Module):
        def forward(self, input_ids, inputs_embeds):
            del input_ids
            return inputs_embeds

    config = Gemma4InputProcessorConfig(
        embed_tokens=embed_tokens,
        input_sequence_length=6,
        context_max_length=32,
        sliding_window=3,
        image_token_id=-1,
        audio_token_id=-1,
        video_token_id=-1,
        past_key_caches=CacheList([torch.zeros((1, 2, 32, 8), dtype=torch.float16)]),
        past_value_caches=CacheList([torch.zeros((1, 2, 32, 8), dtype=torch.float16)]),
        per_layer_input_builder=PassthroughPerLayerInputBuilder(),
        pad_token_id=0,
    )
    preprocess = Gemma4DataPreprocess(config)

    (
        per_layer_inputs,
        inputs_embeds,
        position_ids,
        past_seq_length,
        current_input_length,
        local_attention_mask,
        global_attention_mask,
        _,
        _,
    ) = preprocess(
        {
            "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            "past_seq_length": 0,
        }
    )

    assert per_layer_inputs.shape == (1, 6, 4)
    assert inputs_embeds.shape == (1, 6, 4)
    assert position_ids.shape == (1, 6)
    assert past_seq_length.item() == 0
    assert current_input_length.item() == 4
    assert local_attention_mask.shape == (1, 1, 6, 16)
    assert global_attention_mask.shape == (1, 1, 6, 32)
    assert len(preprocess({"input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long), "past_seq_length": 0})) == 9


def test_gemma4_preprocess_can_emit_legacy_attention_masks():
    from xhmodel_merak.xh_llm.models.gemma4e.data_preprocess import (
        Gemma4DataPreprocess,
        Gemma4InputProcessorConfig,
    )

    embed_tokens = nn.Embedding(32, 4)

    class PassthroughPerLayerInputBuilder(nn.Module):
        def forward(self, input_ids, inputs_embeds):
            del input_ids
            return inputs_embeds

    config = Gemma4InputProcessorConfig(
        embed_tokens=embed_tokens,
        input_sequence_length=6,
        context_max_length=32,
        sliding_window=3,
        use_explicit_attention_mask=True,
        image_token_id=-1,
        audio_token_id=-1,
        video_token_id=-1,
        past_key_caches=CacheList([torch.zeros((1, 2, 32, 8), dtype=torch.float16)]),
        past_value_caches=CacheList([torch.zeros((1, 2, 32, 8), dtype=torch.float16)]),
        per_layer_input_builder=PassthroughPerLayerInputBuilder(),
        pad_token_id=0,
    )
    preprocess = Gemma4DataPreprocess(config)

    outputs = preprocess({"input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long), "past_seq_length": 0})

    assert len(outputs) == 9
    local_attention_mask = outputs[5]
    global_attention_mask = outputs[6]
    assert local_attention_mask.shape == (1, 1, 6, 16)
    assert global_attention_mask.shape == (1, 1, 6, 32)


def test_gemma4_preprocess_keeps_vision_attention_causal():
    from xhmodel_merak.xh_llm.models.gemma4e.data_preprocess import (
        Gemma4DataPreprocess,
        Gemma4InputProcessorConfig,
    )

    embed_tokens = nn.Embedding(32, 4)

    class PassthroughPerLayerInputBuilder(nn.Module):
        def forward(self, input_ids, inputs_embeds):
            del input_ids
            return inputs_embeds

    config = Gemma4InputProcessorConfig(
        embed_tokens=embed_tokens,
        input_sequence_length=6,
        context_max_length=16,
        sliding_window=8,
        image_token_id=9,
        audio_token_id=-1,
        video_token_id=-1,
        past_key_caches=CacheList([torch.zeros((1, 2, 16, 8), dtype=torch.float16)]),
        past_value_caches=CacheList([torch.zeros((1, 2, 16, 8), dtype=torch.float16)]),
        per_layer_input_builder=PassthroughPerLayerInputBuilder(),
        pad_token_id=0,
    )
    preprocess = Gemma4DataPreprocess(config)

    outputs = preprocess(
        {
            "input_ids": torch.tensor([[1, 9, 9, 2]], dtype=torch.long),
            "mm_token_type_ids": torch.tensor([[0, 1, 1, 0]], dtype=torch.long),
            "past_seq_length": 0,
        }
    )

    local_attention_mask = outputs[5]
    global_attention_mask = outputs[6]

    assert global_attention_mask[0, 0, 1, 2] < 0
    assert local_attention_mask[0, 0, 1, 2] < 0
    assert local_attention_mask[0, 0, 2, 1] == 0


def test_gemma4_preprocess_preserves_sequence_metadata_without_attention_masks():
    from xhmodel_merak.xh_llm.models.gemma4e.data_preprocess import (
        Gemma4DataPreprocess,
        Gemma4InputProcessorConfig,
    )

    embed_tokens = nn.Embedding(64, 4)

    class PassthroughPerLayerInputBuilder(nn.Module):
        def forward(self, input_ids, inputs_embeds):
            del input_ids
            return inputs_embeds

    config = Gemma4InputProcessorConfig(
        embed_tokens=embed_tokens,
        input_sequence_length=17,
        context_max_length=64,
        sliding_window=3,
        image_token_id=-1,
        audio_token_id=-1,
        video_token_id=-1,
        past_key_caches=CacheList([torch.zeros((1, 2, 64, 8), dtype=torch.float16)]),
        past_value_caches=CacheList([torch.zeros((1, 2, 64, 8), dtype=torch.float16)]),
        per_layer_input_builder=PassthroughPerLayerInputBuilder(),
        pad_token_id=0,
    )
    preprocess = Gemma4DataPreprocess(config)

    (
        _,
        _,
        _,
        past_seq_length,
        current_input_length,
        _,
        _,
        _,
        _,
    ) = preprocess(
        {
            "input_ids": torch.tensor([[1]], dtype=torch.long),
            "past_seq_length": 0,
        }
    )

    assert past_seq_length.item() == 0
    assert current_input_length.item() == 1
