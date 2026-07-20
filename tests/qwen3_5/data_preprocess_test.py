import torch

from xhmodel_merak.xh_llm.models.qwen3_5.data_preprocess import Qwen3_5_DataPreprocess
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model import _Qwen3_5HFCompatible
from xhmodel_merak.xh_llm.types import CacheList


def _build_data_processor(input_sequence_length: int = 256) -> Qwen3_5_DataPreprocess:
    return Qwen3_5_DataPreprocess(
        token_embedding=torch.nn.Embedding(32, 4),
        input_sequence_length=input_sequence_length,
        past_key_caches=CacheList(),
        past_value_caches=CacheList(),
        past_conv_caches=CacheList(),
        past_recurrent_states=CacheList(),
    )


def test_continuation_keeps_unpadded_current_sequence_length():
    data_processor = _build_data_processor()
    data_processor.rope_deltas = torch.zeros((1, 1), dtype=torch.long)
    input_ids = torch.arange(44, dtype=torch.long).remainder(31).add(1).unsqueeze(0)

    outputs = data_processor(
        {
            "input_ids": input_ids,
            "past_seq_length": 256,
        }
    )

    inputs_embeds = outputs[0]
    current_input_length = outputs[5]
    linear_attn_mask = outputs[6]

    assert inputs_embeds.shape[1] == 256
    assert current_input_length.tolist() == [44]
    assert linear_attn_mask.sum().item() == 44


class _FakeDataProcessor:
    image_token_id = -1

    def __init__(self):
        self.rope_deltas = None
        self.past_key_caches = CacheList()
        self.past_value_caches = CacheList()
        self.past_conv_caches = CacheList()
        self.past_recurrent_states = CacheList()

    def get_rope_index(
        self,
        input_ids,
        inputs_embeds,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
    ):
        del input_ids, image_grid_thw, video_grid_thw, attention_mask
        seq_length = inputs_embeds.shape[1]
        position_ids = torch.arange(seq_length, device=inputs_embeds.device)
        position_ids = position_ids.view(1, 1, -1).expand(3, 1, -1)
        return position_ids, torch.zeros((1, 1), dtype=torch.long, device=inputs_embeds.device)


class _FakeLLMModel:
    def __init__(self):
        self.data_processor = _FakeDataProcessor()
        self.forward_calls = []

    def get_data_preprocessor(self):
        return self.data_processor

    def get_input_sequence_length(self):
        return 256

    def get_num_logits_to_keep(self):
        return 1

    def forward(self, *args):
        past_seq_length = int(args[4].item())
        current_input_length = int(args[5].item())
        linear_mask_length = int(args[6].sum().item())
        self.forward_calls.append((past_seq_length, current_input_length, linear_mask_length))
        logits = torch.zeros((1, 1, 8), dtype=args[0].dtype, device=args[0].device)
        return logits, [], []


class _FakeHFCompatible:
    def __init__(self):
        self._llm_model = _FakeLLMModel()
        self._past_seq_length = 0
        self.embedding = torch.nn.Embedding(32, 4)

    def get_input_embeddings(self):
        return self.embedding


def test_hf_compatible_prefill_splits_long_prompt_with_real_tail_length():
    model = _FakeHFCompatible()
    input_ids = torch.arange(300, dtype=torch.long).remainder(31).add(1).unsqueeze(0)

    output = _Qwen3_5HFCompatible.forward(
        model,
        input_ids=input_ids,
    )

    assert model._llm_model.forward_calls == [(0, 256, 256), (256, 44, 44)]
    assert output.logits.shape == (1, 1, 8)
