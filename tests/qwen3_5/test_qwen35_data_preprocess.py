from __future__ import annotations

import torch

from xhmodel_merak.xh_llm.models.qwen3_5.data_preprocess import (
    Qwen3_5_DataPreprocess,
)
from xhmodel_merak.xh_llm.types import CacheList


def test_page_attention_partial_continuation_preserves_actual_current_length():
    processor = Qwen3_5_DataPreprocess(
        token_embedding=torch.nn.Embedding(16, 8, dtype=torch.float16),
        input_sequence_length=256,
        past_conv_caches=CacheList(),
        past_recurrent_states=CacheList(),
        enable_page_attention=True,
    )
    processor.rope_deltas = torch.zeros((1, 1), dtype=torch.long)

    net_inputs = processor(
        {
            "input_ids": torch.ones((1, 35), dtype=torch.long),
            "past_seq_length": 1792,
        }
    )

    assert net_inputs[0].shape[1] == 256
    assert net_inputs[1].shape[0] == 256
    torch.testing.assert_close(net_inputs[5], torch.tensor([35], dtype=torch.int32))
