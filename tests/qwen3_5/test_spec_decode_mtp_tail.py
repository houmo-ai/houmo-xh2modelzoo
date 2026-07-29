from types import MethodType

import torch

from xhmodel_merak.xh_llm.models.qwen3_5._spec_decode_shared import (
    SpecDecodeVerifyResult,
)
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_spec_decode_onnx_model import (
    Qwen3_5SpecDecodeONNXModel,
)


def _model_with_prefill_capture():
    model = object.__new__(Qwen3_5SpecDecodeONNXModel)
    torch.nn.Module.__init__(model)
    calls = []

    def capture(self, hidden_states, next_token_ids, past_seq_len):
        del self
        calls.append((hidden_states.clone(), next_token_ids.clone(), past_seq_len))

    model._prefill_mtp_chunk = MethodType(capture, model)
    return model, calls


def _verify_result() -> SpecDecodeVerifyResult:
    return SpecDecodeVerifyResult(
        initial_seq_len=20,
        verify_token_ids=[101, 102, 103, 104, 105],
        predicted_token_ids=[102, 103, 104, 105, 106],
        verify_hidden=torch.arange(15, dtype=torch.float32).reshape(1, 5, 3),
        raw_result={},
    )


def test_mtp_full_accept_materializes_missing_tail_pair() -> None:
    model, calls = _model_with_prefill_capture()

    model._complete_mtp_full_accept_tail(
        _verify_result(),
        accepted_steps=5,
        mtp_past_seq_len=20,
    )

    assert len(calls) == 1
    hidden_states, next_token_ids, past_seq_len = calls[0]
    torch.testing.assert_close(
        hidden_states,
        torch.tensor([[[9.0, 10.0, 11.0]]]),
    )
    assert next_token_ids.tolist() == [[105]]
    assert past_seq_len == 24


def test_mtp_rejection_does_not_append_unaccepted_tail_pair() -> None:
    model, calls = _model_with_prefill_capture()

    model._complete_mtp_full_accept_tail(
        _verify_result(),
        accepted_steps=3,
        mtp_past_seq_len=20,
    )

    assert calls == []
