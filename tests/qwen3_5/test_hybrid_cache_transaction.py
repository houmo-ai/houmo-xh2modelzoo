from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from xhmodel_merak.xh_llm.models.qwen3_5.hybrid_cache_runtime import (
    abort_hybrid_cache_transaction,
    begin_hybrid_cache_output_passthrough,
    begin_hybrid_cache_transaction,
    commit_hybrid_cache_outputs,
    commit_hybrid_cache_transaction,
    detach_hybrid_cache_transaction,
    end_hybrid_cache_output_passthrough,
)


def _runtime(output_names: list[str]):
    session = SimpleNamespace(get_output_names=lambda: output_names)
    return SimpleNamespace(
        meta_info=SimpleNamespace(
            spec_decode={"mode": "mtp", "num_draft_tokens": 2},
            model_config=SimpleNamespace(
                prefill_recurrent_state_uses_cache=False
            ),
        ),
        _kvcache_mixin=SimpleNamespace(
            split_conv_cache=True,
            past_conv_caches=[
                (torch.zeros(1), torch.zeros(1), torch.zeros(1))
            ],
            past_recurrent_states=[torch.zeros(1)],
        ),
        decode_model=SimpleNamespace(hmonnx_session=session),
        is_prefill=lambda: False,
    )


def test_verify_transaction_commits_accepted_snapshot_only() -> None:
    names = ["logits"]
    names += [f"conv_cache_out_q_0_{step}" for step in range(3)]
    names += [f"conv_cache_out_k_0_{step}" for step in range(3)]
    names += [f"conv_cache_out_v_0_{step}" for step in range(3)]
    names += [f"recurrent_state_out_0_{step}" for step in range(3)]
    names += ["post_norm_hidden"]
    runtime = _runtime(names)
    outputs = [torch.tensor([[-1.0]])]
    outputs += [torch.tensor([float(step)]) for step in range(3)]
    outputs += [torch.tensor([float(10 + step)]) for step in range(3)]
    outputs += [torch.tensor([float(20 + step)]) for step in range(3)]
    outputs += [torch.tensor([float(30 + step)]) for step in range(3)]
    hidden = torch.tensor([[[99.0]]])
    outputs += [hidden]

    begin_hybrid_cache_transaction(runtime)
    forwarded = commit_hybrid_cache_outputs(
        runtime,
        tuple(outputs),
        model_label="test",
    )

    assert forwarded[-1] is hidden
    q_cache, k_cache, v_cache = runtime._kvcache_mixin.past_conv_caches[0]
    recurrent_cache = runtime._kvcache_mixin.past_recurrent_states[0]
    assert [q_cache.item(), k_cache.item(), v_cache.item()] == [0, 0, 0]
    assert recurrent_cache.item() == 0

    conv, recurrent = commit_hybrid_cache_transaction(
        runtime,
        accepted_steps=2,
        model_label="test",
    )

    assert [item.item() for item in conv] == [1, 11, 21]
    assert [item.item() for item in recurrent] == [31]
    assert [q_cache.item(), k_cache.item(), v_cache.item()] == [1, 11, 21]
    assert recurrent_cache.item() == 31


def test_verify_transaction_zero_draft_acceptance_commits_snapshot_zero() -> None:
    names = ["logits"]
    names += [f"conv_cache_out_q_0_{step}" for step in range(3)]
    names += [f"conv_cache_out_k_0_{step}" for step in range(3)]
    names += [f"conv_cache_out_v_0_{step}" for step in range(3)]
    names += [f"recurrent_state_out_0_{step}" for step in range(3)]
    runtime = _runtime(names)
    outputs = [torch.tensor([[-1.0]])]
    outputs += [torch.tensor([float(step)]) for step in range(3)]
    outputs += [torch.tensor([float(10 + step)]) for step in range(3)]
    outputs += [torch.tensor([float(20 + step)]) for step in range(3)]
    outputs += [torch.tensor([float(30 + step)]) for step in range(3)]

    begin_hybrid_cache_transaction(runtime)
    commit_hybrid_cache_outputs(runtime, outputs, model_label="test")
    commit_hybrid_cache_transaction(
        runtime,
        accepted_steps=1,
        model_label="test",
    )

    assert [
        cache.item()
        for cache in runtime._kvcache_mixin.past_conv_caches[0]
    ] == [0, 10, 20]
    assert runtime._kvcache_mixin.past_recurrent_states[0].item() == 30


def test_verify_transaction_rejects_zero_target_steps_without_mutation() -> None:
    names = ["logits"]
    names += [f"conv_cache_out_q_0_{step}" for step in range(3)]
    names += [f"conv_cache_out_k_0_{step}" for step in range(3)]
    names += [f"conv_cache_out_v_0_{step}" for step in range(3)]
    names += [f"recurrent_state_out_0_{step}" for step in range(3)]
    runtime = _runtime(names)
    outputs = [torch.tensor([[-1.0]])]
    outputs += [torch.tensor([float(step)]) for step in range(3)]
    outputs += [torch.tensor([float(10 + step)]) for step in range(3)]
    outputs += [torch.tensor([float(20 + step)]) for step in range(3)]
    outputs += [torch.tensor([float(30 + step)]) for step in range(3)]
    for cache in runtime._kvcache_mixin.past_conv_caches[0]:
        cache.fill_(-1)
    runtime._kvcache_mixin.past_recurrent_states[0].fill_(-1)

    begin_hybrid_cache_transaction(runtime)
    commit_hybrid_cache_outputs(runtime, outputs, model_label="test")
    with pytest.raises(ValueError, match=r"accepted_steps must be in \[1"):
        commit_hybrid_cache_transaction(
            runtime,
            accepted_steps=0,
            model_label="test",
        )

    assert [
        cache.item()
        for cache in runtime._kvcache_mixin.past_conv_caches[0]
    ] == [-1, -1, -1]
    assert runtime._kvcache_mixin.past_recurrent_states[0].item() == -1
    abort_hybrid_cache_transaction(runtime)


def test_verify_transaction_can_be_aborted_without_mutation() -> None:
    runtime = _runtime(
        [
            "logits",
            "conv_cache_out_q_0_0",
            "conv_cache_out_q_0_1",
            "conv_cache_out_q_0_2",
            "conv_cache_out_k_0_0",
            "conv_cache_out_k_0_1",
            "conv_cache_out_k_0_2",
            "conv_cache_out_v_0_0",
            "conv_cache_out_v_0_1",
            "conv_cache_out_v_0_2",
            "recurrent_state_out_0_0",
            "recurrent_state_out_0_1",
            "recurrent_state_out_0_2",
        ]
    )
    begin_hybrid_cache_transaction(runtime)
    abort_hybrid_cache_transaction(runtime)

    assert runtime._hybrid_cache_transaction is None
    assert runtime._defer_hybrid_cache_commit is False


def test_detached_transaction_can_commit_after_runtime_state_switch() -> None:
    names = ["logits"]
    names += [f"conv_cache_out_q_0_{step}" for step in range(3)]
    names += [f"conv_cache_out_k_0_{step}" for step in range(3)]
    names += [f"conv_cache_out_v_0_{step}" for step in range(3)]
    names += [f"recurrent_state_out_0_{step}" for step in range(3)]
    runtime = _runtime(names)
    outputs = [torch.tensor([[-1.0]])]
    outputs += [torch.tensor([float(step)]) for step in range(3)]
    outputs += [torch.tensor([float(10 + step)]) for step in range(3)]
    outputs += [torch.tensor([float(20 + step)]) for step in range(3)]
    outputs += [torch.tensor([float(30 + step)]) for step in range(3)]

    begin_hybrid_cache_transaction(runtime)
    commit_hybrid_cache_outputs(runtime, outputs, model_label="test")
    transaction = detach_hybrid_cache_transaction(runtime)

    for cache in runtime._kvcache_mixin.past_conv_caches[0]:
        cache.fill_(-1)
    runtime._kvcache_mixin.past_recurrent_states[0].fill_(-1)
    commit_hybrid_cache_transaction(
        runtime,
        accepted_steps=2,
        model_label="test",
        transaction=transaction,
    )

    assert [
        cache.item()
        for cache in runtime._kvcache_mixin.past_conv_caches[0]
    ] == [1, 11, 21]
    assert runtime._kvcache_mixin.past_recurrent_states[0].item() == 31


def test_output_passthrough_keeps_hidden_and_commits_real_decode_step_zero() -> None:
    names = ["logits"]
    names += [f"conv_cache_out_q_0_{step}" for step in range(3)]
    names += [f"conv_cache_out_k_0_{step}" for step in range(3)]
    names += [f"conv_cache_out_v_0_{step}" for step in range(3)]
    names += [f"recurrent_state_out_0_{step}" for step in range(3)]
    names += ["post_norm_hidden"]
    runtime = _runtime(names)
    outputs = [torch.tensor([[-1.0]])]
    outputs += [torch.tensor([float(step)]) for step in range(3)]
    outputs += [torch.tensor([float(10 + step)]) for step in range(3)]
    outputs += [torch.tensor([float(20 + step)]) for step in range(3)]
    outputs += [torch.tensor([float(30 + step)]) for step in range(3)]
    hidden = torch.tensor([[[99.0]]])
    outputs += [hidden]

    begin_hybrid_cache_output_passthrough(runtime, selected_step=0)
    forwarded = commit_hybrid_cache_outputs(
        runtime,
        tuple(outputs),
        model_label="test",
    )
    end_hybrid_cache_output_passthrough(runtime)

    assert len(forwarded) == len(outputs)
    assert all(
        actual is expected
        for actual, expected in zip(forwarded, outputs, strict=True)
    )
    assert forwarded[-1] is hidden
    assert [
        cache.item()
        for cache in runtime._kvcache_mixin.past_conv_caches[0]
    ] == [0, 10, 20]
    assert runtime._kvcache_mixin.past_recurrent_states[0].item() == 30
