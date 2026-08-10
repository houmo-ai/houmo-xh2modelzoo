# Copyright 2025 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from xhmodel_merak.xh_llm.models.hunyuan_ocr import (
    HunyuanOCRDraftCacheController,
    HunyuanOCRDraftGraphs,
    HunyuanOCRSpeculativeDecoder,
    HunyuanOCRSpeculativeRuntimeError,
    HunyuanOCRTargetVerifyController,
    HunyuanOCRTextExportMeta,
    XHHunYuanOCRModel,
    load_draft_graphs,
    resolve_draft_runtime_contract,
)
from xhmodel_merak.xh_llm.models.hunyuan_ocr.data_preprocess import HunyuanOCRTextDataPreprocess
from xhmodel_merak.xh_llm.models.hunyuan_ocr.hunyuan_ocr_speculative_runtime import (
    FALLBACK_CAPACITY_SHORTFALL,
    FALLBACK_DRAFT_EXECUTION_FAILED,
    STOP_EOS,
    STOP_MAX_LENGTH,
)


BLOCK_SIZE = 16
NUM_DRAFT_TOKENS = 15
HIDDEN_WIDTH = 6
DRAFT_HIDDEN_SIZE = 2
VOCAB_SIZE = 64
CAPACITY = 64
NUM_LAYERS = 2
EOS_TOKEN_ID = 7
MASK_TOKEN_ID = 63


def _install_hmonnx_optimizer_stub() -> None:
    module = types.ModuleType("xhquant.xhonnxruntime.hmonnx_optimizer")
    module.materialize_parallel_linear_fusion = lambda path: path
    sys.modules.setdefault("xhquant.xhonnxruntime.hmonnx_optimizer", module)


class _FakeCache:
    def __init__(self, data: torch.Tensor) -> None:
        self.data = data


class _FakeInput:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class _FakeSession:
    def __init__(
        self,
        mode: str,
        log: list[dict[str, int | str]],
        *,
        candidates: list[list[int]] | None = None,
        fail: bool = False,
    ) -> None:
        self.mode = mode
        self.log = log
        self.candidates = candidates or []
        self.fail = fail
        self.calls = 0

    def get_input(self, name: str) -> _FakeInput:
        assert name == "target_hidden"
        return _FakeInput((1, BLOCK_SIZE, HIDDEN_WIDTH))

    def get_output_names(self) -> list[str]:
        if self.mode == "decode":
            return ["draft_logits"]
        return [
            *[f"present_key_cache_{index}" for index in range(NUM_LAYERS)],
            *[f"present_value_cache_{index}" for index in range(NUM_LAYERS)],
        ]

    def run(self, feed: dict[str, object]) -> list[torch.Tensor]:
        if self.fail:
            raise RuntimeError("scripted draft failure")
        self.log.append(
            {
                "mode": self.mode,
                "past_seq_length": int(feed["past_seq_length"].item()),
                "current_input_length": int(feed["current_input_length"].item()),
            }
        )
        if self.mode == "decode":
            token_ids = self.candidates[self.calls]
            self.calls += 1
            logits = torch.zeros(1, NUM_DRAFT_TOKENS, VOCAB_SIZE, dtype=torch.float16)
            for position, token_id in enumerate(token_ids):
                logits[0, position, token_id] = 1.0
            return [logits]
        return [
            feed[name].data.clone()
            for name in (
                *[f"past_key_cache_{index}" for index in range(NUM_LAYERS)],
                *[f"past_value_cache_{index}" for index in range(NUM_LAYERS)],
            )
        ]


class _ContractSession:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def get_input_names(self) -> list[str]:
        cache_names = [
            *[f"past_key_cache_{index}" for index in range(NUM_LAYERS)],
            *[f"past_value_cache_{index}" for index in range(NUM_LAYERS)],
        ]
        if self.mode == "decode":
            return ["noise_embedding", "past_seq_length", "current_input_length", "attn_mask", *cache_names]
        return ["target_hidden", "past_seq_length", "current_input_length", *cache_names]

    def get_output_names(self) -> list[str]:
        if self.mode == "decode":
            return ["draft_logits"]
        return [
            *[f"present_key_cache_{index}" for index in range(NUM_LAYERS)],
            *[f"present_value_cache_{index}" for index in range(NUM_LAYERS)],
        ]


class _Target:
    def __init__(self, predictions: list[list[int]], decode_tokens: list[int] | None = None) -> None:
        self.predictions = predictions
        self.decode_tokens = list(decode_tokens or [])
        self.verify_calls = 0
        self.decode_calls = 0

    def verify(self, *, input_token_ids, position_ids, past_seq_length, current_input_length):
        predicted = self.predictions[self.verify_calls]
        self.verify_calls += 1
        logits = torch.zeros(1, BLOCK_SIZE, VOCAB_SIZE, dtype=torch.float16)
        for position, token_id in enumerate(predicted):
            logits[0, position, token_id] = 1.0
        hidden = torch.ones(1, BLOCK_SIZE, HIDDEN_WIDTH, dtype=torch.float16)
        return logits, hidden

    def decode(self, token_id: int) -> torch.Tensor:
        next_token_id = self.decode_tokens[self.decode_calls]
        self.decode_calls += 1
        logits = torch.zeros(1, 1, VOCAB_SIZE, dtype=torch.float16)
        logits[0, 0, next_token_id] = 1.0
        return logits


def _metadata() -> dict[str, object]:
    cache_input_names = [
        *[f"past_key_cache_{index}" for index in range(NUM_LAYERS)],
        *[f"past_value_cache_{index}" for index in range(NUM_LAYERS)],
    ]
    return {
        "mode": "dflash",
        "target_hidden_concat_size": HIDDEN_WIDTH,
        "draft": {
            "block_size": BLOCK_SIZE,
            "num_draft_tokens": NUM_DRAFT_TOKENS,
            "mask_token_id": MASK_TOKEN_ID,
            "generation_eos_token_id": [EOS_TOKEN_ID],
            "cache": {
                "capacity": CAPACITY,
                "shape": [1, 1, CAPACITY, DRAFT_HIDDEN_SIZE],
                "input_names": cache_input_names,
                "output_names": [name.replace("past_", "present_") for name in cache_input_names],
            },
            "output_head": {"source_shape": [VOCAB_SIZE, DRAFT_HIDDEN_SIZE]},
        },
    }


class _Harness:
    def __init__(
        self,
        *,
        candidates: list[list[int]],
        predictions: list[list[int]],
        decode_tokens: list[int] | None = None,
        capacity: int = CAPACITY,
        draft_failure: bool = False,
        context_decode_failure: bool = False,
    ) -> None:
        contract = resolve_draft_runtime_contract(_metadata())
        self.log: list[dict[str, int | str]] = []
        cache_tensors = [
            _FakeCache(torch.zeros(1, 1, capacity, DRAFT_HIDDEN_SIZE, dtype=torch.float16))
            for _ in contract["cache_input_names"]
        ]
        self.graphs = HunyuanOCRDraftGraphs(
            context=_FakeSession("context", self.log),
            context_decode=_FakeSession("context_decode", self.log, fail=context_decode_failure),
            decode=_FakeSession("decode", self.log, candidates=candidates, fail=draft_failure),
            cache_input_names=contract["cache_input_names"],
            cache_output_names=contract["cache_output_names"],
            cache_tensors=cache_tensors,
            hidden_width=HIDDEN_WIDTH,
            draft_hidden_size=DRAFT_HIDDEN_SIZE,
            capacity=capacity,
        )
        self.draft_cache = HunyuanOCRDraftCacheController(capacity=capacity, block_size=BLOCK_SIZE)
        self.target_controller = HunyuanOCRTargetVerifyController(
            max_sequence_length=capacity,
            verify_input_length=BLOCK_SIZE,
            generation_eos_token_ids=(EOS_TOKEN_ID,),
            logits_width=VOCAB_SIZE,
            target_hidden_width=HIDDEN_WIDTH,
        )
        self.target = _Target(predictions, decode_tokens)
        self.decoder = HunyuanOCRSpeculativeDecoder(
            graphs=self.graphs,
            draft_cache=self.draft_cache,
            embedding_weight=torch.zeros(VOCAB_SIZE, DRAFT_HIDDEN_SIZE, dtype=torch.float16),
            mask_token_id=MASK_TOKEN_ID,
            generation_eos_token_ids=(EOS_TOKEN_ID,),
            block_size=BLOCK_SIZE,
            num_draft_tokens=NUM_DRAFT_TOKENS,
            run_target_verify=self._run_target_verify,
            commit_verify_prefix=self.target_controller.commit_verify_prefix,
            discard_verify_result=self.target_controller.discard_verify_result,
            run_decode_step=self.target.decode,
            target_logical_past_length=lambda: self.target_controller.logical_past_length,
        )

    def _run_target_verify(self, *, current_token_id: int, draft_token_ids: list[int]):
        return self.target_controller.run_target_verify(
            current_token_id=current_token_id,
            draft_token_ids=draft_token_ids,
            executor=self.target.verify,
        )

    def prefill(self, length: int) -> None:
        self.target_controller.restore_request_state(past_seq_length=length, rope_delta=0)
        self.decoder.append_context(torch.zeros(1, length, HIDDEN_WIDTH, dtype=torch.float16))


def _predictions(drafts: list[int], accepted: int, bonus: int) -> list[int]:
    return [*drafts[:accepted], bonus]


def test_load_draft_graphs_supports_injected_factories() -> None:
    contract = resolve_draft_runtime_contract(_metadata())
    opened: list[str] = []
    modes = iter(("context", "context_decode", "decode"))

    graphs = load_draft_graphs(
        SimpleNamespace(
            dflash_context_hmonnx="context.hmonnx",
            dflash_context_decode_hmonnx="context_decode.hmonnx",
            dflash_decode_hmonnx="decode.hmonnx",
        ),
        contract,
        device="cpu",
        session_factory=lambda path: opened.append(path) or _ContractSession(next(modes)),
        cache_factory=_FakeCache,
    )

    assert opened == ["context.hmonnx", "context_decode.hmonnx", "decode.hmonnx"]
    assert len(graphs.cache_tensors) == NUM_LAYERS * 2
    assert graphs.capacity == CAPACITY


def test_full_accept_commits_target_and_draft_lengths() -> None:
    drafts = list(range(10, 25))
    harness = _Harness(candidates=[drafts], predictions=[[*drafts, 5]])
    harness.prefill(2)

    produced = harness.decoder.run(first_token_id=3, max_new_tokens=BLOCK_SIZE + 1)

    assert produced == [3, *drafts, 5]
    assert harness.target_controller.logical_past_length == 2 + BLOCK_SIZE
    assert harness.draft_cache.committed_length == 2 + BLOCK_SIZE
    assert harness.decoder.stats.accepted_draft_tokens == NUM_DRAFT_TOKENS


@pytest.mark.parametrize("accepted", [1, 0])
def test_partial_and_zero_accept_commit_only_verified_prefix(accepted: int) -> None:
    drafts = list(range(10, 25))
    harness = _Harness(candidates=[drafts], predictions=[_predictions(drafts, accepted, 5)])
    harness.prefill(2)

    produced = harness.decoder.run(first_token_id=3, max_new_tokens=accepted + 2)

    assert produced == [3, *drafts[:accepted], 5]
    assert harness.target_controller.logical_past_length == 3 + accepted
    assert harness.draft_cache.committed_length == 3 + accepted
    assert harness.decoder.stats.accept_length_histogram[accepted] == 1


def test_capacity_shortfall_falls_back_without_fabricating_lengths() -> None:
    harness = _Harness(candidates=[], predictions=[], decode_tokens=[6, EOS_TOKEN_ID], capacity=20)
    harness.prefill(5)

    produced = harness.decoder.run(first_token_id=3, max_new_tokens=8)

    assert produced == [3, 6, EOS_TOKEN_ID]
    assert harness.decoder.stats.fallback_reason == FALLBACK_CAPACITY_SHORTFALL
    assert harness.target_controller.logical_past_length == 5
    assert harness.draft_cache.committed_length == 5


def test_draft_failure_falls_back_with_target_transaction_idle() -> None:
    harness = _Harness(
        candidates=[],
        predictions=[],
        decode_tokens=[6, EOS_TOKEN_ID],
        draft_failure=True,
    )
    harness.prefill(2)

    produced = harness.decoder.run(first_token_id=3, max_new_tokens=8)

    assert produced == [3, 6, EOS_TOKEN_ID]
    assert harness.decoder.stats.fallback_reason == FALLBACK_DRAFT_EXECUTION_FAILED
    assert harness.target_controller.has_pending_transaction is False
    assert harness.target_controller.logical_past_length == 2
    assert harness.draft_cache.committed_length == 2


def test_context_decode_failure_discards_both_transactions_before_plain_ar() -> None:
    harness = _Harness(
        candidates=[list(range(10, 25))],
        predictions=[[*range(10, 25), 5]],
        decode_tokens=[6, EOS_TOKEN_ID],
        context_decode_failure=True,
    )
    harness.prefill(2)

    produced = harness.decoder.run(first_token_id=3, max_new_tokens=8)

    assert produced == [3, 6, EOS_TOKEN_ID]
    assert harness.decoder.stats.fallback_reason == FALLBACK_DRAFT_EXECUTION_FAILED
    assert harness.target_controller.logical_past_length == 2
    assert harness.target_controller.has_pending_transaction is False
    assert harness.draft_cache.committed_length == 2
    assert harness.draft_cache.pending_transaction_id is None


def test_eos_and_max_new_tokens_stop_at_visible_boundary() -> None:
    eos_drafts = [10, EOS_TOKEN_ID, *range(12, 25)]
    eos_harness = _Harness(candidates=[eos_drafts], predictions=[[*eos_drafts, 5]])
    eos_harness.prefill(2)

    assert eos_harness.decoder.run(first_token_id=3, max_new_tokens=8) == [3, 10, EOS_TOKEN_ID]
    assert eos_harness.decoder.stats.stop_reason == STOP_EOS

    drafts = list(range(10, 25))
    max_harness = _Harness(candidates=[drafts], predictions=[[*drafts, 5]])
    max_harness.prefill(2)

    assert max_harness.decoder.run(first_token_id=3, max_new_tokens=3) == [3, 10, 11]
    assert max_harness.decoder.stats.stop_reason == STOP_MAX_LENGTH
    assert max_harness.decoder.stats.truncated_tokens == 14


def test_stats_summary_and_length_invariant() -> None:
    drafts = list(range(10, 25))
    harness = _Harness(candidates=[drafts], predictions=[_predictions(drafts, 1, 5)])
    harness.prefill(2)
    harness.decoder.run(first_token_id=3, max_new_tokens=3)

    summary = harness.decoder.stats.as_summary(enabled=True)

    assert summary["enabled"] is True
    assert summary["blocks"] == 1
    assert summary["proposed_draft_tokens"] == NUM_DRAFT_TOKENS
    assert summary["accepted_draft_tokens"] == 1
    assert summary["acceptance_rate"] == pytest.approx(1 / NUM_DRAFT_TOKENS)
    assert summary["block_token_counts"] == [2]
    assert summary["fallback_reason"] is None
    harness.decoder.assert_lengths_agree(draft_length=4)
    with pytest.raises(HunyuanOCRSpeculativeRuntimeError, match="dual-cache length divergence"):
        harness.decoder.assert_lengths_agree(draft_length=5)


def test_model_metadata_stays_verify_ready_until_draft_contract_exists() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            dflash_target_contract={
                "target_layer_ids": [0, 1],
                "target_hidden_size": 3,
                "target_hidden_concat_size": 6,
                "hidden_layout": "concat_last_dim",
                "reference_dtype": "bfloat16",
                "deployment_dtype": "float16",
            },
            dflash_config={},
            num_draft_tokens=15,
            verify_input_length=16,
        )
    )

    verify_ready = XHHunYuanOCRModel._spec_decode_metadata(model)
    assert verify_ready["status"] == "target_verify_ready"
    assert "draft" not in verify_ready

    model.config.dflash_config = {
        "draft_graphs": {
            "context": "draft_context.onnx",
            "context_decode": "draft_context_decode.onnx",
            "decode": "draft_decode.onnx",
            "contract": _metadata()["draft"],
        }
    }
    ready = XHHunYuanOCRModel._spec_decode_metadata(model)
    assert ready["status"] == "speculative_runtime_ready"
    assert ready["draft"]["block_size"] == BLOCK_SIZE


def test_plain_metadata_does_not_construct_speculative_runtime() -> None:
    metadata = SimpleNamespace(spec_decode={"mode": "dflash", "status": "target_only"})

    assert HunyuanOCRTextExportMeta._runtime_ready(metadata) is False
    ready = {
        **_metadata(),
        "status": "speculative_runtime_ready",
        "capabilities": {
            "target_hidden": True,
            "target_verify": True,
            "draft_graphs": True,
            "speculative_runtime": True,
        },
    }
    assert HunyuanOCRTextExportMeta._runtime_ready(SimpleNamespace(spec_decode=ready)) is True
    disabled = {**ready, "capabilities": {**ready["capabilities"], "speculative_runtime": False}}
    assert HunyuanOCRTextExportMeta._runtime_ready(SimpleNamespace(spec_decode=disabled)) is False


def test_prefill_plan_preserves_unpadded_sequence_and_rope_delta() -> None:
    processor = HunyuanOCRTextDataPreprocess(
        token_embedding=torch.nn.Embedding(32, 4),
        input_sequence_length=4,
        context_max_length=16,
        past_key_caches=None,
        past_value_caches=None,
        pad_token_id=0,
    ).to("cpu", torch.float16)

    plan = processor.build_prefill_plan({"input_ids": torch.tensor([[1, 2, 3, 4, 5]])})

    assert plan.inputs_embeds.shape == (1, 5, 4)
    assert plan.position_ids.shape == (4, 1, 5)
    assert plan.valid_length == 5
    assert plan.rope_delta.tolist() == [[0]]


def test_hmonnx_runtime_dispatches_only_explicit_ready_requests(monkeypatch) -> None:
    _install_hmonnx_optimizer_stub()
    from xhmodel_merak.xh_llm.hmonnx.vision_llm_hmonnx_model import VisonLLMHMONNXModel
    from xhmodel_merak.xh_llm.models.hunyuan_ocr.hunyuan_ocr_hmonnx_inference import XHHunYuanOCRHMONNXModel

    runtime = object.__new__(XHHunYuanOCRHMONNXModel)
    runtime._speculative_runtime_ready = True
    runtime._last_request_summary = None
    calls = []
    monkeypatch.setattr(runtime, "_generate_speculative", lambda args, kwargs: calls.append("spec") or "spec")
    monkeypatch.setattr(VisonLLMHMONNXModel, "generate", lambda self, *args, **kwargs: calls.append("plain") or "plain")

    assert runtime.generate(dflash_enabled=True, max_new_tokens=4) == "spec"
    assert runtime.generate(dflash_enabled=False, max_new_tokens=4) == "plain"
    assert calls == ["spec", "plain"]

    runtime._speculative_runtime_ready = False
    with pytest.raises(HunyuanOCRSpeculativeRuntimeError, match="runtime-ready"):
        runtime.generate(dflash_enabled=True, max_new_tokens=4)


def test_hmonnx_runtime_rejects_generation_options_speculative_path_cannot_honor(monkeypatch) -> None:
    _install_hmonnx_optimizer_stub()
    from xhmodel_merak.xh_llm.models.hunyuan_ocr.hunyuan_ocr_hmonnx_inference import XHHunYuanOCRHMONNXModel

    runtime = object.__new__(XHHunYuanOCRHMONNXModel)
    runtime._speculative_runtime_ready = True
    runtime.meta_info = SimpleNamespace(generation_eos_token_id=EOS_TOKEN_ID)
    monkeypatch.setattr(runtime, "_generate_speculative", lambda args, kwargs: "unexpected")

    with pytest.raises(HunyuanOCRSpeculativeRuntimeError, match="return_dict_in_generate"):
        runtime.generate(dflash_enabled=True, return_dict_in_generate=True, max_new_tokens=2)
    with pytest.raises(HunyuanOCRSpeculativeRuntimeError, match="num_return_sequences"):
        runtime.generate(dflash_enabled=True, num_return_sequences=2, max_new_tokens=2)
    with pytest.raises(HunyuanOCRSpeculativeRuntimeError, match="repetition_penalty"):
        runtime.generate(dflash_enabled=True, repetition_penalty=2.0, max_new_tokens=2)
    with pytest.raises(HunyuanOCRSpeculativeRuntimeError, match="stopping_criteria"):
        runtime.generate(dflash_enabled=True, stopping_criteria=[object()], max_new_tokens=2)
    with pytest.raises(HunyuanOCRSpeculativeRuntimeError, match="streamer"):
        runtime.generate(dflash_enabled=True, streamer=object(), max_new_tokens=2)
    with pytest.raises(HunyuanOCRSpeculativeRuntimeError, match="generation_eos_token_id"):
        runtime.generate(dflash_enabled=True, eos_token_id=123, max_new_tokens=2)
    with pytest.raises(HunyuanOCRSpeculativeRuntimeError, match="min_new_tokens"):
        runtime.generate(dflash_enabled=True, min_new_tokens=2, max_new_tokens=2)

    with pytest.raises(HunyuanOCRSpeculativeRuntimeError, match="dflash_num_draft_tokens"):
        runtime.generate(dflash_enabled=True, dflash_num_draft_tokens="7", max_new_tokens=2)