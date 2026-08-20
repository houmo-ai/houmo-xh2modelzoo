from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast


OFFICIAL_ROOT = Path(os.environ["MINICPM_O45_MODEL_DIR"]) if os.environ.get("MINICPM_O45_MODEL_DIR") else None


def _run_llm_decoder_graph(*args, **kwargs):
    """Exercise the shared traversal plus MiniCPM-specific output adapter."""
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_text_decoder import _iter_decoder_graph_outputs
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_llm import _collect_llm_decoder_outputs

    outputs = _iter_decoder_graph_outputs(*args, **kwargs)
    return _collect_llm_decoder_outputs(outputs, args[2].shape[1])


def _require_official_root() -> Path:
    if OFFICIAL_ROOT is None:
        pytest.skip("set MINICPM_O45_MODEL_DIR to run official remote-code tests")
    if not (OFFICIAL_ROOT / "utils.py").is_file():
        pytest.skip(f"official MiniCPM-o-4.5 source not found: {OFFICIAL_ROOT}")
    return OFFICIAL_ROOT


def _official_utils():
    official_root = _require_official_root()
    package_name = "official_minicpm_o_4_5"
    package = types.ModuleType(package_name)
    package.__path__ = [str(official_root)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(f"{package_name}.utils", official_root / "utils.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Tokenizer:
    eos_token_id = 4
    unk_token_id = -1
    all_special_ids = [4]
    all_special_tokens = ["</s>"]

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|chunk_eos|>": 5, "<|chunk_tts_eos|>": 6, "<|turn_eos|>": 7, "<|speak|>": 8}.get(token, 0)


class _FakeSession:
    def __init__(self, hidden_size: int = 8, vocab_size: int = 11) -> None:
        self.calls: list[tuple[int, int, int]] = []
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def __call__(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current = int(inputs[2].item())
        past = int(inputs[1].item())
        self.calls.append((past, current, inputs[0].shape[1]))
        logits = torch.arange(self.vocab_size, dtype=inputs[0].dtype).reshape(1, 1, -1)
        hidden = inputs[0].clone()
        return logits, hidden


def _runtime(prefill: _FakeSession, decode: _FakeSession):
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_cache import FixedCapacityKVCacheMixin
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_llm import (
        MiniCPMO45LLMHMONNXRuntime,
    )
    from xhmodel_merak.xh_llm.types import KVCacheConfig

    runtime = object.__new__(MiniCPMO45LLMHMONNXRuntime)
    runtime.prefill_model = prefill
    runtime.decode_model = decode
    runtime.prefill_length = 4
    runtime.input_sequence_length = 4
    runtime._kvcache_mixin = FixedCapacityKVCacheMixin(
        KVCacheConfig(num_layers=1, kv_cache_shape=[1, 2, 16, 4]),
        "llm",
    )
    runtime._kvcache_mixin.prepare_fixed_cache("cpu")
    return runtime


def test_forward_hf_returns_official_output_and_commits_hmonnx_cache_length() -> None:
    prefill = _FakeSession()
    runtime = _runtime(prefill, _FakeSession())

    output = runtime.forward_hf(
        inputs_embeds=torch.ones((1, 3, 8), dtype=torch.float16),
        past_key_values=None,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )

    assert isinstance(output, CausalLMOutputWithPast)
    assert output.logits.shape[:2] == (1, 1)
    assert output.hidden_states is not None
    assert output.hidden_states[-1].shape == (1, 3, 8)
    assert output.past_key_values is runtime.hf_cache
    assert runtime.hf_cache.get_seq_length() == 3
    assert prefill.calls == [(0, 3, 4)]


def test_forward_hf_rejects_foreign_cache_and_input_ids() -> None:
    runtime = _runtime(_FakeSession(), _FakeSession())

    # A non-empty foreign cache is rejected: HMONNX state must live in the runtime cache.
    with pytest.raises(ValueError, match="runtime-owned"):
        runtime.forward_hf(
            inputs_embeds=torch.ones((1, 1, 8), dtype=torch.float16),
            past_key_values=SimpleNamespace(get_seq_length=lambda: 3),
        )

    # transformers generate() feeds an empty foreign Cache before the first forward;
    # that is accepted because the HMONNX runtime cache owns the actual state.
    output = runtime.forward_hf(
        inputs_embeds=torch.ones((1, 1, 8), dtype=torch.float16),
        past_key_values=SimpleNamespace(get_seq_length=lambda: 0),
    )
    assert output.logits.shape[0] == 1 and output.logits.shape[1] == 1

    with pytest.raises(ValueError, match="inputs_embeds"):
        runtime.forward_hf(input_ids=torch.ones((1, 1), dtype=torch.long))


def test_official_chunk_and_stream_decoder_use_hmonnx_forward_without_native_model() -> None:
    official = _official_utils()
    runtime = _runtime(_FakeSession(), _FakeSession())

    class FakeLLM:
        def __init__(self) -> None:
            self.model = SimpleNamespace(
                embed_tokens=lambda token_ids: torch.zeros((*token_ids.shape, 8), dtype=torch.float16)
            )

        def get_input_embeddings(self):
            return self.model.embed_tokens

        def __call__(self, **kwargs):
            return runtime.forward_hf(**kwargs)

    llm = FakeLLM()
    tokenizer = _Tokenizer()
    chunker = official.ChunkPrefillChunkGenerate(llm, tokenizer, ["<|chunk_eos|>"])
    chunk = chunker.chunk_generate(
        torch.ones((1, 2, 8), dtype=torch.float16),
        runtime.hf_cache,
        is_first_generate_chunk=True,
        chunk_size=2,
        return_hidden_states=True,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
    )
    assert chunk.chunk_token_ids.shape == (1, 2)
    assert runtime.hf_cache.get_seq_length() == 3

    decoder = official.StreamDecoder(llm, tokenizer)
    decoder.feed(torch.ones((2, 8), dtype=torch.float16))
    assert decoder.get_cache_length() == 2
    decoder.reset()
    assert decoder.get_cache_length() == 0
    decoder.feed(torch.ones((1, 8), dtype=torch.float16))
    assert runtime.hf_cache.get_seq_length() == 1


def test_reset_session_syncs_physical_caches_after_official_reset() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime import MiniCPMO45HMONNXRuntime

    calls: list[str] = []
    runtime = object.__new__(MiniCPMO45HMONNXRuntime)
    runtime.host_model = SimpleNamespace(reset_session=lambda value=True: calls.append(f"host:{value}"))
    runtime.audio = SimpleNamespace(reset_state=lambda: calls.append("audio"))
    runtime.llm = SimpleNamespace(reset_state=lambda: calls.append("llm"))
    runtime.tts = SimpleNamespace(reset_state=lambda: calls.append("tts"))

    runtime.reset_session(False)

    assert calls == ["host:False", "audio", "llm", "tts"]


def test_sample_forward_embeds_input_ids_before_hmonnx_forward() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.hf_compatible import create_llm_wraped_cls

    class Base:
        pass

    embedded = torch.full((1, 2, 8), 3, dtype=torch.float16)
    observed: dict[str, object] = {}
    wrapped = object.__new__(create_llm_wraped_cls(Base))
    wrapped._prefill = True
    wrapped._past_seq_length = 0
    wrapped.config = SimpleNamespace(output_hidden_states=False, use_cache=True)
    wrapped.embed_tokens = lambda input_ids: observed.update(input_ids=input_ids) or embedded
    wrapped._llm_model = SimpleNamespace(
        set_input_sequence_length=lambda value: observed.update(sequence_length=value),
        forward_hf=lambda **kwargs: (
            observed.update(forward=kwargs)
            or CausalLMOutputWithPast(logits=torch.ones((1, 1, 11)), past_key_values=runtime_cache)
        ),
    )
    runtime_cache = SimpleNamespace()

    output = wrapped._sample_forward(input_ids=torch.tensor([[2, 3]], dtype=torch.long))

    assert output.past_key_values is runtime_cache
    assert torch.equal(observed["input_ids"], torch.tensor([[2, 3]]))
    assert torch.equal(observed["forward"]["inputs_embeds"], embedded)


def test_official_drop_tokens_compacts_fixed_cache_and_allows_followup_feed() -> None:
    runtime = _runtime(_FakeSession(), _FakeSession())
    cache = runtime.hf_cache
    cache._key_caches[0][0, 0, :5] = torch.arange(5, dtype=torch.float16).unsqueeze(1)
    cache._value_caches[0][0, 0, :5] = torch.arange(10, 15, dtype=torch.float16).unsqueeze(1)
    cache.commit_length(5)

    official = _official_utils()
    _, _, success = official.drop_tokens_from_cache(cache, length=2, preserve=1, position_offset=0)

    assert success
    assert cache.get_seq_length() == 3
    assert cache.value_cache[0][0, 0, :, 0].tolist() == [10, 13, 14]

    class FakeLLM:
        config = SimpleNamespace(rope_theta=10000.0)

        def __call__(self, **kwargs):
            return runtime.forward_hf(**kwargs)

    decoder = official.StreamDecoder(FakeLLM(), _Tokenizer())
    decoder.cache = cache
    decoder.feed(torch.ones((1, 8), dtype=torch.float16))
    assert cache.get_seq_length() == 4


def test_official_drop_tokens_compacts_all_layers_before_committing_length() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_cache import FixedCapacityDynamicCache

    keys = [torch.zeros((1, 1, 8, 4), dtype=torch.float16) for _ in range(2)]
    values = [torch.zeros((1, 1, 8, 4), dtype=torch.float16) for _ in range(2)]
    cache = FixedCapacityDynamicCache(keys, values, capacity=8, component="llm")
    for layer in range(2):
        cache._key_caches[layer][0, 0, :5] = torch.arange(layer * 10, layer * 10 + 5).unsqueeze(1)
        cache._value_caches[layer][0, 0, :5] = torch.arange(layer * 10 + 20, layer * 10 + 25).unsqueeze(1)
    cache.commit_length(5)
    identity = id(cache)

    official = _official_utils()
    returned, _, success = official.drop_tokens_from_cache(cache, length=2, preserve=1, position_offset=0)

    assert success
    assert id(returned) == identity
    assert cache.get_seq_length() == 3
    assert [layer[0, 0, :, 0].tolist() for layer in cache.value_cache] == [
        [20, 23, 24],
        [30, 33, 34],
    ]


def test_multi_layer_compaction_validation_failure_is_atomic() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_cache import FixedCapacityDynamicCache

    keys = [torch.full((1, 1, 8, 4), float(layer + 1), dtype=torch.float16) for layer in range(2)]
    values = [torch.full((1, 1, 8, 4), float(layer + 11), dtype=torch.float16) for layer in range(2)]
    cache = FixedCapacityDynamicCache(keys, values, capacity=8, component="llm")
    cache.commit_length(5)
    key_before = [tensor.clone() for tensor in keys]
    value_before = [tensor.clone() for tensor in values]

    key_views = cache.key_cache
    value_views = cache.value_cache
    key_views[0] = torch.zeros((1, 1, 3, 4), dtype=torch.float16)
    key_views[1] = torch.zeros((1, 1, 3, 4), dtype=torch.float16)
    value_views[0] = torch.zeros((1, 1, 3, 4), dtype=torch.float16)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        value_views[1] = torch.zeros((1, 2, 3, 4), dtype=torch.float16)

    assert cache.get_seq_length() == 5
    assert all(torch.equal(current, original) for current, original in zip(keys, key_before, strict=True))
    assert all(torch.equal(current, original) for current, original in zip(values, value_before, strict=True))


def test_multi_chunk_prefill_returns_logits_for_all_real_input_tokens() -> None:
    class Session:
        def __call__(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            length = int(inputs[2].item())
            logits = torch.arange(length * 5, dtype=torch.float16).reshape(1, length, 5)
            return logits, inputs[0].clone()

    logits, hidden = _run_llm_decoder_graph(
        Session(),
        lambda *inputs: (_ for _ in ()).throw(AssertionError("decode must not run")),
        torch.ones((1, 6, 8), dtype=torch.float16),
        past_seq_length=0,
        current_input_length=6,
        past_key_caches=[torch.zeros((1, 1, 8, 1), dtype=torch.float16)],
        past_value_caches=[torch.zeros((1, 1, 8, 1), dtype=torch.float16)],
        prefill_length=4,
    )

    assert logits.shape == (1, 6, 5)
    assert hidden.shape == (1, 6, 8)


def test_multi_chunk_prefill_slices_fixed_shape_logits_to_real_tokens() -> None:
    class Session:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            self.calls += 1
            logits = torch.full((1, 4, 2), float(self.calls), dtype=torch.float16)
            return logits, inputs[0].clone()

    logits, _ = _run_llm_decoder_graph(
        Session(),
        lambda *inputs: (_ for _ in ()).throw(AssertionError("decode must not run")),
        torch.ones((1, 6, 8), dtype=torch.float16),
        past_seq_length=0,
        current_input_length=6,
        past_key_caches=[torch.zeros((1, 1, 8, 1), dtype=torch.float16)],
        past_value_caches=[torch.zeros((1, 1, 8, 1), dtype=torch.float16)],
        prefill_length=4,
    )

    assert logits.shape == (1, 6, 2)
    assert logits[0, :, 0].tolist() == [1, 1, 1, 1, 2, 2]


def test_multi_token_streaming_prefill_uses_prefill_graph_with_existing_cache() -> None:
    calls: list[str] = []

    def prefill(*inputs: torch.Tensor):
        calls.append("prefill")
        return torch.zeros(1, 4, 8), inputs[0]

    def decode(*inputs: torch.Tensor):
        calls.append("decode")
        return torch.zeros(1, 1, 8), inputs[0]

    _run_llm_decoder_graph(
        prefill,
        decode,
        torch.zeros(1, 3, 8),
        past_seq_length=7,
        current_input_length=3,
        past_key_caches=(),
        past_value_caches=(),
        prefill_length=4,
    )

    assert calls == ["prefill"]
