from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast


OFFICIAL_ROOT = Path(os.environ["MINICPM_O45_MODEL_DIR"]) if os.environ.get("MINICPM_O45_MODEL_DIR") else None


def _require_official_root() -> Path:
    if OFFICIAL_ROOT is None:
        pytest.skip("set MINICPM_O45_MODEL_DIR to run official remote-code tests")
    if not all((OFFICIAL_ROOT / name).is_file() for name in ("utils.py", "modeling_minicpmo.py")):
        pytest.skip(f"official MiniCPM-o-4.5 source not found: {OFFICIAL_ROOT}")
    return OFFICIAL_ROOT


def _official_utils():
    official_root = _require_official_root()
    package_name = "official_minicpm_o_4_5_tts"
    package = types.ModuleType(package_name)
    package.__path__ = [str(official_root)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(f"{package_name}.utils", official_root / "utils.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Session:
    def __init__(self, hidden_size: int = 8, vocab_size: int = 7) -> None:
        self.calls: list[tuple[int, int, int, torch.Tensor]] = []
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def __call__(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        past = int(inputs[1].item())
        current = int(inputs[2].item())
        self.calls.append((past, current, inputs[0].shape[1], inputs[-1].clone()))
        return torch.zeros((1, inputs[0].shape[1], self.hidden_size), dtype=inputs[0].dtype), inputs[0].clone()


class _ProjectionSession:
    def __init__(self, output_size: int) -> None:
        self.output_size = output_size

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros((*inputs.shape[:-1], self.output_size), dtype=inputs.dtype, device=inputs.device)


def _runtime(prefill: _Session, decode: _Session):
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_cache import FixedCapacityKVCacheMixin
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_tts import MiniCPMO45TTSHMONNXRuntime
    from xhmodel_merak.xh_llm.types import KVCacheConfig

    runtime = object.__new__(MiniCPMO45TTSHMONNXRuntime)
    torch.nn.Module.__init__(runtime)
    runtime._device = torch.device("cpu")
    runtime.prefill_model = prefill
    runtime.decode_model = decode
    runtime.prefill_length = 4
    runtime.input_sequence_length = 4
    runtime._kvcache_mixin = FixedCapacityKVCacheMixin(
        KVCacheConfig(num_layers=1, kv_cache_shape=[1, 2, 16, 4]),
        "tts",
    )
    runtime._kvcache_mixin.prepare_fixed_cache("cpu")
    runtime.config = SimpleNamespace(rope_theta=10000.0, hidden_size=8, num_attention_heads=2)
    runtime.projector_semantic_session = _ProjectionSession(8)
    runtime.head_code_session = _ProjectionSession(4)
    runtime.projection_seq_capacity = 16
    return runtime


def _seed_cache(runtime, length: int) -> None:
    runtime.hf_cache.commit_length(length)
    for cache in [*runtime.past_key_caches, *runtime.past_value_caches]:
        cache[:, :, :length].fill_(7)


def _wrapped_tts(runtime):
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.hf_compatible import create_tts_model_wraped_cls

    class Base:
        generate_chunk = None

    wrapped = object.__new__(create_tts_model_wraped_cls(Base))
    wrapped.device = torch.device("cpu")
    wrapped.num_vq = 1
    wrapped.config = SimpleNamespace(num_audio_tokens=4, backbone_model="llama")
    wrapped.emb_code = torch.nn.ModuleList([torch.nn.Embedding(4, 8).to(torch.float16)])
    wrapped.head_code = torch.nn.ModuleList([torch.nn.Linear(8, 4, bias=False).to(torch.float16)])
    wrapped.model = runtime
    wrapped._tts_llama_model = runtime
    with torch.no_grad():
        wrapped.head_code[0].weight.zero_()
        wrapped.head_code[0].weight[1].fill_(1)
    return wrapped


def test_tts_forward_hf_returns_official_output_and_commits_cache() -> None:
    prefill = _Session()
    runtime = _runtime(prefill, _Session())

    output = runtime.forward_hf(
        inputs_embeds=torch.ones((1, 3, 8), dtype=torch.float16),
        past_key_values=None,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )

    assert isinstance(output, BaseModelOutputWithPast)
    assert output.last_hidden_state.shape == (1, 3, 8)
    assert output.past_key_values is runtime.hf_cache
    assert runtime.hf_cache.get_seq_length() == 3
    assert prefill.calls[0][:3] == (0, 3, 4)
    assert prefill.calls[0][3].shape == (1, 1, 4, 16)


def test_tts_forward_hf_routes_decode_and_rejects_foreign_cache() -> None:
    decode = _Session()
    runtime = _runtime(_Session(), decode)
    runtime.forward_hf(inputs_embeds=torch.ones((1, 2, 8), dtype=torch.float16))

    output = runtime.forward_hf(
        inputs_embeds=torch.ones((1, 1, 8), dtype=torch.float16),
        past_key_values=runtime.hf_cache,
        position_ids=torch.tensor([[2]]),
        attention_mask=torch.zeros((1, 1, 1, 16), dtype=torch.float16),
    )

    assert output.past_key_values is runtime.hf_cache
    assert decode.calls[-1][:3] == (2, 1, 1)
    with pytest.raises(ValueError, match="runtime-owned"):
        runtime.forward_hf(inputs_embeds=torch.ones((1, 1, 8)), past_key_values=SimpleNamespace())


def test_official_tts_streaming_generator_uses_hmonnx_model_and_yields_buffer() -> None:
    official = _official_utils()
    runtime = _runtime(_Session(), _Session())

    class FakeTTS:
        device = torch.device("cpu")
        num_vq = 1
        num_audio_tokens = 4
        recomputed_chunks = 0
        attention_type = "full_attention"
        chunk_window_size = 2
        token_window_size = 32
        audio_bos_token_id = 2
        config = SimpleNamespace(text_eos_token_id=3)
        emb_text = torch.nn.Embedding(8, 8)
        emb_code = torch.nn.ModuleList([torch.nn.Embedding(4, 8)])
        head_code = torch.nn.ModuleList([torch.nn.Linear(8, 4, bias=False)])
        model = runtime

    with torch.no_grad():
        FakeTTS.head_code[0].weight.zero_()
        FakeTTS.head_code[0].weight[1].fill_(1)
    generator = official.TTSStreamingGenerator(FakeTTS(), 1.0, eos_token=3, chunk_size=2)
    chunks = list(generator.generate_with_buffer(torch.ones((1, 1, 8)), text_finished=True, max_new_token=2))

    assert chunks
    assert all(chunk.shape[0] == 1 for chunk, _ in chunks)
    assert runtime.hf_cache.get_seq_length() >= 2


def test_generate_chunk_shape_preserves_cache_identity_and_positions(monkeypatch) -> None:
    official_root = _require_official_root()
    spec = importlib.util.spec_from_file_location("official_tts_model", official_root / "modeling_minicpmo.py")
    assert spec is not None and spec.loader is not None

    runtime = _runtime(_Session(), _Session())
    wrapped = _wrapped_tts(runtime)
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import hf_compatible

    monkeypatch.setattr(hf_compatible, "_official_tts_gen_logits", lambda *_args: ([], []))

    official_source = (official_root / "modeling_minicpmo.py").read_text(encoding="utf-8")
    assert "def generate_chunk(" in official_source
    del spec

    tokens, returned = wrapped.generate_chunk(
        inputs_embeds=torch.ones((1, 2, 8), dtype=torch.float16),
        temperature=torch.tensor([1.0]),
        repetition_penalty=1.0,
        eos_token=torch.tensor([3]),
        force_no_stop=True,
        max_new_token=2,
        past_key_values=None,
        text_start_pos=0,
    )

    assert tokens.shape == (1, 1, 1)
    assert returned is runtime.hf_cache
    assert runtime.hf_cache.get_seq_length() == 3


def test_generate_chunk_ignores_duplex_gen_logits_tuple_argument(monkeypatch) -> None:
    """Duplex passes the whole gen_logits() (warpers, processors) tuple as logits_processors;
    like the official generate_chunk, the wrapper must use gen_logits' processors, not iterate
    the tuple."""
    runtime = _runtime(_Session(), _Session())
    wrapped = _wrapped_tts(runtime)
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import hf_compatible

    applied: list[bool] = []

    def processor(previous, logits):
        applied.append(True)
        return logits

    monkeypatch.setattr(hf_compatible, "_official_tts_gen_logits", lambda *_args: ([], [processor]))

    tokens, _ = wrapped.generate_chunk(
        inputs_embeds=torch.ones((1, 2, 8), dtype=torch.float16),
        temperature=torch.tensor([1.0]),
        repetition_penalty=1.0,
        eos_token=torch.tensor([3]),
        force_no_stop=True,
        max_new_token=2,
        past_key_values=None,
        text_start_pos=0,
        logits_processors=(["warper_placeholder"], ["processor_placeholder"]),
    )

    assert tokens.shape == (1, 1, 1)
    assert applied


def test_tts_forward_hf_builds_absolute_query_masks_for_nonzero_multichunk_past() -> None:
    prefill = _Session()
    runtime = _runtime(prefill, _Session())
    _seed_cache(runtime, 5)

    runtime.forward_hf(inputs_embeds=torch.ones((1, 6, 8), dtype=torch.float16), past_key_values=runtime.hf_cache)

    assert [call[:3] for call in prefill.calls] == [(5, 4, 4), (9, 2, 4)]
    first_mask = prefill.calls[0][3]
    second_mask = prefill.calls[1][3]
    assert torch.all(first_mask[:, :, 0, :6] == 0)
    assert torch.all(first_mask[:, :, 1, :7] == 0)
    assert torch.all(second_mask[:, :, 0, :10] == 0)
    assert torch.all(second_mask[:, :, 1, :11] == 0)


def test_tts_forward_hf_without_cache_preserves_live_buffers_and_length() -> None:
    runtime = _runtime(_Session(), _Session())
    _seed_cache(runtime, 3)
    before = [cache.clone() for cache in [*runtime.past_key_caches, *runtime.past_value_caches]]

    output = runtime.forward_hf(
        inputs_embeds=torch.ones((1, 2, 8), dtype=torch.float16),
        use_cache=False,
    )

    assert output.past_key_values is None
    assert runtime.hf_cache.get_seq_length() == 3
    assert all(
        torch.equal(current, original)
        for current, original in zip([*runtime.past_key_caches, *runtime.past_value_caches], before, strict=True)
    )


def test_tts_forward_hf_rejects_capacity_before_graph_call() -> None:
    prefill = _Session()
    runtime = _runtime(prefill, _Session())
    _seed_cache(runtime, 15)

    with pytest.raises(RuntimeError, match="tts cache capacity exceeded"):
        runtime.forward_hf(inputs_embeds=torch.ones((1, 2, 8), dtype=torch.float16), past_key_values=runtime.hf_cache)

    assert prefill.calls == []


@pytest.mark.parametrize("attention_type", ["full_attention", "sliding_window", "sliding_recompute", "reindex"])
def test_official_tts_attention_modes_have_explicit_runtime_support(attention_type: str) -> None:
    official = _official_utils()
    runtime = _runtime(_Session(), _Session())
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.hf_compatible import (
        patch_tts_streaming_generator_cache_compat,
    )

    patch_tts_streaming_generator_cache_compat(official.TTSStreamingGenerator)

    class FakeTTS:
        device = torch.device("cpu")
        num_vq = 1
        num_audio_tokens = 4
        recomputed_chunks = 0
        attention_type = "full_attention"
        chunk_window_size = 2
        token_window_size = 2
        audio_bos_token_id = 2
        config = SimpleNamespace(text_eos_token_id=3)
        emb_text = torch.nn.Embedding(8, 8)
        emb_code = torch.nn.ModuleList([torch.nn.Embedding(4, 8)])
        head_code = torch.nn.ModuleList([torch.nn.Linear(8, 4, bias=False)])
        model = runtime

    FakeTTS.attention_type = attention_type

    generator = official.TTSStreamingGenerator(FakeTTS(), 1.0, eos_token=3, chunk_size=1)
    cache_identity = id(runtime.hf_cache)
    list(generator.generate_with_buffer(torch.ones((1, 1, 8)), text_finished=False, max_new_token=1))
    list(generator.generate_with_buffer(torch.ones((1, 1, 8)), text_finished=True, max_new_token=1))

    assert id(runtime.hf_cache) == cache_identity
    assert generator.past_key_values is runtime.hf_cache
    assert runtime.hf_cache.get_seq_length() > 0


def test_official_tts_rejects_unsupported_attention_type_before_graph_call() -> None:
    official = _official_utils()
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.hf_compatible import (
        patch_tts_streaming_generator_cache_compat,
    )

    patch_tts_streaming_generator_cache_compat(official.TTSStreamingGenerator)
    prefill = _Session()
    decode = _Session()

    class FakeTTS:
        device = torch.device("cpu")
        num_vq = 1
        num_audio_tokens = 4
        recomputed_chunks = 0
        attention_type = "unsupported"
        chunk_window_size = 2
        token_window_size = 2
        audio_bos_token_id = 2
        config = SimpleNamespace(text_eos_token_id=3)
        emb_text = torch.nn.Embedding(8, 8)
        emb_code = torch.nn.ModuleList([torch.nn.Embedding(4, 8)])
        head_code = torch.nn.ModuleList([torch.nn.Linear(8, 4, bias=False)])
        model = SimpleNamespace(config=SimpleNamespace(rope_theta=10000.0, hidden_size=8, num_attention_heads=2))

    with pytest.raises(ValueError, match="unsupported attention_type"):
        list(
            official.TTSStreamingGenerator(FakeTTS(), 1.0, eos_token=3).generate_with_buffer(
                torch.ones((1, 1, 8)), max_new_token=1
            )
        )

    assert prefill.calls == []
    assert decode.calls == []


def test_fixed_cache_update_validates_value_before_key_write() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_cache import FixedCapacityDynamicCache

    key = torch.full((1, 2, 8, 4), 3, dtype=torch.float16)
    value = torch.full((1, 2, 8, 4), 5, dtype=torch.float16)
    cache = FixedCapacityDynamicCache([key], [value], capacity=8, component="tts")
    key_before = key.clone()
    value_before = value.clone()

    with pytest.raises(RuntimeError, match="value"):
        cache.update(
            torch.ones((1, 2, 2, 4), dtype=torch.float16),
            torch.ones((1, 3, 2, 4), dtype=torch.float16),
            0,
        )

    assert torch.equal(key, key_before)
    assert torch.equal(value, value_before)
    assert cache.get_seq_length() == 0


def test_import_cache_is_atomic_when_later_layer_is_invalid() -> None:
    runtime = _runtime(_Session(), _Session())
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_cache import FixedCapacityKVCacheMixin
    from xhmodel_merak.xh_llm.types import KVCacheConfig

    runtime._kvcache_mixin = FixedCapacityKVCacheMixin(
        KVCacheConfig(num_layers=2, kv_cache_shape=[1, 2, 16, 4]),
        "tts",
    )
    runtime._kvcache_mixin.prepare_fixed_cache("cpu")
    _seed_cache(runtime, 3)
    before = [cache.clone() for cache in [*runtime.past_key_caches, *runtime.past_value_caches]]
    foreign = [
        (torch.ones((1, 2, 2, 4), dtype=torch.float16), torch.ones((1, 2, 2, 4), dtype=torch.float16)),
        (torch.ones((1, 3, 2, 4), dtype=torch.float16), torch.ones((1, 3, 2, 4), dtype=torch.float16)),
    ]

    with pytest.raises(ValueError, match="layer 1 shape"):
        runtime.forward_hf(inputs_embeds=torch.ones((1, 1, 8), dtype=torch.float16), past_key_values=foreign)

    assert runtime.hf_cache.get_seq_length() == 3
    assert all(
        torch.equal(current, original)
        for current, original in zip([*runtime.past_key_caches, *runtime.past_value_caches], before, strict=True)
    )


def test_generate_chunk_requests_official_repetition_processor(monkeypatch) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import hf_compatible

    requested: dict[str, float] = {}
    monkeypatch.setattr(
        hf_compatible,
        "_official_tts_gen_logits",
        lambda num_code, repetition_penalty: (
            requested.update(
                num_code=num_code,
                repetition_penalty=repetition_penalty,
            )
            or ([], [])
        ),
        raising=False,
    )
    wrapped = _wrapped_tts(_runtime(_Session(), _Session()))

    wrapped.generate_chunk(
        inputs_embeds=torch.ones((1, 2, 8), dtype=torch.float16),
        temperature=torch.tensor([1.0]),
        repetition_penalty=1.3,
        eos_token=torch.tensor([3]),
        max_new_token=1,
        text_start_pos=0,
    )

    assert requested == {"num_code": 4, "repetition_penalty": 1.3}


def test_generate_chunk_surfaces_official_sampling_import_failure(monkeypatch) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import hf_compatible

    def fail(_num_code: int, _repetition_penalty: float):
        raise RuntimeError("official TTS sampling integration unavailable")

    monkeypatch.setattr(hf_compatible, "_official_tts_gen_logits", fail, raising=False)
    wrapped = _wrapped_tts(_runtime(_Session(), _Session()))

    with pytest.raises(RuntimeError, match="sampling integration unavailable"):
        wrapped.generate_chunk(
            inputs_embeds=torch.ones((1, 2, 8), dtype=torch.float16),
            temperature=torch.tensor([1.0]),
            repetition_penalty=1.3,
            eos_token=torch.tensor([3]),
            max_new_token=1,
            text_start_pos=0,
        )


def test_non_streaming_generation_surfaces_official_sampling_import_failure(monkeypatch) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import hf_compatible

    def fail(_num_code: int, _repetition_penalty: float):
        raise RuntimeError("official TTS sampling integration unavailable")

    monkeypatch.setattr(hf_compatible, "_official_tts_gen_logits", fail)
    monkeypatch.setattr(
        hf_compatible,
        "__name__",
        "test",
        raising=False,
    )
    wrapped = _wrapped_tts(_runtime(_Session(), _Session()))

    with pytest.raises(RuntimeError, match="sampling integration unavailable"):
        wrapped._generate_non_streaming(
            inputs_embeds=torch.ones((1, 2, 8), dtype=torch.float16),
            eos_token=torch.tensor([3]),
            max_new_token=1,
        )
