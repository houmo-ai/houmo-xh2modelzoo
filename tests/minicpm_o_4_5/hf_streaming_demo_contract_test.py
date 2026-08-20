from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from transformers.cache_utils import DynamicCache, EncoderDecoderCache


RUNNER_PATH = Path("examples_merak/llm/minicpm_o_4_5/minicpm_o_4_5_hf_streaming_demo.py")
CASE_NAMES = (
    "session_audio_text",
    "session_audio_reply",
    "duplex_audio_text",
    "duplex_audio_reply",
    "duplex_omni_reply",
)
MODEL_DIR = Path(os.environ.get("MINICPM_O45_MODEL_DIR", "/models/MiniCPM-o-4_5"))


def _load_runner():
    spec = importlib.util.spec_from_file_location("minicpm_o_4_5_hf_streaming_demo", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_contains_all_certified_cases_and_deterministic_parameters() -> None:
    runner = _load_runner()

    manifest = runner.build_manifest(MODEL_DIR)

    assert tuple(item["name"] for item in manifest["cases"]) == CASE_NAMES
    assert manifest["parameters"] == {
        "seed": 42,
        "sampling": False,
        "do_sample": False,
        "chunk_ms": 1000,
        "first_chunk_ms": 1035,
        "cnn_redundancy_ms": 20,
        "sample_rate": 16000,
        "batch_size": 1,
    }
    assert {item["prompt"] for item in manifest["cases"]}
    assert manifest["assets"]["skiing_video"].endswith("assets/Skiing.mp4")
    assert manifest["assets"]["system_ref_audio"].endswith("assets/system_ref_audio.wav")
    assert manifest["assets"]["duplex_omni_1"].endswith("assets/omni_duplex1.mp4")
    assert manifest["assets"]["duplex_omni_2"].endswith("assets/omni_duplex2.mp4")


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_case_manifest_exposes_api_family_and_required_prompt(case_name: str) -> None:
    runner = _load_runner()

    case = runner.case_manifest(case_name, MODEL_DIR)

    assert case["name"] == case_name
    assert case["api_family"] in {"session", "duplex"}
    assert case["prompt"]
    assert case["asset_paths"]
    assert case["generate_audio"] is ("reply" in case_name)


def test_failure_packet_contains_ld_escalation_fields(tmp_path: Path) -> None:
    runner = _load_runner()
    output_paths = (tmp_path / "result.json", tmp_path / "api_events.jsonl")

    packet = runner.make_failure_packet(
        case_name="session_audio_text",
        command=("python", "runner.py", "--case", "session_audio_text"),
        asset_paths=("/assets/Skiing.mp4",),
        api_stage="streaming_prefill",
        error=RuntimeError("model unavailable"),
        last_successful_stage="model_load",
        output_paths=output_paths,
        environment={"transformers": "4.57.1"},
        timestamp="2026-08-09T00:00:00Z",
    )
    path = runner.write_failure_packet(tmp_path, packet)

    decoded = json.loads(path.read_text(encoding="utf-8"))
    assert decoded["status"] == "failed"
    assert decoded["case"] == "session_audio_text"
    assert decoded["api_stage"] == "streaming_prefill"
    assert decoded["exception_type"] == "RuntimeError"
    assert decoded["exception_message"] == "model unavailable"
    assert decoded["last_successful_stage"] == "model_load"
    assert decoded["traceback"]
    assert decoded["output_paths"] == [str(item) for item in output_paths]
    assert decoded["environment_versions"] == {"transformers": "4.57.1"}


def test_dynamic_cache_patch_restores_official_legacy_length_method() -> None:
    runner = _load_runner()

    class Cache:
        def get_seq_length(self, layer_idx: int = 0) -> int:
            return 7 + layer_idx

    runner.patch_dynamic_cache_legacy_methods(Cache)

    assert Cache().get_usable_length(100, 2) == 9


def test_dynamic_cache_patch_exposes_legacy_tensor_views_from_layers() -> None:
    runner = _load_runner()
    cache = DynamicCache()
    keys = torch.ones((1, 2, 3, 4))
    values = torch.full((1, 2, 3, 4), 2.0)
    cache.update(keys, values, 0)

    runner.patch_dynamic_cache_legacy_methods(DynamicCache)

    assert len(cache.key_cache) == 1
    assert len(cache.value_cache) == 1
    assert torch.equal(cache.key_cache[0], keys)
    assert torch.equal(cache.value_cache[0], values)


def test_dynamic_cache_patch_reaches_encoder_decoder_nested_caches() -> None:
    runner = _load_runner()
    cache = EncoderDecoderCache(DynamicCache(), DynamicCache())

    runner.patch_dynamic_cache_legacy_methods(DynamicCache)

    assert cache.self_attention_cache.get_usable_length(1) == 0
    assert cache.cross_attention_cache.key_cache == []


def test_remote_cache_helper_uses_transformers_457_cache_contract() -> None:
    runner = _load_runner()
    module = ModuleType("test_minicpm_remote_module")
    sys.modules[module.__name__] = module

    class Model:
        __module__ = module.__name__

    module.get_kv_cache_length = lambda cache: (_ for _ in ()).throw(AttributeError("legacy-only"))

    class Cache:
        def get_seq_length(self) -> int:
            return 11

    runner.patch_remote_cache_helpers(Model())

    assert module.get_kv_cache_length(Cache()) == 11


def test_empty_audio_cache_is_normalized_before_remote_audio_embedding() -> None:
    runner = _load_runner()

    class Cache:
        def __len__(self) -> int:
            return 0

    class Model:
        audio_past_key_values = Cache()

        def get_audio_embedding_streaming(self, data):
            assert self.audio_past_key_values is None
            return data

    model = Model()
    runner.patch_empty_audio_cache(model)
    assert model.get_audio_embedding_streaming("ok") == "ok"


def test_zero_length_initialized_audio_cache_is_normalized() -> None:
    runner = _load_runner()

    class Cache:
        def __len__(self) -> int:
            return 1

        def get_seq_length(self) -> int:
            return 0

    class Model:
        audio_past_key_values = Cache()

        def get_audio_embedding_streaming(self, data):
            assert self.audio_past_key_values is None
            return data

    model = Model()
    runner.patch_empty_audio_cache(model)

    assert model.get_audio_embedding_streaming("ok") == "ok"


def test_nested_audio_cache_with_initialized_self_attention_is_unwrapped() -> None:
    runner = _load_runner()
    self_attention = DynamicCache()
    self_attention.update(torch.ones((1, 2, 3, 4)), torch.ones((1, 2, 3, 4)), 0)
    cache = EncoderDecoderCache(self_attention, DynamicCache())

    class Model:
        audio_past_key_values = cache

        def get_audio_embedding_streaming(self, data):
            assert self.audio_past_key_values is self_attention
            return data

    model = Model()
    runner.patch_empty_audio_cache(model)

    assert model.get_audio_embedding_streaming("ok") == "ok"


def test_audio_self_attention_cache_is_preserved_across_chunks() -> None:
    runner = _load_runner()
    self_attention = DynamicCache()
    self_attention.update(torch.ones((1, 2, 3, 4)), torch.ones((1, 2, 3, 4)), 0)
    cache = EncoderDecoderCache(self_attention, DynamicCache())

    class Model:
        audio_past_key_values = cache

        def get_audio_embedding_streaming(self, data):
            assert self.audio_past_key_values is self_attention
            return data

    model = Model()
    runner.patch_empty_audio_cache(model)

    assert model.get_audio_embedding_streaming("chunk-1") == "chunk-1"
    assert model.get_audio_embedding_streaming("chunk-2") == "chunk-2"


def test_flatten_audio_chunks_removes_singleton_dimensions() -> None:
    runner = _load_runner()
    chunks = [torch.zeros((1, 4)), torch.ones((1, 2))]

    flattened = runner.flatten_audio_chunks(chunks)

    assert flattened.shape == (6,)
