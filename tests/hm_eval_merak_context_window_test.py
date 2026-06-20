from __future__ import annotations

import torch
import pytest

from hm_eval.core.backends import HMONNXBackend, _resolve_user_config_path


def _backend_with_runtime_meta(meta: dict):
    backend = object.__new__(HMONNXBackend)
    backend._merak_runtime_meta = meta
    backend._merak_runtime_meta_path = "unit-test-meta.json"
    backend._merak_prefill_chunk_length = backend._infer_merak_prefill_chunk_length()
    backend.max_context_tokens = backend._infer_merak_max_context_tokens()
    return backend


def test_merak_context_uses_model_context_not_slice_kv_cache_length():
    backend = _backend_with_runtime_meta(
        {
            "model_config": {
                "model_type": "Gemma4ForConditionalGeneration",
                "context_max_length": 2048,
                "prefill_chunk_length": 256,
            },
            "kv_cache": {"kv_cache_shape": [1, 8, 1280, 256]},
            "kv_cache_shapes_per_layer": [[1, 8, 1280, 256], [1, 2, 2048, 512]],
        }
    )

    assert backend.max_context_tokens == 2048
    assert backend._merak_max_prefill_tokens() == 2047

    input_ids = torch.arange(1500).reshape(1, 1500)
    assert backend._truncate_merak_text_input_ids(input_ids).shape[-1] == 1500


def test_merak_context_falls_back_to_cache_shapes_without_model_context():
    backend = _backend_with_runtime_meta(
        {
            "model_config": {"model_type": "Gemma4ForConditionalGeneration"},
            "kv_cache": {"kv_cache_shape": [1, 8, 1280, 256]},
            "kv_cache_shapes_per_layer": [[1, 8, 1280, 256], [1, 2, 2048, 512]],
        }
    )

    assert backend.max_context_tokens == 2048


def test_hmonnx_config_path_expands_environment(monkeypatch, tmp_path):
    meta = tmp_path / "golden_meta_info.json"
    meta.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HM_EVAL_TEST_META", str(meta))

    assert _resolve_user_config_path("${HM_EVAL_TEST_META}", field="export_meta_info") == meta.resolve()


def test_hmonnx_config_path_reports_unresolved_environment(monkeypatch):
    monkeypatch.delenv("HM_EVAL_MISSING_META", raising=False)

    with pytest.raises(FileNotFoundError, match="unresolved environment variable"):
        _resolve_user_config_path("${HM_EVAL_MISSING_META}", field="export_meta_info")
