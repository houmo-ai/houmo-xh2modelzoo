from __future__ import annotations

import json
from pathlib import Path

import pytest


def _standard_meta() -> dict[str, object]:
    return {
        "create_time": "2026-09-01 00:00:00",
        "model_config": {
            "model_name": "minicpm_o_4_5_test",
            "model_type": "MiniCPMO45LLMModel",
            "prefill_chunk_length": 8,
            "context_max_length": 32,
            "num_logits_to_keep": 1,
            "use_cache": True,
        },
        "hf_config": "hf_config",
        "quant_embedding": "quant_embedding.pt",
        "kv_cache": {
            "num_layers": 2,
            "kv_cache_shape": [1, 2, 32, 4],
            "cache_axis": 2,
            "batch_size": 1,
            "cache_dtype": "float16",
            "use_cache": True,
        },
        "prefill_hmonnx": "Prefill/prefill.hmonnx",
        "decode_hmonnx": "Decode/decode.hmonnx",
        "pad_token_id": 7,
        "meta": {"class_name": "LLMModelMeta"},
    }


def test_llm_standard_child_metadata_uses_shared_parser(tmp_path: Path) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_llm import _resolve_llm_meta

    child_dir = tmp_path / "LLM"
    child_dir.mkdir()
    meta_path = child_dir / "golden_meta_info.json"
    meta_path.write_text(json.dumps(_standard_meta()), encoding="utf-8")

    resolved = _resolve_llm_meta(tmp_path, {"metadata": "LLM/golden_meta_info.json"})

    assert resolved.model_config.prefill_chunk_length == 8
    assert resolved.kv_cache.num_layers == 2
    assert resolved.prefill_hmonnx == str(child_dir / "Prefill/prefill.hmonnx")
    assert resolved.decode_hmonnx == str(child_dir / "Decode/decode.hmonnx")
    assert resolved.quant_embedding == str(child_dir / "quant_embedding.pt")
    assert resolved.pad_token_id == 7


def test_llm_incomplete_child_metadata_is_rejected(tmp_path: Path) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_llm import _resolve_llm_meta

    child_dir = tmp_path / "LLM"
    child_dir.mkdir()
    (child_dir / "golden_meta_info.json").write_text(
        json.dumps(
            {
                "hf_config": "hf_config",
                "quant_embedding": "quant_embedding.pt",
                "prefill_hmonnx": "Prefill/prefill.hmonnx",
                "decode_hmonnx": "Decode/decode.hmonnx",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="complete LLMModelMeta artifact"):
        _resolve_llm_meta(tmp_path, {"metadata": "LLM/golden_meta_info.json"})


def test_tts_standard_child_metadata_uses_shared_parser(tmp_path: Path) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_tts import _resolve_tts_meta

    child_dir = tmp_path / "TTS"
    child_dir.mkdir()
    payload = _standard_meta()
    model_config = payload["model_config"]
    assert isinstance(model_config, dict)
    model_config["model_type"] = "MiniCPMO45TTSModel"
    payload["quant_embedding"] = ""
    (child_dir / "golden_meta_info.json").write_text(json.dumps(payload), encoding="utf-8")

    resolved = _resolve_tts_meta(tmp_path, {"metadata": "TTS/golden_meta_info.json"})

    assert resolved.model_config.model_type == "MiniCPMO45TTSModel"
    assert resolved.kv_cache.num_layers == 2
    assert resolved.prefill_hmonnx == str(child_dir / "Prefill/prefill.hmonnx")
    assert resolved.decode_hmonnx == str(child_dir / "Decode/decode.hmonnx")


def test_tts_missing_child_metadata_is_rejected(tmp_path: Path) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_tts import _resolve_tts_meta

    with pytest.raises(ValueError, match="must provide standard child metadata"):
        _resolve_tts_meta(tmp_path, {})
