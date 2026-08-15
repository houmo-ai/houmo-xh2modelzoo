from __future__ import annotations

from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs_merak/workflows/xh2a/llm_models"
DEFAULT_CONTEXT = 2048
CONTEXT_256K = 262144
FA8 = {"q_bits": 8, "k_bits": 8, "v_bits": 8, "s_bits": 8, "p_bits": 8}


QWEN_CONFIGS = (
    "qwen3_5/9b/qwen3_5_9b_full.yaml",
    "qwen3_5/9b/qwen3_5_9b_full_gptq.yaml",
    "qwen3_5/9b/qwen3_5_9b_full_mtp.yaml",
    "qwen3_5/9b/qwen3_5_9b_full_mtp_gptq.yaml",
    "qwen3_5/9b/qwen3_5_9b_full_dflash.yaml",
    "qwen3_5/9b/qwen3_5_9b_full_dflash_gptq.yaml",
    "qwen3_5/27b/qwen3_6_27b_full.yaml",
    "qwen3_5/27b/qwen3_6_27b_full_gptq.yaml",
    "qwen3_5/27b/qwen3_6_27b_full_mtp.yaml",
    "qwen3_5/27b/qwen3_6_27b_full_mtp_gptq.yaml",
    "qwen3_5/27b/qwen3_6_27b_full_dflash.yaml",
    "qwen3_5/27b/qwen3_6_27b_full_dflash_gptq.yaml",
    "qwen3_5/27b/qwen3_8_27b_full.yaml",
    "qwen3_5/27b/qwen3_8_27b_full_mtp.yaml",
    "qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full.yaml",
    "qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_gptq.yaml",
    "qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_mtp.yaml",
    "qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_mtp_gptq.yaml",
    "qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_dflash.yaml",
    "qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_dflash_gptq.yaml",
    "qwen3_5_moe/122b_a10b/qwen3_5_122b_a10b_full.yaml",
    "qwen3_5_moe/122b_a10b/qwen3_5_122b_a10b_full_gptq.yaml",
    "qwen3_5_moe/122b_a10b/qwen3_5_122b_a10b_full_mtp.yaml",
    "qwen3_5_moe/122b_a10b/qwen3_5_122b_a10b_full_dflash.yaml",
    "qwen3_5_moe/122b_a10b/qwen3_5_122b_a10b_full_dflash_gptq.yaml",
    "qwen3_next/80b_a3b/qwen3_next_80b_a3b_full_gptq.yaml",
    "qwen3_next/80b_a3b/qwen3_next_80b_a3b_full_mtp_gptq.yaml",
)

GEMMA_PAGE_CONFIGS = (
    "gemma4_series/12b_unified/gemma4_12b_unified_full_mtp_page_attention.yaml",
    "gemma4_series/e2b/gemma4_e2b_full_mtp_page_attention.yaml",
    "gemma4_series/e4b/gemma4_e4b_full_mtp_page_attention.yaml",
    "gemma4_series/26b_a4b/gemma4_26b_a4b_full_mtp_page_attention.yaml",
    "gemma4_series/31b/gemma4_31b_full_mtp_page_attention.yaml",
)

GEMMA_PAGE_BASE_CONFIGS = {
    "gemma4_series/12b_unified/gemma4_12b_unified_full_mtp_page_attention.yaml":
        "gemma4_series/12b_unified/gemma4_12b_unified_full_mtp.yaml",
    "gemma4_series/e2b/gemma4_e2b_full_mtp_page_attention.yaml":
        "gemma4_series/e2b/gemma4_e2b_full_mtp.yaml",
    "gemma4_series/e4b/gemma4_e4b_full_mtp_page_attention.yaml":
        "gemma4_series/e4b/gemma4_e4b_full_mtp.yaml",
    "gemma4_series/26b_a4b/gemma4_26b_a4b_full_mtp_page_attention.yaml":
        "gemma4_series/26b_a4b/gemma4_26b_a4b_full_mtp.yaml",
    "gemma4_series/31b/gemma4_31b_full_mtp_page_attention.yaml":
        "gemma4_series/31b/gemma4_31b_full_mtp.yaml",
}


def _workflow_config(relative_path: str) -> dict:
    return yaml.safe_load(
        (CONFIG_ROOT / relative_path).read_text(encoding="utf-8")
    )


def _model_config(relative_path: str) -> dict:
    return _workflow_config(relative_path)["export"]["model"]


@pytest.mark.parametrize("relative_path", QWEN_CONFIGS)
def test_qwen_yaml_keeps_small_default_and_derives_draft_capacity(
    relative_path: str,
) -> None:
    model = _model_config(relative_path)

    assert model["context_max_length"] == DEFAULT_CONTEXT
    assert model["fuse_gdr_block_recurrent_ops"] is False
    assert model["flash_attention"]["enable"] is False
    assert {name: model["flash_attention"][name] for name in FA8} == FA8

    mode = model.get("spec_decode_mode")
    if mode == "mtp":
        assert "context_max_length" not in model["mtp_config"]
    elif mode == "dflash":
        assert "max_sequence_length" not in model["dflash_config"]
        assert model["num_draft_tokens"] == 9


@pytest.mark.parametrize("relative_path", GEMMA_PAGE_CONFIGS)
def test_gemma_page_yaml_selects_flash_without_hardcoding_context_or_abi(
    relative_path: str,
) -> None:
    model = _model_config(relative_path)

    assert model["context_max_length"] == DEFAULT_CONTEXT
    assert model["flash_attention"]["enable"] is True
    assert {name: model["flash_attention"][name] for name in FA8} == FA8
    assert "attention_contract_version" not in model
    assert "context_max_length" not in model["mtp_config"]
    assert "readonly_attention_lowering" not in model["mtp_config"]


@pytest.mark.parametrize(
    ("page_path", "base_path"),
    GEMMA_PAGE_BASE_CONFIGS.items(),
)
def test_gemma_page_yaml_preserves_family_quantization_recipe(
    page_path: str,
    base_path: str,
) -> None:
    """PageAttention is an export variant, not a new calibration recipe."""

    page = _workflow_config(page_path)
    base = _workflow_config(base_path)

    assert page["quant"] == base["quant"]
    assert (
        page["export"]["model"]["model_name"]
        == base["export"]["model"]["model_name"]
    )


def test_gemma_nonflash_yaml_does_not_select_flash_contract() -> None:
    model = _model_config(
        "gemma4_series/12b_unified/gemma4_12b_unified_full.yaml"
    )

    assert model["context_max_length"] == DEFAULT_CONTEXT
    assert "flash_attention" not in model
    assert "attention_contract_version" not in model


def test_context_cli_override_has_one_source_of_truth() -> None:
    from examples_merak.llm.qwen3_5.qwen3_5_workflow import (
        _add_context_length_overrides,
    )

    overrides: dict[str, object] = {}
    _add_context_length_overrides(overrides, CONTEXT_256K)

    assert overrides == {
        "export.model.context_max_length": CONTEXT_256K,
    }


def test_qwen3_next_mtp_checkpoint_is_derived_from_target() -> None:
    model = _model_config(
        "qwen3_next/80b_a3b/qwen3_next_80b_a3b_full_mtp_gptq.yaml"
    )

    assert "hf_model" not in model["mtp_config"]
    assert "context_max_length" not in model["mtp_config"]
