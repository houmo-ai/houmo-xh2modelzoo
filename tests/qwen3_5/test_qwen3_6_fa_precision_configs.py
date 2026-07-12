"""Configuration-matrix regressions for Qwen3.6 35B FlashAttention."""

from __future__ import annotations

import hashlib
from pathlib import Path

from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig
from xhquant.api import Config


CONFIG_DIR = (
    Path(__file__).resolve().parents[2]
    / "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b"
)
BASE_CONFIG = CONFIG_DIR / "qwen3_6_35b_a3b_full.yaml"
BASE_SHA256 = "251b75389b65bf60091d386463773cc82394d1a5a51bc9d1fb65ffe8bac7d814"
def test_qwen3_6_35b_a3b_full_config_is_byte_for_byte_unchanged():
    assert hashlib.sha256(BASE_CONFIG.read_bytes()).hexdigest() == BASE_SHA256


def test_qwen3_6_all16_config_loads_through_workflow_entrypoint():
    workflow_cfg = WorkflowConfig.from_file(
        str(CONFIG_DIR / "qwen3_6_35b_a3b_full_fa_all16.yaml")
    )
    model_cfg = Config(workflow_cfg.build_export_dict()).model

    assert model_cfg.model_name == "qwen3_6_35b_a3b"
    assert model_cfg.flash_attention.enable is True
    fa = model_cfg.flash_attention
    assert (fa.q_bits, fa.k_bits, fa.v_bits, fa.s_bits, fa.p_bits) == (16,) * 5


def test_qwen3_6_all16_config_keeps_every_matmul_activation_operand_sefp16():
    workflow_cfg = WorkflowConfig.from_file(str(CONFIG_DIR / "qwen3_6_35b_a3b_full_fa_all16.yaml"))
    model_cfg = Config(workflow_cfg.build_export_dict()).model

    assert model_cfg.quant_scheme.quant_type == "w8a16h1_sefp"
    assert model_cfg.quant_scheme.nodes.lm_head.quant_type == "w8a16h1_sefp"
    for quant_scheme in (model_cfg.quant_scheme, model_cfg.visual_config.quant_scheme):
        matmul = quant_scheme.ops.MatMul
        assert (matmul.act_scheme.bits, matmul.act_scheme.fp_mode) == (16, "sefp")
        assert (matmul.act_schema_2.bits, matmul.act_schema_2.fp_mode) == (16, "sefp")
    assert model_cfg.only_first_block is False
