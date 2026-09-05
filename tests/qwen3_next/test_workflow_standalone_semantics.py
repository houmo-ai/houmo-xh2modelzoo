from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs_merak/workflows/xh2a/llm_models"
Q35_35B_ROOT = CONFIG_ROOT / "qwen3_5_moe/35b_a3b"
Q35_122B_ROOT = CONFIG_ROOT / "qwen3_5_moe/122b_a10b"
NEXT_ROOT = CONFIG_ROOT / "qwen3_next/80b_a3b"


AUTOROUND_QUANT = {
    "algorithm": "gptqmodel",
    "method": "autoround",
    "output_format": "gptqmodel_hf",
    "artifact_format": "gptqmodel_hf",
    "bits": 4,
    "group_size": 64,
    "rotation": False,
    "calibration": {
        "jsonl": "xh2modelzoo://data/calib_data/NeelNanda-pile-10k.jsonl",
        "nsamples": 128,
        "seqlen": 2048,
    },
    "runtime": {
        "batch_size": 8,
        "gradient_accumulate_steps": 1,
        "trust_remote_code": True,
        "device_map": "0",
        "low_gpu_mem_usage": True,
    },
    "seed": 42,
    "quant_nontext_module": False,
    "sym": True,
    "iters": 200,
    "format": "auto_round:gptqmodel",
    "moe": {"attn_bits": 8, "shared_expert_bits": 8},
}

GPTQ_QUANT = {
    "algorithm": "gptqmodel",
    "method": "gptq",
    "output_format": "gptqmodel_hf",
    "artifact_format": "gptqmodel_hf",
    "preset": "full_vlm",
    "bits": 4,
    "group_size": 64,
    "rotation": False,
    "hessian_mse": True,
    "calibration": {
        "jsonl": "gptqmodel/quantization/calibration/moe_ebss/gen_data/Qwen3-Next-80B-A3B-Instruct.jsonl",
        "text_key": "text",
        "nsamples": 512,
        "seqlen": 1024,
    },
    "runtime": {
        "batch_size": 1,
        "trust_remote_code": True,
        "device_map": "auto",
        "offload_to_disk": False,
    },
    "validation": {
        "check_quant_vision_demo": True,
        "check_rotation_ppl": False,
    },
    "seed": 42,
    "quant_nontext_module": False,
    "moe": {
        "attn_bits": 8,
        "shared_expert_bits": 8,
        "expert_bits": 4,
        "expert_down_bits": 5,
        "routing": "bypass",
    },
}

BASE_EXPORT_MODEL = {
    "chip_arch": "XH2a",
    "model_type": "Qwen3_5MoeForConditionalGeneration",
    "hf_model": None,
    "context_max_length": 2048,
    "prefill_chunk_length": 256,
    "max_pe_length": 262144,
    "support_long_context_over_fp16_limit": True,
    "use_cache": True,
    "num_logits_to_keep": 1,
    "linear_attention_mode": "auto",
    "linear_chunk_size": 64,
    "flash_attention": {
        "enable": False, "q_bits": 8, "k_bits": 8, "v_bits": 8, "s_bits": 8, "p_bits": 8,
    },
    "split_conv_cache": True,
    "normalize_force_fp32": False,
    "use_manual_depthwise_conv1d": False,
    "fuse_gdr_ops": True,
    "fuse_gdr_block_recurrent_ops": True,
    "quant_scheme": {
        "quant_type": "w8a8h1_sefp",
        "nodes": {"lm_head": {"quant_type": "w8a8h1_sefp"}},
        "ops": {},
    },
    "visual_config": {
        "visual_input_mode": "patches",
        "image_token_gears": [96, 196, 384, 704, 1536],
        "image_token_capacity": 1536,
        "spatial_merge_size": 2,
        "quant_scheme": {"quant_type": "w8a8h1_sefp", "ops": {}},
    },
    "only_first_block": False,
}


def _q35_release_expected(
    *,
    model_name: str,
    profile: str,
    autoround_dataset: bool = False,
    expert_down_bits: int = 5,
) -> dict:
    quant = copy.deepcopy(AUTOROUND_QUANT if profile in {"autoround", "fa_all16"} else GPTQ_QUANT)
    model = copy.deepcopy(BASE_EXPORT_MODEL)
    model["model_name"] = model_name
    if autoround_dataset:
        quant["calibration"] = {
            "dataset": "NeelNanda/pile-10k",
            "nsamples": 128,
            "seqlen": 2048,
        }
    if profile in {"gptq", "mtp_gptq"}:
        quant["moe"]["expert_down_bits"] = expert_down_bits
    if profile == "fa_all16":
        model["flash_attention"] = {
            "enable": True,
            "q_bits": 16,
            "k_bits": 16,
            "v_bits": 16,
            "s_bits": 16,
            "p_bits": 16,
        }
        matmul = {
            "act_scheme": {"bits": 16, "fp_mode": "sefp"},
            "act_schema_2": {"bits": 16, "fp_mode": "sefp"},
        }
        model["quant_scheme"] = {
            "quant_type": "w8a16h1_sefp",
            "nodes": {"lm_head": {"quant_type": "w8a16h1_sefp"}},
            "ops": {"MatMul": matmul},
        }
        model["visual_config"]["quant_scheme"] = {
            "quant_type": "w8a16h1_sefp",
            "ops": {"MatMul": copy.deepcopy(matmul)},
        }
    if profile == "mtp_gptq":
        model.update(
            {
                "spec_decode_mode": "mtp",
                "num_draft_tokens": 4,
                "spec_draft_head_weight_bits": 4,
                "output_post_norm_hidden": True,
                "mtp_config": {
                    "hidden_size": 2048,
                    "num_key_value_heads": 2,
                    "head_dim": 256,
                    "batch_size": 1,
                    "input_sequence_length": 1,
                    "max_pe_length": 262144,
                    "use_cache": True,
                },
            }
        )
    return {"quant": quant, "export": {"model": model}}


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            Q35_35B_ROOT / "qwen3_6_35b_a3b_full.yaml",
            _q35_release_expected(model_name="qwen3_6_35b_a3b", profile="autoround"),
        ),
        (
            Q35_35B_ROOT / "qwen3_6_35b_a3b_full_fa_all16.yaml",
            _q35_release_expected(model_name="qwen3_6_35b_a3b", profile="fa_all16"),
        ),
        (
            Q35_35B_ROOT / "qwen3_6_35b_a3b_full_gptq.yaml",
            _q35_release_expected(model_name="qwen3_6_35b_a3b", profile="gptq"),
        ),
        (
            Q35_35B_ROOT / "qwen3_6_35b_a3b_full_mtp_gptq.yaml",
            _q35_release_expected(model_name="qwen3_6_35b_a3b", profile="mtp_gptq"),
        ),
        (
            Q35_122B_ROOT / "qwen3_5_122b_a10b_full.yaml",
            _q35_release_expected(
                model_name="qwen3_5_122b_a10b",
                profile="autoround",
                autoround_dataset=True,
            ),
        ),
        (
            Q35_122B_ROOT / "qwen3_5_122b_a10b_full_gptq.yaml",
            _q35_release_expected(
                model_name="qwen3_5_122b_a10b",
                profile="gptq",
                expert_down_bits=4,
            ),
        ),
    ],
)
def test_standalone_qwen35_leaf_matches_release_contract(path: Path, expected: dict):
    assert WorkflowConfig.from_file(str(path)).data == expected


def test_qwen35_and_qwen3_next_workflows_are_standalone_yaml():
    workflow_paths = sorted(
        [
            *Q35_35B_ROOT.glob("*.yaml"),
            *Q35_122B_ROOT.glob("*.yaml"),
            *NEXT_ROOT.glob("*.yaml"),
        ]
    )

    assert workflow_paths
    for path in workflow_paths:
        with path.open(encoding="utf-8") as stream:
            direct = yaml.safe_load(stream)
        assert "_base_" not in direct, path
        assert "extends" not in direct, path
        assert set(direct) == {"quant", "export"}, path
        assert WorkflowConfig.from_file(str(path)).data == direct


def test_workflow_tree_has_no_yaml_inheritance_directives():
    for path in CONFIG_ROOT.rglob("*.yaml"):
        with path.open(encoding="utf-8") as stream:
            direct = yaml.safe_load(stream)
        assert "_base_" not in direct, path
        assert "extends" not in direct, path


def test_next_autoround_uses_dedicated_recipe_without_fake_qwen35_topology(monkeypatch, tmp_path):
    from xhmodel_merak.xh_llm.models.qwen3_next.workflow import Qwen3NextWorkflow

    captured = {}
    recipe = types.ModuleType("gptqmodel.recipes.qwen3_next")

    def fake_quantize_qwen3_next(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(output_dir=tmp_path / "quantized")

    recipe.quantize_qwen3_next = fake_quantize_qwen3_next
    monkeypatch.setitem(sys.modules, "gptqmodel.recipes.qwen3_next", recipe)

    workflow = Qwen3NextWorkflow.__new__(Qwen3NextWorkflow)
    workflow.model_dir = "/weights/Qwen3-Next-80B-A3B-Instruct"
    config = WorkflowConfig.from_file(str(NEXT_ROOT / "qwen3_next_80b_a3b_full.yaml"))
    result = workflow._quant_autoround_api(
        str(tmp_path),
        "cuda:0",
        config.quant,
        config.export["model"],
    )

    assert result.quanted_model_dir == str(tmp_path / "quantized")
    assert captured["method"] == "autoround"
    assert captured["model_dir"] == workflow.model_dir
    assert captured["dry_run"] is False
    assert "model_type" not in captured
    assert "topology" not in captured


def test_next_autoround_resolved_yaml_contains_only_recipe_consumed_keys():
    from xhmodel_merak.xh_llm.models.qwen3_next.workflow import Qwen3NextWorkflow

    config = WorkflowConfig.from_file(str(NEXT_ROOT / "qwen3_next_80b_a3b_full.yaml"))
    assert config.quant is not None
    Qwen3NextWorkflow._validate_qwen3_next_quant_config(config.quant)
    assert config.quant["method"] == "autoround"
    assert config.quant["runtime"]["trust_remote_code"] is True
    assert config.quant["calibration"]["dataset"] == "NeelNanda/pile-10k"
    resolved_keys = set(config.quant)
    for section in config.quant.values():
        if isinstance(section, dict):
            resolved_keys.update(section)
    assert not (
        {
            "quant_nontext_module",
            "preset",
            "hessian_mse",
            "validation",
            "check_quant_vision_demo",
            "check_rotation_ppl",
            "expert_down_bits",
            "routing",
        }
        & resolved_keys
    )


def test_next_autoround_recipe_dry_run_keeps_native_identity_and_mtp(tmp_path):
    python = Path("/data01/home/yujy/miniconda3/envs/xhquant_55/bin/python")
    gptqmodel_root = Path("/data01/home/yujy/work/gptqmodel")
    if not python.is_file() or not gptqmodel_root.is_dir():
        pytest.skip("Qwen3-Next cross-repo dry-run requires xhquant_55 and the GPTQModel checkout")

    script = """
import copy
import json
import sys

from xhmodel_merak.xh_llm.models.qwen3_next.workflow import Qwen3NextWorkflow
from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig

config = WorkflowConfig.from_file(sys.argv[1])
quant = copy.deepcopy(config.quant)
quant["runtime"]["dry_run"] = True
workflow = Qwen3NextWorkflow.__new__(Qwen3NextWorkflow)
workflow.model_dir = "weights/Qwen3-Next-80B-A3B-Instruct"
result = workflow._quant_autoround_api(sys.argv[2], "cuda:0", quant, config.export["model"])
print("__QWEN3_NEXT_RESULT__=" + json.dumps({
    "method": result.meta.method,
    "model_family": result.meta.model_family,
    "has_mtp": result.meta.has_mtp,
    "provenance": result.meta.provenance,
}))
"""
    env = os.environ.copy()
    pythonpath = [str(REPO_ROOT), str(gptqmodel_root)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    completed = subprocess.run(
        [
            str(python),
            "-c",
            script,
            str(NEXT_ROOT / "qwen3_next_80b_a3b_full.yaml"),
            str(tmp_path / "autoround"),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    result_line = next(line for line in completed.stdout.splitlines() if line.startswith("__QWEN3_NEXT_RESULT__="))
    result = json.loads(result_line.removeprefix("__QWEN3_NEXT_RESULT__="))

    assert result["method"] == "autoround"
    assert result["model_family"] == "qwen3_next"
    assert result["has_mtp"] is True
    provenance = result["provenance"]
    assert provenance["backend"] == "auto_round.AutoRound"
    assert provenance["expected_model_type"] == "qwen3_next"
    assert "model_type" not in provenance["invocation"]
    assert provenance["invocation"]["dataset"] == "NeelNanda/pile-10k"
