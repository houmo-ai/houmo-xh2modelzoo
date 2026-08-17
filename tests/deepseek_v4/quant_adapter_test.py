from __future__ import annotations

from pathlib import Path

import pytest

from xhmodel_merak.xh_llm.models.deepseek_v4.quant_adapter import (
    DeepSeekV4FlashQuantSpec,
    quantization_plan,
    quantize_deepseek_v4_flash,
)


def _spec(tmp_path: Path, **overrides) -> DeepSeekV4FlashQuantSpec:
    values = {
        "model_dir": str(tmp_path / "native"),
        "prepared_model_dir": str(tmp_path / "bf16"),
        "output_dir": str(tmp_path / "quantized"),
        "dry_run": True,
    }
    values.update(overrides)
    return DeepSeekV4FlashQuantSpec(**values)


@pytest.mark.parametrize(
    ("method", "normalized"),
    [("gptq", "gptq"), ("autoround", "auto_round"), ("auto-round", "auto_round")],
)
def test_quant_adapter_uses_one_gptqmodel_recipe_for_both_methods(
    tmp_path: Path,
    method: str,
    normalized: str,
) -> None:
    result = quantize_deepseek_v4_flash(_spec(tmp_path, method=method))

    assert result.method == normalized
    assert result.offload_to_disk is False
    assert result.precision["routed_non_shared_experts"] == "W4"
    assert result.precision["shared_expert_and_other_quantized_linears"] == "W8"
    assert not (tmp_path / "quantized").exists()


def test_quant_adapter_plan_records_static_deployment_profile(tmp_path: Path) -> None:
    plan = quantization_plan(_spec(tmp_path, method="autoround", num_layers=6))

    assert plan["backend"] == "gptqmodel.recipes.deepseek_v4_flash"
    assert plan["method"] == "auto_round"
    assert plan["num_layers"] == 6
    assert plan["offload_to_disk"] is False
    assert plan["auto_forward_data_parallel"] is False
    assert plan["precision"] == {
        "routed_non_shared_experts": "W4",
        "shared_expert_and_other_quantized_linears": "W8",
        "group_size": 64,
        "symmetric": True,
        "control_paths": "FP16",
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"method": "awq"}, "method must"),
        ({"base_bits": 4}, "requires base W8"),
        ({"expert_bits": 8}, "routed experts W4"),
        ({"group_size": 128}, "group_size=64"),
    ],
)
def test_quant_adapter_rejects_non_deployment_profiles(
    tmp_path: Path,
    overrides: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        quantization_plan(_spec(tmp_path, **overrides))
