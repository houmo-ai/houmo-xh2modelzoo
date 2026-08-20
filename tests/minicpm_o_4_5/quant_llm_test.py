from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest


def _install_fake_minicpm_recipe(monkeypatch: pytest.MonkeyPatch, callback):
    gptqmodel = types.ModuleType("gptqmodel")
    gptqmodel.__path__ = []
    recipes = types.ModuleType("gptqmodel.recipes")
    recipes.__path__ = []
    minicpm_o_4_5 = types.ModuleType("gptqmodel.recipes.minicpm_o_4_5")
    minicpm_o_4_5.quantize_minicpm_o_4_5 = callback
    monkeypatch.setitem(sys.modules, "gptqmodel", gptqmodel)
    monkeypatch.setitem(sys.modules, "gptqmodel.recipes", recipes)
    monkeypatch.setitem(
        sys.modules,
        "gptqmodel.recipes.minicpm_o_4_5",
        minicpm_o_4_5,
    )


def test_quant_adapter_resolves_repository_uri_and_uses_standard_api(tmp_path: Path, monkeypatch):
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import quant_llm

    calibration = tmp_path / "data" / "calibration.jsonl"
    calibration.parent.mkdir()
    calibration.write_text(
        "".join(json.dumps({"text": f"sample {index}"}) + "\n" for index in range(6)),
        encoding="utf-8",
    )
    monkeypatch.setenv("XH2MODELZOO_ROOT", str(tmp_path))
    captured: dict[str, object] = {}

    def fake_recipe(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(output_dir=kwargs["output_dir"])

    _install_fake_minicpm_recipe(monkeypatch, fake_recipe)
    output = quant_llm.quantize_minicpm_llm_gptq(
        model_dir="/models/minicpm-o-4_5",
        output_dir=str(tmp_path / "quant"),
        quant_cfg={
            "algorithm": "gptqmodel",
            "bits": 4,
            "group_size": 64,
            "nsamples": 4,
            "seqlen": 768,
            "batch_size": 2,
            "sym": False,
            "calibration_jsonl": "xh2modelzoo://data/calibration.jsonl",
            "offload_to_disk": False,
        },
        device="cuda:0",
    )

    assert output == str(tmp_path / "quant" / "gptq_llm")
    assert captured["model_dir"] == "/models/minicpm-o-4_5"
    assert captured["output_dir"] == str(tmp_path / "quant" / "gptq_llm")
    assert captured["bits"] == 4
    assert captured["group_size"] == 64
    assert captured["sym"] is False
    assert captured["nsamples"] == 4
    assert captured["seqlen"] == 768
    assert captured["batch_size"] == 2
    assert captured["damp_percent"] == 0.01
    assert captured["device"] == "cuda:0"
    assert captured["offload_to_disk"] is False
    assert captured["offload_to_disk_path"] is None
    assert captured["calibration_data"] == [
        "sample 0",
        "sample 1",
        "sample 2",
        "sample 3",
    ]


def test_quant_adapter_rejects_insufficient_calibration_rows(tmp_path: Path, monkeypatch):
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import quant_llm

    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text('{"text": "only one"}\n', encoding="utf-8")
    _install_fake_minicpm_recipe(monkeypatch, lambda **kwargs: None)

    with pytest.raises(ValueError, match="need 2"):
        quant_llm.quantize_minicpm_llm_gptq(
            model_dir="/models/minicpm-o-4_5",
            output_dir=str(tmp_path / "quant"),
            quant_cfg={
                "calibration_jsonl": str(calibration),
                "nsamples": 2,
                "group_size": 64,
            },
            device="cpu",
        )
