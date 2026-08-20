from __future__ import annotations

import json
from pathlib import Path

import torch


def test_load_llm_calibration_prompts_uses_requested_jsonl_samples(tmp_path) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.export_llm import (
        _load_llm_calibration_prompts,
    )

    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text(
        "".join(json.dumps({"text": f"sample {index}"}) + "\n" for index in range(6)),
        encoding="utf-8",
    )

    prompts, info = _load_llm_calibration_prompts({"calibration_jsonl": str(calibration), "calibration_samples": 4})

    assert prompts == ["sample 0", "sample 1", "sample 2", "sample 3"]
    assert info["sample_count"] == 4
    assert info["resolved_path"] == str(calibration.resolve())


def test_repo_calibration_uri_is_independent_of_cwd(tmp_path, monkeypatch) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.resource_path import resolve_repo_resource

    monkeypatch.chdir(tmp_path)
    resolved = resolve_repo_resource(
        "xh2modelzoo://data/calib_data/minicpm_o_4_5_qwen_vl_style_mix80.jsonl",
        description="test calibration",
    )

    repo_root = Path(__file__).resolve().parents[2]
    assert resolved == (repo_root / "data/calib_data/minicpm_o_4_5_qwen_vl_style_mix80.jsonl").resolve()


def test_llm_calibration_embeds_balances_prompts() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.export_llm import _llm_calibration_embeds

    class Tokenizer:
        def __call__(self, prompt, **kwargs):
            base = 10 if prompt == "first" else 20
            return {"input_ids": torch.arange(base, base + 8).reshape(1, 8)}

    embedding = torch.nn.Embedding(64, 4)
    result = _llm_calibration_embeds(
        Tokenizer(), embedding, prefill_length=8, hidden_size=4, device="cpu", prompts=["first", "second"]
    )

    expected_ids = torch.tensor([[10, 11, 12, 13, 20, 21, 22, 23]])
    assert torch.equal(result, embedding(expected_ids).to(torch.float16))


def test_load_gptq_llm_weights_uses_shared_in_memory_gptq_path(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    from xhmodel_merak.xh_llm.base_model import XHBaseModel
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.export_llm import _load_gptq_llm_weights

    gptq_dir = tmp_path / "gptq_llm"
    gptq_dir.mkdir()
    source_state = {"model.layers.0.weight": torch.full((2, 2), 3.0)}
    loaded: dict[str, object] = {}

    class HostLLM:
        def state_dict(self):
            return {"model.layers.0.weight": torch.zeros(2, 2)}

        def load_state_dict(self, state, strict):
            loaded["state"] = state
            loaded["strict"] = strict

        def to(self, dtype):
            loaded["dtype"] = dtype
            return self

    def fake_load(cls, model_dir, **kwargs):
        loaded["model_dir"] = model_dir
        loaded["load_kwargs"] = kwargs
        return SimpleNamespace(state_dict=lambda: source_state)

    def fake_dequantize(cls, model):
        loaded["dequantized"] = model
        return model

    monkeypatch.setattr(XHBaseModel, "_load_gptqmodel", classmethod(fake_load))
    monkeypatch.setattr(XHBaseModel, "_dequantize_gptqmodel_hf_model", classmethod(fake_dequantize))

    _load_gptq_llm_weights(SimpleNamespace(llm=HostLLM()), gptq_dir)

    assert loaded["model_dir"] == str(gptq_dir)
    assert loaded["load_kwargs"] == {"device_map": "cpu", "trust_remote_code": True}
    assert loaded["dequantized"].state_dict() == source_state
    assert torch.equal(loaded["state"]["model.layers.0.weight"], source_state["model.layers.0.weight"])
    assert loaded["strict"] is True
    assert loaded["dtype"] == torch.float16
