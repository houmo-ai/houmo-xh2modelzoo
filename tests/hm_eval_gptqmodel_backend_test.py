from __future__ import annotations

import sys
import types

import pytest
import torch

from hm_eval.core.model_registry import BackendConfig, ModelConfig


class _FakeTokenizer:
    eos_token_id = 7

    def __init__(self):
        self.decoded = None

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["add_generation_prompt"] is True
        return "rendered prompt"

    def __call__(self, prompt, return_tensors):
        assert prompt == "rendered prompt"
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([[10, 11]])}

    def decode(self, token_ids, skip_special_tokens=True):
        self.decoded = token_ids.tolist() if torch.is_tensor(token_ids) else list(token_ids)
        assert skip_special_tokens is True
        return "decoded-tail"


class _FakeModel:
    device = "cpu"

    def __init__(self):
        self.generate_kwargs = None

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return torch.tensor([[10, 11, 21, 22, 23]])


class _FakeGPTQModel:
    calls = []

    @classmethod
    def load(cls, hf_model_dir, **kwargs):
        cls.calls.append((hf_model_dir, kwargs))
        return types.SimpleNamespace(model=_FakeModel())


def test_create_gptqmodel_backend_uses_processor_fallback_and_device_map(monkeypatch):
    import transformers

    tokenizer = _FakeTokenizer()

    monkeypatch.setitem(sys.modules, "gptqmodel", types.SimpleNamespace(GPTQModel=_FakeGPTQModel))
    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("processor unavailable"))),
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: tokenizer),
    )

    from hm_eval.core.backends import create_backend

    config = ModelConfig(
        config_id="gemma4_quant",
        display_name="Gemma4 GPTQ",
        model_family="gemma4_series",
        hf_model_dir="/models/gemma4-gptq",
        transformers_version="5.5.0",
        model_class="Gemma4ForConditionalGeneration",
        processor_class="AutoProcessor",
        disable_thinking=True,
        backends={
            "gptqmodel": BackendConfig(
                type="gptqmodel",
                device_map="auto",
                extra_args={"device": "cuda:3"},
            )
        },
    )

    backend = create_backend("gptqmodel", config)
    output = backend.generate([{"role": "user", "content": "hello"}], max_tokens=128)

    assert output == "decoded-tail"
    assert tokenizer.decoded == [21, 22, 23]
    assert _FakeGPTQModel.calls[-1] == (
        "/models/gemma4-gptq",
        {"trust_remote_code": True, "device_map": "auto"},
    )
    assert backend.model.generate_kwargs["max_new_tokens"] == 128
    assert backend.model.generate_kwargs["do_sample"] is False
    assert backend.model.generate_kwargs["pad_token_id"] == tokenizer.eos_token_id


def test_gptqmodel_backend_reports_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "gptqmodel", None)

    from hm_eval.core.backends import GPTQModelBackend

    with pytest.raises(ImportError, match="requires the gptqmodel package"):
        GPTQModelBackend(hf_model_dir="/models/missing")

