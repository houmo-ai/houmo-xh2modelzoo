from xhmodel_merak.xh_llm.hmonnx import base_llm_hmonnx_model


def _tokenizer_owner():
    owner = object.__new__(base_llm_hmonnx_model.BaseLLMHMONNXModel)
    owner.hf_model_dir = "/tmp/exported-hf-config"
    return owner


def test_exported_tokenizer_trusts_packaged_remote_code_by_default(monkeypatch):
    calls = []
    sentinel = object()
    monkeypatch.setattr(
        base_llm_hmonnx_model.AutoTokenizer,
        "from_pretrained",
        lambda path, **kwargs: calls.append((path, kwargs)) or sentinel,
    )

    assert _tokenizer_owner().get_tokenizer() is sentinel
    assert calls == [
        ("/tmp/exported-hf-config", {"trust_remote_code": True}),
    ]


def test_exported_tokenizer_preserves_explicit_remote_code_opt_out(monkeypatch):
    calls = []
    monkeypatch.setattr(
        base_llm_hmonnx_model.AutoTokenizer,
        "from_pretrained",
        lambda path, **kwargs: calls.append((path, kwargs)),
    )

    _tokenizer_owner().get_tokenizer(trust_remote_code=False)
    assert calls == [
        ("/tmp/exported-hf-config", {"trust_remote_code": False}),
    ]
