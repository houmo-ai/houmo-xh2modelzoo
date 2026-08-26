import importlib.util
from argparse import Namespace
from pathlib import Path

from xhmodel_merak.xh_llm.workflows.result import ExportResult


MODULE_PATH = Path(__file__).parents[1] / "examples_merak/llm/minicpm5/minicpm5_workflow.py"
SPEC = importlib.util.spec_from_file_location("minicpm5_workflow", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
_overrides = MODULE._overrides


def test_minicpm5_workflow_can_override_num_logits_to_keep_zero():
    args = Namespace(
        context_max_length=None,
        prefill_chunk_length=None,
        quant_type=None,
        only_first_block=False,
        max_layers=None,
        num_logits_to_keep=0,
    )

    assert _overrides(args) == {"export.model.num_logits_to_keep": 0}


def test_minicpm5_quant_type_override_preserves_lm_head_scheme():
    args = Namespace(
        context_max_length=None,
        prefill_chunk_length=None,
        quant_type="w4a8h0_ssfp",
        only_first_block=False,
        max_layers=None,
        num_logits_to_keep=None,
    )

    assert _overrides(args) == {"export.model.quant_scheme.quant_type": "w4a8h0_ssfp"}


def test_minicpm5_dump_golden_uses_exported_golden_metadata(tmp_path, monkeypatch):
    import transformers

    import xhmodel_merak.xh_llm as llm_api
    import xhquant.utils

    meta_file = tmp_path / "hmquant_minicpm5" / "golden_meta_info.json"
    meta_file.parent.mkdir()
    meta_file.write_text("{}", encoding="utf-8")

    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            assert messages == [{"role": "user", "content": "hello"}]
            assert tokenize is False
            assert add_generation_prompt is True
            return "hello"

        def __call__(self, texts, return_tensors=None):
            assert texts == ["hello"]
            assert return_tensors == "pt"
            return _Inputs()

    class _Inputs(dict):
        def to(self, device):
            assert device == "cuda:0"
            return self

    class _Model:
        device = "cuda:0"

        def __init__(self):
            self.generated = False

        def get_tokenizer(self):
            return _Tokenizer()

        def generate(self, **kwargs):
            self.generated = True
            assert kwargs["max_new_tokens"] == 2
            assert kwargs["do_sample"] is False
            assert kwargs["pad_token_id"] == 0

    model = _Model()
    calls = {}

    class _AutoModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.update(path=path, kwargs=kwargs)
            return model

    class _Context:
        def __init__(self, value):
            self.value = value

        def __enter__(self):
            return self.value

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(llm_api, "AutoLLMHONNXModel", _AutoModel)
    monkeypatch.setattr(llm_api, "LLMInferenceContextManager", lambda value: _Context(value))
    monkeypatch.setattr(xhquant.utils, "ContextManagers", lambda values: _Context(values))
    monkeypatch.setattr(transformers, "TextStreamer", lambda tokenizer: object())

    from xhmodel_merak.xh_llm.models.minicpm5.workflow import MiniCPM5Workflow

    workflow = MiniCPM5Workflow.__new__(MiniCPM5Workflow)
    result = workflow.dump_golden(
        ExportResult(work_dir=str(tmp_path), config_file=str(tmp_path / "config.yaml")),
        "cuda:0",
        {"text": "hello"},
    )

    assert result == str(meta_file)
    assert calls == {"path": str(meta_file), "kwargs": {"device_map": ["cuda:0"], "enable_golden": True}}
    assert model.generated
