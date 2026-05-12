import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = Path("examples/llm/qwen3omni/qwen3_omni_validate_text_hmonnx_replacement.py")


def _load_module(monkeypatch, *, resolved_device_map="cuda:5", validation_max_memory=None):
    if validation_max_memory is None:
        validation_max_memory = {0: "40.0GiB", "cpu": "64.0GiB"}

    fake_pipeline = types.ModuleType("_hmonnx_pipeline")
    fake_pipeline._create_hmonnx_session = lambda *args, **kwargs: None
    fake_pipeline._build_dense_deepstack_tensors = lambda *args, **kwargs: []
    fake_pipeline._ensure_tensor = lambda tensor, *args, **kwargs: tensor
    fake_pipeline._ensure_mistral_common_reasoning_effort = lambda *args, **kwargs: None
    fake_pipeline._extract_outputs = lambda output: output
    fake_pipeline._extract_primary_output = lambda output: output
    fake_pipeline._resolve_validation_device_map = lambda device_map, logger=None: resolved_device_map
    fake_pipeline._build_safe_validation_max_memory = lambda logger=None: validation_max_memory
    fake_pipeline.build_conversation = lambda case, text_prompt=None: ([], False)
    fake_pipeline.discover_artifacts = lambda work_dir: {"text": {"_meta_path": "meta.json"}}
    fake_pipeline.save_json = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "_hmonnx_pipeline", fake_pipeline)

    fake_api = types.ModuleType("xhquant.api")
    fake_api.CacheTensor = lambda value: value
    fake_api.get_root_logger = lambda: SimpleNamespace(info=lambda *args, **kwargs: None)
    fake_api.set_random_seed = lambda *args, **kwargs: None
    fake_api.xhquant_init = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "xhquant.api", fake_api)

    spec = importlib.util.spec_from_file_location(
        "qwen3_omni_validate_text_hmonnx_replacement_testmod",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_main_uses_safe_model_loading_for_auto_device_map(monkeypatch, tmp_path):
    module = _load_module(
        monkeypatch,
        resolved_device_map="auto",
        validation_max_memory={0: "40.0GiB", 1: "41.0GiB", "cpu": "64.0GiB"},
    )
    captured = {}

    class DummyModel:
        def __init__(self):
            self.thinker = SimpleNamespace(generate=lambda *args, **kwargs: None)
            self.config = SimpleNamespace(talker_config=SimpleNamespace(accept_hidden_layer=0))

        def eval(self):
            return self

    class DummyModelClass:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            captured["model_path"] = model_path
            captured["kwargs"] = kwargs
            return DummyModel()

    class DummyTokenizer:
        pad_token_id = 0
        eos_token = "</s>"
        padding_side = "left"
        chat_template = "dummy"

        @staticmethod
        def from_pretrained(*args, **kwargs):
            return DummyTokenizer()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = DummyTokenizer
    fake_transformers.Qwen3OmniMoeForConditionalGeneration = DummyModelClass
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    monkeypatch.setattr(module, "_run_dialogue_case", lambda *args, **kwargs: {"rendered_text": "", "output_ids": [[]], "output_text": [""]})
    monkeypatch.setattr(module, "_build_text_hmonnx_generate_patch", lambda *args, **kwargs: (lambda *a, **k: None))
    monkeypatch.setattr(module, "_write_markdown_report", lambda *args, **kwargs: None)

    args = SimpleNamespace(
        model="/tmp/fake-model",
        work_dir=str(tmp_path),
        prompt=["hello"],
        case=None,
        max_new_tokens=8,
        talker_max_new_tokens=16,
        device_map="auto",
        allow_mismatch_report_only=True,
        seed=1234,
        debug=False,
    )

    module.main(args)

    assert captured["model_path"] == "/tmp/fake-model"
    assert captured["kwargs"]["device_map"] == "auto"
    assert captured["kwargs"]["max_memory"] == {0: "40.0GiB", 1: "41.0GiB", "cpu": "64.0GiB"}


def test_main_raises_when_replacement_text_mismatches(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)

    class DummyModel:
        def __init__(self):
            self.thinker = SimpleNamespace(generate=lambda *args, **kwargs: None)
            self.config = SimpleNamespace(talker_config=SimpleNamespace(accept_hidden_layer=0))

        def eval(self):
            return self

    class DummyModelClass:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            return DummyModel()

    class DummyTokenizer:
        pad_token_id = 0
        eos_token = "</s>"
        padding_side = "left"
        chat_template = "dummy"

        @staticmethod
        def from_pretrained(*args, **kwargs):
            return DummyTokenizer()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = DummyTokenizer
    fake_transformers.Qwen3OmniMoeForConditionalGeneration = DummyModelClass
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    call_results = iter(
        [
            {"rendered_text": "", "output_ids": [[1, 2]], "output_text": ["baseline"]},
            {"rendered_text": "", "output_ids": [[3, 4]], "output_text": ["replacement"]},
        ]
    )
    monkeypatch.setattr(module, "_run_dialogue_case", lambda *args, **kwargs: next(call_results))
    monkeypatch.setattr(module, "_build_text_hmonnx_generate_patch", lambda *args, **kwargs: (lambda *a, **k: None))
    monkeypatch.setattr(module, "_write_markdown_report", lambda *args, **kwargs: None)

    args = SimpleNamespace(
        model="/tmp/fake-model",
        work_dir=str(tmp_path),
        prompt=["hello"],
        case=None,
        max_new_tokens=8,
        talker_max_new_tokens=16,
        device_map="cuda:0",
        allow_mismatch_report_only=False,
        seed=1234,
        debug=False,
    )

    with pytest.raises(RuntimeError, match="Text HMONNX replacement mismatch detected"):
        module.main(args)
