"""Regression tests for qwen3.5 HMONNX generate CLI helpers."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATE_PATH = REPO_ROOT / "examples_merak" / "llm" / "qwen3_5" / "qwen3_5_xh_hmonnx_generate.py"


def _install_stub(monkeypatch, name: str, attrs: dict | None = None) -> types.ModuleType:
    parts = name.split(".")
    for index in range(1, len(parts) + 1):
        sub = ".".join(parts[:index])
        if sub not in sys.modules:
            monkeypatch.setitem(sys.modules, sub, types.ModuleType(sub))
    module = sys.modules[name]
    if attrs:
        for key, value in attrs.items():
            setattr(module, key, value)
    return module


def _load_generate_module(monkeypatch):
    torch_stub = _install_stub(monkeypatch, "torch")
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    _install_stub(monkeypatch, "transformers", {"TextStreamer": object})
    _install_stub(
        monkeypatch,
        "xhmodel_merak.xh_llm",
        {
            "AutoLLMHONNXModel": object,
            "LLMInferenceContextManager": object,
        },
    )
    _install_stub(
        monkeypatch,
        "xhquant.api",
        {
            "get_xhquant_logger": lambda: None,
            "xhquant_init": lambda *args, **kwargs: None,
        },
    )
    _install_stub(
        monkeypatch,
        "xhquant.utils",
        {
            "ContextManagers": object,
            "TimeProfiler": object,
        },
    )
    _install_stub(monkeypatch, "xhquant.utils.memory_tracker", {"MemoryTracker": object})

    spec = importlib.util.spec_from_file_location("qwen35_hmonnx_generate", GENERATE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_image_path_is_none(monkeypatch):
    module = _load_generate_module(monkeypatch)
    parser = module.build_parser()

    args = parser.parse_args(["--config", "meta.json"])

    assert args.image_path is None


def test_resolve_prompt_reads_file(monkeypatch, tmp_path):
    module = _load_generate_module(monkeypatch)
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("  hello\n", encoding="utf-8")

    assert module._resolve_prompt(str(prompt_file)) == "hello"


def test_resolve_prompt_keeps_literal_when_file_missing(monkeypatch, tmp_path):
    module = _load_generate_module(monkeypatch)
    prompt = str(tmp_path / "missing prompt.txt")

    assert module._resolve_prompt(prompt) == prompt


def test_resolve_image_path_requires_existing_file(monkeypatch, tmp_path):
    module = _load_generate_module(monkeypatch)
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"x")

    assert module._resolve_image_path(str(image_path)) == str(image_path)
    assert module._resolve_image_path(str(tmp_path / "missing.jpg")) is None
    assert module._resolve_image_path(None) is None


def test_supports_multimodal_inputs_requires_callable_processor(monkeypatch):
    module = _load_generate_module(monkeypatch)

    assert module._supports_multimodal_inputs(types.SimpleNamespace(get_tf_processor=lambda: object())) is True
    assert module._supports_multimodal_inputs(types.SimpleNamespace(get_tf_processor=None)) is False
    assert module._supports_multimodal_inputs(types.SimpleNamespace()) is False
