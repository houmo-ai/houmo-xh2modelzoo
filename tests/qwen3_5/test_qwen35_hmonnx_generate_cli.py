"""Regression tests for qwen3.5 HMONNX generate CLI helpers."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATE_PATH = (
    REPO_ROOT / "examples_merak" / "llm" / "qwen3_5" / "qwen3_5_xh_hmonnx_generate.py"
)


def _install_stub(monkeypatch, name: str, attrs: dict | None = None) -> types.ModuleType:
    parts = name.split(".")
    for index in range(1, len(parts) + 1):
        sub = ".".join(parts[:index])
        if sub not in sys.modules:
            module = types.ModuleType(sub)
            if index < len(parts):
                module.__path__ = []
            monkeypatch.setitem(sys.modules, sub, module)
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
        "xhmodel_merak.xh_llm.models.qwen3_5.workflow_runtime",
        {
            "hmonnx_generate": lambda *args, **kwargs: None,
            "print_quick_test_result": lambda *args, **kwargs: None,
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


def test_default_image_path_keeps_demo_image(monkeypatch):
    module = _load_generate_module(monkeypatch)
    parser = module.build_parser()

    args = parser.parse_args(["--config", "meta.json"])

    assert args.image_path == "./data/images/demo_qwen3_vl.jpeg"
    assert args.prompt == "Describe this image."
    assert args.max_new_tokens == 1024


def test_parse_device_arg_handles_cpu_and_gpu_lists(monkeypatch):
    module = _load_generate_module(monkeypatch)

    assert module._parse_device_arg("cpu") == ["cpu"]
    assert module._parse_device_arg("0") == [0]
    assert module._parse_device_arg("cuda:0,1,1") == [0, 1]


@pytest.mark.parametrize("device_arg", ["", "cpu,0", "cuda:abc"])
def test_parse_device_arg_rejects_invalid_values(monkeypatch, device_arg):
    module = _load_generate_module(monkeypatch)

    with pytest.raises(ValueError):
        module._parse_device_arg(device_arg)
