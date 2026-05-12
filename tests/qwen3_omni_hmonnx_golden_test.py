import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


MODULE_PATH = Path("examples/llm/qwen3omni/qwen3_omni_hmonnx_golden.py")


def _load_module(monkeypatch):
    fake_api = types.ModuleType("xhquant.api")
    fake_api.CacheTensor = lambda value: value
    fake_api.get_root_logger = lambda: SimpleNamespace(info=lambda *args, **kwargs: None)
    fake_api.xhquant_init = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "xhquant.api", fake_api)

    fake_runtime = types.ModuleType("xhquant.xhonnxruntime")
    fake_runtime.HMONNXGraphGoldenInference = object
    monkeypatch.setitem(sys.modules, "xhquant.xhonnxruntime", fake_runtime)

    spec = importlib.util.spec_from_file_location(
        "qwen3_omni_hmonnx_golden_testmod",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_run_session_with_golden_moves_session_back_to_cpu(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    calls = []

    class FakeSession:
        def __init__(self, onnx_file):
            self.onnx_file = onnx_file
            self.exec_device = None
            self.save_golden = False
            self.golden_dir = None

        def to(self, device):
            calls.append(("to", str(device)))
            return self

        def initialize(self):
            calls.append(("initialize", None))

        def get_input_names(self):
            return ["input_ids"]

        def get_input(self, name):
            return SimpleNamespace(shape=[1, 2], dtype=torch.float16)

        def run(self, feed):
            calls.append(("run", tuple(feed)))
            return (torch.zeros(1, 2),)

        def get_output_names(self):
            return ["logits"]

    monkeypatch.setattr(module, "HMONNXGoldenInference", FakeSession)
    monkeypatch.setattr(module, "gc", SimpleNamespace(collect=lambda: calls.append(("collect", None))), raising=False)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "empty_cache", lambda: calls.append(("empty_cache", None)))

    logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    module._run_session_with_golden(tmp_path / "model.onnx", tmp_path / "golden", logger, torch.device("cuda:0"))

    assert ("to", "cuda:0") in calls
    assert ("to", "cpu") in calls
    assert calls[-2:] == [("collect", None), ("empty_cache", None)]


def test_run_session_with_golden_cleans_up_after_failure(monkeypatch, tmp_path):
    module = _load_module(monkeypatch)
    calls = []

    class FakeSession:
        def __init__(self, onnx_file):
            self.onnx_file = onnx_file

        def to(self, device):
            calls.append(("to", str(device)))
            return self

        def initialize(self):
            calls.append(("initialize", None))

        def get_input_names(self):
            return ["input_ids"]

        def get_input(self, name):
            return SimpleNamespace(shape=[1, 2], dtype=torch.float16)

        def run(self, feed):
            raise RuntimeError("boom")

    monkeypatch.setattr(module, "HMONNXGoldenInference", FakeSession)
    monkeypatch.setattr(module, "gc", SimpleNamespace(collect=lambda: calls.append(("collect", None))), raising=False)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "empty_cache", lambda: calls.append(("empty_cache", None)))

    logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="boom"):
        module._run_session_with_golden(
            tmp_path / "model.onnx",
            tmp_path / "golden",
            logger,
            torch.device("cuda:0"),
        )

    assert ("to", "cpu") in calls
    assert calls[-2:] == [("collect", None), ("empty_cache", None)]
