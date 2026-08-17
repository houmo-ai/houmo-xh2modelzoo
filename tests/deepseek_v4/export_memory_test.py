from __future__ import annotations

import gc
import weakref
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch

import xhmodel_merak.xh_llm.base_llm_model as base_llm_model
from xhmodel_merak.xh_llm.types import ModelSwitcher


class _FixedModel:
    def is_fixed(self) -> bool:
        return True

    def to(self, _device: str):
        return self


class _Processor:
    input_sequence_length = 1

    def __call__(self, _inputs):
        return [torch.zeros(1)]


class _CacheMixin:
    @contextmanager
    def kv_cache_scope(self, *, device: str):
        assert device == "meta"
        yield


class _ExportHarness:
    _export_hmonnx = base_llm_model.BaseLLMModel._export_hmonnx

    def __init__(self) -> None:
        self._quanted_model = _FixedModel()
        self.kvcache_config = SimpleNamespace()
        self.wrap_cfg = SimpleNamespace(prefill_chunk_length=256)
        self.config = SimpleNamespace(prefill_chunk_length=256)

    def _save_export_token_embedding(self, _output_dir, meta_info):
        return meta_info

    def get_kvcache_mixin(self):
        return _CacheMixin()

    def set_input_sequence_length(self, _length: int) -> None:
        pass

    def set_prefill(self) -> None:
        if isinstance(self._quanted_model, ModelSwitcher):
            self._quanted_model.set_activate_model("prefill")

    def set_decode(self) -> None:
        if isinstance(self._quanted_model, ModelSwitcher):
            self._quanted_model.set_activate_model("decode")

    def get_data_preprocessor(self):
        return _Processor()

    def get_dummy_inputs(self):
        return {}

    def get_export_cfg(self):
        return {"input_names": []}

    @staticmethod
    def xh1_hmonnx_compatible(names):
        return names

    @staticmethod
    def _trim_cpu_allocator() -> None:
        gc.collect()

    @staticmethod
    def _release_prefill_quanted_model_after_export() -> bool:
        return False


class _ExportedProgram:
    pass


def test_prefill_exported_program_is_released_before_decode_trace(tmp_path, monkeypatch) -> None:
    calls = 0
    prefill_ref = None

    def fake_to_export_graph(_model, _inputs):
        nonlocal calls, prefill_ref
        calls += 1
        if calls == 2:
            assert prefill_ref is not None
            assert prefill_ref() is None
        exported = _ExportedProgram()
        if calls == 1:
            prefill_ref = weakref.ref(exported)
        return exported

    def fake_to_export_hmonnx_v2(_model, _inputs, output_path, _cfg, **_kwargs):
        Path(output_path).write_bytes(b"")
        return output_path

    monkeypatch.setattr(base_llm_model, "to_export_graph", fake_to_export_graph)
    monkeypatch.setattr(base_llm_model, "to_export_hmonnx_v2", fake_to_export_hmonnx_v2)

    exported_info = SimpleNamespace(
        meta=SimpleNamespace(),
        model_name="deepseek-v4-memory-test",
        exported_dir=str(tmp_path),
    )
    _ExportHarness()._export_hmonnx(exported_info)

    assert calls == 2


def test_completed_prefill_quanted_model_can_be_released_before_decode_trace(
    tmp_path, monkeypatch
) -> None:
    prefill = _FixedModel()
    decode = _FixedModel()
    prefill_ref = weakref.ref(prefill)
    switcher = ModelSwitcher({"prefill": prefill, "decode": decode})
    switcher.set_activate_model("prefill")
    del prefill

    harness = _ExportHarness()
    harness._quanted_model = switcher
    harness._release_prefill_quanted_model_after_export = lambda: True
    calls = 0

    def fake_to_export_graph(model, _inputs):
        nonlocal calls
        calls += 1
        if calls == 1:
            assert model is prefill_ref()
        else:
            assert model is decode
            assert prefill_ref() is None
        return _ExportedProgram()

    def fake_to_export_hmonnx_v2(_model, _inputs, output_path, _cfg, **_kwargs):
        Path(output_path).write_bytes(b"")
        return output_path

    monkeypatch.setattr(base_llm_model, "to_export_graph", fake_to_export_graph)
    monkeypatch.setattr(base_llm_model, "to_export_hmonnx_v2", fake_to_export_hmonnx_v2)

    exported_info = SimpleNamespace(
        meta=SimpleNamespace(),
        model_name="deepseek-v4-memory-test",
        exported_dir=str(tmp_path),
    )
    harness._export_hmonnx(exported_info)

    assert calls == 2
    assert tuple(switcher.keys()) == ("decode",)
    assert switcher.activate_model is decode
