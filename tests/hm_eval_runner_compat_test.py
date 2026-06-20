from __future__ import annotations

import types
import sys


def test_modelscope_compat_patch_is_skipped_for_huggingface(monkeypatch):
    from hm_eval.core import eval_runner

    datasets_load = types.SimpleNamespace(_ALL_ALLOWED_EXTENSIONS=["jsonl"])
    datasets_pkg = types.ModuleType("datasets")
    datasets_pkg.load = datasets_load
    monkeypatch.setitem(sys.modules, "datasets", datasets_pkg)
    monkeypatch.setitem(sys.modules, "datasets.load", datasets_load)

    eval_runner._patch_modelscope_datasets_compat("huggingface")

    assert not hasattr(datasets_load, "ALL_ALLOWED_EXTENSIONS")


def test_modelscope_compat_patch_is_scoped_to_modelscope(monkeypatch):
    from hm_eval.core import eval_runner

    datasets_load = types.SimpleNamespace(_ALL_ALLOWED_EXTENSIONS=["jsonl"])
    datasets_pkg = types.ModuleType("datasets")
    datasets_pkg.load = datasets_load
    modelscope = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "datasets", datasets_pkg)
    monkeypatch.setitem(sys.modules, "datasets.load", datasets_load)
    monkeypatch.setitem(sys.modules, "modelscope", modelscope)

    eval_runner._patch_modelscope_datasets_compat("modelscope")

    assert datasets_load.ALL_ALLOWED_EXTENSIONS == ["jsonl"]
    assert hasattr(modelscope, "MsDataset")
