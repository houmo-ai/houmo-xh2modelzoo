# Copyright 2025 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Argparse/CLI unit tests for ``qwen3_vl_xh2a_multi_image_demo``.

These tests exercise only the pure CLI / input-resolution helpers and do
not require a real model, GPU, or real image data.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
DEMO_PATH = REPO_ROOT / "examples" / "llm" / "qwen3vl" / "qwen3_vl_xh2a_multi_image_demo.py"


def _install_stub(name: str, attrs: dict | None = None) -> types.ModuleType:
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        sub = ".".join(parts[:i])
        if sub not in sys.modules:
            mod = types.ModuleType(sub)
            sys.modules[sub] = mod
    mod = sys.modules[name]
    if attrs:
        for key, value in attrs.items():
            setattr(mod, key, value)
    return mod


@pytest.fixture(scope="module")
def demo_module():
    """Import the demo module while stubbing heavy external dependencies."""
    if "torch" not in sys.modules:
        torch_stub = _install_stub("torch")
        torch_stub.float16 = "float16"
        torch_stub.float32 = "float32"
        torch_stub.bfloat16 = "bfloat16"
        torch_stub.dtype = type("dtype", (), {})
        torch_stub.Tensor = type("Tensor", (), {})
        torch_stub.device = lambda *a, **kw: "cpu"

        class _Cuda:
            @staticmethod
            def is_available():
                return False

        torch_stub.cuda = _Cuda()
        nn_stub = _install_stub("torch.nn")
        nn_stub.Embedding = type("Embedding", (), {})
        nn_stub.Module = type("Module", (), {})

    if "PIL" not in sys.modules:
        pil_stub = _install_stub("PIL")

        class _Image:
            MAX_IMAGE_PIXELS = 0
            BICUBIC = 0

            @staticmethod
            def open(*args, **kwargs):
                raise NotImplementedError

            class Image:
                pass

        pil_stub.Image = _Image
        _install_stub("PIL.Image", {"MAX_IMAGE_PIXELS": 0})
        _install_stub("PIL.ImageOps", {"expand": lambda *a, **kw: None})
        sys.modules["PIL"].ImageOps = sys.modules["PIL.ImageOps"]

    if "transformers" not in sys.modules:
        _install_stub("transformers", {"AutoConfig": type("AutoConfig", (), {})})

    if "xhquant" not in sys.modules:
        _install_stub("xhquant")
        printing = _install_stub("xhquant.utils.suppress_printing")
        printing.disable_printing = False
        _install_stub("xhquant.utils", {"suppress_printing": printing})

    if "xh_model_zoo" not in sys.modules:
        _install_stub("xh_model_zoo")
        api_stub = _install_stub("xh_model_zoo.api")
        api_stub.ConfigDict = dict
        api_stub.get_root_logger = lambda: logging.getLogger("qwen3vl-test")
        api_stub.xhquant_llm_init = lambda *args, **kwargs: None
        models_stub = _install_stub("xh_model_zoo.xh_llm.models.qwen3_vl")
        models_stub.Qwen3VLONNXModel = type("Qwen3VLONNXModel", (), {})
        models_stub.Qwen3VLProcessor = type("Qwen3VLProcessor", (), {})

    spec = importlib.util.spec_from_file_location("qwen3vl_multi_image_demo", DEMO_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _ns(demo_module, **overrides):
    """Build an argparse.Namespace pre-populated with defaults from the parser."""
    parser = demo_module.build_parser()
    ns = parser.parse_args([])  # all defaults; we patch later
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def test_help_lists_all_input_modes(demo_module, capsys):  # noqa: ARG001 - fixture needed for stubs
    parser = demo_module.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    out = capsys.readouterr().out
    for flag in ("--image-paths", "--image-dir", "--task-json", "--scene-name", "--prompt", "--prompt-file"):
        assert flag in out, f"missing {flag} in --help"


def test_select_image_mode_rejects_no_input(demo_module):
    args = _ns(demo_module)
    with pytest.raises(ValueError, match="one of --image-paths"):
        demo_module.select_image_mode(args)


def test_select_image_mode_rejects_multiple_inputs(demo_module, tmp_path):
    args = _ns(demo_module, image_paths="a.jpg", image_dir=str(tmp_path))
    with pytest.raises(ValueError, match="mutually exclusive"):
        demo_module.select_image_mode(args)


def test_select_image_mode_task_requires_task_id(demo_module, tmp_path):
    args = _ns(demo_module, task_json=str(tmp_path / "x.json"))
    with pytest.raises(ValueError, match="--task-id is required"):
        demo_module.select_image_mode(args)


def test_select_image_mode_image_glob_requires_dir(demo_module):
    args = _ns(demo_module, image_paths="a.jpg", image_glob="*.jpg")
    with pytest.raises(ValueError, match="--image-glob is only valid"):
        demo_module.select_image_mode(args)


def test_resolve_prompt_mutex(demo_module, tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("hi", encoding="utf-8")
    args = _ns(demo_module, prompt="hello", prompt_file=str(f))
    with pytest.raises(ValueError, match="mutually exclusive"):
        demo_module.resolve_prompt(args)


def test_resolve_prompt_requires_one(demo_module):
    args = _ns(demo_module)
    with pytest.raises(ValueError, match="one of --prompt"):
        demo_module.resolve_prompt(args)


def test_resolve_prompt_from_file(demo_module, tmp_path):
    f = tmp_path / "prompt.txt"
    f.write_text("  Describe the images.  \n", encoding="utf-8")
    args = _ns(demo_module, prompt_file=str(f))
    assert demo_module.resolve_prompt(args) == "Describe the images."


def test_parse_image_paths_list_missing_file(demo_module, tmp_path):
    real = tmp_path / "a.jpg"
    real.write_bytes(b"x")
    spec = f"{real},{tmp_path / 'missing.jpg'}"
    with pytest.raises(FileNotFoundError):
        demo_module.parse_image_paths_list(spec)


def test_parse_image_paths_list_strips_and_dedupes_blanks(demo_module, tmp_path):
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    spec = f"  {a} , , {b}"
    out = demo_module.parse_image_paths_list(spec)
    assert [p.name for p in out] == ["a.jpg", "b.jpg"]


def test_scan_image_dir_sorts_and_filters(demo_module, tmp_path):
    (tmp_path / "z.jpg").write_bytes(b"x")
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "ignore.txt").write_text("nope", encoding="utf-8")
    paths = demo_module.scan_image_dir(tmp_path, None)
    assert [p.name for p in paths] == ["a.png", "z.jpg"]


def test_scan_image_dir_with_glob(demo_module, tmp_path):
    (tmp_path / "keep.jpg").write_bytes(b"x")
    (tmp_path / "drop.png").write_bytes(b"x")
    paths = demo_module.scan_image_dir(tmp_path, "*.jpg")
    assert [p.name for p in paths] == ["keep.jpg"]
