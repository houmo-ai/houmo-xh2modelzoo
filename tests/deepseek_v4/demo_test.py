from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


_DEMO = Path(__file__).resolve().parents[2] / "examples_merak/llm/deepseekv4/deepseek_v4_xh_hmonnx_generate.py"
_SPEC = importlib.util.spec_from_file_location("deepseek_v4_hmonnx_demo", _DEMO)
assert _SPEC is not None and _SPEC.loader is not None
demo = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(demo)


def _write_export(root: Path) -> Path:
    package = root / "hmquant_test"
    package.mkdir(parents=True)
    meta_path = package / "golden_meta_info.json"
    meta_path.write_text(json.dumps({"model_config": {"context_max_length": 262144}}))
    return meta_path


def test_demo_resolves_one_export_metadata_file(tmp_path: Path) -> None:
    meta_path = _write_export(tmp_path)

    assert demo.resolve_meta_path(tmp_path) == meta_path
    assert demo.resolve_meta_path(meta_path) == meta_path


def test_demo_rejects_ambiguous_export_directory(tmp_path: Path) -> None:
    _write_export(tmp_path / "one")
    _write_export(tmp_path / "two")

    with pytest.raises(ValueError, match="exactly one"):
        demo.resolve_meta_path(tmp_path)


def test_demo_parses_deduplicated_cuda_devices() -> None:
    assert demo.parse_devices("0,cuda:1,2,2") == [0, 1, 2]
    assert demo.parse_devices("cpu") == ["cpu"]
    with pytest.raises(ValueError, match="cannot mix"):
        demo.parse_devices("cpu,0")


def test_demo_validates_packed_prefill_decode_sharing() -> None:
    valid = {
        "runtime_w4_planned_references": 6,
        "runtime_w4_packed_references": 6,
        "runtime_w4_packed_initializers": 3,
        "runtime_w4_shared_references": 3,
    }
    demo.validate_packed_shared_weights(valid)

    with pytest.raises(RuntimeError, match="packed/shared W4"):
        demo.validate_packed_shared_weights(dict(valid, runtime_w4_shared_references=2))


def test_demo_places_real_context_before_question(tmp_path: Path) -> None:
    context = tmp_path / "context.txt"
    context.write_text("第一段。\n第二段。", encoding="utf-8")

    content = demo.build_user_content("请概括。", str(context))

    assert content == "第一段。\n第二段。\n\n请根据以上内容回答：请概括。"


def test_demo_uses_native_deepseek_v4_chat_encoding_without_jinja_template() -> None:
    class Tokenizer:
        chat_template = None
        bos_token = "<｜begin▁of▁sentence｜>"
        eos_token = "<｜end▁of▁sentence｜>"

    assert demo.render_prompt(Tokenizer(), "测试问题", raw_prompt=False) == (
        "<｜begin▁of▁sentence｜><｜User｜>测试问题<｜Assistant｜></think>"
    )


def test_demo_rejects_unknown_tokenizer_without_template_unless_raw() -> None:
    class Tokenizer:
        chat_template = None
        bos_token = "<bos>"
        eos_token = "<eos>"

    with pytest.raises(ValueError, match="not a recognized DeepSeek-V4 tokenizer"):
        demo.render_prompt(Tokenizer(), "测试问题", raw_prompt=False)

    assert demo.render_prompt(Tokenizer(), "测试问题", raw_prompt=True) == "测试问题"


def test_demo_defaults_to_model_parallel_friendly_runtime() -> None:
    args = demo.build_parser().parse_args(["--config", "unused"])

    assert args.device == "0,1,2,3"
    assert args.pack_w4 is True
    assert args.cuda_graph is False
    assert args.max_new_tokens == 128


def test_demo_accepts_workflow_style_dump_golden_alias() -> None:
    args = demo.build_parser().parse_args(["--config", "unused", "--dump-golden"])

    assert args.golden is True
