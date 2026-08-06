import json
import sys

import pytest

from examples_merak.llm.qwen3_5.qwen3_5_dynamic_prune_hmonnx_generate import (
    _runtime_device,
)
from examples_merak.llm.qwen3_5.qwen3_5_dynamic_prune_hmonnx_generate import (
    build_parser as build_generate_parser,
)
from examples_merak.llm.qwen3_5.qwen3_5_dynamic_prune_workflow import (
    DynamicPruneTextOnlyHMONNXModel,
    _is_llm_only_meta,
    dump_dynamic_prune_golden,
    parse_args,
    use_dynamic_prune_text_only_hmonnx_runtime,
)
from xhmodel_merak.xh_llm.models.qwen3_5_moe_dynamic_prune.qwen3_5_moe_dynamic_prune_model import (
    XHQwen3_5MoeDynamicPruneModel,
)


@pytest.mark.parametrize(
    ("visual_config", "expected"),
    [(None, True), ({"hmonnx": "visual/model.onnx"}, False)],
)
def test_is_llm_only_meta(tmp_path, visual_config, expected):
    meta_file = tmp_path / "golden_meta_info.json"
    meta_file.write_text(json.dumps({"visual_config": visual_config}), encoding="utf-8")

    assert _is_llm_only_meta(meta_file) is expected


def test_llm_only_golden_temporarily_selects_text_runtime(tmp_path):
    meta_file = tmp_path / "golden_meta_info.json"
    meta_file.write_text(json.dumps({"visual_config": None}), encoding="utf-8")
    original_runtime = XHQwen3_5MoeDynamicPruneModel.HMONNXINFERENCE_CLS

    class Workflow:
        def _find_golden_meta_file(self, export_result):
            assert export_result == "export-result"
            return str(meta_file)

        def dump_golden(self, export_result, device, input_messages):
            assert export_result == "export-result"
            assert device == "cuda"
            assert input_messages == "prompt"
            assert (
                XHQwen3_5MoeDynamicPruneModel.HMONNXINFERENCE_CLS
                is DynamicPruneTextOnlyHMONNXModel
            )
            return str(meta_file)

    result = dump_dynamic_prune_golden(Workflow(), "export-result", "cuda", "prompt")

    assert result == str(meta_file)
    assert XHQwen3_5MoeDynamicPruneModel.HMONNXINFERENCE_CLS is original_runtime


def test_full_vlm_golden_keeps_registered_runtime(tmp_path):
    meta_file = tmp_path / "golden_meta_info.json"
    meta_file.write_text(
        json.dumps({"visual_config": {"hmonnx": "visual/model.onnx"}}),
        encoding="utf-8",
    )
    original_runtime = XHQwen3_5MoeDynamicPruneModel.HMONNXINFERENCE_CLS

    class Workflow:
        def _find_golden_meta_file(self, export_result):
            return str(meta_file)

        def dump_golden(self, export_result, device, input_messages):
            assert XHQwen3_5MoeDynamicPruneModel.HMONNXINFERENCE_CLS is original_runtime
            return str(meta_file)

    result = dump_dynamic_prune_golden(Workflow(), object(), "cuda", "prompt")

    assert result == str(meta_file)
    assert XHQwen3_5MoeDynamicPruneModel.HMONNXINFERENCE_CLS is original_runtime


def test_dynamic_prune_cli_accepts_both_gdr_fusion_switches(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen3_5_dynamic_prune_workflow.py",
            "--model-dir",
            "model",
            "--config-path",
            "workflow.yaml",
            "--enable-fuse-gdr-ops",
            "--enable-fuse-gdr-block-recurrent-ops",
        ],
    )

    args = parse_args()

    assert args.fuse_gdr_ops is True
    assert args.enable_fuse_gdr_block_recurrent_ops is True


def test_dynamic_prune_generate_cli_has_conversation_validation_defaults():
    args = build_generate_parser().parse_args(["--config", "export-dir"])

    assert args.max_new_tokens == 64
    assert args.min_output_tokens == 8
    assert args.do_sample is False
    assert args.prompt == "用中文简单介绍 Qwen3.5。"


def test_dynamic_prune_generate_runtime_device_uses_first_device():
    assert _runtime_device([0, 1]) == "cuda:0"
    assert _runtime_device(["cpu"]) == "cpu"


def test_text_only_runtime_context_restores_dynamic_prune_registration():
    original_runtime = XHQwen3_5MoeDynamicPruneModel.HMONNXINFERENCE_CLS

    with use_dynamic_prune_text_only_hmonnx_runtime():
        assert (
            XHQwen3_5MoeDynamicPruneModel.HMONNXINFERENCE_CLS
            is DynamicPruneTextOnlyHMONNXModel
        )

    assert XHQwen3_5MoeDynamicPruneModel.HMONNXINFERENCE_CLS is original_runtime
