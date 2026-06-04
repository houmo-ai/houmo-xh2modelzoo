from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from xh_model_zoo.xh_llm.models.qwen3_next.qwen3_next_convert_config import (
    Qwen3NextConvertConfig,
)
from xh_model_zoo.xh_llm.models.qwen3_next.qwen3_next_onnx_model import (
    Qwen3NextONNXModel,
    _alloc_cache_inputs,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT_SCRIPT = REPO_ROOT / "examples/llm/qwen3_next/qwen3_next_xh2a_export_hmonnx.py"
CONVERTER = REPO_ROOT / "xh_model_zoo/xh_llm/models/qwen3_next/qwen3_next_converter.py"
LLM_MODEL = REPO_ROOT / "xh_model_zoo/xh_llm/models/qwen3_next/qwen3_next_llm_model.py"
CONVERT_CONFIG = REPO_ROOT / "xh_model_zoo/xh_llm/models/qwen3_next/qwen3_next_convert_config.py"
DEMO_RUNTIME = REPO_ROOT / "examples/llm/qwen3_next/_runtime.py"


def _load_export_script():
    spec = importlib.util.spec_from_file_location("qwen3_next_export_hmonnx", EXPORT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_demo_runtime():
    spec = importlib.util.spec_from_file_location("qwen3_next_demo_runtime", DEMO_RUNTIME)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _expected_conv_io_names(num_linear_layers: int, split_conv_cache: bool):
    input_names = []
    output_names = []
    for layer_idx in range(num_linear_layers):
        if split_conv_cache:
            for branch in ("q", "k", "v"):
                input_names.append(f"past_conv_cache_{branch}_{layer_idx}")
                output_names.append(f"conv_cache_out_{branch}_{layer_idx}")
        else:
            input_names.append(f"past_conv_cache_{layer_idx}")
            output_names.append(f"conv_cache_out_{layer_idx}")
    return input_names, output_names



def _load_function_from_source(path: Path, name: str):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(module)
            namespace = {}
            exec(compile(module, filename=str(path), mode="exec"), namespace)
            return namespace[name]
    raise AssertionError(f"function {name!r} not found in {path}")


def test_qwen3_next_split_cache_dims_prefer_actual_split_modules():
    converter_dim_resolver = _load_function_from_source(CONVERTER, "_linear_split_conv_dims")
    llm_dim_resolver = _load_function_from_source(LLM_MODEL, "_linear_split_conv_dims")

    class Conv:
        def __init__(self, in_channels: int):
            self.in_channels = in_channels

    class LinearAttn:
        key_dim = 4096
        value_dim = 4096
        head_k_dim = 64
        head_v_dim = 128
        num_v_heads = 32
        conv1d_q = Conv(2048)
        conv1d_k = Conv(2048)
        conv1d_v = Conv(4096)

    class PreSplitLinearAttn:
        key_dim = 4096
        value_dim = 4096
        head_k_dim = 64
        head_v_dim = 128
        num_v_heads = 32

    for dim_resolver in (converter_dim_resolver, llm_dim_resolver):
        assert dim_resolver(LinearAttn()) == (2048, 2048, 4096)
        assert dim_resolver(PreSplitLinearAttn()) == (2048, 2048, 4096)


def test_qwen3_next_demo_runtime_accepts_current_golden_meta_fields(tmp_path, monkeypatch):
    runtime = _load_demo_runtime()
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    (release_dir / "quant_embedding.pt").touch()
    hf_model_dir = tmp_path / "weights" / "qwen3-next"
    hf_model_dir.mkdir(parents=True)

    monkeypatch.chdir(tmp_path)

    assert runtime.resolve_path(release_dir, "quant_embedding.pt") == (release_dir / "quant_embedding.pt").resolve()
    assert runtime.resolve_path(release_dir, "weights/qwen3-next") == hf_model_dir.resolve()

    runtime_src = DEMO_RUNTIME.read_text()
    assert 'meta_info.get("hf_model")' in runtime_src
    assert 'meta_info.get("quant_embedding")' in runtime_src


def test_qwen3_next_config_defaults_are_explicit_and_self_consistent():
    cfg = Qwen3NextConvertConfig()

    assert cfg.normalize_force_fp32 is False
    assert cfg.split_conv_cache is True
    assert cfg.use_manual_depthwise_conv1d is False
    assert cfg.fuse_gdr_ops is False


def test_qwen3_next_export_parser_accepts_split_and_merged_modes():
    module = _load_export_script()
    parser = module.parse_arguments()

    assert parser.parse_args([]).split_conv_cache is True
    assert parser.parse_args(["--split_conv_cache"]).split_conv_cache is True
    assert parser.parse_args(["--split-conv-cache"]).split_conv_cache is True
    assert parser.parse_args(["--split_conv_cache=false"]).split_conv_cache is False
    assert parser.parse_args(["--no_split_conv_cache"]).split_conv_cache is False
    assert parser.parse_args(["--no-split-conv-cache"]).split_conv_cache is False
    assert parser.parse_args([]).normalize_force_fp32 is False
    assert parser.parse_args(["--normalize-force-fp32=False"]).normalize_force_fp32 is False
    assert parser.parse_args(["--normalize-force-fp32"]).normalize_force_fp32 is True
    assert parser.parse_args([]).use_manual_depthwise_conv1d is False
    assert parser.parse_args(["--use_manual_depthwise_conv1d=False"]).use_manual_depthwise_conv1d is False
    assert parser.parse_args([]).fuse_gdr_ops is False
    assert parser.parse_args(["--fuse_gdr_ops=False"]).fuse_gdr_ops is False


def test_qwen3_next_export_help_smoke_documents_both_split_conv_modes():
    result = subprocess.run(
        [sys.executable, str(EXPORT_SCRIPT), "--help"],
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    help_text = result.stdout
    assert "--split_conv_cache" in help_text
    assert "--no_split_conv_cache" in help_text
    assert "Default True" in help_text
    assert "--no_split_conv_cache" in help_text
    assert "--normalize-force-fp32" in help_text
    assert "--use_manual_depthwise_conv1d" in help_text
    assert "--fuse_gdr_ops" in help_text


def test_qwen3_next_converter_and_export_script_forward_split_conv_flags():
    convert_config_src = CONVERT_CONFIG.read_text()
    converter_src = CONVERTER.read_text()
    export_script_src = EXPORT_SCRIPT.read_text()

    assert "normalize_force_fp32: bool = False" in convert_config_src
    assert "split_conv_cache: bool = True" in convert_config_src
    assert "use_manual_depthwise_conv1d: bool = False" in convert_config_src
    assert "fuse_gdr_ops: bool = False" in convert_config_src
    assert "normalize_force_fp32=self.config.normalize_force_fp32" in converter_src
    assert "split_conv_cache=self.config.split_conv_cache" in converter_src
    assert "use_manual_depthwise_conv1d=self.config.use_manual_depthwise_conv1d" in converter_src
    assert "fuse_gdr_ops=self.config.fuse_gdr_ops" in converter_src
    assert 'cfg.model.wrap_cfg.split_conv_cache = getattr(args, "split_conv_cache", True)' in export_script_src
    assert 'cfg.model.wrap_cfg.normalize_force_fp32 = normalize_force_fp32' in export_script_src
    assert 'cfg.model.wrap_cfg.fuse_gdr_ops = getattr(args, "fuse_gdr_ops", False)' in export_script_src
    assert 'dest="split_conv_cache"' in export_script_src
    assert 'meta_info.num_linear_attention_layers = len(qwen3_next_model.past_recurrent_states)' in export_script_src


def test_qwen3_next_export_conv_cache_names_cover_split_and_merged_modes():
    converter_src = CONVERTER.read_text()
    assert 'input_names.append(f"past_conv_cache_{branch}_{layer_idx}")' in converter_src
    assert 'input_names.append(f"past_conv_cache_{layer_idx}")' in converter_src
    assert 'output_names.append(f"conv_cache_out_{branch}_{layer_idx}")' in converter_src
    assert 'output_names.append(f"conv_cache_out_{layer_idx}")' in converter_src

    merged_inputs, merged_outputs = _expected_conv_io_names(2, split_conv_cache=False)
    split_inputs, split_outputs = _expected_conv_io_names(2, split_conv_cache=True)

    assert merged_inputs == ["past_conv_cache_0", "past_conv_cache_1"]
    assert merged_outputs == ["conv_cache_out_0", "conv_cache_out_1"]
    assert split_inputs == [
        "past_conv_cache_q_0",
        "past_conv_cache_k_0",
        "past_conv_cache_v_0",
        "past_conv_cache_q_1",
        "past_conv_cache_k_1",
        "past_conv_cache_v_1",
    ]
    assert split_outputs == [
        "conv_cache_out_q_0",
        "conv_cache_out_k_0",
        "conv_cache_out_v_0",
        "conv_cache_out_q_1",
        "conv_cache_out_k_1",
        "conv_cache_out_v_1",
    ]


def test_qwen3_next_runtime_updates_split_and_merged_conv_cache_outputs():
    model = Qwen3NextONNXModel.__new__(Qwen3NextONNXModel)
    merged_state = {
        "past_conv_cache_0": torch.zeros(1),
        "past_recurrent_state_0": torch.zeros(1),
    }
    split_state = {
        "past_conv_cache_q_0": torch.zeros(1),
        "past_conv_cache_k_0": torch.zeros(1),
        "past_conv_cache_v_0": torch.zeros(1),
        "past_recurrent_state_0": torch.zeros(1),
    }

    model._update_linear_cache(
        merged_state,
        {
            "conv_cache_out_0": torch.ones(1),
            "recurrent_state_out_0": torch.full((1,), 2.0),
        },
    )
    model._update_linear_cache(
        split_state,
        {
            "conv_cache_out_q_0": torch.full((1,), 3.0),
            "conv_cache_out_k_0": torch.full((1,), 4.0),
            "conv_cache_out_v_0": torch.full((1,), 5.0),
            "recurrent_state_out_0": torch.full((1,), 6.0),
        },
    )

    assert merged_state["past_conv_cache_0"].item() == 1.0
    assert merged_state["past_recurrent_state_0"].item() == 2.0
    assert split_state["past_conv_cache_q_0"].item() == 3.0
    assert split_state["past_conv_cache_k_0"].item() == 4.0
    assert split_state["past_conv_cache_v_0"].item() == 5.0
    assert split_state["past_recurrent_state_0"].item() == 6.0


class _FakeSession:
    def __init__(self, names):
        self._names = names

    def get_input_names(self):
        return list(self._names)

    def get_input(self, name):
        return SimpleNamespace(shape=(1,), dtype=torch.float16)


def test_qwen3_next_runtime_allocates_split_and_merged_conv_cache_inputs():
    merged_cache = _alloc_cache_inputs(
        _FakeSession(["past_conv_cache_0", "past_recurrent_state_0"]),
        torch.device("cpu"),
    )
    split_cache = _alloc_cache_inputs(
        _FakeSession(["past_conv_cache_q_0", "past_conv_cache_k_0", "past_conv_cache_v_0"]),
        torch.device("cpu"),
    )

    assert set(merged_cache) == {"past_conv_cache_0", "past_recurrent_state_0"}
    assert set(split_cache) == {"past_conv_cache_q_0", "past_conv_cache_k_0", "past_conv_cache_v_0"}


def test_qwen3_next_export_meta_records_split_and_merged_conv_cache_shapes():
    module = _load_export_script()

    def make_cfg(split_conv_cache: bool):
        return SimpleNamespace(
            target_device="XH2a",
            dtype="float16",
            model=SimpleNamespace(
                wrap_cfg=SimpleNamespace(
                    split_conv_cache=split_conv_cache,
                    normalize_force_fp32=False,
                    use_manual_depthwise_conv1d=False,
                    fuse_gdr_ops=False,
                    max_sequence_length=512,
                    num_logits_to_keep=1,
                )
            ),
        )

    args = SimpleNamespace(hf_model_dir="weights/Qwen3-Next-80B-A3B-Instruct")
    split_meta_info = module.ConfigDict(
        dict(
            create_time="now",
            source_quant_method="gptq",
            prefill_onnx_file="prefill.onnx",
            decode_onnx_file="decode.onnx",
            hf_config="hf_config",
            token_embedding_file="token_embedding.pt",
            pad_token_id=151645,
            kv_cache_shape=[1, 1, 128, 128],
            num_full_attention_layers=2,
            conv_cache_shapes=[[1, 2048, 4], [1, 2048, 4], [1, 4096, 4]],
            recurrent_state_shape=[1, 32, 64, 128],
            num_linear_attention_layers=1,
            linear_cache_layers=[
                dict(
                    layer_idx=0,
                    conv_shapes=[[1, 2048, 4], [1, 2048, 4], [1, 4096, 4]],
                    recurrent_shape=[1, 32, 64, 128],
                )
            ],
        )
    )
    split_meta = module._build_normalized_meta(split_meta_info, make_cfg(True), args)

    assert split_meta["split_conv_cache"] is True
    assert split_meta["wrap_cfg"]["split_conv_cache"] is True
    assert split_meta["linear_cache"]["conv_shapes"] == [[1, 2048, 4], [1, 2048, 4], [1, 4096, 4]]
    assert "conv_shape" not in split_meta["linear_cache"]
    assert split_meta["linear_cache"]["layers"][0]["conv_shapes"][2] == [1, 4096, 4]

    merged_meta_info = module.ConfigDict(
        dict(
            create_time="now",
            source_quant_method="gptq",
            prefill_onnx_file="prefill.onnx",
            decode_onnx_file="decode.onnx",
            hf_config="hf_config",
            token_embedding_file="token_embedding.pt",
            pad_token_id=151645,
            kv_cache_shape=[1, 1, 128, 128],
            num_full_attention_layers=2,
            conv_cache_shape=[1, 8192, 4],
            recurrent_state_shape=[1, 32, 64, 128],
            num_linear_attention_layers=1,
        )
    )
    merged_meta = module._build_normalized_meta(merged_meta_info, make_cfg(False), args)

    assert merged_meta["split_conv_cache"] is False
    assert merged_meta["wrap_cfg"]["split_conv_cache"] is False
    assert merged_meta["linear_cache"]["conv_shape"] == [1, 8192, 4]
    assert "conv_shapes" not in merged_meta["linear_cache"]
