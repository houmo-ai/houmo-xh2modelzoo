import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "xh_model_zoo/xh_llm/models/qwen3_5_moe_prune/_moe_model_prune.py"
CONVERTER = ROOT / "xh_model_zoo/xh_llm/models/qwen3_5_moe_prune/qwen3_5_moe_prune_converter.py"
EXPORT = ROOT / "examples/llm/qwen3_5_moe_prune/qwen3_5_moe_prune_xh2a_export_hmonnx.py"


def test_wrapper_uses_merged_moeblock_prune_attributes():
    source = MODEL.read_text()

    assert "from xhquant.nn.modules.moeblock import MoeBlock" in source
    assert "prune_threshold=0.0" in source
    assert "self.moeblock(hidden_states, routing_weights)" in source
    assert "dynamic_prune_threshold" not in source
    assert "PruningRouter" not in source
    assert "from xhquant.nn.modules.moeblock_prune" not in source


def test_converter_sets_moeblock_threshold_attribute():
    source = CONVERTER.read_text()

    assert "module.moeblock.prune_threshold = float(threshold)" in source
    assert "module.moeblock.s_scalar" in source


def test_export_forces_full_network_and_removes_num_blocks_option():
    source = EXPORT.read_text()
    tree = ast.parse(source)

    assert "max_layers=None" in source
    assert "--num-blocks" not in source
    assert tree is not None


def test_golden_embedding_lookup_stays_on_embedding_device():
    source = EXPORT.read_text()

    assert "embedding_device = token_embedding.weight.device" in source
    assert "token_embedding(input_ids.to(embedding_device))" in source
    assert "token_embedding = _load_token_embedding(token_embedding_file).to(dtype)" in source
