import importlib.util
import sys
from pathlib import Path


_INFER_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples_merak"
    / "llm"
    / "ling_3_flash"
    / "infer.py"
)


def _load_infer_module():
    spec = importlib.util.spec_from_file_location("ling_3_flash_infer", _INFER_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_infer_cli_exposes_v2_w4_and_cuda_graph_modes(monkeypatch):
    module = _load_infer_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "infer.py",
            "--meta",
            "/tmp/golden_meta_info.json",
            "--use-v2",
            "--cuda-graph",
            "--raw-prompt",
            "--print-token-ids",
        ],
    )

    args = module.parse_args()

    assert args.use_v2 is True
    assert args.pack_w4 is True
    assert args.cuda_graph is True
    assert args.enable_thinking is False
    assert args.raw_prompt is True
    assert args.print_token_ids is True


def test_infer_cli_can_enable_thinking_explicitly(monkeypatch):
    module = _load_infer_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "infer.py",
            "--meta",
            "/tmp/golden_meta_info.json",
            "--enable-thinking",
        ],
    )

    assert module.parse_args().enable_thinking is True


def test_infer_cli_defaults_to_v2_w4_runtime(monkeypatch):
    module = _load_infer_module()
    monkeypatch.setattr(
        sys,
        "argv",
        ["infer.py", "--meta", "/tmp/golden_meta_info.json"],
    )

    args = module.parse_args()

    assert args.use_v2 is True
    assert args.pack_w4 is True
    assert args.cuda_graph is True


def test_infer_cli_can_explicitly_select_legacy_runtime(monkeypatch):
    module = _load_infer_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "infer.py",
            "--meta",
            "/tmp/golden_meta_info.json",
            "--no-use-v2",
        ],
    )

    assert module.parse_args().use_v2 is False
