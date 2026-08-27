from pathlib import Path

from xhmodel_merak.xh_llm.hmonnx import base_llm_hmonnx_model
from xhmodel_merak.xh_llm.hmonnx.base_llm_hmonnx_model import BaseLLMHMONNXModel


def test_prepare_hmonnx_paths_fuses_before_page_attention(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def materialize(path: str) -> Path:
        calls.append(("linear_fusion", path))
        return Path(f"{path}_fused")

    def lower(path: str) -> str:
        calls.append(("page_attention", path))
        return f"{path}_page"

    monkeypatch.setattr(
        base_llm_hmonnx_model,
        "materialize_parallel_linear_fusion",
        materialize,
    )
    monkeypatch.setattr(
        BaseLLMHMONNXModel,
        "_convert_to_page_attention_hmonnx",
        lower,
    )

    paths = BaseLLMHMONNXModel._prepare_hmonnx_paths(
        "prefill.onnx",
        "decode.onnx",
        enable_parallel_linear_fusion=True,
        enable_page_attention=True,
    )

    assert paths == (
        "prefill.onnx_fused_page",
        "decode.onnx_fused_page",
    )
    assert calls == [
        ("linear_fusion", "prefill.onnx"),
        ("linear_fusion", "decode.onnx"),
        ("page_attention", "prefill.onnx_fused"),
        ("page_attention", "decode.onnx_fused"),
    ]


def test_prepare_hmonnx_paths_keeps_sources_when_disabled() -> None:
    assert BaseLLMHMONNXModel._prepare_hmonnx_paths(
        Path("prefill.onnx"),
        Path("decode.onnx"),
        enable_parallel_linear_fusion=False,
        enable_page_attention=False,
    ) == ("prefill.onnx", "decode.onnx")


def test_page_attention_derivation_replaces_complete_temporary_file(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "decode.onnx"
    source.write_bytes(b"source")
    converted_paths: list[Path] = []

    def convert(input_path: Path, output_path: Path, **kwargs) -> None:
        assert input_path == source
        assert kwargs == {"convert_unfused_sliding_kv_cache": False}
        assert output_path.parent == source.parent
        assert output_path.name.startswith(".decode_page_attention_")
        assert output_path.suffix == ".tmp"
        converted_paths.append(output_path)
        output_path.write_bytes(b"complete converted graph")

    monkeypatch.setattr(base_llm_hmonnx_model, "convert_to_page_attention", convert)

    result = Path(BaseLLMHMONNXModel._convert_to_page_attention_hmonnx(source))

    assert result.parent == source.parent
    assert result.name.startswith("decode_page_attention_")
    assert result.suffix == ".onnx"
    assert result.read_bytes() == b"complete converted graph"
    assert len(converted_paths) == 1
    assert not converted_paths[0].exists()
