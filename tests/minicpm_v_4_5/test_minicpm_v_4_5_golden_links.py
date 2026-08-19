from __future__ import annotations

from pathlib import Path


def test_link_graph_files_into_steps_uses_resolvable_relative_links(tmp_path: Path):
    from xhmodel_merak.xh_llm.models.minicpm_v_4_5.inference import _link_graph_files_into_steps

    (tmp_path / "prefill" / "step_0").mkdir(parents=True)
    (tmp_path / "vision" / "step_0").mkdir(parents=True)
    (tmp_path / "decode" / "step_0").mkdir(parents=True)

    (tmp_path / "prefill" / "model.onnx").write_bytes(b"onnx")
    (tmp_path / "prefill" / "model_external_data").write_bytes(b"weights")
    (tmp_path / "vision" / "vision_with_act.onnx").write_bytes(b"onnx")
    # HMONNX renames the graph but preserves this pre-rename location.
    (tmp_path / "vision" / "vision_external_data").write_bytes(b"weights")
    (tmp_path / "decode" / "decode.onnx").write_bytes(b"onnx")

    meta = {
        "vision": {"hmonnx": "vision/vision_with_act.onnx"},
        "llm": {
            "prefill_hmonnx": "prefill/model.onnx",
            "decode_hmonnx": "decode/decode.onnx",
        },
    }
    _link_graph_files_into_steps(tmp_path, meta)

    expected = {
        tmp_path / "prefill" / "step_0" / "model.onnx": "../model.onnx",
        tmp_path / "prefill" / "step_0" / "model_external_data": "../model_external_data",
        tmp_path / "vision" / "step_0" / "vision_with_act.onnx": "../vision_with_act.onnx",
        tmp_path / "vision" / "step_0" / "vision_external_data": "../vision_external_data",
        tmp_path / "decode" / "step_0" / "decode.onnx": "../decode.onnx",
    }
    for link, target in expected.items():
        assert link.is_symlink()
        assert link.readlink() == Path(target)
        assert link.resolve(strict=True).is_file()


def test_link_graph_files_repairs_dangling_existing_link(tmp_path: Path):
    from xhmodel_merak.xh_llm.models.minicpm_v_4_5.inference import _link_graph_files_into_steps

    graph_dir = tmp_path / "prefill"
    step_dir = graph_dir / "step_0"
    step_dir.mkdir(parents=True)
    graph = graph_dir / "model.onnx"
    graph.write_bytes(b"onnx")
    (step_dir / graph.name).symlink_to("model.onnx")

    _link_graph_files_into_steps(
        tmp_path,
        {"llm": {"prefill_hmonnx": "prefill/model.onnx", "decode_hmonnx": "prefill/model.onnx"}},
    )

    link = step_dir / graph.name
    assert link.readlink() == Path("../model.onnx")
    assert link.resolve(strict=True) == graph
