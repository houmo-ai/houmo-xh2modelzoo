from pathlib import Path
from types import SimpleNamespace

from xhmodel_merak.xh_llm.models.deepseek_v4 import deepseek_v4_model


def test_deepseek_v4_finalizes_both_graphs_before_recording_md5(tmp_path, monkeypatch):
    prefill = tmp_path / "prefill/model.onnx"
    decode = tmp_path / "decode/model.onnx"
    prefill.parent.mkdir()
    decode.parent.mkdir()
    prefill.write_bytes(b"prefill")
    decode.write_bytes(b"decode")
    finalized = []

    def fake_canonicalize(path: str | Path, logger=None):
        del logger
        path = Path(path)
        finalized.append(path)
        path.write_bytes(path.read_bytes() + b"-canonical")
        return {"model": str(path)}

    monkeypatch.setattr(deepseek_v4_model, "canonicalize_hmonnx_artifact", fake_canonicalize)
    exported_info = SimpleNamespace(
        exported_dir=str(tmp_path),
        meta=SimpleNamespace(
            prefill_hmonnx="prefill/model.onnx",
            decode_hmonnx="decode/model.onnx",
            prefill_hmonnx_md5="stale",
            decode_hmonnx_md5="stale",
        ),
    )

    deepseek_v4_model.XHDeepSeekV4Model._finalize_deterministic_hmonnx(exported_info)

    assert finalized == [prefill, decode]
    assert exported_info.meta.prefill_hmonnx_md5 == deepseek_v4_model.calculate_file_md5(str(prefill))
    assert exported_info.meta.decode_hmonnx_md5 == deepseek_v4_model.calculate_file_md5(str(decode))
    assert exported_info.meta.prefill_hmonnx_md5 != "stale"
    assert exported_info.meta.decode_hmonnx_md5 != "stale"
