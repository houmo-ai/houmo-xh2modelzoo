from pathlib import Path

import numpy as np

from xhmodel_merak.xh_llm.models.gemma4_series.workflow import Gemma4SeriesWorkflow


def _write_step(root: Path, module: str, step: int, valid: int, current: int) -> None:
    step_dir = root / module / f"step_{step}"
    step_dir.mkdir(parents=True, exist_ok=True)
    np.save(step_dir / "hmquant_prefill_valid_length_input.npy", np.array([valid], dtype=np.int32))
    np.save(step_dir / "hmquant_prefill_current_length_input.npy", np.array([current], dtype=np.int32))
    np.save(step_dir / "dummy.npy", np.array([1], dtype=np.int32))


def _write_decode(root: Path) -> None:
    step_dir = root / "decode" / "step_0"
    step_dir.mkdir(parents=True, exist_ok=True)
    np.save(step_dir / "dummy.npy", np.array([1], dtype=np.int32))


def test_short_text_golden_is_not_considered_long_context(tmp_path):
    _write_step(tmp_path, "prefill", 0, valid=0, current=74)
    _write_step(tmp_path, "prefill", 1, valid=256, current=24)
    _write_decode(tmp_path)

    assert not Gemma4SeriesWorkflow._text_golden_covers_long_context(tmp_path)


def test_long_text_golden_covers_slice_window_boundary(tmp_path):
    _write_step(tmp_path, "prefill", 0, valid=0, current=256)
    _write_step(tmp_path, "prefill", 1, valid=1024, current=256)
    _write_decode(tmp_path)

    assert Gemma4SeriesWorkflow._text_golden_covers_long_context(tmp_path)


def test_build_cases_replaces_short_text_golden(tmp_path):
    _write_step(tmp_path, "prefill", 0, valid=0, current=74)
    _write_decode(tmp_path)
    workflow = Gemma4SeriesWorkflow.__new__(Gemma4SeriesWorkflow)

    cases = workflow._build_golden_message_cases(
        {},
        "这是一个超过长上下文边界的真实问题占位。",
        golden_root=tmp_path,
    )

    assert cases == [("text", [{"role": "user", "content": "这是一个超过长上下文边界的真实问题占位。"}])]
    assert (tmp_path / "prefill").exists()
    assert not any((tmp_path / "prefill").glob("step_*"))
    assert (tmp_path / "decode").exists()
    assert not any((tmp_path / "decode").glob("step_*"))


def test_normalize_module_step_dirs_removes_even_step_gaps(tmp_path):
    module = tmp_path / "prefill"
    for step in (0, 2, 4):
        (module / f"step_{step}").mkdir(parents=True)

    Gemma4SeriesWorkflow._normalize_module_step_dirs(module)

    assert sorted(path.name for path in module.glob("step_*")) == ["step_0", "step_1", "step_2"]
