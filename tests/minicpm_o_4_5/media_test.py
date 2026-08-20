from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def test_media_preserves_frame_audio_stacked_order(monkeypatch) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import media

    frame_0 = Image.new("RGB", (2, 2), color="red")
    frame_1 = Image.new("RGB", (2, 2), color="green")
    audio_0 = np.array([0.1], dtype=np.float32)
    audio_1 = np.array([0.2], dtype=np.float32)
    stacked = Image.new("RGB", (2, 2), color="blue")
    monkeypatch.setattr(
        media,
        "_load_official_video_helper",
        lambda: lambda *args, **kwargs: ([frame_0, frame_1], [audio_0, audio_1], [stacked, None]),
    )

    output = media.normalize_minicpmo_video(Path("fixture.mp4"), include_audio=True, stack_frames=2)

    assert output == [frame_0, audio_0, stacked, frame_1, audio_1]


def test_media_does_not_insert_silence_when_audio_is_absent(monkeypatch) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import media

    frame = Image.new("RGB", (2, 2), color="white")
    monkeypatch.setattr(
        media,
        "_load_official_video_helper",
        lambda: lambda *args, **kwargs: ([frame], None, None),
    )

    assert media.normalize_minicpmo_video(Path("fixture.mp4"), include_audio=True, stack_frames=1) == [frame]


def test_new_runtime_package_has_no_legacy_runtime_dependency() -> None:
    package = Path("xhmodel_merak/xh_llm/models/minicpm_o_4_5")
    forbidden = ("xhquant_llm", "xh_model_zoo", "spec_from_file_location", "importlib.util")

    matches = {
        token: str(path)
        for path in package.glob("*.py")
        for token in forbidden
        if token in path.read_text(encoding="utf-8")
    }

    assert matches == {}
