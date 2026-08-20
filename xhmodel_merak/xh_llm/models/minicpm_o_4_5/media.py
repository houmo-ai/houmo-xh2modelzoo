from __future__ import annotations

import math
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any


VideoSegments = tuple[list[Any], list[Any] | None, list[Any] | None]


def _load_official_video_helper() -> Callable[..., VideoSegments]:
    try:
        from minicpmo.utils import get_video_frame_audio_segments
    except (ImportError, OSError) as error:
        if not any(token in str(error).lower() for token in ("shared object", "libav", "decord")):
            raise
        sys.modules.pop("minicpmo.utils", None)
        decord = ModuleType("decord")
        decord.cpu = lambda index: index
        decord.VideoReader = None
        sys.modules["decord"] = decord
        try:
            from minicpmo.utils import get_video_frame_audio_segments
        finally:
            sys.modules.pop("decord", None)

    return get_video_frame_audio_segments


def _moviepy_video_segments(path: Path, include_audio: bool) -> VideoSegments:
    import numpy as np
    from moviepy import VideoFileClip
    from PIL import Image

    with VideoFileClip(str(path)) as video:
        duration = float(video.duration or 0.0)
        count = max(1, math.ceil(duration))
        frames = [
            Image.fromarray(video.get_frame(min(float(index + 1), max(duration - 1e-6, 0.0))).astype(np.uint8))
            for index in range(count)
        ]
        if not include_audio or video.audio is None:
            return frames, None, None
        waveform = video.audio.to_soundarray(fps=16000)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        return frames, [waveform[index * 16000 : (index + 1) * 16000] for index in range(count)], None


def normalize_minicpmo_video(path: Path, *, include_audio: bool, stack_frames: int) -> list[Any]:
    try:
        helper = _load_official_video_helper()
        frames, audio, stacked = helper(
            str(path),
            stack_frames=stack_frames,
            use_ffmpeg=True,
        )
    except (FileNotFoundError, ImportError, OSError):
        frames, audio, stacked = _moviepy_video_segments(path, include_audio)
    contents: list[Any] = []
    for index, frame in enumerate(frames):
        contents.append(frame)
        if include_audio and audio is not None:
            contents.append(audio[index])
        if stacked is not None and index < len(stacked) and stacked[index] is not None:
            contents.append(stacked[index])
    return contents


__all__ = ["normalize_minicpmo_video"]
