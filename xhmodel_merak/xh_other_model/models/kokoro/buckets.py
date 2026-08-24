from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .host import SAMPLE_RATE, WAVEFORM_SAMPLES_PER_FRAME


TOKEN_BUCKETS = (32, 64, 128, 256)
AUDIO_BUCKET_SECONDS = (4, 8, 16, 32)
LSTM_VARIANTS = ("native", "decomposed")


def audio_seconds_to_frames(seconds: int) -> int:
    seconds = int(seconds)
    if seconds <= 0:
        raise ValueError("audio bucket seconds must be positive")
    samples = seconds * SAMPLE_RATE
    frames, remainder = divmod(samples, WAVEFORM_SAMPLES_PER_FRAME)
    if remainder:
        raise ValueError(
            f"{seconds}s at {SAMPLE_RATE}Hz is not divisible by "
            f"{WAVEFORM_SAMPLES_PER_FRAME} samples/frame"
        )
    return frames


FRAME_BUCKETS = tuple(audio_seconds_to_frames(seconds) for seconds in AUDIO_BUCKET_SECONDS)
_SECONDS_BY_FRAME = dict(zip(FRAME_BUCKETS, AUDIO_BUCKET_SECONDS, strict=True))


@dataclass(frozen=True, order=True)
class KokoroBucketRoute:
    token_max_length: int
    frame_max_length: int

    def __post_init__(self) -> None:
        if self.token_max_length <= 0 or self.frame_max_length <= 0:
            raise ValueError("Kokoro bucket dimensions must be positive")

    @property
    def audio_seconds(self) -> int:
        return audio_seconds_for_frame(self.frame_max_length)

    @property
    def key(self) -> str:
        return f"{token_bucket_key(self.token_max_length)}_{frame_bucket_key(self.frame_max_length)}"

    def as_dict(self) -> dict[str, int | str]:
        return {
            "key": self.key,
            "token_max_length": self.token_max_length,
            "frame_max_length": self.frame_max_length,
            "audio_seconds": self.audio_seconds,
        }


def token_bucket_key(token_max_length: int) -> str:
    token_max_length = int(token_max_length)
    if token_max_length <= 0:
        raise ValueError("token_max_length must be positive")
    return f"t{token_max_length:04d}"


def frame_bucket_key(frame_max_length: int) -> str:
    frame_max_length = int(frame_max_length)
    if frame_max_length <= 0:
        raise ValueError("frame_max_length must be positive")
    return f"f{frame_max_length:04d}"


def audio_seconds_for_frame(frame_max_length: int) -> int:
    try:
        return _SECONDS_BY_FRAME[int(frame_max_length)]
    except KeyError as error:
        raise ValueError(f"F={frame_max_length} is not a preset Kokoro audio bucket") from error


BUCKET_ROUTES = tuple(
    KokoroBucketRoute(token, frame)
    for token, frame in zip(TOKEN_BUCKETS, FRAME_BUCKETS, strict=True)
)


def _normalize_subset(
    values: Iterable[int] | None,
    presets: tuple[int, ...],
    *,
    label: str,
) -> tuple[int, ...]:
    if values is None:
        return presets
    requested = tuple(int(value) for value in values)
    if not requested:
        raise ValueError(f"Kokoro {label} bucket selection must not be empty")
    if len(requested) != len(set(requested)):
        raise ValueError(f"Kokoro {label} buckets must be unique")
    unsupported = sorted(set(requested).difference(presets))
    if unsupported:
        raise ValueError(f"unsupported Kokoro {label} buckets {unsupported}; presets are {list(presets)}")
    requested_set = set(requested)
    return tuple(value for value in presets if value in requested_set)


def normalize_token_buckets(values: Iterable[int] | None) -> tuple[int, ...]:
    return _normalize_subset(values, TOKEN_BUCKETS, label="token")


def normalize_audio_seconds_buckets(values: Iterable[int] | None) -> tuple[int, ...]:
    return _normalize_subset(values, AUDIO_BUCKET_SECONDS, label="audio-seconds")


def normalize_frame_buckets(values: Iterable[int] | None) -> tuple[int, ...]:
    return _normalize_subset(values, FRAME_BUCKETS, label="frame")


def normalize_bucket_routes(
    token_values: Iterable[int] | None,
    audio_seconds_values: Iterable[int] | None,
) -> tuple[KokoroBucketRoute, ...]:
    """Resolve a canonical subset of the four approved T/F capacity gears."""

    requested_tokens = (
        set(normalize_token_buckets(token_values)) if token_values is not None else None
    )
    requested_seconds = (
        set(normalize_audio_seconds_buckets(audio_seconds_values))
        if audio_seconds_values is not None
        else None
    )
    routes = tuple(
        route
        for route in BUCKET_ROUTES
        if (requested_tokens is None or route.token_max_length in requested_tokens)
        and (requested_seconds is None or route.audio_seconds in requested_seconds)
    )
    if not routes:
        raise ValueError(
            "token_buckets and audio_seconds_buckets must select matching Kokoro T/F routes"
        )
    if requested_tokens is not None and {route.token_max_length for route in routes} != requested_tokens:
        raise ValueError("token_buckets and audio_seconds_buckets must select matching Kokoro T/F routes")
    if requested_seconds is not None and {route.audio_seconds for route in routes} != requested_seconds:
        raise ValueError("token_buckets and audio_seconds_buckets must select matching Kokoro T/F routes")
    return routes


__all__ = [
    "AUDIO_BUCKET_SECONDS",
    "BUCKET_ROUTES",
    "FRAME_BUCKETS",
    "KokoroBucketRoute",
    "LSTM_VARIANTS",
    "TOKEN_BUCKETS",
    "audio_seconds_for_frame",
    "audio_seconds_to_frames",
    "frame_bucket_key",
    "normalize_audio_seconds_buckets",
    "normalize_bucket_routes",
    "normalize_frame_buckets",
    "normalize_token_buckets",
    "token_bucket_key",
]
