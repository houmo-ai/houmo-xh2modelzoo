from .hf_compatible import (
    MiniCPMOHFCompatible,
    ensure_tts_sampling_config,
    patch_audio_attention_return_compat,
    patch_dynamic_cache_legacy_methods,
    patch_dynamic_cache_seen_tokens,
    patch_empty_audio_cache,
    patch_remote_cache_helpers,
    patch_speech_generation_capture,
    patch_tts_cache_position_compat,
)
from .media import normalize_minicpmo_video
from .model import (
    MiniCPMO45AudioModel,
    MiniCPMO45LLMModel,
    MiniCPMO45Model,
    MiniCPMO45TTSModel,
    MiniCPMO45VisionModel,
)
from .runtime import (
    STREAMING_CASE_DUPLEX_AUDIO_REPLY,
    STREAMING_CASE_DUPLEX_AUDIO_TEXT,
    STREAMING_CASE_DUPLEX_OMNI_REPLY,
    STREAMING_CASE_SESSION_AUDIO_REPLY,
    STREAMING_CASE_SESSION_AUDIO_TEXT,
    STREAMING_CASES,
    MiniCPMO45HMONNXRuntime,
)
from .runtime_audio import MiniCPMO45AudioHMONNXRuntime
from .runtime_llm import MiniCPMO45LLMHMONNXRuntime
from .runtime_token2wav import MiniCPMO45Token2WavHMONNXRuntime, Token2WavExecutionReport
from .runtime_tts import MiniCPMO45TTSHMONNXRuntime
from .runtime_vision import MiniCPMO45VisionHMONNXRuntime


__all__ = [
    "MiniCPMO45AudioHMONNXRuntime",
    "MiniCPMO45AudioModel",
    "MiniCPMO45HMONNXRuntime",
    "MiniCPMO45LLMHMONNXRuntime",
    "MiniCPMO45LLMModel",
    "MiniCPMO45Model",
    "MiniCPMO45TTSHMONNXRuntime",
    "MiniCPMO45TTSModel",
    "MiniCPMO45Token2WavHMONNXRuntime",
    "MiniCPMO45VisionHMONNXRuntime",
    "MiniCPMO45VisionModel",
    "MiniCPMOHFCompatible",
    "STREAMING_CASES",
    "STREAMING_CASE_DUPLEX_AUDIO_REPLY",
    "STREAMING_CASE_DUPLEX_AUDIO_TEXT",
    "STREAMING_CASE_DUPLEX_OMNI_REPLY",
    "STREAMING_CASE_SESSION_AUDIO_REPLY",
    "STREAMING_CASE_SESSION_AUDIO_TEXT",
    "Token2WavExecutionReport",
    "ensure_tts_sampling_config",
    "normalize_minicpmo_video",
    "patch_audio_attention_return_compat",
    "patch_dynamic_cache_legacy_methods",
    "patch_dynamic_cache_seen_tokens",
    "patch_empty_audio_cache",
    "patch_remote_cache_helpers",
    "patch_speech_generation_capture",
    "patch_tts_cache_position_compat",
]
