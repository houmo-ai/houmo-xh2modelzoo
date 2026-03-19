from ._thinker_model import register_wrap_modules as register_thinker_wrap_modules
from ._vision_model import register_wrap_modules as register_vision_wrap_modules
from ._audio_model import register_wrap_modules as register_audio_wrap_modules
from ._talker_model import register_wrap_modules as register_talker_wrap_modules
from ._code2wav_model import register_wrap_modules as register_code2wav_wrap_modules


def register_wrap_modules():
    """Register all Qwen3 Omni module wrappers."""
    register_thinker_wrap_modules()
    register_vision_wrap_modules()
    register_audio_wrap_modules()
    register_talker_wrap_modules()
    register_code2wav_wrap_modules()
    return None
