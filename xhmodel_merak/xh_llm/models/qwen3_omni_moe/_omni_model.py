from ._audio_model import register_wrap_modules as register_audio_wrap_modules
from ._talker_model import register_wrap_modules as register_talker_wrap_modules
from ._talker_prediction import register_wrap_modules as register_talker_prediction_wrap_modules
from ._text_model import register_wrap_modules as register_text_wrap_modules
from ._vision_model import register_wrap_modules as register_vision_wrap_modules


def register_wrap_modules():
    register_text_wrap_modules()
    register_vision_wrap_modules()
    register_audio_wrap_modules()
    register_talker_wrap_modules()
    register_talker_prediction_wrap_modules()
