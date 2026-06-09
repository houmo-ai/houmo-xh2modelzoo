_base_ = [
    "./qwen3_tts_12hz_model_xh2a.py",
]
from config.llm._components import CODE_PREDICTOR

model = CODE_PREDICTOR
