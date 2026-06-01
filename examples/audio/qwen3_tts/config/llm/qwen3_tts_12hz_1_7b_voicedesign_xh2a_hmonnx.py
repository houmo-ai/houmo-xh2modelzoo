_base_ = [
    "./qwen3_tts_12hz_1_7B_voicedesign_xh2a.py",
]

model = dict(
    type="Qwen3TTSHMONNXInference",
    text_projection=dict(
        type="Qwen3TTSTextProjectionInference",
        onnx_file="work_dirs/qwen3_tts_12hz_1_7B_text_projection_xh2a/hmonnx/text_projection_XH2a.onnx",
    ),
    code_predictor=dict(
        type="Qwen3TTSCodePredictorInference",
        model_cfg="work_dirs/qwen3_tts_12hz_1_7B_voicedesign_code_predictor_2k_xh2a/meta.json",
    ),
    talker=dict(
        type="Qwen3TTSTalkerInference",
        model_cfg="work_dirs/qwen3_tts_12hz_1_7B_voicedesign_talker_2k_xh2a/meta.json",
    ),
    speech_tokenizer=dict(
        type="Qwen3TTSSpeechTokenizerInference",
        model_cfg="work_dirs/qwen3_tts_12hz_1_7B_speech_tokenizer_xh2a/meta.json",
    ),
)
