# Parametrized HMONNX full-pipeline config (replaces the base/customvoice/voicedesign trio).
# Kept pure (non-lazy). The work_dirs product paths below are cv placeholders;
# at runtime the demo/eval scripts read --variant and call
# apply_variant_hmonnx(cfg, variant) to override them with paths from
# config.llm._components.WORKNAME (which reproduces the 1.7B naming asymmetry).
_base_ = [
    "./qwen3_tts_12hz_model_xh2a.py",
]

model = dict(
    type="Qwen3TTSHMONNXInference",
    text_projection=dict(
        type="Qwen3TTSTextProjectionInference",
        onnx_file="work_dirs/qwen3_tts_12hz_0_6B_customvoice_text_projection_xh2a/hmonnx/text_projection_XH2a.onnx",
    ),
    code_predictor=dict(
        type="Qwen3TTSCodePredictorInference",
        model_cfg="work_dirs/qwen3_tts_12hz_0_6B_customvoice_code_predictor_2k_xh2a/meta.json",
    ),
    talker=dict(
        type="Qwen3TTSTalkerInference",
        model_cfg="work_dirs/qwen3_tts_12hz_0_6B_customvoice_talker_2k_xh2a/meta.json",
    ),
    speech_tokenizer=dict(
        type="Qwen3TTSSpeechTokenizerInference",
        model_cfg="work_dirs/qwen3_tts_12hz_0_6B_customvoice_speech_tokenizer_xh2a/meta.json",
    ),
)
