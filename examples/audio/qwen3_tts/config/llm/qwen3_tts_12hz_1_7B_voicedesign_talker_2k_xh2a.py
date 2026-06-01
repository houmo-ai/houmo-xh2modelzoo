_base_ = [
    "./qwen3_tts_12hz_1_7B_voicedesign_xh2a.py",
]
frontend_type = "TorchFX"
# hf_model_dir = "./data/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign/"  # 模型路径

quant_config = dict()

model = dict(
    type="XHQwen3TTSTalker",
    # hf_model=hf_model_dir,
    wrap_cfg=dict(
        max_sequence_length=2048,
        input_sequence_length=256,
        use_cache=True,
        num_logits_to_keep=1,
        kv_cache=dict(
            cache_axis=2,
        ),
        enable_rope=True,
    ),  # wrap模型时，需要传入的配置参数
    quant_config=quant_config,
    frontend_type=frontend_type,
    export_cfg=dict(
        input_names=[
            "inputs_embeds",
            "past_seq_length",
            "current_input_length",
        ],
        output_names=[
            "logits",
            "past_hidden",
        ],
    ),
)
