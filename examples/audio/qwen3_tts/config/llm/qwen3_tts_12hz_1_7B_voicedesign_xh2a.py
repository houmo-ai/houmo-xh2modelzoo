_base_ = [
    "../../../../../configs/_base_/xh2a_base.py",
]
hf_model_dir = "./data/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign"  # 模型路径

model = dict(
    hf_model=hf_model_dir,
)
