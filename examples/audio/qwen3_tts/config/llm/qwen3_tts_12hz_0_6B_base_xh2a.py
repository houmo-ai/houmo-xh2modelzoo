_base_ = [
    "../../../../configs/_base_/xh2a_base.py",
]
hf_model_dir = "./data/models/Qwen3-TTS-12HZ_0.6B-Base"  # 模型路径

model = dict(
    hf_model=hf_model_dir,
)
