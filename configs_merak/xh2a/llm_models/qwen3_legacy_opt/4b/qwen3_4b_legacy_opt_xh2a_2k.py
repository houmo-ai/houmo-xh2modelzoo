_base_ = [
    "../_qwen3_legacy_opt_xh2a_2k.py",
]
hf_model_dir = "./data/models/Qwen3-4B"  # 模型路径

model = dict(
    hf_model=hf_model_dir,
)
