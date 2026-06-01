_base_ = [
    "./qwen2_7b_xh2a_4k.py",
]
hf_model_dir = "./data/models/Qwen2-7B-Instruct"  # 模型路径

model = dict(
    hf_model=hf_model_dir,
)  # wrap模型时，需要传入的配置参数
