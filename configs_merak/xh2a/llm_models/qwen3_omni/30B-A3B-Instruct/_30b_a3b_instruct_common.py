_base_ = [
    "../../../_base_/xh2a_base.py",
]

hf_model_dir = "./data/models/Qwen3-Omni-30B-A3B-Instruct/"

model = dict(
    hf_model=hf_model_dir,
)
