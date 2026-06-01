_base_ = [
    "../_qwen3_moe_xh2a_2k.py",
]
hf_model_dir = "./data/models/Qwen3-30B-A3B"

model = dict(
    hf_model=hf_model_dir,
    model_name="xh2_qwen3-30b-a3b_w8a8_256_2k",
)
