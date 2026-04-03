_base_ = [
    "../_gpt_oss_xh2a_2k.py",
]

hf_model_dir = "./data/models/gpt-oss-120b"

model = dict(
    hf_model=hf_model_dir,
    model_name="xh2_gpt-oss-120b_w8a8_256_2k",
)
