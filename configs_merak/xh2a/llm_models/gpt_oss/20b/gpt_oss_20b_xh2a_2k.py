_base_ = [
    "../_gpt_oss_xh2a_2k.py",
]

hf_model_dir = "./data/models/gpt-oss-20b-bfloat16"
# hf_model_dir = "/data02/datasets/chuyuan.wei/gpt-oss-20b-BF16/"

model = dict(
    hf_model=hf_model_dir,
    model_name="xh2_gpt-oss-20b_w8a8_256_2k",
    num_experts_per_tok=4,
)

