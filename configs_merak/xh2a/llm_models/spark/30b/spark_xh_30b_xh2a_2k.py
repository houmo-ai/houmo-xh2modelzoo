_base_ = [
    "../_spark_xh_xh2a_2k.py",
]
hf_model_dir = "./data/models/ipt_30b"

model = dict(
    hf_model=hf_model_dir,
    model_name="xh2_ipt-30b_w8a8_256_2k",
)
