_base_ = [
    "./qwen3_5_moe_35b_a3b_instruct_xh2a_2k.py",
]

hf_model_dir = "data/models/Qwen3.5-35B-A3B-int4-AutoRound"

model = dict(
    hf_model=hf_model_dir,
    model_name="xh2_Qwen3.5-35B-A3B_w4a8_256_2k",
)
