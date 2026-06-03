_base_ = [
    "./qwen3_5_moe_122b_a10b_instruct_xh2a_2k.py",
]

hf_model_dir = "data/models/Qwen3.5-122B-A10B-int4-AutoRound_4layers"

model = dict(
    hf_model=hf_model_dir,
    model_name="xh2_Qwen3.5-122B-A10B_4layers_w4a8_256_2k",
)
