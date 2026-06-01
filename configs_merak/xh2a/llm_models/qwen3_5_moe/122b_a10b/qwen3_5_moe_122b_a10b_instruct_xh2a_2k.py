_base_ = [
    "../_qwen3_5_moe_xh2a_2k.py",
]

hf_model_dir = "data/models/Qwen3.5-122B-A10B/"

model = dict(
    hf_model=hf_model_dir,
    model_name="xh2_Qwen3.5-122B-A10B_w8a8_256_2k",
    visual_config=dict(
        max_size_w=448,
        max_size_h=448,
        quant_scheme=dict(
            quant_type="w8a8h1_sefp",
            ops={},
        ),
    ),
)
