_base_ = [
    "../_qwen3_5_moe_xh2a_2k.py",
]

hf_model_dir = "weights/Qwen3.6-35B-A3B"

model = dict(
    hf_model=hf_model_dir,
    model_name="xh2_Qwen3.6-35B-A3B_w8a8_256_2k_1_layer",
    visual_config=dict(
        max_size_w=448,
        max_size_h=448,
        quant_scheme=dict(
            quant_type="w8a8h0_ssfp",
            ops={},
        ),
    ),
    only_first_block=True,
)
