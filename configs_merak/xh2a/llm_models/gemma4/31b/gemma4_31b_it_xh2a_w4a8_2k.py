_base_ = [
    "../_gemma4_xh2a_w4a8_2k.py",
]

hf_model_dir = "./weights/gemma-4-31B-it"

model = dict(
    model_type="Gemma4ForConditionalGeneration",
    hf_model=hf_model_dir,
    model_name="xh2_gemma4_31b_it_w4a8_256_2k",
    visual_config=dict(
        image_seq_length=280,
        patch_size=16,
        pooling_kernel_size=3,
        quant_scheme=dict(
            quant_type="w8a8h1_sefp",
            ops={},
        ),
    ),
)
