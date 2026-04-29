_base_ = [
    "../../../_base_/xh2a_base.py",
]

hf_model_dir = "/data01/datasets/gemma-4-26B-A4B-it"

model = dict(
    model_type="Gemma4ForConditionalGeneration_visual",
    hf_model=hf_model_dir,
    model_name="xh2_gemma4_moe_visual_26b_a4b_it_w8a8",
    quant_scheme=dict(
        quant_type="w8a8h1_sefp",
        ops={},
    ),
    max_size_w=448,
    max_size_h=448,
    upsample_token=False,
    fuse_norm=True,
)