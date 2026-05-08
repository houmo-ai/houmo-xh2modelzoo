_base_ = [
    "../../../_base_/xh2a_base.py",
]

hf_model_dir = "/data01/datasets/gemma-4-26B-A4B-it"

model = dict(
    model_type="Gemma4ForConditionalGeneration_with_mask",
    hf_model=hf_model_dir,
    fallback_hf_model=hf_model_dir,
    model_name="xh2_gemma4_moe_with_mask_26b_a4b_it_w8a8_256_2k",
    context_max_length=2048,
    prefill_chunk_length=256,
    use_cache=True,
    num_logits_to_keep=1,
    quant_scheme=dict(
        quant_type="w4a8h1_sefp",
        ops={},
    ),
    visual_config=dict(
        model_type="Gemma4ForConditionalGeneration_visual",
        hf_model=hf_model_dir,
        model_name="xh2_gemma4_moe_visual_26b_a4b_it_w8a8",
        max_size_w=448,
        max_size_h=448,
        upsample_token=False,
        fuse_norm=True,
        quant_scheme=dict(
            quant_type="w8a8h1_sefp",
            ops={},
        ),
    ),
)