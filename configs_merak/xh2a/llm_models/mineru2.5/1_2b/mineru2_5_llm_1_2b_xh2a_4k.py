_base_ = [
    "../../../_base_/xh2a_base.py",
]

hf_model_dir = "/data02/datasets/MinerU2.5-Pro-2604-1.2B"

model = dict(
    model_type="Qwen2VLForConditionalGeneration",
    hf_model=hf_model_dir,
    model_name="xh2_mineru2_5_pro_1_2b_w8a8_256_4k",
    context_max_length=32768,
    prefill_chunk_length=256,
    use_cache=True,
    num_logits_to_keep=1,
    quant_scheme=dict(
        quant_type="w16a16h0_sefp",
        nodes=dict(
            lm_head=dict(
                quant_type="w16a16h0_sefp",
            )
        ),
        # 建议使用16-bit激活值matmul，否则attention会有严重的精度问题
        ops = dict(
                MatMul=dict(
                    act_scheme=dict(
                        bits=16,
                        fp_mode="sefp",
                    ),
                    act_schema_2=dict(
                        bits=16,
                        fp_mode="sefp",
                    ),
                )
        )
    ),
    visual_config=dict(
        max_size_w=1036,
        max_size_h=1036,
        patch_size=14,
        temporal_patch_size=2,
        quant_scheme=dict(
            quant_type="w16a16h0_sefp",
            # 建议使用16-bit激活值matmul，否则attention会有严重的精度问题
            ops = dict(
                MatMul=dict(
                    act_scheme=dict(
                        bits=16,
                        fp_mode="sefp",
                    ),
                    act_schema_2=dict(
                        bits=16,
                        fp_mode="sefp",
                    ),
                )
            )
        ),
    ),
    only_first_block=False,
)
