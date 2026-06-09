_base_ = [
    "../../../_base_/xh2a_base.py",
]

hf_model_dir = "/data02/datasets/MinerU2.5-Pro-2604-1.2B"

model = dict(
    model_type="Qwen2VLForConditionalGeneration_visual",
    hf_model=hf_model_dir,
    model_name="xh2_mineru2_5_pro_vision_1_2b_w8a8",
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
    max_size_w=1036,
    max_size_h=1036,
    patch_size=14,
)
