_base_ = [
    "../../../_base_/xh2a_base.py",
]
hf_model_dir = "/data02/datasets/Qwen2-VL-2B-Instruct"

model = dict(
    model_type="Qwen2VLForConditionalGeneration_visual",
    hf_model=hf_model_dir,
    model_name="xh2_qwen2_vl_vision_2b_w8a8",
    quant_scheme=dict(
        quant_type="w8a8h0_sefp",
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
    max_size_w=448,
    max_size_h=448,
)
