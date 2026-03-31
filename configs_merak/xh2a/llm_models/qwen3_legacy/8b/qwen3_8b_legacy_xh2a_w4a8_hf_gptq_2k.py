_base_ = [
    "./qwen3_8b_legacy_xh2a_2k.py",
]
hf_model_dir = "./data/models/Qwen3-8B-GPTQ-Int4"  # 模型路径
model = dict(
    model_name="xh2_Qwen3-8B-GPTQ-Int4_w4a8_256_2k",
    hf_model=hf_model_dir,
    quant_scheme=dict(
        # quant_type="w4a8_ssfp",  # Node默认量化类型
        _delete_=True,
        w_scheme=dict(
            bits=4,
            fp_mode="ssfp",
        ),
        act_scheme=dict(
            bits=8,
            fp_mode="sefp",
        ),
        # nodes=dict(
        #     lm_head=dict(
        #         # quant_type="w8a8h1_sefp",
        #         w_scheme=dict(
        #             bits=8,
        #             fp_mode="sefp",
        #         ),
        #         act_scheme=dict(
        #             bits=8,
        #             fp_mode="sefp",
        #         ),
        #     )
        # ),
        ops={},
    ),
)
