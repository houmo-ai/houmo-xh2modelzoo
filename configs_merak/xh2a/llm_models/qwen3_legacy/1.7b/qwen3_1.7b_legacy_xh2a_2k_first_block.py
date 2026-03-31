_base_ = [
    "./qwen3_1.7b_legacy_xh2a_2k.py",
]

model = dict(
    model_name="xh2_Qwen3-1.7B_first_block_w4a8_256_2k",
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
        nodes=dict(
            lm_head=dict(
                # quant_type="w8a8h1_sefp",
                w_scheme=dict(
                    bits=8,
                    fp_mode="sefp",
                ),
                act_scheme=dict(
                    bits=8,
                    fp_mode="sefp",
                ),
            )
        ),
        ops={},
    ),
    # 内部调试参数
    only_first_block=True,  # 仅包裹第一层，调试用
)
