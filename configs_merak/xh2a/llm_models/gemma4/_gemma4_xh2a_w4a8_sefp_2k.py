_base_ = [
    "./_gemma4_xh2a_2k.py",
]

model = dict(
    model_name="xh2_gemma4_31b_w4a8_sefp",
    quant_scheme=dict(
        _delete_=True,
        w_scheme=dict(
            bits=4,
            fp_mode="sefp",
        ),
        act_scheme=dict(
            bits=8,
            fp_mode="sefp",
        ),
        nodes=dict(
            lm_head=dict(
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
)
