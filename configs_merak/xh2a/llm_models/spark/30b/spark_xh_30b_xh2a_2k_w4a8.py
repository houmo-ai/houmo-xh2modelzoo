_base_ = [
    "../_spark_xh_xh2a_2k.py",
]
hf_model_dir = "./data/models/ipt_30b_unfuse"

model = dict(
    hf_model=hf_model_dir,
    model_name="xh2_ipt-30b_w4a8_256_2k",
    quant_scheme=dict(
        quant_type="w4a8h0_ssfp",
        nodes=dict(
            lm_head=dict(
                quant_type="w8a8h1_sefp",
            )
        ),
        ops={},
    ),
)
