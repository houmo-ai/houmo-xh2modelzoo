target_device = "XH2a"
quant_config = dict(
    w_scheme=dict(
        bits=8,
        fp_mode="sefp",
    ),
    act_scheme=dict(
        bits=8,
        fp_mode="sefp",
    ),
)
