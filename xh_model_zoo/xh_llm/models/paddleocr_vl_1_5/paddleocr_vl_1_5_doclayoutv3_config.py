_base_ = ["../_base_/xh2a_base.py"]

input_shapes = {
    "im_shape": [1, 2],
    "image": [1, 3, 800, 800],
    "scale_factor": [1, 2],
}

quant_config = dict(
    inputs=dict(
        im_shape=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="float16"),
            )
        ),
        image=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="float16"),
            )
        ),
        scale_factor=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="float16"),
            )
        ),
    ),
    w_schema=dict(
        bits=8,
        fp_mode="sefp",
    ),
    act_schema=dict(
        bits=8,
        fp_mode="sefp",
    ),
    ops_cfg=dict(
        MatMul=dict(
            act_schema=dict(
                bits=16,
                fp_mode="sefp",
            ),
            w_schema=dict(
                bits=16,
                fp_mode="sefp",
            ),
        ),
    ),
)

model = dict(
    type="PP-DocLayoutV3",
    wrap_cfg=dict(
        input_size=800,
        num_queries=300,
    ),
    quant_config=quant_config,
)
