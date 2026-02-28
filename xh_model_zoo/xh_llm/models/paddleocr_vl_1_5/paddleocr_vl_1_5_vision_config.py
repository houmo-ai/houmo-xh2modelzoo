_base_ = ["../_base_/xh2a_base.py"]
trace_type = "TorchFX"

quant_config = dict(
    inputs=dict(
        pixel_values=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="float16"),
            )
        ),
    )
)

model = dict(
    type="XHPaddleOCRVLVisionModel",
    wrap_cfg=dict(
        max_sequence_length=2048,
        temporal_patch_size=2,
        patch_size=14,
    ),
    quant_config=quant_config,
)
