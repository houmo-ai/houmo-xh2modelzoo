_base_ = [
    "../../base_xh2a.py",
]
hf_model_dir = "./weights/MiniCPM-o-2_6"  # 模型路径
quant_config = dict(
    inputs=dict(
        pixel_values=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="float16"),
            )
        ),
        attention_mask=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="int32"),
            )
        ),
        position_ids=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="int32"),
            )
        ),
        resampler_pos_embed=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="float16"),
            )
        ),
        resampler_key_padding_mask=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="int32"),
            )
        ),
        # tgt_sizes=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="int32"),
        #     )
        # ),
    )
)
model = dict(
    type="XHMiniCPMOVisionModel",
    hf_model=hf_model_dir,
    frontend_type="TorchFX",
    wrap_cfg=dict(
        image_slice_max_size=[40, 40],  # 切片图像最大尺寸
    ),  # wrap模型时，需要传入的配置参数
    quant_config=quant_config,
    export_cfg=dict(
        input_names=[
            "pixel_values",
            "attention_mask",
            "position_ids",
            "resampler_pos_embed",
            "resampler_key_padding_mask",
        ],
        output_names=["features"],
    ),
)
