_base_ = [
    "../../base_xh2a.py",
]
hf_model_dir = "./weights/MiniCPM-o-2_6"  # 模型路径
quant_config = dict(
    inputs=dict(
        input_features=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="float16"),
            )
        ),
        audio_attention_mask=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="float16"),
            )
        ),
    )
)
model = dict(
    type="XHMiniCPMOAudioModel",
    hf_model=hf_model_dir,
    frontend_type="TorchFX",
    wrap_cfg=dict(
        image_slice_max_size=[40, 40],  # 切片图像最大尺寸
    ),  # wrap模型时，需要传入的配置参数
    quant_config=quant_config,
    export_cfg=dict(
        input_names=[
            "input_features",
            "attention_mask",
        ],
        output_names=["audio_embeddings"],
    ),
)
