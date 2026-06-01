_base_ = [
    "../../base_xh2a.py",
]
hf_model_dir = "./weights/MiniCPM-o-2_6"  # 模型路径
quant_config = dict(
    inputs=dict(
        indices=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="int32"),
            )
        ),
    )
)
model = dict(
    type="XHMiniCPMOTTSDVAEModel",
    hf_model=hf_model_dir,
    frontend_type="TorchFX",
    quant_config=quant_config,
    wrap_cfg=dict(
        batch_size=1,
        max_sequence_length=512,
        input_sequence_length=12,
        use_cache=True,
        num_logits_to_keep=1,
        kv_cache=dict(
            cache_axis=2,
        ),
        image_slice_max_size=[40, 40], 
    ),  
    export_cfg=dict(
        input_names=[
            "indices",
        ],
        output_names=["outputs"],
    ),
)
