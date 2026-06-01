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
        past_seq_length=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="int32"),
            )
        ),
        current_input_length=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="int32"),
            )
        ),
        attention_mask=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="float16"),
            )
        ),
    )
)
model = dict(
    type="XHMiniCPMOTTSModel",
    hf_model=hf_model_dir,
    frontend_type="TorchFX",
    quant_config=quant_config,
    wrap_cfg=dict(
        batch_size=1,
        max_sequence_length=2048,
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
            "inputs_embeds",
            "past_seq_length",
            "current_input_length",
            "attention_mask",
        ],
        output_names=["logits"],
    ),
)
