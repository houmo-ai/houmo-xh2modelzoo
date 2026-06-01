_base_ = [
    "../../base_xh2a.py",
]
frontend_type = "TorchFX"
hf_model_dir = "./weights/MiniCPM-o-2_6"  # 模型路径

quant_config = dict(
    inputs=dict(
        inputs_embeds=dict(
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
    ),
    w_schema=dict(
        bits=4,
        fp_mode="ssfp",
    ),
    nodes_cfg=dict(
        lm_head=dict(
            w_schema=dict(
                bits=8,
                fp_mode="sefp",
            ),
        ),
    ),
)

model = dict(
    type="XHMiniCPMOLLMModel",
    hf_model=hf_model_dir,
    wrap_cfg=dict(
        max_sequence_length=4096,
        input_sequence_length=256,
        use_cache=True,
        num_logits_to_keep=1,
        kv_cache=dict(
            cache_axis=2,
        ),
        image_slice_max_size = [40, 40]
    ),  
    quant_config=quant_config,
    frontend_type=frontend_type,
    export_cfg=dict(
        input_names=[
            "inputs_embeds",
            "past_seq_length",
            "current_input_length",
        ],
        output_names=["logits", "hidden_states"],
    ),
)
