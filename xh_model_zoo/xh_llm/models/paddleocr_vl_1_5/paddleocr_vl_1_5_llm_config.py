_base_ = ["../_base_/xh2a_base.py"]
# _base_ =["../_base_/yuehui_base.py"]
frontend_type = "TorchFX"

quant_config = dict(
    inputs=dict(
        inputs_embeds=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="float16"),
            )
        ),
        time_position_ids=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="float16"),
            )
        ),
        hight_position_ids=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="float16"),
            )
        ),
        width_position_ids=dict(
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
        position_ids=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="int32"),
            )
        ),
    ),
    w_schema=dict(
        bits=8,
        fp_mode="sefp",
    ),
    act_schema=dict(
        bits=16,
        fp_mode="sefp",
    ),
    nodes_cfg=dict(
        lm_head=dict(
            w_schema=dict(
                bits=8,
                fp_mode="sefp",
            ),
            act_schema=dict(
                bits=16,
                fp_mode="sefp",
            ),
        ),
    ),
)

model = dict(
    type="XHPaddleOCRVLLLMModel",
    wrap_cfg=dict(
        max_sequence_length=2048,
        max_pe_length=32768,
        input_sequence_length=256,
        patch_size=14,
        use_cache=True,
        num_logits_to_keep=1,
        kv_cache=dict(
            cache_axis=2,
        ),
        # bfp_flash_attention=dict(
        #     enable=True,
        #     sefp_manbit=8,
        #     out_fp_manbit=8,
        #     out_fp_expbit=5,
        # ),
    ),
    quant_config=quant_config,
    frontend_type=frontend_type,
    export_cfg=dict(
        input_names=[
            "inputs_embeds",
            "time_position_ids",
            "hight_position_ids",
            "width_position_ids",
            "past_seq_length",
            "current_input_length",
        ],
        output_names=[
            "logits",
        ],
    ),
)
