_base_ = [
    "../../_base_/xh2a_base.py",
]
frontend_type = "TorchFX"
hf_model_dir = "./data/models/Qwen2-7B"  # 模型路径

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
        # past_key_cache_0=dict(quantizer=dict(qspec=dict(fake_dtype="float16"),)),
        # past_value_cache_0=dict(quantizer=dict(qspec=dict(fake_dtype="float16"),)),
        # past_key_cache_0=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_value_cache_0=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_key_cache_1=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_value_cache_1=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_key_cache_2=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_value_cache_2=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_key_cache_3=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_value_cache_3=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_key_cache_4=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_value_cache_4=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_key_cache_5=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_value_cache_5=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_key_cache_6=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_value_cache_6=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_key_cache_7=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_value_cache_7=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_key_cache_8=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_value_cache_8=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_key_cache_9=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_value_cache_9=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_key_cache_10=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_value_cache_10=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_key_cache_11=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_value_cache_11=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_key_cache_12=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_value_cache_12=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_key_cache_13=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_value_cache_13=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_key_cache_14=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_value_cache_14=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_key_cache_15=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_value_cache_15=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_key_cache_16=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_value_cache_16=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_key_cache_17=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_value_cache_17=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_key_cache_18=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_value_cache_18=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_key_cache_19=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_value_cache_19=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_key_cache_20=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_value_cache_20=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_key_cache_21=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_value_cache_21=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_key_cache_22=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_value_cache_22=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_key_cache_23=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_value_cache_23=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_key_cache_24=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_value_cache_24=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_key_cache_25=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_value_cache_25=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_key_cache_26=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_value_cache_26=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_key_cache_27=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
        # past_value_cache_27=dict(
        #     quantizer=dict(
        #         qspec=dict(fake_dtype="float16"),
        #     )
        # ),
    )
)

model = dict(
    type="XHQwen2LegacyModel",
    hf_model=hf_model_dir,
    wrap_cfg=dict(
        max_sequence_length=4096,
        input_sequence_length=256,
        use_cache=True,
        num_logits_to_keep=1,
        kv_cache=dict(
            cache_axis=2,
        ),
    ),  # wrap模型时，需要传入的配置参数
    quant_config=quant_config,
    frontend_type=frontend_type,
    export_cfg=dict(
        input_names=[
            "inputs_embeds",
            "past_seq_length",
            "current_input_length",
            # "past_key_cache_0",
            # "past_key_cache_1",
            # "past_key_cache_2",
            # "past_key_cache_3",
            # "past_key_cache_4",
            # "past_key_cache_5",
            # "past_key_cache_6",
            # "past_key_cache_7",
            # "past_key_cache_8",
            # "past_key_cache_9",
            # "past_key_cache_10",
            # "past_key_cache_11",
            # "past_key_cache_12",
            # "past_key_cache_13",
            # "past_key_cache_14",
            # "past_key_cache_15",
            # "past_key_cache_16",
            # "past_key_cache_17",
            # "past_key_cache_18",
            # "past_key_cache_19",
            # "past_key_cache_20",
            # "past_key_cache_21",
            # "past_key_cache_22",
            # "past_key_cache_23",
            # "past_key_cache_24",
            # "past_key_cache_25",
            # "past_key_cache_26",
            # "past_key_cache_27",
            # "past_value_cache_0",
            # "past_value_cache_1",
            # "past_value_cache_2",
            # "past_value_cache_3",
            # "past_value_cache_4",
            # "past_value_cache_5",
            # "past_value_cache_6",
            # "past_value_cache_7",
            # "past_value_cache_8",
            # "past_value_cache_9",
            # "past_value_cache_10",
            # "past_value_cache_11",
            # "past_value_cache_12",
            # "past_value_cache_13",
            # "past_value_cache_14",
            # "past_value_cache_15",
            # "past_value_cache_16",
            # "past_value_cache_17",
            # "past_value_cache_18",
            # "past_value_cache_19",
            # "past_value_cache_20",
            # "past_value_cache_21",
            # "past_value_cache_22",
            # "past_value_cache_23",
            # "past_value_cache_24",
            # "past_value_cache_25",
            # "past_value_cache_26",
            # "past_value_cache_27",
        ],
        output_names=["logits"],
    ),
)
