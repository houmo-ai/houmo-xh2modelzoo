_base_ = [
    "../../_base_/xh2a_base.py",
]
frontend_type = "TorchFX"
# hf_model_dir = "/data02/datasets/Qwen3-30B-A3B/"  # 模型路径
hf_model_dir = "/data02/datasets/kimi"  # 模型路径


quant_config = dict(
    w_cfg=dict(
        quantizer=dict(
            qspec=dict(
                fp_mode="ssfp",  # sefp or ssfp
                hidden_bit=False,
                man_bit=4,  # 量化参数的mantissa位数
                # nshare=128,
            ),
        ),
    ),
    i_cfg=dict(
        quantizer=dict(
            qspec=dict(
                fp_mode="sefp",  # sefp or ssfp
                hidden_bit=True,
                man_bit=8,  # 量化参数的mantissa位数
                # nshare=128,
            ),
        ),
    ),
    ops_cfg=dict(RMSNorm=dict(compute_mode="fast"))
)

model = dict(
    type="XHKimiMoeModel",
    hf_model=hf_model_dir,
    wrap_cfg=dict(
        batch_size=1,
        # only_first_block=True,
        max_sequence_length=2048,
        input_sequence_length=256,
        use_cache=True,
        num_logits_to_keep=1,
        kv_cache=dict(
            cache_axis=2,
        ),
        enable_rope=True,
    ),  # wrap模型时，需要传入的配置参数
    quant_config=quant_config,
    frontend_type=frontend_type,
    export_cfg=dict(
        input_names=[
            "inputs_embeds",
            "past_seq_length",
            "current_input_length",
            "position_ids",
        ],
        output_names=["logits"],
    ),
)
resume_from = None
# resume_from = "/data02/chenzx/ssfp_weight/qwen3_moequant/quarot_gptq-state-dict.safetensors"