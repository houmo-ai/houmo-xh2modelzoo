_base_ = [
    "../_base_/xh2a_base.py",
]

frontend_type = "TorchFX"

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
        linear_attn_mask=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="float16"),
            )
        ),
    ),
    w_schema=dict(
        fp_mode="ssfp",
        hidden_bit=False,
        bits=8,
    ),
    act_schema=dict(
        fp_mode="sefp",
        hidden_bit=True,
        bits=8,
    ),
)

release = dict(
    xh_version="xh2",
    modelscope_name="qwen3next",
)

model = dict(
    type="XHQwen3NextModel",
    wrap_cfg=dict(
        batch_size=1,
        max_pe_length=262144,
        max_sequence_length=2048,
        input_sequence_length=256,
        use_cache=True,
        num_logits_to_keep=1,
        linear_attention_mode="auto",
        linear_chunk_size=64,
        kv_cache=dict(
            cache_axis=2,
        ),
        enable_rope=True,
        enable_auto_offload=True,
        auto_offload_max_memory=None,
    ),
    quant_config=quant_config,
    frontend_type=frontend_type,
    export_cfg=dict(
        input_names=[
            "inputs_embeds",
            "past_seq_length",
            "current_input_length",
            "linear_attn_mask",
        ],
        output_names=["logits"],
    ),
)
