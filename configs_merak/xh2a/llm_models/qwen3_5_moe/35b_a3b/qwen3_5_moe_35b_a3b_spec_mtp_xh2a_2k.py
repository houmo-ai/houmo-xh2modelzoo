_base_ = [
    "./qwen3_5_moe_35b_a3b_instruct_xh2a_2k.py",
]

model = dict(
    model_name="qwen3_6_35b_a3b_spec_mtp_test",
    max_pe_length=262144,
    spec_decode_mode="mtp",
    num_draft_tokens=4,
    spec_draft_head_weight_bits=4,
    output_post_norm_hidden=True,
    mtp_config=dict(
        hidden_size=2048,
        num_key_value_heads=2,
        head_dim=256,
        batch_size=1,
        input_sequence_length=1,
        context_max_length=2048,
        max_pe_length=262144,
        use_cache=True,
    ),
)
