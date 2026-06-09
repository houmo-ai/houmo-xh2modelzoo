_base_ = [
    "./qwen3_5_moe_122b_a10b_instruct_hf_gptq_xh2a_w4a8_2k.py",
]


model = dict(
    model_name="xh2_Qwen3.5-122B-A10B_spec_mtp_w4a8_256_2k",
    quant_scheme=dict(
        quant_type="w4a8h0_ssfp",
        nodes=dict(
            lm_head=dict(
                quant_type="w8a8h1_sefp",
            )
        ),
        ops={},
    ),
    spec_decode_mode="mtp",
    num_draft_tokens=4,
    output_post_norm_hidden=True,
    mtp_config=dict(
        hidden_size=2048,
        num_key_value_heads=2,
        head_dim=256,
        batch_size=1,
        input_sequence_length=1,
        context_max_length=2048,
        max_pe_length=32768,
        use_cache=True,
    ),
)
