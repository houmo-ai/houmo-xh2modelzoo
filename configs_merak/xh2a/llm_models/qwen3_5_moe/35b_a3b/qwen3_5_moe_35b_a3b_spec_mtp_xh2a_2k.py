_base_ = [
    "../_qwen3_5_moe_xh2a_2k.py",
]

hf_model_dir = "weights/Qwen3.6-35B-A3B"

model = dict(
    hf_model=hf_model_dir,
    model_name="qwen3_6_35b_a3b_spec_mtp_test",
    context_max_length=2048,
    prefill_chunk_length=256,
    max_pe_length=32768,
    visual_config=dict(
        max_size_w=448,
        max_size_h=448,
        quant_scheme=dict(
            quant_type="w8a8h0_ssfp",
            ops={},
        ),
        enable=True,
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
    only_first_block=False,
)
