_base_ = [
    "../_qwen3_5_xh2a_2k.py",
]

hf_model_dir = "weights/Qwen3.5-9B"

model = dict(
    model_type="Qwen3_5ForConditionalGeneration",
    hf_model=hf_model_dir,
    model_name="qwen3_5_9b_spec_mtp_test",
    context_max_length=2048,
    prefill_chunk_length=256,
    max_pe_length=32768,
    visual_config=dict(
        max_size_w=448,
        max_size_h=448,
        quant_scheme=dict(
            quant_type="w8a8h1_sefp",
            ops={},
        ),
        enable=True,
    ),
    # spec decode: MTP mode
    spec_decode_mode="mtp",
    num_draft_tokens=4,
    spec_draft_head_weight_bits=4,
    output_post_norm_hidden=True,
    mtp_config=dict(
        hidden_size=4096,
        num_key_value_heads=4,
        head_dim=256,
        batch_size=1,
        input_sequence_length=1,
        context_max_length=2048,
        max_pe_length=32768,
        use_cache=True,
    ),
    only_first_block=False,
)
