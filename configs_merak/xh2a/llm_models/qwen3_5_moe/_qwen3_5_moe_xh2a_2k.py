_base_ = [
    "../../_base_/xh2a_base.py",
]

model = dict(
    model_type="Qwen3_5MoeForConditionalGeneration",
    model_name="xh2_Qwen3.6-35B-A3B_w8a8",
    context_max_length=2048,
    prefill_chunk_length=256,
    use_cache=True,
    num_logits_to_keep=1,
    linear_attention_mode="auto",
    linear_chunk_size=64,
    split_conv_cache=True,
    normalize_force_fp32=False,
    use_manual_depthwise_conv1d=False,
    quant_scheme=dict(
        quant_type="w8a8h1_sefp",
        nodes=dict(
            lm_head=dict(
                quant_type="w8a8h1_sefp",
            )
        ),
        ops={},
    ),
    only_first_block=False,
)
