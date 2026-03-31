_base_ = [
    "../../_base_/xh2a_base.py",
]
model = dict(
    model_type="Qwen3MoeForCausalLM",
    context_max_length=2048,
    prefill_chunk_length=256,
    use_cache=True,
    num_logits_to_keep=1,
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
