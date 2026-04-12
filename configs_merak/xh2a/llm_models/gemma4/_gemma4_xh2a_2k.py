_base_ = [
    "../../_base_/xh2a_base.py",
]

model = dict(
    model_type="Gemma4ForConditionalGeneration",
    model_name="xh2_gemma4_31b_w8a8",
    context_max_length=2048,
    prefill_chunk_length=256,
    max_pe_length=32768,
    use_cache=True,
    num_logits_to_keep=0,
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
