_base_ = [
    "../../../_base_/xh2a_base.py",
]

hf_model_dir = "./data/models/MiniCPM5-15B-A2.5B-0518-job_366159_step_12000_hybrid_thinking"

model = dict(
    model_type="MiniCPM5MoEForCausalLM",
    hf_model=hf_model_dir,
    model_name="xh2_MiniCPM5-15B-A2.5B_w8a8",
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
