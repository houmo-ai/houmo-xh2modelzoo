_base_ = [
    "../../_base_/xh2a_base.py",
]
model = dict(
    model_type="Qwen3ForCausalLM_legacy_opt",
    model_name="xh2_Qwen3-4B_w8a8",
    context_max_length=2048,
    prefill_chunk_length=256,
    use_cache=True,
    num_logits_to_keep=1,
    quant_scheme=dict(
        quant_type="w8a8h1_sefp",  # Node默认量化类型
        nodes=dict(
            lm_head=dict(
                quant_type="w8a8h1_sefp",
            )
        ),
        ops={},
    ),
    # 内部调试参数
    only_first_block=False,  # 仅包裹第一层，调试用
    use_flash_attention=True,  # 是否使用FlashAttention
)
