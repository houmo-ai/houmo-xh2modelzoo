_base_ = [
    "../../../_base_/xh2a_base.py",
]
hf_model_dir = "./data/models/Qwen3-VL-2B-Instruct"  # 模型路径

model = dict(
    model_type="Qwen3VLForConditionalGeneration",
    hf_model=hf_model_dir,
    model_name="xh2_qwen3_vl_2b_w8a8_256_2k",
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
    visual_config=dict(
        max_size_w=448,
        max_size_h=448,
        quant_scheme=dict(
            quant_type="w8a8h1_sefp",  # Node默认量化类型
            ops={},
        ),
    ),
    only_first_block=False,  # 仅包裹第一层，调试用
)
