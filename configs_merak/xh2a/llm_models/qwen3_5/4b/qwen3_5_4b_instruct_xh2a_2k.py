_base_ = [
    "../_qwen3_5_xh2a_2k.py",
]

hf_model_dir = "weights/Qwen3.5-4B"

model = dict(
    model_type="Qwen3_5ForConditionalGeneration",
    hf_model=hf_model_dir,
    model_name="xh2_Qwen3.5-4B_w8a8_256_2k",
    # 内部调试参数
    visual_config=dict(
        max_size_w=448,
        max_size_h=448,
        quant_scheme=dict(
            quant_type="w8a8h1_sefp",  # Node默认量化类型
            ops={},
        ),
        # enable=False,
    ),
    # only_first_block=True,  # 仅包裹第一层，调试用
)
