_base_ = [
    "../../../_base_/xh2a_base.py",
]
hf_model_dir = "./data/models/Qwen3-VL-2B-Instruct"  # 模型路径

model = dict(
    model_type="Qwen3VLForConditionalGeneration_visual",
    hf_model=hf_model_dir,
    model_name="xh2_qwen3_vl_vision_2b_w8a8",
    quant_scheme=dict(
        quant_type="w8a8h1_sefp",  # Node默认量化类型
        ops={},
    ),
    max_size_w=448,
    max_size_h=448,
)
