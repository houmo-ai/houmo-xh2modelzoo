_base_ = [
    "../../../_base_/xh2a_base.py",
]
hf_model_dir = "/data03/nfs_shared/llm_models/Qwen/Qwen3.5-9B"

model = dict(
    model_type="Qwen3_5ForConditionalGeneration_visual",
    hf_model=hf_model_dir,
    model_name="xh2_qwen3_5_vl_vision_9b_w8a8",
    quant_scheme=dict(
        quant_type="w8a8h1_sefp",  # Node默认量化类型
        ops={},
    ),
    max_size_w=448,
    max_size_h=448,
)
