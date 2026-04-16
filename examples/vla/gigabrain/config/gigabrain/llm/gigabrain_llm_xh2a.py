"""
GigaBrain-0.1 LLM 导出配置
PaliGemma2 主体部分 (26层, 2304 hidden)
"""

quant_config = dict(inputs=dict())
target_device = "XH2a"
frontend_type = "TorchFX"

# GigaBrain 模型路径
hf_model_dir = "/data01/home/she.gao/.cache/huggingface/hub/models--open-gigaai--GigaBrain-0.1-3.5B-Base/snapshots/e705989fe052d53a8677db41d497a4f1ee519b66"

# PaliGemma 配置路径 (tokenizer等)
config_dir = "/data01/home/she.gao/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c"

model = dict(
    type="XHGemma2LLMModel",  # GigaBrain PaliGemma2 LLM 模型
    hf_model=hf_model_dir,
    wrap_cfg=dict(
        max_sequence_length=2048,
        input_sequence_length=1024,  # 根据实际任务调整
        use_cache=True,
        num_logits_to_keep=1,
        kv_cache=dict(
            cache_axis=2,
        ),
    ),
    quant_config=quant_config,
    frontend_type=frontend_type,
    export_cfg=dict(
        input_names=[
            "inputs_embeds",
            "past_seq_length",
            "current_input_length",
            "attention_mask",
        ],
        output_names=["last_hidden_state"],
    ),
)
