"""
GigaBrain-0.1 Action Expert 导出配置
Action Expert 部分 (26层, 1024 hidden, 与 LLM 共享 Attention)
"""

quant_config = dict(inputs=dict())
target_device = "XH2a"
frontend_type = "TorchFX"

# GigaBrain 模型路径
hf_model_dir = "/data01/home/she.gao/.cache/huggingface/hub/models--open-gigaai--GigaBrain-0.1-3.5B-Base/snapshots/e705989fe052d53a8677db41d497a4f1ee519b66"

# PaliGemma2 配置路径
config_dir = "/data01/home/she.gao/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c"

model = dict(
    type="XHGemma2CondLLMModel",  # GigaBrain Action Expert 模型
    hf_model=hf_model_dir,
    wrap_cfg=dict(
        max_sequence_length=2048,
        input_sequence_length=50,  # n_action_steps
        use_cache=True,
        num_logits_to_keep=0,  # 输出所有 action steps
        kv_cache=dict(
            cache_axis=2,
        ),
        # Action Expert 特有配置
        expert_hidden_size=1024,
        proj_width=1024,
        n_action_steps=50,
        max_action_dim=32,
    ),
    quant_config=quant_config,
    frontend_type=frontend_type,
    export_cfg=dict(
        input_names=[
            "inputs_embeds",      # action embeddings
            "past_seq_length",
            "current_input_length",
            "cond",               # AdaRMS condition (time embedding)
            "attention_mask",
        ],
        output_names=["last_hidden_state"],
    ),
)
