# 路径设置
hf_model_dir = "/data02/datasets/xvla-libero"  # 你的 XVLA 权重路径
work_dir = "./work_dirs/xvla_florence2"

# 目标设备
target_device = "XH2a"
frontend_type = "TorchFX"

# 量化配置 (Standard W8A8 for XH2A)
quant_config = dict()

model = dict(
    # 注意：这里不能再写 XHGemmaLLMModel，因为我们会在脚本里手动 Wrap
    # 这里只是一个占位符，或者给脚本读取参数用
    type="XHFlorence2EncoderLLMModel", 
    hf_model=hf_model_dir,
    
    # Florence-2 Decoder 的特定参数
    wrap_cfg=dict(
        max_sequence_length=2048,
        input_sequence_length=100, #256,
        use_cache=False,
        num_logits_to_keep=0,
        # kv_cache=dict(
        #     cache_axis=2,
        # ),
    ),
    
    quant_config=quant_config,
    frontend_type=frontend_type,

    export_cfg=dict(
        input_names=["inputs_embeds"],
        output_names=["last_hidden_state"],
    ),
)