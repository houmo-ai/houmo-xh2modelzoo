quant_config = dict(inputs=dict())
target_device = "XH2a"
frontend_type = "TorchFX"
hf_model_dir = "/data01/home/binghu.ji/models/Qwen/Qwen3-ASR-0.6B"
config_dir = "/data01/home/binghu.ji/models/Qwen/Qwen3-ASR-0.6B"

quant_config = dict()
model = dict(
    type="XHQwen3ASRLLMModel",
    hf_model=hf_model_dir,
    wrap_cfg=dict(
        max_sequence_length=2048,
        input_sequence_length=216,  # 参考 15s 音频，在导出中会被动态覆盖
        use_cache=True,
        num_logits_to_keep=1,
        kv_cache=dict(
            cache_axis=2,
        ),
        # only_first_block=True,
    ),  # wrap模型时，需要传入的配置参数
    quant_config=quant_config,
    frontend_type=frontend_type,
    export_cfg = dict(
            input_names=[
                "inputs_embeds",
                "past_seq_length",
                "current_input_length",
                "past_key_cache",
                "past_value_cache",
            ],
            output_names=["last_hidden_state"],
    )
)



