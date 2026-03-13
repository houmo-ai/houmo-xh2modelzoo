quant_config = dict(inputs=dict())
target_device = "XH2a"
frontend_type = "TorchFX"
hf_model_dir = "/data/gexinyu_workspace/modelzoo/llm/glm-asr-nano-2512"
config_dir = "/data/gexinyu_workspace/modelzoo/llm/glm-asr-nano-2512"

quant_config = dict()
model = dict(
    type="XHGlmAsrLLMModel",
    hf_model=hf_model_dir,
    wrap_cfg=dict(
        max_sequence_length=2048,
        input_sequence_length=411,
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
            "past_key_cache",
            "past_value_cache",
        ],
        output_names=["last_hidden_state"],
    )
)
