quant_config = dict(inputs=dict())
target_device = "XH2a"
frontend_type = "TorchFX"
hf_model_dir = "/data01/home/she.gao/.cache/huggingface/hub/models--lerobot--pi05_libero_finetuned/snapshots/d8419fc249cbb1f29b0c528f05c0d2fe50f46855" # 模型路径
config_dir = "/data01/home/she.gao/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c"

quant_config = dict()
model = dict(
    type="XHGemma05CLLMModel",
    hf_model=hf_model_dir,
    wrap_cfg=dict(
        max_sequence_length=1024,
        input_sequence_length=50,
        use_cache=True,
        num_logits_to_keep=1,
        kv_cache=dict(
            cache_axis=2,
        ),
        # only_first_block=True,
    ),  # wrap模型时，需要传入的配置参数
    quant_config=quant_config,
    frontend_type=frontend_type,
    export_cfg=dict(
        input_names=[
            "inputs_embeds",
            "past_seq_length",
            "current_input_length",
            "cond",
            "attention_mask",
        ],
        output_names=["last_hidden_state"],
    ),
)

