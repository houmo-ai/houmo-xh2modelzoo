config_dir = '/data01/home/binghu.ji/models/Qwen/Qwen3-ASR-0.6B'
device = 'cuda'
dtype = 'float16'
exec_device = 'cuda'
frontend_type = 'TorchFX'
hf_model_dir = '/data01/home/binghu.ji/models/Qwen/Qwen3-ASR-0.6B'
model = dict(
    export_cfg=dict(
        input_names=[
            'inputs_embeds',
            'past_seq_length',
            'current_input_length',
            'past_key_cache',
            'past_value_cache',
        ],
        output_names=[
            'last_hidden_state',
        ]),
    frontend_type='TorchFX',
    hf_model='/data01/home/binghu.ji/models/Qwen/Qwen3-ASR-0.6B',
    quant_config=dict(),
    type='XHQwen3ASRLLMModel',
    wrap_cfg=dict(
        input_sequence_length=411,
        kv_cache=dict(cache_axis=2),
        max_sequence_length=2048,
        num_logits_to_keep=1,
        use_cache=True))
quant_config = dict()
target_device = 'XH2a'
work_dir = 'work_dirs/qwen3_asr_decode_xh2a'
