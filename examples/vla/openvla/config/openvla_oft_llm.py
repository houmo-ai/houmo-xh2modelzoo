quant_config = dict(inputs=dict())
target_device = "XH2a"
hf_model_dir = "/data01/home/she.gao/.cache/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-spatial/snapshots/6d0231af0e48c5985f1ff86908f4674b84bc049b"  # 模型路径
frontend_type = "TorchFX"
# resume_from = "/data01/home/she.gao/xhquant_llm/examples/work_dirs/fa5ae1e7509348889295bba8e08621d8b55e9baf_quarot_gptq_transformers-4.53.3/quarot_gptq-state-dict.safetensors"
quant_config = dict(
    inputs=dict(
        inputs_embeds=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="float16"),
            )
        ),
        past_seq_length=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="int32"),
            )
        ),
        current_input_length=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="int32"),
            )
        ),
        position_ids=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="int32"),
            )
        ),
    )
)


model = dict(
    type="XHLlamaModel",
    wrap_cfg=dict(
        batch_size=1,
        max_sequence_length=1024,
        input_sequence_length=720,
        use_cache=True,
        num_logits_to_keep=1,
        kv_cache=dict(
            cache_axis=2,
        ),
    ),  # wrap模型时，需要传入的配置参数
    hf_model=hf_model_dir,
    frontend_type=frontend_type,
    quant_config=quant_config,
    export_cfg=dict(
        input_names=[
            "inputs_embeds",
            "past_seq_length",
            "current_input_length",
            "position_ids",
            "attention_mask",
        ],
        output_names=["hidden_states"],
    ),
)
