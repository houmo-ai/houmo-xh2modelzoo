_base_ = [
    "../_qwen3_5_xh2a_2k.py",
]

hf_model_dir = "weights/Qwen3.5-9B"
dflash_model_dir = "weights/Qwen3.5-9B-DFlash"

model = dict(
    model_type="Qwen3_5ForConditionalGeneration",
    hf_model=hf_model_dir,
    model_name="qwen3_5_9b_spec_dflash_test",
    context_max_length=2048,
    prefill_chunk_length=256,
    max_pe_length=32768,
    visual_config=dict(
        max_size_w=448,
        max_size_h=448,
        quant_scheme=dict(
            quant_type="w8a8h1_sefp",
            ops={},
        ),
        enable=True,
    ),
    # spec decode: DFlash mode
    spec_decode_mode="dflash",
    num_draft_tokens=9,
    spec_draft_head_weight_bits=4,
    output_hidden_state_indices=[1, 8, 15, 22, 29],
    dflash_config=dict(
        hf_model=dflash_model_dir,
        target_model_dir=hf_model_dir,
        mode="context",
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        num_hidden_layers=5,
        # DFlash consumes the selected target hidden layers, not all target
        # decoder layers.  Qwen3.5-9B-DFlash config.json uses five ids:
        # [1, 8, 15, 22, 29].
        num_target_layers=5,
        block_size=16,
        batch_size=1,
        input_sequence_length=256,
        max_sequence_length=2048,
        max_pe_length=32768,
    ),
    only_first_block=False,
)
