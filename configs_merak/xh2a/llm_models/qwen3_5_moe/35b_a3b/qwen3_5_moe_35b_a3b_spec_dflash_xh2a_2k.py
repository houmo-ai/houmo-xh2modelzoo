_base_ = [
    "./qwen3_5_moe_35b_a3b_instruct_xh2a_2k.py",
]

hf_model_dir = "weights/Qwen3.6-35B-A3B"
dflash_model_dir = "weights/Qwen3.6-35B-A3B-DFlash"

model = dict(
    model_name="qwen3_6_35b_a3b_spec_dflash_test",
    max_pe_length=262144,
    spec_decode_mode="dflash",
    num_draft_tokens=9,
    spec_draft_head_weight_bits=4,
    output_hidden_state_indices=[1, 10, 19, 28, 37],
    dflash_config=dict(
        hf_model=dflash_model_dir,
        target_model_dir=hf_model_dir,
        mode="context",
        hidden_size=2048,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
        num_hidden_layers=8,
        # DFlash consumes selected target hidden layers, not every target layer.
        num_target_layers=5,
        block_size=16,
        batch_size=1,
        input_sequence_length=256,
        max_sequence_length=2048,
        max_pe_length=262144,
    ),
)
