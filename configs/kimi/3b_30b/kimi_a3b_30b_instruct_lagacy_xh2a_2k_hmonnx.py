# hf_model_dir = "./data/models/Qwen3-8B"  # 模型路径
# 推理HMONNX时不依赖HF官方模型，相关配置已包含在hf_model_config_dir指定的目录下, hf_config由导出时生成
hmonnx_export_dir = "work_dirs/kimi_a3b_30b_instruct_legacy_xh2a_2k_batch_eval/"
hf_model_config_dir = "/data02/datasets/kimi"
embed_tokens = hmonnx_export_dir + "token_embedding.pt"
batch_size = 1
model = dict(
    type="KimiMoeHMONNXModel",
    prefill=dict(
        onnx=hmonnx_export_dir + "prefill_onnx/kimi_a3b_30b_instruct_legacy_xh2a_2k_batch_eval_prefill.onnx",
        input_sequence_length=256,
    ),
    decode=dict(
        onnx=hmonnx_export_dir + "decode_onnx/kimi_a3b_30b_instruct_legacy_xh2a_2k_batch_eval_decode.onnx",
    ),
    kv_cache=dict(
        num_hidden_layers=27,
        shape=[1, 16, 2048, 192], # 128
    ),
    v_cache=dict(
        num_hidden_layers=27,
        shape=[1, 16, 2048, 128], # 128
    ),
)
