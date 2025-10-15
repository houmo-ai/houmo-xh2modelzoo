# 1. Qwen2_5_VL量化教程

## 1.1 导出HMONNX

### 1.1.1 visual w8a8 + llm w8a8
    ```shell
        python examples/llm/qwen2_5_vl/qwen2_5_vl_xh2a_export_hmonnx.py --quant-type w8a8h1_sefp 
    ```
### 1.1.2 visual w8a8 + llm w4a8
    w48a8h0_ssfp依赖gptq量化的模型
#### 1.1.2.1 生成gptq量化的模型
    ```shell
        python examples/llm/qwen2_5_vl/qwen2_5_vl_common_quant.py
    ```
#### 1.1.2.2 导出hmonnx
    ```shell
        python examples/llm/qwen2_5_vl/qwen2_5_vl_xh2a_export_hmonnx.py --image_max_size_h 448 --image_max_size_w 448  --quant-type w4a8h0_ssfp --quant_weight work_dirs/Qwen2.5-VL-7B-Instruct_quarot_gptq_transformers-4.56.2/quarot_gptq-state-dict.safetensors
    ```

## 1.2 Load HMONNX And Demo
    ```shell
    python examples/llm/qwen2_5_vl/qwen2_5_vl_xh2a_demo.py --model_dir work_dirs/Qwen2.5-VL-7B-Instruct-XH2a --visual_onnx work_dirs/Qwen2.5-VL-7B-Instruct-XH2a/hmonnx/Qwen2.5-VL-7B-Instruct-XH2a-w4a8h0_ssfp_vision.onnx --prefill_onnx work_dirs/Qwen2.5-VL-7B-Instruct-XH2a/hmonnx/Qwen2.5-VL-7B-Instruct-XH2a-w4a8h0_ssfp-llm-prefill.onnx --decode_onnx work_dirs/Qwen2.5-VL-7B-Instruct-XH2a/hmonnx/Qwen2.5-VL-7B-Instruct-XH2a-w4a8h0_ssfp-llm-decode.onnx --blocks_num 28 --input_sequence_length 256 --cache_len 2048 --image_size_w 448 --image_size_h 448 --max_size_t 2 --temporal_patch_size 2 --patch_size 14
    ```