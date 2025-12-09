# 依赖
- torch 2.8
- transformers 4.57.1
# 校准数据准备

## 默认提供了5个数据集的评测结果，作为校准数据，在data/calib_data
- data/calib_data/Qwen2.5-VL-7B-Instruct_CMMMU_VAL_20250923102519_struct.json
- data/calib_data/Qwen2.5-VL-7B-Instruct_COCO_VAL_20250923104643_struct.json
- data/calib_data/Qwen2.5-VL-7B-Instruct_DocVQA_VAL_20250923102720_struct.json
- data/calib_data/Qwen2.5-VL-7B-Instruct_MMMU_DEV_VAL_20250923102615_struct.json
- data/calib_data/Qwen2.5-VL-7B-Instruct_OCRBench_20250923102641_struct.json
对应的图片数据在http://10.10.1.53:8081/artifactory/model_zoo2/houmo/vllm_datasets/LMUData.tar.gz，下载到home目录

# 模型量化
### 3.1.1 2B
    ```shell
        python examples/llm/qwen3vl/qwen3_vl_common_quant.py --model weights/Qwen3-VL-2B-Instruct --data_files data/calib_data/Qwen2.5-VL-7B-Instruct_CMMMU_VAL_20250923102519_struct.json data/calib_data/Qwen2.5-VL-7B-Instruct_COCO_VAL_20250923104643_struct.json data/calib_data/Qwen2.5-VL-7B-Instruct_DocVQA_VAL_20250923102720_struct.json data/calib_data/Qwen2.5-VL-7B-Instruct_MMMU_DEV_VAL_20250923102615_struct.json --calib_samples 64

    ```       
### 3.1.2 4B
    ```shell
        python examples/qwen3_vl/qwen3_vl_common_quant.py --model weights/Qwen3-VL-4B-Instruct --data_files data/calib_data/Qwen2.5-VL-7B-Instruct_CMMMU_VAL_20250923102519_struct.json data/calib_data/Qwen2.5-VL-7B-Instruct_COCO_VAL_20250923104643_struct.json data/calib_data/Qwen2.5-VL-7B-Instruct_DocVQA_VAL_20250923102720_struct.json data/calib_data/Qwen2.5-VL-7B-Instruct_MMMU_DEV_VAL_20250923102615_struct.json --calib_samples 64
    ```
### 3.1.2 8B
    ```shell
        python examples/qwen3_vl/qwen3_vl_common_quant.py --model weights/Qwen3-VL-8B-Instruct --data_files data/calib_data/Qwen2.5-VL-7B-Instruct_CMMMU_VAL_20250923102519_struct.json data/calib_data/Qwen2.5-VL-7B-Instruct_COCO_VAL_20250923104643_struct.json data/calib_data/Qwen2.5-VL-7B-Instruct_DocVQA_VAL_20250923102720_struct.json data/calib_data/Qwen2.5-VL-7B-Instruct_MMMU_DEV_VAL_20250923102615_struct.json --calib_samples 64
    ```

# 模型导出
## w8a8 sefp
### 2B
```shell
    python ./examples/llm/qwen3vl/qwen3_vl_xh2a_export_hmonnx.py --model weights/Qwen3-VL-2B-Instruct --quant-type w8a8h1_sefp
```

## w4a8 ssfp
### 2B
```shell
    python ./examples/llm/qwen3vl/qwen3_vl_xh2a_export_hmonnx.py --model weights/Qwen3-VL-2B-Instruct --quant-type w4a8h0_ssfp --quant_weight work_dirs/Qwen3-VL-2B-Instruct_quarot_gptq_transformers-4.57.1/quarot_gptq_use_hession_mse_False_calib_samples_64_heading_gptq_False-state-dict.safetensors 
```

# Demo

### 2B
```shell
    python examples/llm/qwen3vl/qwen3_vl_xh2a_demo.py --model_dir work_dirs/Qwen3-VL-2B-Instruct-XH2a --visual_onnx work_dirs/Qwen3-VL-2B-Instruct-XH2a/golden/Qwen3-VL-2B-Instruct-XH2a-w4a8h0_ssfp_vision/hmquant_Qwen3-VL-2B-Instruct-XH2a-w4a8h0_ssfp_vision_with_act.onnx --prefill_onnx work_dirs/Qwen3-VL-2B-Instruct-XH2a/golden/Qwen3-VL-2B-Instruct-XH2a-w4a8h0_ssfp-llm-prefill/hmquant_Qwen3-VL-2B-Instruct-XH2a-w4a8h0_ssfp-llm-prefill_with_act.onnx --decode_onnx work_dirs/Qwen3-VL-2B-Instruct-XH2a/golden/Qwen3-VL-2B-Instruct-XH2a-w4a8h0_ssfp-llm-decode/hmquant_Qwen3-VL-2B-Instruct-XH2a-w4a8h0_ssfp-llm-decode_with_act.onnx --model_type 2B --image_size_w 448 --image_size_h 448


    python examples/llm/qwen3vl/qwen3_vl_xh2a_demo.py --model_dir work_dirs/Qwen3-VL-2B-Instruct-XH2a --visual_onnx work_dirs/Qwen3-VL-2B-Instruct-XH2a/golden/Qwen3-VL-2B-Instruct-XH2a-w8a8h1_sefp_vision/hmquant_Qwen3-VL-2B-Instruct-XH2a-w8a8h1_sefp_vision_with_act.onnx --prefill_onnx work_dirs/Qwen3-VL-2B-Instruct-XH2a/golden/Qwen3-VL-2B-Instruct-XH2a-w8a8h1_sefp-llm-prefill/hmquant_Qwen3-VL-2B-Instruct-XH2a-w8a8h1_sefp-llm-prefill_with_act.onnx --decode_onnx work_dirs/Qwen3-VL-2B-Instruct-XH2a/golden/Qwen3-VL-2B-Instruct-XH2a-w8a8h1_sefp-llm-decode/hmquant_Qwen3-VL-2B-Instruct-XH2a-w8a8h1_sefp-llm-decode_with_act.onnx --model_type 2B --image_size_w 448 --image_size_h 448
```