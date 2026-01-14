# 依赖
- torch 2.7.0
- transformers 4.57.0
# 校准数据准备

## 默认提供了5个数据集的评测结果，作为校准数据，在data/calib_data
- data/calib_data/Qwen2.5-VL-7B-Instruct_CMMMU_VAL_20250923102519_struct.json
- data/calib_data/Qwen2.5-VL-7B-Instruct_COCO_VAL_20250923104643_struct.json
- data/calib_data/Qwen2.5-VL-7B-Instruct_DocVQA_VAL_20250923102720_struct.json
- data/calib_data/Qwen2.5-VL-7B-Instruct_MMMU_DEV_VAL_20250923102615_struct.json
- data/calib_data/Qwen2.5-VL-7B-Instruct_OCRBench_20250923102641_struct.json
对应的图片数据在http://10.10.1.53:8081/artifactory/model_zoo2/houmo/vllm_datasets/LMUData.tar.gz，下载到home目录

# 模型量化
## Qwen3-VL-30B-A3B-Instruct 
    ```shell
        python examples/llm/qwen3vl_moe/qwen3_vl_moe_common_quant.py --model weights/Qwen3-VL-30B-A3B-Instruct --data_files data/calib_data/Qwen2.5-VL-7B-Instruct_CMMMU_VAL_20250923102519_struct.json data/calib_data/Qwen2.5-VL-7B-Instruct_COCO_VAL_20250923104643_struct.json data/calib_data/Qwen2.5-VL-7B-Instruct_DocVQA_VAL_20250923102720_struct.json data/calib_data/Qwen2.5-VL-7B-Instruct_MMMU_DEV_VAL_20250923102615_struct.json --calib_samples 256 --heading_gptq,--use_hession_mse,--self_attn_weight_bits 8
    ```
其中最后三项在`Qwen3-VL-30B-A3B-Instruct`的量化中建议开启，保证一定的量化精度

# 模型导出
## w8a8 sefp
```shell
    python examples/llm/qwen3vl_moe/qwen3_vl_moe_xh2a_export_hmonnx.py --model weights/Qwen3-VL-30B-A3B-Instruct --quant-type w8a8h1_sefp
```

## w4a8 ssfp
```shell
    python examples/llm/qwen3vl_moe/qwen3_vl_moe_xh2a_export_hmonnx.py --model weights/Qwen3-VL-30B-A3B-Instruct --quant-type w4a8h0_ssfp --quant_weight work_dirs/Qwen3-VL-30B-A3B-Instruct_quarot_gptq_transformers-4.57.1/quarot_gptq_use_hession_mse_False_calib_samples_64_heading_gptq_False-state-dict.safetensors 
```

# Demo
```shell
    python examples/llm/qwen3vl_moe/qwen3_vl_moe_xh2a_demo.py 
        --model_dir=/data01/home/chenzx/project/xh2modelzoo/work_dirs/Qwen3-VL-30B-A3B-Instruct-XH2a  
        --visual_onnx=./work_dirs/Qwen3-VL-30B-A3B-Instruct-XH2a/hmonnx/Qwen3-VL-30B-A3B-Instruct-XH2a-w4a8h0_ssfp_vision.onnx, --prefill_onnx=/data01/home/chenzx/project/xh2modelzoo/work_dirs/Qwen3-VL-30B-A3B-Instruct-XH2a/hmonnx/Qwen3-VL-30B-A3B-Instruct-XH2a-w4a8h0_ssfp-llm-prefill.onnx
        --decode_onnx=/data01/home/chenzx/project/xh2modelzoo/work_dirs/Qwen3-VL-30B-A3B-Instruct-XH2a/hmonnx/Qwen3-VL-30B-A3B-Instruct-XH2a-w4a8h0_ssfp-llm-decode.onnx
        --model_type moe --image_size_w 448 --image_size_h 448
```