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
### 2B
```shell
    
```