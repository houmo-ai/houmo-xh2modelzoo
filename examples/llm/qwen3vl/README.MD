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

导出产物目录结构：
```
work_dirs/Qwen3-VL-2B-Instruct-XH2a/
├── hmonnx/                          # ONNX 模型
│   ├── *_vision.onnx                # Vision Encoder
│   ├── *-llm-prefill.onnx           # LLM Prefill
│   └── *-llm-decode.onnx            # LLM Decode
├── golden/                          # Golden 参考数据（用于 hmonnx 验证）
│   ├── *_vision/
│   ├── *-llm-prefill/
│   └── *-llm-decode/
├── hf_config/                       # HF processor 配置
└── token_embedding.pt               # Token embedding 权重
```

## 导出参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `weights/Qwen3-VL-2B-Instruct` | HuggingFace 模型目录 |
| `--quant-type` | `w8a8h1_sefp` | 量化类型 |
| `--context-length` | `2048` | KV cache 长度，影响最大生成长度 |
| `--image_max_size_h` | `448` | 图像/视频空间高度 |
| `--image_max_size_w` | `448` | 图像/视频空间宽度 |
| `--image_max_size_t` | `2` | 时间维度帧数。图像=2，视频常用 4/6/8 |
| `--sample_image_path` | `data/images/qwen2_vl_demo.jpeg` | 图像样例路径（用于生成 golden） |
| `--sample_video_path` | `""` | 视频样例路径；设置后启用视频导出链路 |
| `--quant_weight` | `None` | 预量化权重路径（用于 w4a8 等需要 GPTQ 的方案） |

## w8a8h1_sefp

### 2B 图像导出
```shell
python ./examples/llm/qwen3vl/qwen3_vl_xh2a_export_hmonnx.py \
    --model weights/Qwen3-VL-2B-Instruct \
    --quant-type w8a8h1_sefp
```

### 2B 视频导出
```shell
python ./examples/llm/qwen3vl/qwen3_vl_xh2a_export_hmonnx.py \
    --model weights/Qwen3-VL-2B-Instruct \
    --quant-type w8a8h1_sefp \
    --sample_video_path data/test/test.mp4 \
    --image_max_size_t 4 \
    --image_max_size_h 448 \
    --image_max_size_w 448
```

### 4B 图像导出
```shell
python ./examples/llm/qwen3vl/qwen3_vl_xh2a_export_hmonnx.py \
    --model weights/Qwen3-VL-4B-Instruct \
    --quant-type w8a8h1_sefp
```

### 4B 视频导出
```shell
python ./examples/llm/qwen3vl/qwen3_vl_xh2a_export_hmonnx.py \
    --model weights/Qwen3-VL-4B-Instruct \
    --quant-type w8a8h1_sefp \
    --sample_video_path data/test/test.mp4 \
    --image_max_size_t 4 \
    --image_max_size_h 448 \
    --image_max_size_w 448
```

### 8B 图像导出
```shell
python ./examples/llm/qwen3vl/qwen3_vl_xh2a_export_hmonnx.py \
    --model weights/Qwen3-VL-8B-Instruct \
    --quant-type w8a8h1_sefp
```

### 8B 视频导出
```shell
python ./examples/llm/qwen3vl/qwen3_vl_xh2a_export_hmonnx.py \
    --model weights/Qwen3-VL-8B-Instruct-gptqmodel-4bit-rotation-hessian-mse \
    --quant-type w4a8h0_ssfp \
    --sample_video_path data/test/test.mp4 \
    --image_max_size_t 6 \
    --image_max_size_h 448 \
    --image_max_size_w 448 \
    --use_gptqmodel
```

```shell
python examples/llm/qwen3vl/qwen3_vl_xh2a_demo.py \
    --model_dir work_dirs/Qwen3-VL-8B-Instruct-gptqmodel-4bit-rotation-hessian-mse-XH2a \
    --visual_onnx work_dirs/Qwen3-VL-8B-Instruct-gptqmodel-4bit-rotation-hessian-mse-XH2a/hmonnx/Qwen3-VL-8B-Instruct-gptqmodel-4bit-rotation-hessian-mse-XH2a-w4a8h0_ssfp_vision.onnx \
    --prefill_onnx work_dirs/Qwen3-VL-8B-Instruct-gptqmodel-4bit-rotation-hessian-mse-XH2a/hmonnx/Qwen3-VL-8B-Instruct-gptqmodel-4bit-rotation-hessian-mse-XH2a-w4a8h0_ssfp-llm-prefill.onnx \
    --decode_onnx work_dirs/Qwen3-VL-8B-Instruct-gptqmodel-4bit-rotation-hessian-mse-XH2a/hmonnx/Qwen3-VL-8B-Instruct-gptqmodel-4bit-rotation-hessian-mse-XH2a-w4a8h0_ssfp-llm-decode.onnx \
    --model_type 8B \
    --image_size_w 448 --image_size_h 448 \
    --max_size_t 6 \
    --video data/test/test.mp4 \
    --prompt '视频中婴儿在干嘛，请简短回答' \
    --cache_len 2048
```

## w8a16h1_sefp（推荐，精度最高）

### 2B 图像导出
```shell
python ./examples/llm/qwen3vl/qwen3_vl_xh2a_export_hmonnx.py \
    --model weights/Qwen3-VL-2B-Instruct \
    --quant-type w8a16h1_sefp
```

### 2B 视频导出
```shell
python ./examples/llm/qwen3vl/qwen3_vl_xh2a_export_hmonnx.py \
    --model weights/Qwen3-VL-2B-Instruct \
    --quant-type w8a16h1_sefp \
    --sample_video_path data/test/test.mp4 \
    --image_max_size_t 4 \
    --image_max_size_h 448 \
    --image_max_size_w 448
```

### 4B 图像导出
```shell
python ./examples/llm/qwen3vl/qwen3_vl_xh2a_export_hmonnx.py \
    --model weights/Qwen3-VL-4B-Instruct \
    --quant-type w8a16h1_sefp
```

### 4B 视频导出
```shell
python ./examples/llm/qwen3vl/qwen3_vl_xh2a_export_hmonnx.py \
    --model weights/Qwen3-VL-4B-Instruct \
    --quant-type w8a16h1_sefp \
    --sample_video_path data/test/test.mp4 \
    --image_max_size_t 4 \
    --image_max_size_h 448 \
    --image_max_size_w 448
```

### 8B 图像导出
```shell
python ./examples/llm/qwen3vl/qwen3_vl_xh2a_export_hmonnx.py \
    --model weights/Qwen3-VL-8B-Instruct \
    --quant-type w8a16h1_sefp
```

### 8B 视频导出
```shell
python ./examples/llm/qwen3vl/qwen3_vl_xh2a_export_hmonnx.py \
    --model weights/Qwen3-VL-8B-Instruct \
    --quant-type w8a16h1_sefp \
    --sample_video_path data/test/test.mp4 \
    --image_max_size_t 4 \
    --image_max_size_h 448 \
    --image_max_size_w 448
```

## w4a8h0_ssfp（需要预量化权重）

### 2B
```shell
python ./examples/llm/qwen3vl/qwen3_vl_xh2a_export_hmonnx.py \
    --model weights/Qwen3-VL-2B-Instruct \
    --quant-type w4a8h0_ssfp \
    --quant_weight work_dirs/Qwen3-VL-2B-Instruct_quarot_gptq_transformers-4.57.1/quarot_gptq_use_hession_mse_False_calib_samples_64_heading_gptq_False-state-dict.safetensors
```

## GPTQModel 导出（w4a8h0_ssfp）

### 4B
```shell
python ./examples/llm/qwen3vl/qwen3_vl_xh2a_export_hmonnx.py \
    --model weights/Qwen3-VL-4B-Instruct-gptqmodel-4bit-rotation-hessian-mse \
    --quant-type w4a8h0_ssfp
```

### 8B
```shell
python ./examples/llm/qwen3vl/qwen3_vl_xh2a_export_hmonnx.py \
    --model weights/Qwen3-VL-8B-Instruct-gptqmodel-4bit-all-rotation \
    --quant-type w4a8h0_ssfp
```

## 导出注意事项

- `--sample_video_path` 非空时启用视频导出链路，视频 message 会固定 `nframes=image_max_size_t`
- 视频导出链路会显式注入 clip-level metadata，避免回退到默认 `fps=24`
- 图像和视频导出的 **vision ONNX 形状不同**（temporal 维度不同），不能混用
- 如果需要同时支持图像和视频 demo，需分别导出两套产物

# Demo

## Demo 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_dir` | （必填） | 导出产物根目录，包含 `hf_config/` 和 `token_embedding.pt` |
| `--visual_onnx` | （必填） | Vision ONNX 路径 |
| `--prefill_onnx` | （必填） | LLM Prefill ONNX 路径 |
| `--decode_onnx` | （必填） | LLM Decode ONNX 路径 |
| `--model_type` | `2B` | 模型规模：`2B` / `4B` / `8B` |
| `--image_size_w` | `448` | 图像宽度，需和导出时一致 |
| `--image_size_h` | `448` | 图像高度，需和导出时一致 |
| `--max_size_t` | `2` | 时间维度帧数，需和导出时一致 |
| `--cache_len` | `2048` | KV cache 长度，需和导出时 `--context-length` 一致 |
| `--image` | `data/images/qwen2_vl_demo.jpeg` | 图像路径 |
| `--video` | `None` | 视频路径（设置后进入视频模式） |
| `--prompt` | `请用简洁优美的语言描述这张图片。` | 用户提示词 |
| `--presence_penalty` | `0.0` | 生成惩罚系数，建议设为 `1.05` 以避免重复 |

## 2B 图像 Demo

### w8a16h1_sefp（推荐）
```shell
python examples/llm/qwen3vl/qwen3_vl_xh2a_demo.py \
    --model_dir work_dirs/Qwen3-VL-2B-Instruct-XH2a \
    --visual_onnx work_dirs/Qwen3-VL-2B-Instruct-XH2a/hmonnx/Qwen3-VL-2B-Instruct-XH2a-w8a16h1_sefp_vision.onnx \
    --prefill_onnx work_dirs/Qwen3-VL-2B-Instruct-XH2a/hmonnx/Qwen3-VL-2B-Instruct-XH2a-w8a16h1_sefp-llm-prefill.onnx \
    --decode_onnx work_dirs/Qwen3-VL-2B-Instruct-XH2a/hmonnx/Qwen3-VL-2B-Instruct-XH2a-w8a16h1_sefp-llm-decode.onnx \
    --model_type 2B \
    --image_size_w 448 --image_size_h 448 \
    --image data/images/qwen2_vl_demo.jpeg \
    --prompt '请用简洁优美的语言描述这张图片。'
```

### w8a8h1_sefp
```shell
python examples/llm/qwen3vl/qwen3_vl_xh2a_demo.py \
    --model_dir work_dirs/Qwen3-VL-2B-Instruct-XH2a \
    --visual_onnx work_dirs/Qwen3-VL-2B-Instruct-XH2a/hmonnx/Qwen3-VL-2B-Instruct-XH2a-w8a8h1_sefp_vision.onnx \
    --prefill_onnx work_dirs/Qwen3-VL-2B-Instruct-XH2a/hmonnx/Qwen3-VL-2B-Instruct-XH2a-w8a8h1_sefp-llm-prefill.onnx \
    --decode_onnx work_dirs/Qwen3-VL-2B-Instruct-XH2a/hmonnx/Qwen3-VL-2B-Instruct-XH2a-w8a8h1_sefp-llm-decode.onnx \
    --model_type 2B \
    --image_size_w 448 --image_size_h 448 \
    --image data/images/qwen2_vl_demo.jpeg \
    --prompt '请用简洁优美的语言描述这张图片。'
```

### w4a8h0_ssfp（golden ONNX）
```shell
python examples/llm/qwen3vl/qwen3_vl_xh2a_demo.py \
    --model_dir work_dirs/Qwen3-VL-2B-Instruct-XH2a \
    --visual_onnx work_dirs/Qwen3-VL-2B-Instruct-XH2a/golden/Qwen3-VL-2B-Instruct-XH2a-w4a8h0_ssfp_vision/hmquant_Qwen3-VL-2B-Instruct-XH2a-w4a8h0_ssfp_vision_with_act.onnx \
    --prefill_onnx work_dirs/Qwen3-VL-2B-Instruct-XH2a/golden/Qwen3-VL-2B-Instruct-XH2a-w4a8h0_ssfp-llm-prefill/hmquant_Qwen3-VL-2B-Instruct-XH2a-w4a8h0_ssfp-llm-prefill_with_act.onnx \
    --decode_onnx work_dirs/Qwen3-VL-2B-Instruct-XH2a/golden/Qwen3-VL-2B-Instruct-XH2a-w4a8h0_ssfp-llm-decode/hmquant_Qwen3-VL-2B-Instruct-XH2a-w4a8h0_ssfp-llm-decode_with_act.onnx \
    --model_type 2B \
    --image_size_w 448 --image_size_h 448
```

## 2B 视频 Demo

### w8a16h1_sefp（推荐）
```shell
python examples/llm/qwen3vl/qwen3_vl_xh2a_demo.py \
    --model_dir work_dirs/Qwen3-VL-2B-Instruct-XH2a \
    --visual_onnx work_dirs/Qwen3-VL-2B-Instruct-XH2a/hmonnx/Qwen3-VL-2B-Instruct-XH2a-w8a16h1_sefp_vision.onnx \
    --prefill_onnx work_dirs/Qwen3-VL-2B-Instruct-XH2a/hmonnx/Qwen3-VL-2B-Instruct-XH2a-w8a16h1_sefp-llm-prefill.onnx \
    --decode_onnx work_dirs/Qwen3-VL-2B-Instruct-XH2a/hmonnx/Qwen3-VL-2B-Instruct-XH2a-w8a16h1_sefp-llm-decode.onnx \
    --model_type 2B \
    --image_size_w 448 --image_size_h 448 \
    --max_size_t 4 \
    --video data/test/test.mp4 \
    --prompt 'Describe this video.' \
    --cache_len 2048
```

### w8a8h1_sefp
```shell
python examples/llm/qwen3vl/qwen3_vl_xh2a_demo.py \
    --model_dir work_dirs/Qwen3-VL-2B-Instruct-XH2a \
    --visual_onnx work_dirs/Qwen3-VL-2B-Instruct-XH2a/hmonnx/Qwen3-VL-2B-Instruct-XH2a-w8a8h1_sefp_vision.onnx \
    --prefill_onnx work_dirs/Qwen3-VL-2B-Instruct-XH2a/hmonnx/Qwen3-VL-2B-Instruct-XH2a-w8a8h1_sefp-llm-prefill.onnx \
    --decode_onnx work_dirs/Qwen3-VL-2B-Instruct-XH2a/hmonnx/Qwen3-VL-2B-Instruct-XH2a-w8a8h1_sefp-llm-decode.onnx \
    --model_type 2B \
    --image_size_w 448 --image_size_h 448 \
    --max_size_t 4 \
    --video data/test/test.mp4 \
    --prompt 'Describe this video.' \
    --cache_len 2048
```

## 4B 图像 Demo

### w8a16h1_sefp（推荐）
```shell
python examples/llm/qwen3vl/qwen3_vl_xh2a_demo.py \
    --model_dir work_dirs/Qwen3-VL-4B-Instruct-XH2a \
    --visual_onnx work_dirs/Qwen3-VL-4B-Instruct-XH2a/hmonnx/Qwen3-VL-4B-Instruct-XH2a-w8a16h1_sefp_vision.onnx \
    --prefill_onnx work_dirs/Qwen3-VL-4B-Instruct-XH2a/hmonnx/Qwen3-VL-4B-Instruct-XH2a-w8a16h1_sefp-llm-prefill.onnx \
    --decode_onnx work_dirs/Qwen3-VL-4B-Instruct-XH2a/hmonnx/Qwen3-VL-4B-Instruct-XH2a-w8a16h1_sefp-llm-decode.onnx \
    --model_type 4B \
    --image_size_w 448 --image_size_h 448 \
    --image data/images/qwen2_vl_demo.jpeg \
    --prompt '请用简洁优美的语言描述这张图片。'
```

### w8a8h1_sefp
```shell
python examples/llm/qwen3vl/qwen3_vl_xh2a_demo.py \
    --model_dir work_dirs/Qwen3-VL-4B-Instruct-XH2a \
    --visual_onnx work_dirs/Qwen3-VL-4B-Instruct-XH2a/hmonnx/Qwen3-VL-4B-Instruct-XH2a-w8a8h1_sefp_vision.onnx \
    --prefill_onnx work_dirs/Qwen3-VL-4B-Instruct-XH2a/hmonnx/Qwen3-VL-4B-Instruct-XH2a-w8a8h1_sefp-llm-prefill.onnx \
    --decode_onnx work_dirs/Qwen3-VL-4B-Instruct-XH2a/hmonnx/Qwen3-VL-4B-Instruct-XH2a-w8a8h1_sefp-llm-decode.onnx \
    --model_type 4B \
    --image_size_w 448 --image_size_h 448 \
    --image data/images/qwen2_vl_demo.jpeg \
    --prompt '请用简洁优美的语言描述这张图片。'
```

## 4B 视频 Demo

### w8a16h1_sefp（推荐）
```shell
python examples/llm/qwen3vl/qwen3_vl_xh2a_demo.py \
    --model_dir work_dirs/Qwen3-VL-4B-Instruct-XH2a \
    --visual_onnx work_dirs/Qwen3-VL-4B-Instruct-XH2a/hmonnx/Qwen3-VL-4B-Instruct-XH2a-w8a16h1_sefp_vision.onnx \
    --prefill_onnx work_dirs/Qwen3-VL-4B-Instruct-XH2a/hmonnx/Qwen3-VL-4B-Instruct-XH2a-w8a16h1_sefp-llm-prefill.onnx \
    --decode_onnx work_dirs/Qwen3-VL-4B-Instruct-XH2a/hmonnx/Qwen3-VL-4B-Instruct-XH2a-w8a16h1_sefp-llm-decode.onnx \
    --model_type 4B \
    --image_size_w 448 --image_size_h 448 \
    --max_size_t 4 \
    --video data/test/test.mp4 \
    --prompt 'Describe this video.' \
    --cache_len 2048
```

## 8B 图像 Demo

### w8a16h1_sefp（推荐）
```shell
python examples/llm/qwen3vl/qwen3_vl_xh2a_demo.py \
    --model_dir work_dirs/Qwen3-VL-8B-Instruct-XH2a \
    --visual_onnx work_dirs/Qwen3-VL-8B-Instruct-XH2a/hmonnx/Qwen3-VL-8B-Instruct-XH2a-w8a16h1_sefp_vision.onnx \
    --prefill_onnx work_dirs/Qwen3-VL-8B-Instruct-XH2a/hmonnx/Qwen3-VL-8B-Instruct-XH2a-w8a16h1_sefp-llm-prefill.onnx \
    --decode_onnx work_dirs/Qwen3-VL-8B-Instruct-XH2a/hmonnx/Qwen3-VL-8B-Instruct-XH2a-w8a16h1_sefp-llm-decode.onnx \
    --model_type 8B \
    --image_size_w 448 --image_size_h 448 \
    --image data/images/qwen2_vl_demo.jpeg \
    --prompt '请用简洁优美的语言描述这张图片。'
```

### w8a8h1_sefp
```shell
python examples/llm/qwen3vl/qwen3_vl_xh2a_demo.py \
    --model_dir work_dirs/Qwen3-VL-8B-Instruct-XH2a \
    --visual_onnx work_dirs/Qwen3-VL-8B-Instruct-XH2a/hmonnx/Qwen3-VL-8B-Instruct-XH2a-w8a8h1_sefp_vision.onnx \
    --prefill_onnx work_dirs/Qwen3-VL-8B-Instruct-XH2a/hmonnx/Qwen3-VL-8B-Instruct-XH2a-w8a8h1_sefp-llm-prefill.onnx \
    --decode_onnx work_dirs/Qwen3-VL-8B-Instruct-XH2a/hmonnx/Qwen3-VL-8B-Instruct-XH2a-w8a8h1_sefp-llm-decode.onnx \
    --model_type 8B \
    --image_size_w 448 --image_size_h 448 \
    --image data/images/qwen2_vl_demo.jpeg \
    --prompt '请用简洁优美的语言描述这张图片。'
```

### GPTQModel w4a8h0_ssfp
```shell
python examples/llm/qwen3vl/qwen3_vl_xh2a_demo.py \
    --model_dir work_dirs/Qwen3-VL-8B-Instruct-gptqmodel-4bit-all-rotation-XH2a \
    --visual_onnx work_dirs/Qwen3-VL-8B-Instruct-gptqmodel-4bit-all-rotation-XH2a/hmonnx/Qwen3-VL-8B-Instruct-gptqmodel-4bit-all-rotation-XH2a-w4a8h0_ssfp_vision.onnx \
    --prefill_onnx work_dirs/Qwen3-VL-8B-Instruct-gptqmodel-4bit-all-rotation-XH2a/hmonnx/Qwen3-VL-8B-Instruct-gptqmodel-4bit-all-rotation-XH2a-w4a8h0_ssfp-llm-prefill.onnx \
    --decode_onnx work_dirs/Qwen3-VL-8B-Instruct-gptqmodel-4bit-all-rotation-XH2a/hmonnx/Qwen3-VL-8B-Instruct-gptqmodel-4bit-all-rotation-XH2a-w4a8h0_ssfp-llm-decode.onnx \
    --model_type 8B \
    --image_size_w 448 --image_size_h 448
```

## 8B 视频 Demo

### w8a16h1_sefp（推荐）
```shell
python examples/llm/qwen3vl/qwen3_vl_xh2a_demo.py \
    --model_dir work_dirs/Qwen3-VL-8B-Instruct-XH2a \
    --visual_onnx work_dirs/Qwen3-VL-8B-Instruct-XH2a/hmonnx/Qwen3-VL-8B-Instruct-XH2a-w8a16h1_sefp_vision.onnx \
    --prefill_onnx work_dirs/Qwen3-VL-8B-Instruct-XH2a/hmonnx/Qwen3-VL-8B-Instruct-XH2a-w8a16h1_sefp-llm-prefill.onnx \
    --decode_onnx work_dirs/Qwen3-VL-8B-Instruct-XH2a/hmonnx/Qwen3-VL-8B-Instruct-XH2a-w8a16h1_sefp-llm-decode.onnx \
    --model_type 8B \
    --image_size_w 448 --image_size_h 448 \
    --max_size_t 4 \
    --video data/test/test.mp4 \
    --prompt 'Describe this video.' \
    --cache_len 2048
```

## Demo 注意事项

- **图像 vs 视频**：使用 `--video` 进入视频模式，不要和 `--image` 混用
- **参数对齐**：`--max_size_t`、`--image_size_w/h`、`--cache_len` 必须和导出时一致
- **重复生成**：greedy decode 下 2B 模型易出现重复，属于 HF 原始模型行为（已验证）。建议加 `--presence_penalty 1.05` 缓解
- **ONNX 来源**：hmonnx 目录下为量化后的 ONNX，golden 目录下为带 act 的 golden ONNX；两者都可用于 demo

# Transformers 与本地 xh2 一致性验证

## 目的

用于验证以下两条链路是否完全一致：

- transformers 官方 Processor + Model
- xh2modelzoo 本地 Processor + Model

当前脚本：

- examples/llm/qwen3vl/qwen3_vl_transformers_consistency_demo.py

验证内容包括：

- processor 输出是否一致
- logits 是否一致
- generate 结果是否一致

说明：

- 本地分支仍然保留 hm_pixel_values
- 一致性对比时会自动 pop 掉 hm_pixel_values，只对标准主链做逐项比较

## 环境

```shell
source /home/jiangyong.yu/miniconda3/etc/profile.d/conda.sh
conda activate xhquant
cd /home/jiangyong.yu/xh2_work/xh2modelzoo
export PYTHONPATH=$PWD
```

## 常用参数

```text
--model-path    HuggingFace 模型目录，例如 weights/Qwen3-VL-8B-Instruct
--backend       transformers | xh2 | both
--mode          image | video
--device        cuda | cpu
--dtype         float16 | float32 | bfloat16
--image-path    指定真实图片路径
--video-path    指定真实视频路径
--sample-dir    未显式指定 image/video 路径时，自动从该目录挑选样本，默认 data/test
--video-frames  视频模式下采样帧数，默认 4
--max-new-tokens 生成 token 数，默认 8
--atol          浮点对齐阈值，默认 1e-4
```

## 运行方式

### 1. 只跑官方 transformers

```shell
CUDA_VISIBLE_DEVICES=0 python examples/llm/qwen3vl/qwen3_vl_transformers_consistency_demo.py \
    --model-path weights/Qwen3-VL-8B-Instruct \
    --backend transformers \
    --mode image \
    --device cuda \
    --image-path data/test/image-2025-09-15-14-09-19-086.png
```

### 2. 只跑本地 xh2

```shell
CUDA_VISIBLE_DEVICES=0 python examples/llm/qwen3vl/qwen3_vl_transformers_consistency_demo.py \
    --model-path weights/Qwen3-VL-8B-Instruct \
    --backend xh2 \
    --mode video \
    --device cuda \
    --video-path data/test/test.mp4
```

### 3. 同时对比两端 image

```shell
CUDA_VISIBLE_DEVICES=0 python examples/llm/qwen3vl/qwen3_vl_transformers_consistency_demo.py \
    --model-path weights/Qwen3-VL-8B-Instruct \
    --backend both \
    --mode image \
    --device cuda \
    --max-new-tokens 256 \
    --image-path data/test/image-2025-09-15-14-09-19-086.png
```

### 4. 同时对比两端 video

```shell
CUDA_VISIBLE_DEVICES=0 python examples/llm/qwen3vl/qwen3_vl_transformers_consistency_demo.py \
    --model-path weights/Qwen3-VL-8B-Instruct \
    --backend both \
    --mode video \
    --device cuda \
    --max-new-tokens 4 \
    --video-path data/test/test.mp4
```

### 5. 不指定路径，自动使用 data/test 下样本

```shell
CUDA_VISIBLE_DEVICES=0 python examples/llm/qwen3vl/qwen3_vl_transformers_consistency_demo.py \
    --model-path weights/Qwen3-VL-8B-Instruct \
    --backend both \
    --mode image \
    --device cuda
```

```shell
CUDA_VISIBLE_DEVICES=0 python examples/llm/qwen3vl/qwen3_vl_transformers_consistency_demo.py \
    --model-path weights/Qwen3-VL-8B-Instruct \
    --backend both \
    --mode video \
    --device cuda
```

## 输出解读

当 `--backend both` 时，脚本会输出类似下面这些字段：

```text
processor:attention_mask: max_abs_diff=0, passed=True
processor:image_grid_thw: max_abs_diff=0, passed=True
processor:pixel_values: max_abs_diff=0, passed=True
logits: max_abs_diff=0, passed=True
generation_equal=True
```

判定规则：

- `processor:* passed=True`：两端预处理输出一致
- `logits passed=True`：两端前向结果一致
- `generation_equal=True`：两端生成 token 完全一致

## 已验证命令

以下命令已经在 `xhquant` 环境和 `weights/Qwen3-VL-8B-Instruct` 上验证通过：

```shell
CUDA_VISIBLE_DEVICES=0 python examples/llm/qwen3vl/qwen3_vl_transformers_consistency_demo.py \
    --model-path weights/Qwen3-VL-8B-Instruct \
    --mode image \
    --backend both \
    --device cuda \
    --max-new-tokens 4 \
    --image-path data/test/image-2025-09-15-14-09-19-086.png
```

```shell
CUDA_VISIBLE_DEVICES=0 python examples/llm/qwen3vl/qwen3_vl_transformers_consistency_demo.py \
    --model-path weights/Qwen3-VL-8B-Instruct \
    --mode video \
    --backend both \
    --device cuda \
    --max-new-tokens 4 \
    --video-path data/test/test.mp4
```

对应结果：

- image: processor 全部一致，logits 一致，generation_equal=True
- video: processor 全部一致，logits 一致，generation_equal=True

# Wrapped Model 与 transformers 原生代码对比

## 目的

用于补齐 export 前的一段关键验证：

- 输入侧使用 xh2modelzoo 修改后的 Processor
- 推理侧使用 convert 前的 wrapped vision + wrapped llm 路径
- 对比侧使用未改动的 transformers 原生 Qwen3VL 模型

当前脚本：

- examples/llm/qwen3vl/qwen3_vl_wrapped_model_consistency_demo.py

验证内容包括：

- wrapped vision 输出 vs transformers 原生视觉特征
- wrapped deepstack 输出 vs transformers 原生 deepstack 特征
- wrapped prefill logits vs transformers 原生 logits
- 最后一位 token 的 argmax 是否一致

## image 示例

```shell
CUDA_VISIBLE_DEVICES=0 python examples/llm/qwen3vl/qwen3_vl_wrapped_model_consistency_demo.py \
    --model-path weights/Qwen3-VL-2B-Instruct \
    --mode image \
    --image-path data/test/image-2025-09-15-14-09-19-086.png \
    --device cuda \
    --dtype float16 \
    --input-sequence-length 256 \
    --cache-len 2048
```

## video 示例

```shell
CUDA_VISIBLE_DEVICES=0 python examples/llm/qwen3vl/qwen3_vl_wrapped_model_consistency_demo.py \
    --model-path weights/Qwen3-VL-2B-Instruct \
    --mode video \
    --video-path data/test/test.mp4 \
    --device cuda \
    --dtype float16 \
    --input-sequence-length 512 \
    --cache-len 2048 \
    --max-size-t 4
```

## 当前实测结果

在 `weights/Qwen3-VL-2B-Instruct` 上，image 和 video 两条路径都已经跑通。

- image：wrapped 和原生的最后一位 next token 一致
- video：wrapped 和原生的最后一位 next token 一致
- 但 vision 特征、deepstack 特征、prefill logits 仍存在明显数值漂移

这说明：

- wrapped export 前主链已经能在真实输入上给出与 transformers 原生一致的下一 token 决策
- 但若要求中间张量和 logits 数值级严格对齐，当前仍有进一步排查空间




## 多图推理（HMONNX 图运行时 Demo）

`examples/llm/qwen3vl/qwen3_vl_xh2a_multi_image_demo.py` 是一个通用 CLI 多图推理脚本，基于 `HMONNXGraphInference` + 可选 cuda_graph，支持流式输出。**该脚本不含任何特定业务场景的硬编码**，所有图片路径、提示词通过命令行参数传入。

### 输入方式（三选一，必须且只能选一种）

| 输入模式 | 参数 | 说明 |
|---------|------|------|
| 直接列举 | `--image-paths a.jpg,b.jpg,c.jpg` | 逗号分隔的图像路径，按出现顺序作为多图序列 |
| 目录扫描 | `--image-dir ./images [--image-glob '*.jpg']` | 扫描目录，默认按文件名排序；可选 glob 过滤 |
| 任务 JSON | `--task-json tasks.json --task-id <id>` | 兼容旧任务描述格式，从 JSON 记录中解析 basename 列表，对应文件应位于 `--image-dir`（缺省取 JSON 同级目录） |

### 提示词（二选一）

| 参数 | 说明 |
|------|------|
| `--prompt "Describe these images."` | 直接传入提示词字符串 |
| `--prompt-file ./prompts/desc.txt` | 从 UTF-8 文本文件读入提示词（自动 strip） |

### 通用参数

| 类别 | 参数 | 默认 / 说明 |
|------|------|-----------|
| 模型 | `--model-config export_meta_info.json` | 推荐方式，单一入口指向导出元数据 |
| 模型（兼容） | `--model-dir`, `--visual-onnx`, `--prefill-onnx`, `--decode-onnx`, `--hf-model-dir` | 旧式分参指定 |
| 批处理 | `--images-per-batch 20`, `--max-batches`, `--max-images` | 控制分批与上限 |
| 分辨率 | `--image-size-w 448`, `--image-size-h 448`, `--max-size-t`, `--patch-size`, `--temporal-patch-size` | 视觉编码器输入参数 |
| 生成 | `--max-new-tokens 256`, `--max-context-tokens 32768`, `--temperature`, `--top-p`, `--top-k`, `--repetition-penalty`, `--do-sample/--no-sample` | 标准生成超参 |
| 运行时 | `--dtype bf16`, `--device cuda:0`, `--enable-cuda-graph`, `--cuda-graph-modules prefill decode vision`, `--disable-auto-offload`, `--stream-output/--no-stream-output` | 设备与图运行时控制 |
| 其他 | `--scene-name street_scene`, `--prepare-only`, `--debug` | scene-name 仅用于日志/work_dir 命名 |

### 示例 1：街景多图描述

```bash
CUDA_VISIBLE_DEVICES=0 python examples/llm/qwen3vl/qwen3_vl_xh2a_multi_image_demo.py \
    --model-config ./weights/qwen3vl/Qwen3-VL-2B-Instruct-XH2a/export_meta_info.json \
    --image-dir ./data/street_scene \
    --image-glob '*.jpg' \
    --prompt "请按拍摄顺序描述这几张街景图，并指出各张图之间的差异。" \
    --scene-name street_scene \
    --images-per-batch 8 \
    --image-size-w 448 --image-size-h 448 \
    --max-new-tokens 512 \
    --enable-cuda-graph --cuda-graph-modules prefill decode vision \
    --disable-auto-offload
```

### 示例 2：商品对比

```bash
CUDA_VISIBLE_DEVICES=0 python examples/llm/qwen3vl/qwen3_vl_xh2a_multi_image_demo.py \
    --model-config ./weights/qwen3vl/Qwen3-VL-2B-Instruct-XH2a/export_meta_info.json \
    --image-paths ./data/products/sku_a_front.jpg,./data/products/sku_a_side.jpg,./data/products/sku_b_front.jpg,./data/products/sku_b_side.jpg \
    --prompt-file ./prompts/product_compare.txt \
    --scene-name product_compare \
    --images-per-batch 4 \
    --max-new-tokens 384 \
    --enable-cuda-graph
```

### 示例 3：兼容旧 task-json 输入

```bash
CUDA_VISIBLE_DEVICES=0 python examples/llm/qwen3vl/qwen3_vl_xh2a_multi_image_demo.py \
    --model-config ./weights/qwen3vl/Qwen3-VL-2B-Instruct-XH2a/export_meta_info.json \
    --task-json ./data/tasks.json --task-id demo_task_001 \
    --image-dir ./data/task_images \
    --prompt "请整理这组图片的关键信息。" \
    --scene-name task_demo
```

输出工作目录约定为 `./work_dirs/qwen3_vl_multi_image_demo_<scene-name>/`，其中 `<scene-name>` 经过基础合法化处理（仅保留字母数字与 `-_`）。
