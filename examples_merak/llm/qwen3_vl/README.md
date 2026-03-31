# Qwen3 VL

Qwen3-VL 多模态示例，覆盖校准数据准备、量化、HMONNX 导出、HMONNX 推理和开发调试。

## 环境

所有命令默认在仓库根目录执行：

```bash
source env.sh
```

依赖说明：

- `transformers 4.57.1`

常用模型与配置：

- 模型目录：
  - `data/models/Qwen3-VL-2B-Instruct`
  - `data/models/Qwen3-VL-4B-Instruct`
  - `data/models/Qwen3-VL-8B-Instruct`
- 2B 配置：
  - `configs_merak/xh2a/llm_models/qwen3_vl/2b/qwen3_vl_llm_2b_xh2a_2k.py`
  - `configs_merak/xh2a/llm_models/qwen3_vl/2b/qwen3_vl_visual_2b_xh2a_2k.py`

## 快速上手

推荐顺序：

1. 准备校准数据。
2. 可选：先跑 `qwen3_vl_common_quant.py` 生成量化权重。
3. 导出 HMONNX，产物会落在 `work_dirs/<cfg_name>/`。
4. 使用导出目录中的 `golden_meta_info.json` 做 HMONNX 推理或 Golden 导出。
5. 开发阶段用 debug 脚本验证 visual 和 llm 两条链路。

## 1. 校准数据准备

默认提供了 5 个结构化校准数据文件，位于 `data/calib_data/`：

- `data/calib_data/Qwen2.5-VL-7B-Instruct_CMMMU_VAL_20250923102519_struct.json`
- `data/calib_data/Qwen2.5-VL-7B-Instruct_COCO_VAL_20250923104643_struct.json`
- `data/calib_data/Qwen2.5-VL-7B-Instruct_DocVQA_VAL_20250923102720_struct.json`
- `data/calib_data/Qwen2.5-VL-7B-Instruct_MMMU_DEV_VAL_20250923102615_struct.json`
- `data/calib_data/Qwen2.5-VL-7B-Instruct_OCRBench_20250923102641_struct.json`

对应图片数据可从以下地址下载，再解压到 `data/calib_data/`：

- `http://10.10.1.53:8081/artifactory/model_zoo2/houmo/vllm_datasets/LMUData.tar.gz`

## 2. 模型量化

以下命令使用同一组校准数据，区别仅在模型尺寸。

### 2B

```bash
python examples_merak/llm/qwen3_vl/qwen3_vl_common_quant.py --model data/models/Qwen3-VL-2B-Instruct --data_files data/calib_data/Qwen2.5-VL-7B-Instruct_CMMMU_VAL_20250923102519_struct.json data/calib_data/Qwen2.5-VL-7B-Instruct_COCO_VAL_20250923104643_struct.json data/calib_data/Qwen2.5-VL-7B-Instruct_DocVQA_VAL_20250923102720_struct.json data/calib_data/Qwen2.5-VL-7B-Instruct_MMMU_DEV_VAL_20250923102615_struct.json --calib_samples 64
```

### 4B

```bash
python examples_merak/llm/qwen3_vl/qwen3_vl_common_quant.py --model data/models/Qwen3-VL-4B-Instruct --data_files data/calib_data/Qwen2.5-VL-7B-Instruct_CMMMU_VAL_20250923102519_struct.json data/calib_data/Qwen2.5-VL-7B-Instruct_COCO_VAL_20250923104643_struct.json data/calib_data/Qwen2.5-VL-7B-Instruct_DocVQA_VAL_20250923102720_struct.json data/calib_data/Qwen2.5-VL-7B-Instruct_MMMU_DEV_VAL_20250923102615_struct.json --calib_samples 64
```

### 8B

```bash
python examples_merak/llm/qwen3_vl/qwen3_vl_common_quant.py --model data/models/Qwen3-VL-8B-Instruct --data_files data/calib_data/Qwen2.5-VL-7B-Instruct_CMMMU_VAL_20250923102519_struct.json data/calib_data/Qwen2.5-VL-7B-Instruct_COCO_VAL_20250923104643_struct.json data/calib_data/Qwen2.5-VL-7B-Instruct_DocVQA_VAL_20250923102720_struct.json data/calib_data/Qwen2.5-VL-7B-Instruct_MMMU_DEV_VAL_20250923102615_struct.json --calib_samples 64
```

量化结果默认保存在 `work_dirs/<model_name>_quarot_gptq_transformers-<version>/` 下。

## 3. 导出 HMONNX

### 方式一：基于配置文件导出

适合开发和调试已有配置。

```bash
python examples_merak/llm/qwen3_vl/qwen3_vl_xh_export_hmonnx.py --config configs_merak/xh2a/llm_models/qwen3_vl/2b/qwen3_vl_llm_2b_xh2a_2k.py
```

### 方式二：加载量化权重导出

适合 `w4a8` 流程。`--quant-weight` 替换为量化章节生成的权重文件。

```bash
python examples_merak/llm/qwen3_vl/qwen3_vl_xh_export_hmonnx.py --model data/models/Qwen3-VL-2B-Instruct --context-length 2048 --prefill-chunk-length 256 --quant-type w4a8h0_ssfp --quant-weight work_dirs/Qwen3-VL-2B-Instruct_quarot_gptq_transformers-4.57.6/quarot_gptq_use_hession_mse_False_calib_samples_64_heading_gptq_False-state-dict.safetensors
```

导出完成后，目标目录下通常会包含：

- `golden_meta_info.json`
- 导出的 HMONNX 文件
- 导出时使用的配置副本

## 4. HMONNX 推理

把 `--config` 替换成导出目录中的 `golden_meta_info.json`。

```bash
python examples_merak/llm/qwen3_vl/qwen3_vl_xh_hmonnx_generate.py --config work_dirs/<cfg_name>/<export_dir>/golden_meta_info.json
```

常用附加参数：

- `--fast`：切换到 fast 模式
- `--golden`：导出 Golden 输出
- `--auto-offload`：开发阶段调试显存卸载
- `--image-path`：指定输入图片

## 5. Golden 导出

```bash
python examples_merak/llm/qwen3_vl/qwen3_vl_xh_hmonnx_generate.py --config work_dirs/<cfg_name>/<export_dir>/golden_meta_info.json --golden
```

## 6. 开发调试

### 调试 visual 分支

```bash
python examples_merak/llm/qwen3_vl/debug_scripts/qwen3_vl_visual_xh_generate.py --config configs_merak/xh2a/llm_models/qwen3_vl/2b/qwen3_vl_visual_2b_xh2a_2k.py --eval-type quanted_fast
```

### 调试 llm 分支

```bash
python examples_merak/llm/qwen3_vl/debug_scripts/qwen3_vl_llm_xh_generate.py --config configs_merak/xh2a/llm_models/qwen3_vl/2b/qwen3_vl_llm_2b_xh2a_2k.py --eval-type quanted_fast
```

### 原生 Transformers 对照

```bash
python examples_merak/llm/qwen3_vl/debug_scripts/native_qwen3_vl_generate.py --model-dir data/models/Qwen3-VL-2B-Instruct
```

## 说明

- `--config` 更适合开发和调试复杂参数组合。
- `--model` 更适合快速跑通导出流程。
- 导出目录已存在时，可加 `--force` 重新导出。
