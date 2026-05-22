# GLM-OCR 示例脚本

GLM-OCR 是一个基于视觉语言模型的 OCR 模型，支持图片文字识别。本目录包含完整的模型导出（Vision + LLM + PP-DocLayoutV3）、HMONNX 量化推理、Golden 数据导出以及 OCRBench 评测等脚本。

## 目录结构

```
examples/llm/glm_ocr/
├── README.md                              # 本文档
├── common.py                              # 共享工具函数
├── data/                                  # 测试数据
│   ├── img3.png                           # 测试图片
│   └── 18UF.pdf                           # 测试 PDF
├── GLM-OCR/                               # GLM-OCR 官方 SDK（子仓库，需手动 clone）
├── glm_ocr_hmonnx_vision_xh2a_export.py   # 1. Vision 模块导出
├── glm_ocr_hmonnx_llm_xh2a_export.py      # 2. LLM 模块导出
├── ppdoclayoutv3_hmonnx_xh2a_export.py    # 3. PP-DocLayoutV3 版面检测模型导出
├── glm_ocr_hmonnx_demo.py                 # 4. HMONNX 全量化推理 demo
├── glm_ocr_hmonnx_golden.py               # 5. Golden 参考数据导出
├── glm_ocr_ocrbench_fp_eval.py            # 6. OCRBench 浮点评测
├── glm_ocr_ocrbench_hmonnx_eval.py        # 7. OCRBench HMONNX 评测
└── run_ocrbench_compare.py                # 8. 评测结果对比
```

## 环境准备

```bash
# 激活 xhquant 环境（包含 torch 2.8.0、triton 3.4.0 等依赖）
source env.sh

# 指定 GPU（按需修改）
export CUDA_VISIBLE_DEVICES=0
```

### GLM-OCR 官方 SDK（子仓库）

本目录下的 `GLM-OCR/` 是官方 GLM-OCR SDK 的 git 子仓库（submodule），源地址为 `https://github.com/zai-org/GLM-OCR.git`。该子仓库用于提供完整的文档 OCR 管线（版面检测 → 区域裁切 → OCR → 格式化输出），**不会随主仓库自动 clone**，需要手动拉取：

```bash
# 方式一：通过 submodule 初始化（推荐）
cd examples/llm/glm_ocr
git submodule update --init GLM-OCR

# 方式二：手动 clone 到对应目录
cd examples/llm/glm_ocr
git clone https://github.com/zai-org/GLM-OCR.git GLM-OCR
```

> **注意**：请勿向 `GLM-OCR/` 子仓库推送任何 commit。如需更新 SDK 版本，在 `GLM-OCR/` 目录内手动 `git pull` 即可。

安装 SDK 依赖（可选，仅在需要使用 SDK 管线功能时）：

```bash
cd examples/llm/glm_ocr/GLM-OCR
pip install -e .
```

**默认模型路径**: `/data01/datasets/GLM-OCR/`（HuggingFace 格式）

## 脚本依赖关系

```
┌──────────────────────────────────────┐    ┌─────────────────────────────────────────┐    ┌─────────────────────────────────────────────┐
│  glm_ocr_hmonnx_vision_xh2a_export  │    │  glm_ocr_hmonnx_llm_xh2a_export.py (2)  │    │  ppdoclayoutv3_hmonnx_xh2a_export.py (3)    │
│             .py (1)                  │    │                                         │    │  (PP-DocLayoutV3 版面检测模型)              │
└───────────────┬──────────────────────┘    └──────────────────┬──────────────────────┘    └──────────────────────┬──────────────────────┘
                │                                              │                                                    │
                └──────────────────────────────────────────────┼────────────────────────────────────────────────────┘
                                                               ▼
                                              ┌────────────────────────────────────────┐
                                              │   glm_ocr_hmonnx_demo.py (4)           │
                                              │   glm_ocr_hmonnx_golden.py (5)         │
                                              └────────────────────────────────────────┘

┌─────────────────────────────────┐     ┌─────────────────────────────────────┐
│ glm_ocr_ocrbench_fp_eval.py (6) │     │ glm_ocr_ocrbench_hmonnx_eval.py (7) │
│  (仅需 HF 模型)                 │     │  (需 1 + 2 的导出结果)               │
└───────────────┬─────────────────┘     └──────────────────┬──────────────────┘
                │                                          │
                └──────────────┬───────────────────────────┘
                               ▼
              ┌────────────────────────────────┐
              │   run_ocrbench_compare.py (8)  │
              └────────────────────────────────┘
```

---

## 脚本说明

### 1. Vision 模块导出 (`glm_ocr_hmonnx_vision_xh2a_export.py`)

将 GLM-OCR 的 Vision 编码器导出为 ONNX，并转换为 HMONNX（XH2a 硬件加速格式）。

```bash
python examples/llm/glm_ocr/glm_ocr_hmonnx_vision_xh2a_export.py \
    --hf_model_dir /data01/datasets/GLM-OCR \
    --work_dir work_dirs/glm_ocr_vision_xh2a_export_hmonnx
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--hf_model_dir` | `/data01/datasets/GLM-OCR` | HF 模型目录 |
| `--work_dir` | `work_dirs/glm_ocr_vision_xh2a_export_hmonnx` | 输出目录 |
| `--image_path` | `examples/llm/glm_ocr/data/img3.png` | 校准图片路径 |
| `--image_size_w` | `672` | 图像宽度 |
| `--image_size_h` | `672` | 图像高度 |
| `--max_size_t` | `2` | 最大时间维度 |
| `--patch_size` | `14` | patch 大小 |
| `--temporal_patch_size` | `2` | 时间 patch 大小 |
| `--valid` | `True` | 导出后是否验证 |
| `--skip_golden` | `False` | 跳过 HMONNX golden 生成 |

**产出目录**:
```
work_dirs/glm_ocr_vision_xh2a_export_hmonnx/
├── vision/                          # 量化后的 HMONNX 模型
│   └── glm_ocr_vision_xh2a_export_hmonnx.onnx
└── onnx/                            # 原始 ONNX 模型
    └── visual_1.onnx
```

---

### 2. LLM 模块导出 (`glm_ocr_hmonnx_llm_xh2a_export.py`)

将 GLM-OCR 的 LLM 部分（prefill + decode）导出为量化 ONNX/HMONNX，流程包括：wrap → frontend graph → quant graph → PTQ → export。

```bash
python examples/llm/glm_ocr/glm_ocr_hmonnx_llm_xh2a_export.py \
    --hf_model_dir /data01/datasets/GLM-OCR \
    --work_dir work_dirs/glm_ocr_llm_xh2a_2k_export
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--hf_model_dir` | `/data01/datasets/GLM-OCR` | HF 模型目录 |
| `--work_dir` | `work_dirs/glm_ocr_llm_xh2a_2k_export` | 输出目录 |
| `--image_path` | `examples/llm/glm_ocr/data/img3.png` | 校准图片路径 |
| `--max_sequence_length` | `2048` | 最大序列长度 |
| `--input_sequence_length` | `256` | prefill 输入序列长度 |
| `--image_size_w` | `672` | 图像宽度 |
| `--image_size_h` | `672` | 图像高度 |
| `--target_device` | `XH2a` | 目标硬件 |
| `--valid` | `True` | 导出后是否验证 |
| `--skip_golden` | `False` | 跳过 HMONNX golden 生成 |

**产出目录**:
```
work_dirs/glm_ocr_llm_xh2a_2k_export/
├── prefill_onnx/                    # Prefill ONNX 模型
│   └── glm_ocr_llm_xh2a_2k_export_prefill.onnx
├── decode_onnx/                     # Decode ONNX 模型
│   └── glm_ocr_llm_xh2a_2k_export_decode.onnx
├── hf_config/                       # HF 配置文件
├── token_embedding.pt               # Token Embedding 权重
└── export_meta_info.json            # 导出元信息
```

---

### 3. PP-DocLayoutV3 版面检测模型导出 (`ppdoclayoutv3_hmonnx_xh2a_export.py`)

将文档版面检测模型（PP-DocLayoutV3）导出为 HMONNX，用于文档版面分析。

```bash
python examples/llm/glm_ocr/ppdoclayoutv3_hmonnx_xh2a_export.py \
    --model_dir /data01/datasets/ppdoclayoutv3_safetensors \
    --work_dir work_dirs/ppdoclayoutv3_xh2a_export_hmonnx
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_dir` | `/data01/datasets/ppdoclayoutv3_safetensors` | PP-DocLayoutV3 safetensors 模型目录 |
| `--work_dir` | `work_dirs/ppdoclayoutv3_xh2a_export_hmonnx` | 输出目录 |
| `--image` | `examples/llm/glm_ocr/data/18UF.pdf` | 校准图片或 PDF 路径 |
| `--pdf_page` | `1` | PDF 校准时使用的页码（1-based） |
| `--quant_type` | `w16a16_sefp` | 量化方案 |
| `--device_type` | `XH2a` | 目标硬件 |
| `--force_export` | `False` | 强制重新导出中间 ONNX |
| `--force_convert` | `False` | 强制重新转换 HMONNX |
| `--skip_golden` | `False` | 跳过 HMONNX golden 校验 |

---

### 4. HMONNX 全量化推理 Demo (`glm_ocr_hmonnx_demo.py`)

使用导出的 HMONNX Vision + LLM 模型进行完整的量化推理。

```bash
python examples/llm/glm_ocr/glm_ocr_hmonnx_demo.py \
    --model_dir work_dirs/glm_ocr_llm_xh2a_2k_export \
    --vision_export_dir work_dirs/glm_ocr_vision_xh2a_export_hmonnx \
    --image examples/llm/glm_ocr/data/img3.png \
    --max_new_tokens 256
```

识别 PDF 时，示例会先将页面渲染成图片，再逐页调用同一套 HMONNX Vision + LLM 推理链路：

```bash
python examples/llm/glm_ocr/glm_ocr_hmonnx_demo.py \
    --model_dir work_dirs/glm_ocr_llm_xh2a_2k_export \
    --vision_export_dir work_dirs/glm_ocr_vision_xh2a_export_hmonnx \
    --pdf examples/llm/glm_ocr/data/18UF.pdf \
    --pdf_output_dir work_dirs/glm_ocr_pdf_pages \
    --output_path work_dirs/glm_ocr_xh2a_hmonnx_demo/18UF_hmonnx_ocr.txt \
    --max_new_tokens 256
```

只识别部分页可增加 `--pdf_pages`，例如 `--pdf_pages 1` 或 `--pdf_pages 1-3,5`。PDF 渲染依赖 PyMuPDF；若环境没有 `fitz`，请先安装 `pymupdf`，或手动将 PDF 转成图片后使用 `--image` / `--images_json`。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_dir` | `work_dirs/glm_ocr_llm_xh2a_2k_export` | LLM 导出目录 |
| `--vision_export_dir` | `work_dirs/glm_ocr_vision_xh2a_export_hmonnx` | Vision 导出目录 |
| `--image` | `examples/llm/glm_ocr/data/img3.png` | 输入图片 |
| `--pdf` | `None` | PDF 输入路径；设置后逐页 OCR |
| `--pdf_output_dir` | `work_dirs/glm_ocr_pdf_pages` | PDF 页面图片输出目录 |
| `--pdf_dpi` | `200` | PDF 渲染 DPI |
| `--pdf_pages` | `None` | 页码选择，支持 `1`、`1,3`、`1-3,5` |
| `--prompt` | `None` | 自定义提示词 |
| `--images_json` | `None` | JSON 批量输入 |
| `--max_new_tokens` | `256` | 最大生成 token 数 |
| `--use_fast` | `True` | HMONNX fast 模式 |
| `--do_sample` | `False` | 是否采样解码 |
| `--output_path` | `None` | 输出文本文件路径 |

---

### 5. Golden 参考数据导出 (`glm_ocr_hmonnx_golden.py`)

使用 HMONNX 模型运行推理，同时导出 vision / prefill / decode 各阶段的输入输出 `.npy` 文件，用于硬件端验证和调试。

```bash
python examples/llm/glm_ocr/glm_ocr_hmonnx_golden.py \
    --model_dir work_dirs/glm_ocr_llm_xh2a_2k_export \
    --vision_export_dir work_dirs/glm_ocr_vision_xh2a_export_hmonnx \
    --image examples/llm/glm_ocr/data/img3.png \
    --max_decode_steps 8
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_dir` | `work_dirs/glm_ocr_llm_xh2a_2k_export` | LLM 导出目录 |
| `--vision_export_dir` | `work_dirs/glm_ocr_vision_xh2a_export_hmonnx` | Vision 导出目录 |
| `--image` | `examples/llm/glm_ocr/data/img3.png` | 输入图片 |
| `--max_decode_steps` | `8` | 最大 decode 步数 |
| `--resume` | `False` | 断点续跑 |
| `--skip_layout_golden` | `False` | 跳过 PP-DocLayoutV3 HMONNX golden |
| `--layout_model_dir` | `/data01/datasets/ppdoclayoutv3_safetensors` | 版面检测模型目录 |

**产出目录**:
```
work_dirs/glm_ocr_xh2a_hmonnx_golden/golden/
├── golden_summary.json              # 推理结果摘要
├── vision/                          # Vision 阶段 golden
│   ├── pixel_values.npy
│   ├── image_grid_thw.npy
│   ├── image_embeds.npy
│   └── hmquant_*_input.npy / *_output.npy
├── prefill/                         # Prefill 阶段 golden
│   ├── input_ids.npy
│   ├── logits.npy
│   └── prefill_step_*/
└── decode/                          # Decode 阶段 golden（每步一个子目录）
    ├── decode_0/
    ├── decode_1/
    └── ...
```

---

### 6. OCRBench 浮点评测 (`glm_ocr_ocrbench_fp_eval.py`)

使用 HF 浮点模型在 OCRBench（1000 条样本，10 个 OCR 子任务）上进行评测。

```bash
python examples/llm/glm_ocr/glm_ocr_ocrbench_fp_eval.py \
    --model /data01/datasets/GLM-OCR/ \
    --output_dir ./work_dirs/glm_ocr_ocrbench_fp \
    --max_samples 100   # 可选，限制样本数加速调试
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `/data02/datasets/GLM-OCR/` | HF 模型路径 |
| `--ocrbench_tsv` | `/data01/home/henry/LMUData/OCRBench.tsv` | OCRBench 数据集路径 |
| `--output_dir` | `./work_dirs/glm_ocr_ocrbench_fp` | 输出目录 |
| `--max_new_tokens` | `2048` | 最大生成 token 数 |
| `--max_samples` | `None` | 限制样本数（调试用） |
| `--dtype` | `float16` | 数据类型 |
| `--resume` | `False` | 断点续跑 |

**产出**: `ocrbench_results.json`（逐样本结果）、`ocrbench_scores.json`（分类得分）

---

### 7. OCRBench HMONNX 评测 (`glm_ocr_ocrbench_hmonnx_eval.py`)

使用导出的 HMONNX 量化模型在 OCRBench 上进行评测。

```bash
python examples/llm/glm_ocr/glm_ocr_ocrbench_hmonnx_eval.py \
    --model_dir work_dirs/glm_ocr_llm_xh2a_2k_export \
    --vision_export_dir work_dirs/glm_ocr_vision_xh2a_export_hmonnx \
    --output_dir ./work_dirs/glm_ocr_ocrbench_hmonnx \
    --max_samples 100   # 可选
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_dir` | `work_dirs/glm_ocr_llm_xh2a_2k_export` | LLM 导出目录 |
| `--vision_export_dir` | `work_dirs/glm_ocr_vision_xh2a_export_hmonnx` | Vision 导出目录 |
| `--ocrbench_tsv` | `/data01/home/henry/LMUData/OCRBench.tsv` | 数据集路径 |
| `--output_dir` | `./work_dirs/glm_ocr_ocrbench_hmonnx` | 输出目录 |
| `--max_new_tokens` | `2048` | 最大生成 token 数 |
| `--max_samples` | `None` | 限制样本数 |
| `--vision_mode` | `quantized` | `quantized` / `unquantized` |
| `--use_fast` | `False` | HMONNX fast 模式 |
| `--resume` | `False` | 断点续跑 |

**产出**: `ocrbench_results.json`、`ocrbench_scores.json`

---

### 8. 评测结果对比 (`run_ocrbench_compare.py`)

对比浮点 vs HMONNX 的 OCRBench 评测结果，打印分类精度对比表格并列出差异样本。

```bash
python examples/llm/glm_ocr/run_ocrbench_compare.py \
    --fp_dir ./work_dirs/glm_ocr_ocrbench_fp \
    --hmonnx_dir ./work_dirs/glm_ocr_ocrbench_hmonnx
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--fp_dir` | `./work_dirs/glm_ocr_ocrbench_fp` | 浮点评测结果目录 |
| `--hmonnx_dir` | `./work_dirs/glm_ocr_ocrbench_hmonnx` | HMONNX 评测结果目录 |
| `--show_diffs` | `20` | 显示差异样本的最大数量 |

---

## 使用指南

### 一、HMONNX 模型导出（三步）

将 HuggingFace 模型导出为 XH2a 硬件可执行的 HMONNX 量化模型，需要依次完成以下三步：

#### Step 1：导出 Vision 编码器

```bash
python examples/llm/glm_ocr/glm_ocr_hmonnx_vision_xh2a_export.py \
    --hf_model_dir /data01/datasets/GLM-OCR \
    --work_dir work_dirs/glm_ocr_vision_xh2a_export_hmonnx
```

#### Step 2：导出 LLM（Prefill + Decode）

```bash
python examples/llm/glm_ocr/glm_ocr_hmonnx_llm_xh2a_export.py \
    --hf_model_dir /data01/datasets/GLM-OCR \
    --work_dir work_dirs/glm_ocr_llm_xh2a_2k_export
```

#### Step 3：导出 PP-DocLayoutV3 版面检测模型

```bash
python examples/llm/glm_ocr/ppdoclayoutv3_hmonnx_xh2a_export.py \
    --model_dir /data01/datasets/ppdoclayoutv3_safetensors \
    --work_dir work_dirs/ppdoclayoutv3_xh2a_export_hmonnx
```

**导出完成后，目录结构如下**：

```
work_dirs/
├── glm_ocr_vision_xh2a_export_hmonnx/  # Step 1 产出
│   ├── vision/
│   └── onnx/
├── glm_ocr_llm_xh2a_2k_export/        # Step 2 产出
│   ├── prefill_onnx/
│   ├── decode_onnx/
│   ├── hf_config/
│   ├── token_embedding.pt
│   └── export_meta_info.json
└── ppdoclayoutv3_xh2a_export_hmonnx/   # Step 3 产出
    └── hmonnx/
```

---

### 二、Demo 脚本

#### HMONNX 量化推理 Demo

使用导出的 HMONNX Vision + LLM 模型进行端到端量化推理（需先完成导出 Step 1 + Step 2）：

```bash
# 单张图片
python examples/llm/glm_ocr/glm_ocr_hmonnx_demo.py \
    --model_dir work_dirs/glm_ocr_llm_xh2a_2k_export \
    --vision_export_dir work_dirs/glm_ocr_vision_xh2a_export_hmonnx \
    --image examples/llm/glm_ocr/data/img3.png

# PDF 逐页 OCR
python examples/llm/glm_ocr/glm_ocr_hmonnx_demo.py \
    --model_dir work_dirs/glm_ocr_llm_xh2a_2k_export \
    --vision_export_dir work_dirs/glm_ocr_vision_xh2a_export_hmonnx \
    --pdf examples/llm/glm_ocr/data/18UF.pdf \
    --pdf_pages 1-3 \
    --output_path work_dirs/glm_ocr_xh2a_hmonnx_demo/18UF_hmonnx_ocr.txt
```

---

## 快速上手

**完整流程（从导出到评测）**:

```bash
source env.sh
export CUDA_VISIBLE_DEVICES=0

# Step 1: 导出 Vision 模块
python examples/llm/glm_ocr/glm_ocr_hmonnx_vision_xh2a_export.py

# Step 2: 导出 LLM 模块
python examples/llm/glm_ocr/glm_ocr_hmonnx_llm_xh2a_export.py

# Step 3: 导出 PP-DocLayoutV3 版面检测模型
python examples/llm/glm_ocr/ppdoclayoutv3_hmonnx_xh2a_export.py

# Step 4: HMONNX 推理 demo
python examples/llm/glm_ocr/glm_ocr_hmonnx_demo.py

# Step 5: 导出 golden 参考数据
python examples/llm/glm_ocr/glm_ocr_hmonnx_golden.py

# Step 6: OCRBench 评测（浮点 + HMONNX）
python examples/llm/glm_ocr/glm_ocr_ocrbench_fp_eval.py --max_samples 100
python examples/llm/glm_ocr/glm_ocr_ocrbench_hmonnx_eval.py --max_samples 100

# Step 7: 对比评测结果
python examples/llm/glm_ocr/run_ocrbench_compare.py
```

## 模型参数

| 参数 | 值 |
|------|----|
| 隐藏层数 (`num_hidden_layers`) | 16 |
| KV Head 数 (`num_kv_heads`) | 8 |
| Head 维度 (`head_dim`) | 128 |
| KV Cache 长度 (`cache_len`) | 2048 |
| Prefill 序列长度 (`input_sequence_length`) | 256 |
| 图像尺寸 | 672 × 672 |
| pad_token_id | 59246 |
| eos_token_id | [59246, 59253] |
