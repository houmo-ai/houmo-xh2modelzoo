# Kimi MoE (月肉\_0.2B/2.4B/30B) 示例说明

本目录提供 Kimi MoE 模型在 XH2a 侧的导出与评测示例。

## 1. 环境准备

### 1.1 Python 依赖

```bash
pip install transformers==4.51.0
```

### 1.2 依赖安装

本项目依赖 `xhquant` 和 `xh_model_zoo`，请确保已正确安装：

```bash
pip install xhquant
pip install -e .
```

## 2. 脚本说明

| 脚本 | 功能 |
|------|------|
| `kimi_moe_xh2a_export.py` | 将 Kimi MoE 模型导出为 XH2a 格式 |
| `kimimoe_xh2a_generate.py` | 使用 XH2a 格式模型进行生成任务评测 |
| `kimimoe_onnx_golden.py` | 生成 ONNX Golden 数据用于验证 |
| `configs/` | 配置文件目录 |

## 3. 导出 XH2a 模型

### 3.1 命令行参数

```bash
python examples/llm/kimi_moe/kimi_moe_xh2a_export.py \
    --config configs/kimi/3b_30b/kimi_a3b_30b_instruct_legacy_xh2a_2k_batch.py \
    --seed 1024
```

### 3.2 主要参数说明

- `--config`: 配置文件路径（必需）
- `--seed`: 随机种子，默认 1024
- `--eval-type`: 评测类型，默认 `pytorch`
- `--offload`: 是否启用自动 offload

## 4. 生成任务评测

```bash
python examples/llm/kimi_moe/kimimoe_xh2a_generate.py \
    --config configs/kimi/3b_30b/kimi_a3b_30b_instruct_legacy_xh2a_2k_batch.py \
    --prompt "你多大了？用中文回答。"
```

## 5. 配置文件说明

配置文件位于 `configs/kimi/` 目录，主要包含以下配置项：

- `model`: 模型配置（模型路径、量化参数等）
- `dataset`: 数据集配置
- `work_dir`: 工作目录
- `device`: 运行设备
- `dtype`: 数据类型

## 6. 常见问题

- **CUDA 显存不足**：尝试减小 `input_sequence_length` 或启用 `--offload`
- **模型加载失败**：检查 `hf_model` 路径是否正确
- **量化精度问题**：确认量化配置文件中的 `quant_config` 参数

## 7. 目录结构

```
examples/llm/kimi_moe/
├── configs/                  # 配置文件
│   └── kimi/
│       └── 3b_30b/          # 3B+30B MoE 配置
├── kimi_moe_xh2a_export.py   # 模型导出脚本
├── kimimoe_xh2a_generate.py # 生成评测脚本
├── kimimoe_onnx_golden.py   # ONNX Golden 生成
├── modify.py                # 模型修改工具
└── Readme.md               # 本文档
```
