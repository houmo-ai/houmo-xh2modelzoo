# Qwen3.5

Qwen3.5 文本模型在 `xhmodel_merak` 中的最小迁移示例，覆盖配置、HMONNX 导出、wrap 调试和 HMONNX 生成。

## 环境

所有命令默认在仓库根目录执行：

```bash
source env.sh
```

## 配置

- 基础配置：`configs_merak/xh2a/llm_models/qwen3_5/_qwen3_5_xh2a_2k.py`
- 9B 配置：`configs_merak/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_instruct_xh2a_2k.py`
- 默认模型目录：`data/models/Qwen3.5-9B`

## 导出 HMONNX

使用配置文件导出：

```bash
python examples_merak/llm/qwen3_5/qwen3_5_xh_export_hmonnx.py --config configs_merak/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_instruct_xh2a_2k.py
```

或者直接指定模型目录：

```bash
python examples_merak/llm/qwen3_5/qwen3_5_xh_export_hmonnx.py --model data/models/Qwen3.5-9B --context-length 2048 --prefill-chunk-length 256 --quant-type w8a8h1_sefp
```

## 调试视觉模型

```bash
python examples_merak/llm/qwen3_5/debug_scripts/qwen3_5_visual_xh_generate.py --config configs_merak/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_instruct_visual_xh2a_2k.py --eval-type quanted_fast
```

## 语言模型+视觉模型

```bash
调试语言模型+视觉模型
python examples_merak/llm/qwen3_5/debug_scripts/qwen3_5_llm_xh_generate.py --config configs_merak/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_instruct_xh2a_2k.py --eval-type quanted_fast
```

## 原生 HF 对比

```bash
python examples_merak/llm/qwen3_5/debug_scripts/native_qwen3_5_generate.py --model-dir data/models/Qwen3.5
```

## requirements
- qwen-vl-utils==0.0.14
- transformers==5.3.0
```