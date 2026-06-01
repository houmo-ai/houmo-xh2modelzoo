# GPT-OSS

GPT-OSS 文本模型在 `xhmodel_merak` 中的示例，覆盖 HMONNX 导出、HMONNX 推理和开发调试。

## 环境

所有命令默认在仓库根目录执行：

```bash
source env.sh
```

常用模型与配置：

- 模型目录：
  - `data/models/gpt-oss-20b`
  - `data/models/gpt-oss-120b`
- 基础配置：
  - `configs_merak/xh2a/llm_models/gpt_oss/20b/gpt_oss_20b_xh2a_2k.py`
  - `configs_merak/xh2a/llm_models/gpt_oss/120b/gpt_oss_120b_xh2a_2k.py`

## 快速上手

推荐顺序：

1. 导出 HMONNX，产物会落在 `work_dirs/<cfg_name>/`。
2. 使用导出目录中的 `golden_meta_info.json` 做 HMONNX 推理或 Golden 导出。
3. 开发阶段使用 debug 脚本验证不同运行态。

## 1. 导出 HMONNX

### 方式一：基于配置文件导出

```bash
python examples_merak/llm/gpt_oss/gpt_oss_xh_export_hmonnx.py \
  --config configs_merak/xh2a/llm_models/gpt_oss/20b/gpt_oss_20b_xh2a_2k.py
```

### 方式二：直接指定原始模型导出

```bash
python examples_merak/llm/gpt_oss/gpt_oss_xh_export_hmonnx.py \
  --model data/models/gpt-oss-20b \
  --context-length 2048 \
  --prefill-chunk-length 256 \
  --quant-type w8a8h1_sefp
```

## 2. HMONNX 推理

```bash
python examples_merak/llm/gpt_oss/gpt_oss_xh_hmonnx_generate.py \
  --config work_dirs/<cfg_name>/<export_dir>/golden_meta_info.json
```

常用附加参数：

- `--fast`：切换到 fast 模式
- `--golden`：导出 Golden 输出
- `--auto-offload`：开发阶段调试显存卸载

## 3. Golden 导出

```bash
python examples_merak/llm/gpt_oss/gpt_oss_xh_hmonnx_generate.py \
  --config work_dirs/<cfg_name>/<export_dir>/golden_meta_info.json \
  --golden
```

## 4. 开发调试

### 调试不同运行态

```bash
python examples_merak/llm/gpt_oss/debug_scripts/gpt_oss_xh_generate.py \
  --config configs_merak/xh2a/llm_models/gpt_oss/20b/gpt_oss_20b_xh2a_2k.py \
  --eval-type quanted_fast
```

### 原生 Transformers 对照

```bash
python examples_merak/llm/gpt_oss/debug_scripts/native_gpt_oss_generate.py \
  --model-dir data/models/gpt-oss-20b
```

## 说明

- `--config` 更适合开发和调试复杂参数组合。
- `--model` 更适合快速跑通导出流程。
- 导出目录已存在时，可加 `--force` 重新导出。
