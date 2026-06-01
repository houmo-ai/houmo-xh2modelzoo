# Qwen3 MoE

Qwen3 MoE 文本模型示例，覆盖 HMONNX 导出、HMONNX 推理和开发调试。

## 环境

所有命令默认在仓库根目录执行：

```bash
source env.sh
```

常用模型与配置：

- 模型目录：
  - `data/models/Qwen3-30B-A3B`
- 基础配置：
  - `configs_merak/xh2a/llm_models/qwen3moe/30b_a3b/xh2a_qwen3-30b-a3b_w8a8h1_sefp_256_2k.py`

## 快速上手

推荐顺序：

1. 导出 HMONNX，产物会落在 `work_dirs/<cfg_name>/`。
2. 使用导出目录中的 `golden_meta_info.json` 做 HMONNX 推理或 Golden 导出。
3. 开发阶段使用 debug 脚本验证不同运行态。

## 1. 导出 HMONNX

### 方式一：基于配置文件导出

适合开发和调试已有配置。

```bash
python examples_merak/llm/qwen3moe/qwen3moe_xh_export_hmonnx.py --config configs_merak/xh2a/llm_models/qwen3moe/30b_a3b/xh2a_qwen3-30b-a3b_w8a8h1_sefp_256_2k.py
```

### 方式二：直接指定原始模型导出

适合快速跑通标准 `w8a8` 导出流程。

```bash
python examples_merak/llm/qwen3moe/qwen3moe_xh_export_hmonnx.py --model data/models/Qwen3-30B-A3B --context-length 2048 --prefill-chunk-length 256 --quant-type w8a8h1_sefp
```

导出完成后，目标目录下通常会包含：

- `golden_meta_info.json`
- 导出的 HMONNX 文件
- 导出时使用的配置副本

## 2. HMONNX 推理

把 `--config` 替换成导出目录中的 `golden_meta_info.json`。

```bash
python examples_merak/llm/qwen3moe/qwen3moe_xh_hmonnx_generate.py --config work_dirs/<cfg_name>/<export_dir>/golden_meta_info.json --auto-offload
```

常用附加参数：

- `--fast`：切换到 fast 模式
- `--golden`：导出 Golden 输出
- `--auto-offload`：开发阶段调试显存卸载
- `--think`：开启 think 模式

## 3. Golden 导出

```bash
python examples_merak/llm/qwen3moe/qwen3moe_xh_hmonnx_generate.py --config work_dirs/<cfg_name>/<export_dir>/golden_meta_info.json --golden --auto-offload
```

## 4. 开发调试

### 调试不同运行态

```bash
python examples_merak/llm/qwen3moe/debug_scripts/qwen3moe_legacy_xh_generate.py --config configs_merak/xh2a/llm_models/qwen3moe/30b_a3b/xh2a_qwen3-30b-a3b_w8a8h1_sefp_256_2k.py --eval-type quanted_fast --auto-offload
```

### 开启 prefill chunk 调试

```bash
python examples_merak/llm/qwen3moe/debug_scripts/qwen3moe_legacy_xh_generate.py --config configs_merak/xh2a/llm_models/qwen3moe/30b_a3b/xh2a_qwen3-30b-a3b_w8a8h1_sefp_256_2k.py --eval-type quanted_fast --enable-prefill-chunk --auto-offload
```

### 原生 Transformers 对照

```bash
python examples_merak/llm/qwen3moe/debug_scripts/native_qwen3moe_generate.py --model-dir data/models/Qwen3-30B-A3B
```

## 说明

- `--config` 更适合开发和调试复杂参数组合。
- `--model` 更适合快速跑通导出流程。
- 导出目录已存在时，可加 `--force` 重新导出。
