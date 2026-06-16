# Qwen3 

Qwen3 文本模型在 `xhmodel_merak` 中的示例，覆盖 GPTQ 量化、HMONNX 导出、HMONNX 推理、Golden 导出、评测和调试。

## 环境

所有命令默认在仓库根目录执行：

```bash
source env.sh
```

常用模型与配置：

- 模型目录：
  - `data/models/Qwen3-1.7B`
  - `data/models/Qwen3-8B`
- 基础配置：
  - `configs_merak/xh2a/llm_models/qwen3/1.7b/qwen3_1.7b_xh2a_2k.py`
  - `configs_merak/xh2a/llm_models/qwen3/8b/qwen3_8b_xh2a_2k.py`
- GPTQ 配置示例：
  - `configs_merak/xh2a/llm_models/qwen3/1.7b/qwen3_1.7b_xh2a_w4a8_gptq_2k.py`
  - `configs_merak/xh2a/llm_models/qwen3/8b/qwen3_8b_xh2a_w4a8_hf_gptq_2k.py`

## 快速上手

如果你只想最快跑通一遍，推荐顺序：

1. 可选：先做 GPTQ 量化，拿到 4bit 权重。
2. 导出 HMONNX，产物会落在 `work_dirs/<cfg_name>/`。
3. 使用导出目录下的 `golden_meta_info.json` 做 HMONNX 推理或 Golden 导出。
4. 需要验证运行态时，再使用 debug 或评测脚本。

## 1. GPTQ 量化

4bit GPTQ 示例：

```bash
python examples_merak/llm/qwen3/qwen3_xh_gptq_quant.py --model data/models/Qwen3-8B --bits 4 --group-size 64
```

量化结果会保存在新生成的目录中，名称类似 `Qwen3-8B-4bit-64g/`。

## 2. 导出 HMONNX

### 方式一：基于配置文件导出

适合开发和调试已有配置。

```bash
python examples_merak/llm/qwen3/qwen3_xh_export_hmonnx.py --config configs_merak/xh2a/llm_models/qwen3/1.7b/qwen3_1.7b_xh2a_2k.py
```

### 方式二：直接指定原始模型导出

适合快速导出标准 `w8a8` 流程。

```bash
python examples_merak/llm/qwen3/qwen3_xh_export_hmonnx.py --model data/models/Qwen3-8B --context-length 2048  --prefill-chunk-length 256 --quant-type w8a8h1_sefp
```

### 方式三：加载 GPTQ 权重导出

适合 `w4a8` 流程。`--quant-weight` 替换为你实际生成的量化权重文件。

```bash
python examples_merak/llm/qwen3/qwen3_xh_export_hmonnx.py --model data/models/Qwen3-8B --context-length 2048  --prefill-chunk-length 256 --quant-type w4a8h0_ssfp --quant-weight work_dirs/qwen3_8b_instruct_xh2a_2k_quarot_gptq_4bit/quarot_gptq-state-dict.safetensors
```

导出完成后，目标目录下会包含：

- `golden_meta_info.json`
- 导出的 HMONNX 文件
- 导出时使用的配置副本

## 3. HMONNX 推理

把 `--config` 替换成导出目录中的 `golden_meta_info.json`。

```bash
python examples_merak/llm/qwen3/qwen3_xh_hmonnx_generate.py --config work_dirs/<cfg_name>/<export_dir>/golden_meta_info.json
```

常用附加参数：

- `--fast`：切换到 fast 模式
- `--auto-offload`：开发阶段调试显存卸载
- `--think`：开启 think 模式

## 4. Golden 导出

```bash
python examples_merak/llm/qwen3/qwen3_xh_hmonnx_generate.py --config work_dirs/<cfg_name>/<export_dir>/golden_meta_info.json --golden
```

## 5. 评测

### lm-eval 评测

```bash
python examples_merak/llm/llm_eval.py --config configs_merak/xh2a/llm_models/qwen3/8b/qwen3_8b_xh2a_2k.py --eval-type quanted_fast --task cmmlu
```

常见任务包括：

- `cmmlu`
- `mmlu`
- `gsm8k`
- `arc_challenge`
- `hellaswag`

## 6. 开发调试

### 量化态调试

```bash
c
```

### 开启 prefill chunk 调试

```bash
python examples_merak/llm/qwen3/debug_scripts/qwen3_xh_generate.py --config configs_merak/xh2a/llm_models/qwen3/8b/qwen3_8b_xh2a_2k.py --eval-type quanted_fast --enable-prefill-chunk
```

## 说明

- `--config` 更适合开发和调试复杂参数组合。
- `--model` 更适合快速跑通导出流程。
- 导出目录已存在时，可加 `--force` 重新导出。
