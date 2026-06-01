# GLM-4.7-Flash

GLM-4.7-Flash 文本模型在 `xhmodel_merak` 中的示例，覆盖 HMONNX 导出、HMONNX 推理、Golden 导出和开发调试。

## 环境

所有命令默认在仓库根目录执行：

```bash
source env.sh
```

常用模型与配置：

- 模型目录：`/data02/datasets/chuyuan.wei/GLM-4.7-Flash`
- 基础配置：`configs_merak/xh2a/llm_models/glm_4.7_flash/glm_4_7_flash_xh2a_2k.py`
- GPTQ 配置示例：
  - `configs_merak/xh2a/llm_models/glm_4.7_flash/glm_4_7_flash_xh2a_w4a8_gptq_2k.py`
  - `configs_merak/xh2a/llm_models/glm_4.7_flash/glm_4_7_flash_xh2a_w4a8_hf_gptq_2k.py`

## 1. 导出 HMONNX

基于配置文件导出：

```bash
python examples_merak/llm/glm_4.7_flash/glm_4_7_flash_xh_export_hmonnx.py --config configs_merak/xh2a/llm_models/glm_4.7_flash/glm_4_7_flash_xh2a_2k.py
```

直接指定原始模型导出：

```bash
python examples_merak/llm/glm_4.7_flash/glm_4_7_flash_xh_export_hmonnx.py --model /data02/datasets/chuyuan.wei/GLM-4.7-Flash --context-length 2048 --prefill-chunk-length 256 --quant-type w8a8h1_sefp
```

加载外部 GPTQ 权重导出时，将 `--quant-weight` 替换为实际权重文件：

```bash
python examples_merak/llm/glm_4.7_flash/glm_4_7_flash_xh_export_hmonnx.py --model /data02/datasets/chuyuan.wei/GLM-4.7-Flash --context-length 2048 --prefill-chunk-length 256 --quant-type w4a8h0_ssfp --quant-weight work_dirs/glm_4_7_flash_xh2a_2k_gptq_4bit/gptq-state-dict.safetensors
```

## 2. HMONNX 推理

把 `--config` 替换成导出目录中的 `golden_meta_info.json`：

```bash
python examples_merak/llm/glm_4.7_flash/glm_4_7_flash_xh_hmonnx_generate.py --config work_dirs/<cfg_name>/<export_dir>/golden_meta_info.json
```

Golden 导出：

```bash
python examples_merak/llm/glm_4.7_flash/glm_4_7_flash_xh_hmonnx_generate.py --config work_dirs/<cfg_name>/<export_dir>/golden_meta_info.json --golden
```

## 3. 开发调试

XH wrap 态最小调试：

```bash
python examples_merak/llm/glm_4.7_flash/debug_scripts/glm_4_7_flash_xh_generate.py --config configs_merak/xh2a/llm_models/glm_4.7_flash/glm_4_7_flash_xh2a_2k.py --eval-type wrap
```

原生 Transformers 对照：

```bash
python examples_merak/llm/glm_4.7_flash/debug_scripts/native_glm_4_7_flash_generate.py --model-dir /data02/datasets/chuyuan.wei/GLM-4.7-Flash
```

## 说明

- GPTQ 默认复用 Merak 通用 `quant_weight` 和 HF/GPTQModel checkpoint 加载路径。
- 导出目录已存在时，可加 `--force` 重新导出。
