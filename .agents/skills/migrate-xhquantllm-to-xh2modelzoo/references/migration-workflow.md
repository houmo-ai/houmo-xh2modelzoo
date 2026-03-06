# Migration Workflow (xhquant_llm -> xh2modelzoo)

## 1. Scope
- 目标：把 `xhquant_llm` 中某个模型族迁移到 `xh2modelzoo` 的 `LLMConverter` 架构。
- 入口目录：
  - 源：`xhquant_llm/xhquant_llm/models/<family>`、`xhquant_llm/examples/<family>`
  - 目标：`xh2modelzoo/xh_model_zoo/xh_llm/models/<family>`、`xh2modelzoo/examples/llm/<family>`

先读 [shared-model-matrix.md](./shared-model-matrix.md)，选最近邻模板再开始改。

## 2. 架构差异（必须先统一认知）
- 旧范式（xhquant_llm）：`MODELS.register_module` + `LLMBaseModel` + 大型 `*_xh2a_export.py`。
- 新范式（xh2modelzoo）：`*ConvertConfig` + `*ConverterXH2a(HFTransfromersConverter)` + `LLMConverter.from_pretrained(...)`。
- 迁移核心不是“逐行复制”，而是把旧模型逻辑收敛到新的 Converter 生命周期：
  1. `load_hf_model`
  2. `register_wrap_modules`
  3. `convert_fx_model_to_quanted_model`
  4. `convert_quanted_model_to_hmonnx`
  5. 产出 `meta.json`

## 3. 模板选择
- `llm`：优先对齐 `qwen2_legacy` / `qwen3_legacy` / `llama` / `fm9g`
- `vlm-ocr`：优先对齐 `qwen2_5_vl` / `qwen3_vl` / `glm_ocr` / `paddleocr_vl`
- `moe`：优先对齐 `qwen3moe`
- `multi-component`：优先对齐 `minicpmo`

## 4. 文件落位清单

### 4.1 模型目录
目标最小集（按族型增减）：
- `xh_model_zoo/xh_llm/models/<family>/__init__.py`
- `xh_model_zoo/xh_llm/models/<family>/_model.py` 或 `_model_impl.py`（VLM 含 `_llm_model_impl.py` / `_vision_model_impl.py`）
- `xh_model_zoo/xh_llm/models/<family>/<family>_convert_config.py`
- `xh_model_zoo/xh_llm/models/<family>/<family>_converter.py`（或 `<family>_convert.py`）
- `xh_model_zoo/xh_llm/models/<family>/<family>_hf_compatible.py`（如需 hmonnx generate/eval）
- `xh_model_zoo/xh_llm/models/<family>/inference.py`（如需 hmonnx 测试）

### 4.2 Converter 路由
- 在 `xh_model_zoo/xh_llm/llm_converter.py` 注册 architecture 分支。
- 推荐显式 architecture 字符串，避免只靠 `None` 自动分派。

### 4.3 示例脚本
- 新增 `examples/llm/<family>/<family>_xh2a_export_hmonnx.py`
- 推荐补齐：
  - `examples/llm/<family>/<family>_xh2a_hmonnx_test.py`
  - `examples/llm/<family>/<family>_eval.py`

## 5. 迁移动作顺序（强制）
1. 从源模型抽取 wrap 逻辑（`register_wrap_modules`、cache 结构、特殊算子改写）。
2. 把量化/导出流程收敛到目标 `*ConverterXH2a`，优先复用 `BaseConverter`/`HFTransfromersConverter` 的公共能力。
3. 写 `*ConvertConfig`，确保参数可由 export 脚本透传。
4. 更新 `__init__.py` 暴露 `ConvertConfig` / `Converter` / `Inference` / `HFCompatible`。
5. 在 `llm_converter.py` 注册 architecture 到 converter class。
6. 写 export 脚本（调用 `LLMConverter.from_pretrained`），再写 hmonnx 测试与评测脚本。
7. 跑验证闭环（见 [validation-checklist.md](./validation-checklist.md)）。

## 6. 常见迁移坑
- 只复制 `_model.py`，漏了 `llm_converter.py` 注册。
- 保留旧 `LLMBaseModel` 导出脚本，未切换到 `LLMConverter.from_pretrained`。
- `input_sequence_length` / `context_length` 与 wrap 输入 shape 不一致。
- VLM 只迁移 LLM 分支，漏 `_vision_model_impl.py` 或视觉导出。
- LoRA/GPTQ 权重路径沿用旧字段名，导致 `quant_weight` 未加载。
- 没有同步 `hf_config` 拷贝与 `meta.json` 字段，后续 hmonnx 推理找不到资源。

## 7. 推荐命令（示例）
```bash
# 1) 更新共同模型矩阵
python skills/migrate-xhquantllm-to-xh2modelzoo/scripts/discover_shared_models.py \
  --xh2modelzoo-root /home/jiangyong.yu/xh2_work/xh2modelzoo \
  --xhquant-llm-root /home/jiangyong.yu/xh2_work/xhquant_llm \
  --write skills/migrate-xhquantllm-to-xh2modelzoo/references/shared-model-matrix.md

# 2) 检查某个家族的迁移产物（示例：qwen3_legacy）
python skills/migrate-xhquantllm-to-xh2modelzoo/scripts/check_migration_artifacts.py \
  --xh2modelzoo-root /home/jiangyong.yu/xh2_work/xh2modelzoo \
  --model qwen3_legacy \
  --archetype llm \
  --architecture Qwen3ForCausalLM_legacy \
  --require-example
```
