---
name: migrate-xhquantllm-to-xh2modelzoo
description: 将 xhquant_llm 模型家族迁移到 xh2modelzoo 的 LLMConverter 架构，并完成 converter/convert_config、architecture 注册、examples 导出脚本与 hmonnx 验证闭环。当用户要求把已有 xhquant_llm 模型同步到 xh2modelzoo、补齐 xh2modelzoo 缺失模型支持、排查迁移后导出或推理失败时使用。
---

# xhquant_llm -> xh2modelzoo 迁移

## 目标
把源仓 `xhquant_llm` 的模型支持迁移到目标仓 `xh2modelzoo`，并且保证：
1. 代码落在 `LLMConverter` 体系内，不保留旧 `LLMBaseModel` 导出范式。
2. `llm_converter.py` 可按 architecture 正确分派 converter。
3. 至少有 export + hmonnx test 的可运行闭环。

## 先读这些文件
1. [migration-workflow.md](./references/migration-workflow.md)
2. [validation-checklist.md](./references/validation-checklist.md)
3. [shared-model-matrix.md](./references/shared-model-matrix.md)

## 执行顺序（强制）
1. 先刷新共同模型矩阵，确认你要迁移的家族最接近哪个模板。
2. 按模板创建/修改 `models/<family>/` 下的 `*convert_config.py`、`*converter.py`、`__init__.py`。
3. 在 `xh_model_zoo/xh_llm/llm_converter.py` 增加 architecture 分支。
4. 新增或更新 `examples/llm/<family>/*_xh2a_export_hmonnx.py`。
5. 跑迁移产物检查脚本，修复缺失项。
6. 跑导出烟测，再跑 hmonnx 测试；必要时补 eval。

## 模板选择规则
- `llm`：优先参考 `qwen2_legacy` / `qwen3_legacy` / `llama` / `fm9g`
- `vlm-ocr`：优先参考 `qwen2_5_vl` / `qwen3_vl` / `glm_ocr` / `paddleocr_vl`
- `moe`：优先参考 `qwen3moe`
- `multi-component`：优先参考 `minicpmo`

如果模型在两个仓都已存在，优先按 [shared-model-matrix.md](./references/shared-model-matrix.md) 的同名家族迁移；如果只在源仓存在，选择“结构最接近”的重合模板。

## 必须运行的脚本

1. 刷新矩阵：
```bash
python skills/migrate-xhquantllm-to-xh2modelzoo/scripts/discover_shared_models.py \
  --xh2modelzoo-root /home/jiangyong.yu/xh2_work/xh2modelzoo \
  --xhquant-llm-root /home/jiangyong.yu/xh2_work/xhquant_llm \
  --write skills/migrate-xhquantllm-to-xh2modelzoo/references/shared-model-matrix.md
```

2. 检查迁移产物：
```bash
python skills/migrate-xhquantllm-to-xh2modelzoo/scripts/check_migration_artifacts.py \
  --xh2modelzoo-root /home/jiangyong.yu/xh2_work/xh2modelzoo \
  --model <family> \
  --archetype <llm|vlm-ocr|moe|multi-component> \
  --architecture <ArchitectureString> \
  --require-example
```

## 迁移时的硬约束
1. 不要把 `xhquant_llm/examples/<family>/*_xh2a_export.py` 原样拷到目标仓；必须改成 `LLMConverter.from_pretrained(...)` 风格。
2. 不要只迁 `models/` 不迁 `llm_converter.py` 分支。
3. 不要遗漏 `meta.json` 关键字段（`hf_model_path`、onnx 路径、kv cache 相关信息）。
4. VLM/OCR 必须同时验证 llm 与 vision 分支，不接受只通一个分支。
5. 多组件模型必须按组件拆 export，不接受“一脚本全包但不可复用”。

## 最低交付物
1. `xh_model_zoo/xh_llm/models/<family>/` 下可用 converter 代码。
2. `llm_converter.py` architecture 注册完成。
3. `examples/llm/<family>/` 下至少 1 个可运行 export 脚本。
4. 导出成功产物：prefill/decode ONNX + `meta.json`。
5. 至少 1 次 hmonnx 推理验证通过。

## 默认排障顺序
1. architecture 分支是否命中。
2. converter 的 `load_hf_model` / `register_wrap_modules`。
3. quant config 与输入签名（prefill/decode）是否一致。
4. inference/hf_compatible 对 `meta.json` 字段的读取是否匹配。
