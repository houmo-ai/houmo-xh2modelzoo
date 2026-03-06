# Validation Checklist (xh2modelzoo)

## 1. 静态检查
- [ ] `models/<family>/` 包含 `__init__.py`、`*convert_config.py`、`*converter.py`
- [ ] `llm_converter.py` 已注册目标 architecture
- [ ] `examples/llm/<family>/` 至少有一个 export 脚本

建议先跑：
```bash
python skills/migrate-xhquantllm-to-xh2modelzoo/scripts/check_migration_artifacts.py \
  --xh2modelzoo-root /home/jiangyong.yu/xh2_work/xh2modelzoo \
  --model <family> \
  --archetype <llm|vlm-ocr|moe|multi-component> \
  --architecture <ArchitectureString> \
  --require-example
```

## 2. 导出烟测（必做）
- [ ] `*_xh2a_export_hmonnx.py` 可启动
- [ ] 产出 `work_dirs/<prefix>/meta.json`
- [ ] 产出 prefill/decode ONNX

典型命令：
```bash
python examples/llm/<family>/<family>_xh2a_export_hmonnx.py \
  --model <hf_model_dir> \
  --context-length 2048 \
  --input-sequence-length 256 \
  --quant-type w8a8h1_sefp
```

## 3. HMONNX 推理验证（必做）
- [ ] `*_xh2a_hmonnx_test.py` 可加载 `meta.json`
- [ ] 至少 1 组 prompt 输出正常（非空、非崩溃）

典型命令：
```bash
python examples/llm/<family>/<family>_xh2a_hmonnx_test.py \
  --config work_dirs/<prefix>/meta.json
```

## 4. 评测回归（推荐）
- [ ] `*_eval.py` 或等价评测脚本可执行
- [ ] 保存结果到文件（ppl/accuracy 任一）

典型命令：
```bash
python examples/llm/<family>/<family>_eval.py \
  --config work_dirs/<prefix>/meta.json
```

## 5. 与源仓基线对齐（推荐）
- [ ] 同 prompt 对比 `xhquant_llm` 的 naive/generate 输出风格
- [ ] 若为 VLM/OCR：同图输入下关键字段（bbox/text）可对齐
- [ ] 若为 MoE：至少验证 1 个长上下文场景

## 6. 失败排查顺序
1. 先查 `llm_converter.py` architecture 分支是否命中。
2. 再查 converter 内 `load_hf_model` 和 `register_wrap_modules`。
3. 再查 quant config 与 cache shape（prefill/decode 输入一致性）。
4. 最后查 `meta.json` 字段与 inference/hf_compatible 的读取字段是否一致。
