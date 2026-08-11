# Unlimited-OCR 调试与 golden 对齐脚本

本目录提供 Unlimited-OCR base/no-crop 单图路径的逐段调试脚本，用于定位误差来自
visual、scatter、LLM、frontend/quant 还是 HMONNX runtime。

所有脚本默认使用 `xhmodel312` 环境，示例图 `data/images/qwen2_vl_demo.jpeg`，
提示词统一使用字面量反斜杠 `\\n`：`<image>\\nFree OCR. `。bash 示例请用单引号，避免把 `\\n` 展开成真实换行。

## 阶段与脚本

| 阶段 | 脚本 | 作用 |
|---|---|---|
| HF native | `native_unlimited_ocr_forward.py` | 本地化 HF 模型 base 单图 forward，产出 golden 参考（prefill logits + 下一 token） |
| Visual wrap | `unlimited_ocr_visual_xh_debug.py` | wrap-state visual 子模型输出 image embeddings，可 `--compare-hf` 与 HF visual 对齐 |
| Multimodal prefill | `unlimited_ocr_llm_xh_debug.py` | wrap-state LLM 用 HF visual embeddings scatter 后跑 prefill，和 native golden 比较 argmax/logits |
| HMONNX generate | `../unlimited_ocr_xh_hmonnx_generate.py` | 导出产物端到端 generate（`--golden` 保存 golden） |

## 输出目录

- 每个 wrap 脚本在 `./work_dirs/<config_stem>/` 下写日志（`visual_debug.log` / `llm_debug.log`）。
- golden dump 目录建议统一到 `./work_dirs/unlimited_ocr_debug/`：
  - `native_prefill.pt`：native HF 的 `input_ids`、`images_seq_mask`、`last_logits`、`next_token`。
  - `visual_wrap.pt`：wrap visual 的 image embeddings。

## 推荐对齐顺序

```bash
# 1) native golden（reference）
python examples_merak/llm/unlimited_ocr/debug_scripts/native_unlimited_ocr_forward.py \
  --model data/models/Unlimited-OCR \
  --image-path data/images/qwen2_vl_demo.jpeg \
  --prompt '<image>\nFree OCR. ' \
  --dump work_dirs/unlimited_ocr_debug/native_prefill.pt

# 2) visual wrap vs HF visual
python examples_merak/llm/unlimited_ocr/debug_scripts/unlimited_ocr_visual_xh_debug.py \
  --config configs_merak/xh2a/llm_models/unlimited_ocr/base/unlimited_ocr_visual_base_xh2a_32k.py \
  --image-path data/images/qwen2_vl_demo.jpeg \
  --compare-hf --debug

# 3) multimodal prefill wrap vs native golden
python examples_merak/llm/unlimited_ocr/debug_scripts/unlimited_ocr_llm_xh_debug.py \
  --config configs_merak/xh2a/llm_models/unlimited_ocr/base/unlimited_ocr_llm_base_xh2a_32k.py \
  --image-path data/images/qwen2_vl_demo.jpeg \
  --golden work_dirs/unlimited_ocr_debug/native_prefill.pt --debug

# 4) HMONNX generate smoke
python examples_merak/llm/unlimited_ocr/unlimited_ocr_xh_hmonnx_generate.py \
  --config work_dirs/_unlimited_ocr_xh2a_debug/<hmquant_dir>/golden_meta_info.json \
  --image-path data/images/qwen2_vl_demo.jpeg --prompt '<image>\nFree OCR. ' --debug
```

## 阈值

- Visual wrap vs HF visual：数值对齐用 `--fp32`，`--atol` 默认 `1e-2`。fp32 下 `wrap` 状态误差应接近 0（约 1e-3）；不加 `--fp32` 的 fp16 会有约 0.04 的累积误差（12 层 SAM），属正常半精度噪声，不代表逻辑错误。`quanted_aligned` 状态会更大，需按量化误差放宽。
- Multimodal prefill：优先看 last-token argmax 是否一致（`PASS`/`FAIL`）。注意导出配置 `num_logits_to_keep=1` 时 wrap 只返回最后一个 token 的 logits，last-logits 的绝对差可能较大，argmax 一致即视为通过。
- base/no-crop 图像 token 固定为 273，shape 不符直接说明 processor 或 visual 出错。

## 失败排查

- shape mismatch（visual 输出不是 `(1, 273, 1280)`）：检查 visual wrap 算子替换与静态尺寸。
- token/features mismatch：检查 processor 的 `<image>` token 数与 visual feature 数是否一致。
- prefill 长度 > 256：base 单图约 278 token，`unlimited_ocr_llm_xh_debug.py` 会把 wrap input sequence length 放宽到 prompt 长度跑单次 forward；HMONNX runtime 走 chunked prefill。
- argmax 对不上但 shape 对：多为量化误差，改用 `--eval-type wrap` 隔离量化后再定位。

## 已知限制

- 仅覆盖 base/no-crop 单图；crop/gundam 走 eager，不在此目录做 HMONNX 对齐。
- HMONNX 的逐层 golden dump 依赖 runtime `enable_golden`；本目录聚焦端到端与 wrap 级对齐。
