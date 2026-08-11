# Unlimited-OCR Merak 适配说明

本目录承接 Unlimited-OCR 的 HF remote code 本地化、Merak wrap、XH operator 替换、HMONNX 导出和 runtime 推理。当前支持 **base/no-crop 单图路径**的 LLM + visual 静态图导出；crop/gundam 路径保留 eager/wrap smoke，不产出 HMONNX。

## 源码来源与 patch 边界

以下文件是 Unlimited-OCR HF remote code 及其 DeepSeekV2 依赖的本地化副本，来源为本仓库运行时权重目录
`data/models/Unlimited-OCR` 中随 checkpoint 分发的实现；本地化改动限于相对 import、项目内配置注册和
必要的 transformers 版本兼容，不承载 XH 算子替换逻辑：

- `configuration_deepseek_v2.py`
- `modeling_deepseekv2.py`
- `deepencoder.py`
- `conversation.py`
- `modeling_unlimitedocr.py`

来源基线：当前副本以 `data/models/Unlimited-OCR` checkpoint 目录中的 remote code 为准，该 checkpoint
对应 Baidu `Unlimited-OCR`（README release 记录 2026/06/28 vLLM 支持，模型卡 license 为 MIT）。如果后续
checkpoint 或上游 remote code 更新，先整体同步上述本地化副本，再重新应用下方 Merak/XH patch 文件。
同步时必须保留原始版权/许可声明，确认 `licenses/Unlimited-OCR/LICENSE` 和 `THIRD_PARTY_NOTICES` 仍有效，
并重新运行本目录 README 中的 wrap、token diff 和 HMONNX generate 验证流程。

Merak/XH 专用 patch 和导出逻辑集中在以下文件，后续升级上游模型代码时应优先 diff 这些边界：

- `modeling_unlimitedocr_patch.py`
- `xh_unlimited_ocr_config.py`
- `data_preprocess.py`
- `unlimited_ocr_processor.py`
- `unlimited_ocr_model.py`
- `unlimited_ocr_visual_model.py`
- `unlimited_ocr_hmonnx_inference.py`
- `_llm_model_impl.py`
- `_visual_model_impl.py`

同步策略：上游模型行为变化只落在本地化副本；XH 算子替换、full-KV 决策、HMONNX 导出限制和环境变量
路径覆盖只落在 Merak/XH 专用文件，避免把项目适配逻辑散进大体量上游源码。

## 关键文件

- `modeling_unlimitedocr.py`：本地化 HF `UnlimitedOCRConfig`、`UnlimitedOCRModel`、`UnlimitedOCRForCausalLM` 以及原始推理辅助逻辑。
- `modeling_deepseekv2.py`：Unlimited-OCR 文本侧 DeepSeekV2 decoder 和官方 `SlidingWindowLlamaAttention` ring-buffer 参考实现。
- `deepencoder.py`：SAM ViT-B、CLIP-L 和 projector 视觉编码组件。
- `modeling_unlimitedocr_patch.py`：本地模型 patch 入口。
- `_llm_model_impl.py`：LLM wrap/XH operator 替换，包括 RMSNorm、RoPE、MoE、KV cache、MaskedSoftmax。
- `unlimited_ocr_model.py`：主模型注册、prefill/decode `ModelSwitcher`、visual 子模型串联、HMONNX 导出入口。
- `unlimited_ocr_visual_model.py`：base visual 静态图导出和 crop/gundam eager visual 封装。
- `data_preprocess.py`：base/no-crop 文本 token embedding、图像 embedding scatter、KV cache 输入组织。
- `unlimited_ocr_hmonnx_inference.py`：HMONNX runtime，串联 visual HMONNX 和 LLM prefill/decode HMONNX。

## Sliding Window 处理

官方 Unlimited-OCR 的 `SlidingWindowLlamaAttention` 不是标准“最近 W 个绝对 token”滑窗，而是：

```text
全部 prefill(图像 token + prompt) 永久可见 + 最近 W 个 decode token ring buffer
```

XH 现有 `LLMCacheV2` / `MaskedSoftmax` 只有单个 `attention_max_length`，表达的是标准连续窗口，无法表达“protected prefill prefix”。如果 decode 阶段直接固化 `attention_max_length=128`，图像/prompt 前缀会被滑出 KV 窗口，HMONNX 会退化成短重复或计数循环。

当前采用方案二：在 `_llm_model_impl.py` 的 `_SlidingWindowLlamaAttention` 中，prefill/decode 均使用 full KV：

```text
MaskedSoftmax.attention_max_length = -1
k_cache.attention_max_length = -1
v_cache.attention_max_length = -1
```

这个方案保留图像/prompt 永久可见，牺牲官方 ring buffer 对早期 decode token 的 KV 裁剪。wrap vs native 200 token 对比中，仅观察到坐标数字级 fp16 抖动，没有早期重复退化。

## 推荐导出配置

base/no-crop w8a8 正式导出使用显式 base 配置：

```bash
python examples_merak/llm/llm_export_hmonnx.py \
	--config configs_merak/xh2a/llm_models/unlimited_ocr/base/unlimited_ocr_llm_base_xh2a_32k.py \
	--force --debug
```

关键配置：

```text
model_name              xh2_unlimited_ocr_base_w8a8_256_32k
quant_type              w8a8h1_sefp
visual_quant            w8a8h1_sefp
export_mode             base
crop_mode               False
prefill_chunk_length    256
context_max_length      32768
```

`configs_merak/xh2a/llm_models/unlimited_ocr/_unlimited_ocr_xh2a_calib.py` 使用 `w16a16h0_sefp` 和真实图片 calibration，适合隔离量化误差，不作为 base w8a8 正式产物配置。

## 验证流程

Prompt 口径统一使用字面量反斜杠 `\\n`，即 Python 字符串 `"<image>\\nFree OCR. "`，bash 示例写作 `'<image>\nFree OCR. '`。不要使用 `$'<image>\nFree OCR. '` 或 Python 字符串 `"<image>\nFree OCR. "`，这会变成真实换行并导致 HMONNX 诊断口径不一致。

默认权重路径为 `./data/models/Unlimited-OCR`，可通过 `UNLIMITED_OCR_HF_MODEL` 覆盖；真实图片
PTQ 校准目录可通过 `UNLIMITED_OCR_CALIB_IMAGE_DIR` 覆盖。base/no-crop 是唯一正式 HMONNX 路径，
gundam/crop 配置带 `hmonnx_export=False`，导出入口会明确拒绝。

### 1. wrap vs native decode smoke

先验证 full-KV decode 逻辑没有退化：

```bash
python examples_merak/llm/unlimited_ocr/debug_scripts/unlimited_ocr_sliding_window_verify.py \
	--config configs_merak/xh2a/llm_models/unlimited_ocr/base/unlimited_ocr_llm_base_xh2a_32k.py \
	--image-path data/images/unlimited_ocr_pdf_page1.png \
	--generate-tokens 200
```

预期：不出现早期计数循环；若有单个坐标 token 差异（如 `273` vs `274`），通常属于 fp16 抖动。

### 2. HMONNX token diff

导出后对比 HF native 和 HMONNX greedy token：

```bash
python examples_merak/llm/unlimited_ocr/debug_scripts/unlimited_ocr_hmonnx_token_diff.py \
	--hf-model data/models/Unlimited-OCR \
	--hmonnx-config work_dirs/unlimited_ocr_llm_base_xh2a_32k_debug/hmquant_xh2_unlimited_ocr_base_w8a8_256_32k_<date>/golden_meta_info.json \
	--image-path data/images/unlimited_ocr_pdf_page1.png \
	--prompt '<image>\nFree OCR. ' \
	--max-new-tokens 128 \
	--debug
```

重点看：

```text
input ids equal=True
[comparison]
first N token(s) match
```

w8a8 HMONNX 不保证与 HF native 完全逐 token 一致，但不应在前几个 token 退化成短重复或 `1. 2. 3...` 计数循环。

### 3. HMONNX 单图生成

```bash
python examples_merak/llm/unlimited_ocr/unlimited_ocr_xh_hmonnx_generate.py \
	--config work_dirs/unlimited_ocr_llm_base_xh2a_32k_debug/hmquant_xh2_unlimited_ocr_base_w8a8_256_32k_<date>/golden_meta_info.json \
	--image-path data/images/unlimited_ocr_pdf_page1.png \
	--prompt '<image>\nFree OCR. ' \
	--max-new-tokens 128 \
	--debug
```

输出应包含图像正文内容（例如 `Baidu`、`Unlimited OCR Works` 等），不应很短 EOS 或重复计数。

### 4. 小批量 OmniDocBench smoke

```bash
python examples_merak/llm/unlimited_ocr/unlimited_ocr_omnidocbench_infer.py \
	--mode hmonnx \
	--hmonnx-config work_dirs/unlimited_ocr_llm_base_xh2a_32k_debug/hmquant_xh2_unlimited_ocr_base_w8a8_256_32k_<date>/golden_meta_info.json \
	--image-dir /path/to/OmniDocBench/images \
	--output-dir outputs/omnidocbench/hmonnx_plan2_smoke \
	--limit 20 \
	--prompt-mode free-ocr \
	--save-raw \
	--debug
```

检查 `manifest.json`、`.md` 和 `.raw.txt`，确认 `failed=0`、输出非空且无大面积重复。

## 已知限制

- 当前 HMONNX 只覆盖 base/no-crop 单图路径。
- crop/gundam 有 eager/wrap smoke，但动态 crop 数会破坏静态 shape，暂不导出 HMONNX。
- full-KV decode 会比官方 ring buffer 占用更多 KV cache，换取正确保留图像/prompt 前缀和稳定导出。
- MoE gate 仍可能存在 fp16 routing 抖动；若需要进一步贴近官方，可将 gate linear/softmax 改为 fp32。
