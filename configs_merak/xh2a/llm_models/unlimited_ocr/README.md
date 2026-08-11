# Unlimited-OCR merak 配置

Unlimited-OCR 在 `xhmodel_merak` 体系下的 XH2a 量化/导出配置模板。原模型位于
`./data/models/Unlimited-OCR`，结构基于 DeepSeek-OCR / DeepSeekV2（SAM ViT-B + CLIP-L
视觉塔 + DeepSeekV2 MoE 主干）。

## 目录结构

```text
unlimited_ocr/
├── _unlimited_ocr_xh2a.py                    # 共享 LLM 默认（32k context、w8a8 quant_scheme、base visual_config）
├── _unlimited_ocr_xh2a_calib.py              # 真实图校准的高精度变体（w16a16，calib_config.enable=True）
├── base/
│   ├── unlimited_ocr_llm_base_xh2a_32k.py    # base/no-crop LLM 导出配置（主交付路径）
│   └── unlimited_ocr_visual_base_xh2a_32k.py # base/no-crop visual 导出配置
└── gundam/
    ├── unlimited_ocr_llm_gundam_xh2a_32k.py    # crop/gundam LLM 配置（eager，不产 HMONNX）
    └── unlimited_ocr_visual_gundam_xh2a_32k.py # crop/gundam visual 配置
```

## 固定参数

| 项 | 值 |
|----|-----|
| `hf_model` | `./data/models/Unlimited-OCR` |
| 主模型 `model_type` | `UnlimitedOCRForCausalLM` |
| 视觉子模型 `model_type` | `UnlimitedOCRForCausalLM_visual` |
| image token id | `128815` |
| patch size / downsample | `16` / `4` |
| context 长度 | `32768` |
| `prefill_chunk_length` | `256` |
| quant 方案 | `w8a8h1_sefp`（LLM + visual） |
| sliding window 字段 | `sliding_window_size=128`（见下方 KV 说明） |

## base 与 gundam 差异

| 模式 | `crop_mode` | `image_size` | `base_size` | 视觉 token 数 | HMONNX | 用途 |
|------|-------------|--------------|-------------|---------------|--------|------|
| base | `False` | `1024` | `1024` | 固定 273 | 支持（主交付路径） | 单页 / global OCR |
| gundam | `True` | `640` | `1024` | 随 crop 数变化（273 起） | 不支持（动态 crop 破坏静态 shape），走 eager | 大图 dynamic crop |

配置里同时保留 `hmonnx_export` 标记：base/no-crop 为 `True`，gundam/crop 为 `False`。
导出入口会在 gundam/crop 配置上直接报错，避免自动导出流程误把动态 crop 路径当成正式
HMONNX 产物。

## 配置选择

- **base w8a8 正式导出**：用 `base/unlimited_ocr_llm_base_xh2a_32k.py`（`model_name=xh2_unlimited_ocr_base_w8a8_256_32k`）。
- **共享底座**：`_unlimited_ocr_xh2a.py` 是 base/gundam 的公共父配置，也能直接导 base，但命名不带 `base_` 前缀，正式导出建议用 `base/` 下的显式配置。
- **高精度校准变体**：`_unlimited_ocr_xh2a_calib.py` 用 `w16a16h0_sefp` + 真实文档图 PTQ 校准，适合隔离量化误差做数值对齐验证，不是 w8a8 交付配置。

默认模型路径是 `./data/models/Unlimited-OCR`。如果权重不在默认目录，设置环境变量
`UNLIMITED_OCR_HF_MODEL=/path/to/Unlimited-OCR` 覆盖配置中的 `hf_model`。真实图片校准目录可用
`UNLIMITED_OCR_CALIB_IMAGE_DIR=/path/to/images` 覆盖 `calib_config.image_dir`。

## KV cache / sliding window 说明

原模型 `SlidingWindowLlamaAttention` 用 ring buffer：prefill（图像 + prompt）永久可见，
只对 decode 生成的 token 做 128 窗口 ring。XH 现有 `attention_max_length` 只能表达
「全量 KV」或「最近 W 个绝对位置」，无法表达「保护前缀 + decode ring」这种分段语义。

当前适配在 wrap / 导出图里 decode 都用**全量 KV**（`attention_max_length=-1`），
即保留全部图像 + prompt + 已生成 token。这保证图像/prompt 不被滑掉、数值正确，
代价是超长输出 KV 显存比原 ring buffer 高。`sliding_window_size=128` 字段仍保留在
config 中记录原模型语义，但不会再被固化成 128 绝对滑窗。

提交中的结构测试覆盖 decode 位置超过 128 后仍继续使用绝对 past position；完整端到端验证使用
`examples_merak/llm/unlimited_ocr/unlimited_ocr_xh_hmonnx_generate.py`，固定图片和 prompt 后检查
base visual、LLM prefill/decode 与生成文本。

完全复刻官方 ring buffer（protected-prefix + decode ring，即 R-SWA）需要在
`xhquant` 侧新增算子能力，属于后续增强项。
