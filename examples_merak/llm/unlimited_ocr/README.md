# Unlimited-OCR (XH2a) 适配

Unlimited-OCR 是在 DeepSeek-OCR / DeepSeekV2 基础上改造的视觉-语言 OCR 模型。本目录汇总它在
`xhmodel_merak` 里的适配流程、命令、产物路径和已知限制。

视觉塔来自 `deepencoder.py`（SAM ViT-B + CLIP-L + linear projector + newline/separator），
LLM 主干是 DeepSeekV2 MoE + sliding window。上一代实现参考
`xh_model_zoo/xh_llm/models/deepseek_ocr`。

## 目录结构

```text
xhmodel_merak/xh_llm/models/unlimited_ocr/
├── modeling_unlimitedocr.py / modeling_deepseekv2.py / deepencoder.py   # 本地化 HF 源码（去 remote code）
├── modeling_unlimitedocr_patch.py        # patch 入口
├── xh_unlimited_ocr_config.py            # XHUnlimitedOCRModelConfig / VisualConfig
├── unlimited_ocr_model.py                # 主模型注册 + wrap + 导出 + HF 兼容 generate
├── unlimited_ocr_visual_model.py         # visual 子模型（base 导出 + crop eager）
├── _llm_model_impl.py                    # LLM trace/quant 算子替换（DeepseekV2 + attention）
├── _visual_model_impl.py                 # visual trace/quant 算子替换（SAM/CLIP）
├── data_preprocess.py                    # 多模态 scatter（image embeds -> <image> 位）
├── unlimited_ocr_processor.py            # prompt/image 预处理（base + crop）
└── unlimited_ocr_hmonnx_inference.py     # HMONNX runtime

configs_merak/xh2a/llm_models/unlimited_ocr/
├── _unlimited_ocr_xh2a.py                # 共享 LLM 默认（含 base visual_config）
├── _unlimited_ocr_xh2a_calib.py          # w16a16 真实图校准变体
├── base/   unlimited_ocr_{llm,visual}_base_xh2a_32k.py
└── gundam/ unlimited_ocr_{llm,visual}_gundam_xh2a_32k.py

examples_merak/llm/unlimited_ocr/
├── unlimited_ocr_workflow.py             # 新版统一 quant/export/golden 入口（base）
├── unlimited_ocr_xh_hmonnx_generate.py   # HMONNX generate（base）
├── unlimited_ocr_omnidocbench_infer.py   # OmniDocBench 批量推理（hf / wrap / hmonnx）
└── debug_scripts/                        # 逐段 golden 对齐 + crop smoke
```

## 固定参数

| 项 | 值 |
|----|-----|
| `hf_model` | `./data/models/Unlimited-OCR` |
| 主模型 `model_type` | `UnlimitedOCRForCausalLM` |
| visual `model_type` | `UnlimitedOCRForCausalLM_visual` |
| image token id | `128815` |
| patch size / downsample | `16` / `4` |
| base 视觉 token 数 | `273 = (16+1)*16+1` |
| quant 方案 | `w8a8h1_sefp` |

## base 与 gundam/crop 区别

| | base | gundam/crop |
|--|------|-------------|
| `crop_mode` | `False` | `True` |
| `image_size` / `base_size` | `1024` / `1024` | `640` / `1024` |
| 视觉 token 数 | 固定 273 | 随 crop ratio 变化（273 起） |
| 导出 HMONNX | 支持（量化导出主路径） | 不支持（动态 crop 破坏静态 shape），走 eager |
| 典型用途 | 单页 / global OCR | 大图 dynamic crop |

## Prompt 口径（重要）

`<image>\nFree OCR. ` 里的 `\n` 必须是**字面量反斜杠 + n**，不是真实换行。真实换行
prompt 会让 tokenizer 得到不同 token，量化后 HMONNX 可能退化到 `unable to extract`
模板。所有脚本默认 prompt 已统一为字面量 `\\n`。

- **bash 命令行**：用单引号，`--prompt '<image>\nFree OCR. '`（单引号不展开 `\n`）。
- **Python 字符串**：写成 `"<image>\\nFree OCR. "`。
- markdown 任务对应 `<image>\n<|grounding|>Convert the document to markdown. `（同样字面量 `\n`）。

## 命令

先激活项目 Python 环境，例如 `conda activate xhmodel312`；以下命令用 `python` 表示当前环境解释器。

如果模型权重或校准图片不在默认目录，可在运行前覆盖：

```bash
export UNLIMITED_OCR_HF_MODEL=/path/to/Unlimited-OCR
export UNLIMITED_OCR_CALIB_IMAGE_DIR=/path/to/calib/images
```

### 0. Workflow 导出与 golden（推荐）

新版统一入口使用 Workflow YAML 串联 `quant -> export -> dump_golden`。Unlimited-OCR 当前没有独立的
HF 权重量化阶段，因此 YAML 顶层固定为 `quant: null`；W8A8/W16A16 图量化仍由
`export.model.quant_scheme` 和 `calib_config` 控制。

```bash
python examples_merak/llm/unlimited_ocr/unlimited_ocr_workflow.py \
  --model-dir data/models/Unlimited-OCR \
  --config-path configs_merak/workflows/xh2a/llm_models/unlimited_ocr/base/unlimited_ocr_base_xh2a_w8a8.yaml \
  --export-output-dir work_dirs/unlimited_ocr_workflow_base \
  --dump-golden \
  --image data/images/unlimited_ocr_pdf_page1.png \
  --prompt '<image>\nFree OCR. ' \
  --device cuda \
  --overwrite
```

真实图 W16A16 数值对齐配置：

```text
configs_merak/workflows/xh2a/llm_models/unlimited_ocr/base/unlimited_ocr_base_xh2a_w16a16_calib.yaml
```

使用该配置前必须将 `UNLIMITED_OCR_CALIB_IMAGE_DIR` 指向包含有效图片的目录；校准开启但目录为空或
不存在时，Workflow 会在加载模型前直接报错，避免静默退化为 dummy calibration。

Workflow 仅支持 base/no-crop HMONNX。gundam/crop 继续使用旧 Python 配置执行 eager/debug smoke。
旧 Python 配置和 `llm_export_hmonnx.py` 保留为兼容及底层调试入口。

### 1. 配置加载 smoke

```bash
python - <<'PY'
from xhquant.api import Config
from xhmodel_merak.xh_llm import AutoLLMConfig
cfg = Config.fromfile("configs_merak/xh2a/llm_models/unlimited_ocr/base/unlimited_ocr_llm_base_xh2a_32k.py")
mc = AutoLLMConfig.from_pretrained(cfg.model)
print(type(mc).__name__, type(mc.visual_config).__name__, mc.image_token_id)
PY
```

### 2. HF native golden（base 单图 forward）

```bash
python examples_merak/llm/unlimited_ocr/debug_scripts/native_unlimited_ocr_forward.py \
  --model data/models/Unlimited-OCR \
  --image-path data/images/qwen2_vl_demo.jpeg \
  --prompt '<image>\nFree OCR. ' \
  --dump work_dirs/unlimited_ocr_debug/native_prefill.pt
```

### 3. visual wrap 对齐（vs HF）

```bash
python examples_merak/llm/unlimited_ocr/debug_scripts/unlimited_ocr_visual_xh_debug.py \
  --config configs_merak/xh2a/llm_models/unlimited_ocr/base/unlimited_ocr_visual_base_xh2a_32k.py \
  --image-path data/images/qwen2_vl_demo.jpeg --compare-hf --fp32 --debug
```

### 4. multimodal prefill 对齐（wrap vs native golden）

```bash
python examples_merak/llm/unlimited_ocr/debug_scripts/unlimited_ocr_llm_xh_debug.py \
  --config configs_merak/xh2a/llm_models/unlimited_ocr/base/unlimited_ocr_llm_base_xh2a_32k.py \
  --image-path data/images/qwen2_vl_demo.jpeg \
  --golden work_dirs/unlimited_ocr_debug/native_prefill.pt --debug
```

### 5. HMONNX 导出（base）

```bash
python examples_merak/llm/llm_export_hmonnx.py \
  --config configs_merak/xh2a/llm_models/unlimited_ocr/base/unlimited_ocr_llm_base_xh2a_32k.py --force --debug
```

产物：`work_dirs/unlimited_ocr_llm_base_xh2a_32k_debug/hmquant_<model_name>_<yyyymmdd>/`，含
`golden_meta_info.json`、`quant_embedding.pt`、`prefill/`、`decode/`、`visual/`、`hf_config/`。

### 6. HMONNX generate（base）

```bash
python examples_merak/llm/unlimited_ocr/unlimited_ocr_xh_hmonnx_generate.py \
  --config work_dirs/unlimited_ocr_llm_base_xh2a_32k_debug/<hmquant_dir>/golden_meta_info.json \
  --image-path data/images/unlimited_ocr_pdf_page1.png --prompt '<image>\nFree OCR. ' --max-new-tokens 128 --debug
```

固定样例应至少输出 `Baidu`、`Unlimited OCR Works` 等页面文本。gundam/crop 配置只用于 eager/debug，
不支持 HMONNX 导出；若误用于正式 HMONNX export，导出入口会报错。

### 7. HF vs HMONNX token diff

```bash
python examples_merak/llm/unlimited_ocr/debug_scripts/unlimited_ocr_hmonnx_token_diff.py \
  --hf-model data/models/Unlimited-OCR \
  --hmonnx-config work_dirs/unlimited_ocr_llm_base_xh2a_32k_debug/<hmquant_dir>/golden_meta_info.json \
  --image-path data/images/unlimited_ocr_pdf_page1.png --prompt '<image>\nFree OCR. ' --max-new-tokens 64 --debug
```

### 8. OmniDocBench 批量推理（可选）

```bash
python examples_merak/llm/unlimited_ocr/unlimited_ocr_omnidocbench_infer.py \
  --mode hmonnx \
  --hmonnx-config work_dirs/unlimited_ocr_llm_base_xh2a_32k_debug/<hmquant_dir>/golden_meta_info.json \
  --image-dir <OmniDocBench images> --output-dir outputs/omnidocbench/hmonnx --limit 20 --save-raw
```

### 9. crop/gundam smoke（eager，无 HMONNX）

```bash
python examples_merak/llm/unlimited_ocr/debug_scripts/unlimited_ocr_crop_smoke.py \
  --config configs_merak/xh2a/llm_models/unlimited_ocr/gundam/unlimited_ocr_llm_gundam_xh2a_32k.py \
  --image-path data/images/qwen2_vl_demo.jpeg --debug
```

## 验证方法与阈值

- **visual wrap/fronted vs HF**（`--fp32`）：应接近 0（约 1e-3）；fp16 有约 0.04 累积噪声（12 层 SAM），属正常半精度。
- **visual quanted/HMONNX vs HF**：mean 约 0.015、cosine > 0.99，属 w8a8 量化误差。
- **LLM prefill wrap/fronted/quanted**：以 last-token argmax 一致为准（`num_logits_to_keep=1` 只返回最后一个 token 的 logits）。
- **LLM decode wrap**：用 `debug_scripts/unlimited_ocr_sliding_window_verify.py` 跑多步 decode，应与 native 一致（full-KV 修复后 200 token 对齐）。
- **端到端 HMONNX**：文本内容应与 HF 一致，bbox 坐标可能有量化级小偏移。
- 逐段调试见 `debug_scripts/README.md`。

## KV cache / sliding window

原模型 sliding window 是 ring buffer（prefill 永久可见 + decode token ring）。XH 的
`attention_max_length` 无法表达「保护前缀 + decode ring」分段语义，当前 decode 用**全量 KV**
（`attention_max_length=-1`）保证图像/prompt 不被滑掉、数值正确。完全复刻官方 ring buffer
（protected-prefix + decode ring）需要 `xhquant` 新增算子，属后续增强。

## 已知限制

- crop/gundam 只支持单图、走 eager，不产 HMONNX（动态 crop 数破坏静态 shape）。
- 当前 decode 用 full-KV，超长输出 KV 显存高于原 ring buffer。
- w8a8 量化误差会影响 bbox 坐标精度，并使边界 prompt 的生成稳定性下降。
- 数据集级评测（OmniDocBench 全量 1651 页 + text edit distance / TEDS）尚未跑完全量闭环。
- 未覆盖：多图、长文档连续推理、batch、crop 的逐 token golden 对齐。
