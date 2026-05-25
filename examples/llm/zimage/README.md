# Z-Image HMONNX Export and Demo

本目录提供 Z-Image / Z-Image-Turbo 的 HMONNX 导出与端到端 demo 脚本。导出流程会把模型拆成三部分：

- text encoder: `text_encoder_export.py`
- VAE: `vae_export.py`
- DiT transformer: `dit_export.py`
- HMONNX demo: `text_encoder_demo.py`

建议在仓库根目录运行下面的命令。

```bash
cd /data01/home/huxing/xh2modelzoo
conda activate base
```

## 快速验证 Turbo

Turbo 是当前最适合做 smoke test 的路径。示例模型目录为 `/data01/datasets/Z-Image-Turbo`，输出目录建议单独放在 `work_dirs/zimage_turbo`，不要和 Base 模型混用。

```bash
export CUDA_VISIBLE_DEVICES=1
export MODEL_DIR=/data01/datasets/Z-Image-Turbo
export WORK_DIR=work_dirs/zimage_turbo
```

依次导出 text encoder、VAE、DiT：

```bash
python examples/llm/zimage/text_encoder_export.py \
  --model "$MODEL_DIR" \
  --work-dir "$WORK_DIR"

python examples/llm/zimage/vae_export.py \
  --model "$MODEL_DIR" \
  --work-dir "$WORK_DIR"

python examples/llm/zimage/dit_export.py \
  --model "$MODEL_DIR" \
  --work-dir "$WORK_DIR"
```

运行 HMONNX demo：

```bash
python examples/llm/zimage/text_encoder_demo.py \
  --model "$MODEL_DIR" \
  --work-dir "$WORK_DIR" \
  --output "$WORK_DIR/zimage_turbo_hmonnx.png" \
  --num-inference-steps 9 \
  --guidance-scale 0 \
  --cfg-normalization 0
```

输出图片会保存到：

```text
work_dirs/zimage_turbo/zimage_turbo_hmonnx.png
```

## 导出 Base

Base 模型示例目录为 `/data01/datasets/Z-Image`，建议使用独立输出目录 `work_dirs/zimage`。

```bash
export CUDA_VISIBLE_DEVICES=1
export MODEL_DIR=/data01/datasets/Z-Image
export WORK_DIR=work_dirs/zimage
```

导出顺序与 Turbo 相同：

```bash
python examples/llm/zimage/text_encoder_export.py \
  --model "$MODEL_DIR" \
  --work-dir "$WORK_DIR"

python examples/llm/zimage/vae_export.py \
  --model "$MODEL_DIR" \
  --work-dir "$WORK_DIR"

python examples/llm/zimage/dit_export.py \
  --model "$MODEL_DIR" \
  --work-dir "$WORK_DIR"
```

Base 推荐使用 CFG 参数运行 demo，例如：

```bash
python examples/llm/zimage/text_encoder_demo.py \
  --model "$MODEL_DIR" \
  --work-dir "$WORK_DIR" \
  --output "$WORK_DIR/zimage_base_hmonnx.png" \
  --num-inference-steps 28 \
  --guidance-scale 4 \
  --cfg-normalization 0
```

## 产物说明

导出完成后，`--work-dir` 下会生成以下关键文件：

```text
<WORK_DIR>/
  meta.json
  meta_vae.json
  token_embedding.pt
  hmonnx/
    prefill/
      zimage_llm-XH2a-2k-w8a8h1_sefp_prefill.onnx
    zimage_vae-XH2a-w8a8h1_sefp.onnx
    zimage_vae-XH2a-w8a8h1_sefp_external_data/
    zimage_dit-XH2a-w8a8h1_sefp.onnx
    zimage_dit-XH2a-w8a8h1_sefp_external_data/
  golden/
    zimage_vae-XH2a-w8a8h1_sefp/
    zimage_dit-XH2a-w8a8h1_sefp/
```

`text_encoder_demo.py` 会从同一个 `--work-dir` 读取 `meta.json`、text encoder HMONNX、VAE HMONNX 和 DiT HMONNX。导出和 demo 必须使用同一个模型目录与同一个 `--work-dir`。

## 常用参数

导出脚本常用参数：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model` | Hugging Face / diffusers 模型目录 | `/data02/datasets/zimage` |
| `--work-dir` | HMONNX、meta、golden 输出目录 | `work_dirs/zimage` |
| `--quant-type` | 量化类型 | `w8a8h1_sefp` |
| `--context-length` | text encoder 上下文长度 | `2048` |
| `--input-sequence-length` | text encoder 单次输入长度 | `256` |

demo 脚本常用参数：

| 参数 | 说明 | Turbo 建议 | Base 建议 |
| --- | --- | --- | --- |
| `--num-inference-steps` | 采样步数 | `9` | `28` 到 `50` |
| `--guidance-scale` | CFG 强度 | `0` | `3` 到 `5` |
| `--cfg-normalization` | CFG norm 限制 | `0` | `0` |
| `--height` | 输出高度 | `1024` | `1024` |
| `--width` | 输出宽度 | `1024` | `1024` |
| `--seed` | 随机种子 | `42` | `42` |

自定义 prompt 示例：

```bash
python examples/llm/zimage/text_encoder_demo.py \
  --model "$MODEL_DIR" \
  --work-dir "$WORK_DIR" \
  --output "$WORK_DIR/custom.png" \
  --prompt "一只长得像蝴蝶一样缤纷绚丽的奇异花朵，开在丛林中，散发着柔和的光芒" \
  --negative-prompt "" \
  --num-inference-steps 9 \
  --guidance-scale 0 \
  --cfg-normalization 0 \
  --seed 42
```

## 重新导出

部分 converter 在目标 HMONNX 或 golden 已存在时会跳过导出。需要强制重新导出时，先删除对应 `--work-dir`，或为新模型指定一个新的输出目录。

```bash
rm -rf work_dirs/zimage_turbo
```

然后重新执行 text encoder、VAE、DiT 三个导出命令。

## 注意事项

- Base 和 Turbo 权重不同，必须使用不同的 `--work-dir`，避免 HMONNX、meta、token embedding 混用。
- demo 依赖三个 HMONNX 产物同时存在：text encoder、VAE、DiT。缺少任意一个都会在加载阶段失败。
- `CUDA_VISIBLE_DEVICES` 可以按机器空闲 GPU 调整；脚本内部使用 `cuda`。
- 当前脚本面向 XH2a，默认量化类型为 `w8a8h1_sefp`。