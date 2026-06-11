# FLUX.2-klein

本目录提供 FLUX.2-klein-4B 在仓库中的示例脚本，覆盖以下几类场景：

- 浮点基线推理（文生图、图像编辑）
- HMONNX 导出与加载验证
- 文本编码器单独量化与替换
- EvalMuse 文生图评测

默认示例模型路径为 `/data02/datasets/flux-4b`。

## 目录内容

- `flux2_klein_fp_demo.py`：浮点文生图基线推理。
- `flux2_klein_fp_edit_demo.py`：浮点图像编辑基线推理。
- `flux2_klein_export.py`：导出 text_encoder、transformer、vae 到 HMONNX。
- `flux2_klein_edit_export.py`：导出图像编辑模式所需组件，默认导出 `vae_encoder`。
- `flux2_klein_hmonnx_demo.py`：加载 HMONNX 组件做文生图验证。
- `flux2_klein_hmonnx_edit_demo.py`：加载 HMONNX 组件做图像编辑验证。
- `flux2_evalMuse.py`：通过 EvalScope 跑 EvalMuse 文生图评测。
- `flux2_klein_quant_text_encoder_autoround.py`：使用 AutoRound 对 text_encoder 做 W8 量化，并重新组织为可直接加载的 FLUX 模型目录。

## 依赖

建议至少准备以下 Python 依赖：

```bash
pip install torch torchvision
pip install transformers sentencepiece accelerate safetensors pillow # 4.57.3
pip install git+https://github.com/huggingface/diffusers.git  # 3.8.0
```

说明：

- 示例依赖安装版 `diffusers` 中包含 `Flux2KleinPipeline`。
- 若当前 `diffusers` 不支持 `Flux2KleinPipeline`，脚本会直接退出并提示升级。
- 评测脚本还需要 `evalscope`。
- AutoRound 量化脚本还需要 `auto-round`。

可选依赖：

```bash
pip install evalscope
pip install auto-round
```

## 1. 浮点基线推理

### 文生图

```bash
python examples/aigc/flux2_klein/flux2_klein_fp_demo.py \
  --model /data02/datasets/flux-4b \
  --prompt "A cat holding a sign that says hello world" \
  --steps 4 \
  --height 1024 \
  --width 1024 \
  --output work_dirs/flux2-klein-4b/fp_demo/flux2_klein.png
```

常用参数：

- `--device auto|cuda|cpu`
- `--dtype auto|bfloat16|float16|float32`
- `--cpu-offload`：仅 CUDA 下有效，可降低显存占用
- `--image`：可重复传入多张参考图，进入编辑/多参考模式

### 图像编辑

```bash
python examples/aigc/flux2_klein/flux2_klein_fp_edit_demo.py \
  --model /data02/datasets/flux-4b \
  --image examples/aigc/flux2_klein/RU2A3501.jpg \
  --prompt "make the image more cinematic and detailed" \
  --steps 4 \
  --output work_dirs/flux2-klein-4b/fp_edit_demo/flux2_klein_edit.png
```

## 2. 导出 HMONNX 组件

### 文生图导出

默认支持导出以下组件：

- `text_encoder`
- `transformer`
- `vae`

示例：

```bash
python examples/aigc/flux2_klein/flux2_klein_export.py \
  --model /data02/datasets/flux-4b \
  --width 1024 \
  --height 1024 \
  --steps 4 \
  --guidance-scale 1.0 \
  --components text_encoder transformer vae
```

导出结果默认落在：

```bash
work_dirs/flux-4b_XH2a_1024x1024
```

如果只想先验证文本编码器导出链路，可以先只导出一个组件：

```bash
python examples/aigc/flux2_klein/flux2_klein_export.py \
  --model /data02/datasets/flux-4b \
  --components text_encoder
```

### 图像编辑导出

图像编辑模式默认脚本只导出 `vae_encoder`，因为编辑链路比纯文生图多了图像编码输入。

```bash
python examples/aigc/flux2_klein/flux2_klein_edit_export.py \
  --model /data02/datasets/flux-4b \
  --components vae_encoder \
  --work-dir work_dirs/flux-4b_XH2a_1024x1024_edit
```

## 3. HMONNX 加载验证

### 文生图验证

`--meta` 可以直接传 text_encoder 的 `meta.json`，也可以传根 `meta.json`。如果同时启用 `transformer` 或 `vae`，建议额外传 `--root-meta` 指向导出根目录的 `meta.json`。

```bash
python examples/aigc/flux2_klein/flux2_klein_hmonnx_demo.py \
  --model /data02/datasets/flux-4b \
  --meta work_dirs/flux-4b_XH2a_1024x1024/text_encoder/meta.json \
  --root-meta work_dirs/flux-4b_XH2a_1024x1024/meta.json \
  --components text_encoder,transformer,vae \
  --prompt "A cat holding a sign that says hello world" \
  --output work_dirs/flux2-klein-4b/text_encoder_demo/flux2_klein_hmonnx.png
```

如果只做文本编码器对比，不跑整图生成：

```bash
python examples/aigc/flux2_klein/flux2_klein_hmonnx_demo.py \
  --model /data02/datasets/flux-4b \
  --meta work_dirs/flux-4b_XH2a_1024x1024/text_encoder/meta.json \
  --components text_encoder \
  --skip-image
```

脚本会输出：

- `max_abs_diff`
- `mean_abs_diff`
- `rmse`
- `cosine`
- `allclose`

用于比较 HMONNX text_encoder 与浮点 reference 的 embedding 一致性。

### 图像编辑验证

```bash
python examples/aigc/flux2_klein/flux2_klein_hmonnx_edit_demo.py \
  --model /data02/datasets/flux-4b \
  --meta work_dirs/flux-4b_XH2a_1024x1024/text_encoder/meta.json \
  --root-meta work_dirs/flux-4b_XH2a_1024x1024/meta.json \
  --components text_encoder,transformer,vae_encoder,vae \
  --image examples/aigc/flux2_klein/flux2_klein_hmonnx.png \
  --prompt "make the image more cinematic and detailed" \
  --output work_dirs/flux2-klein-4b/edit_hmonnx_demo/flux2_klein_hmonnx_edit.png
```

这里的默认组件包括：

- `text_encoder`
- `transformer`
- `vae_encoder`
- `vae`

其中 `vae_encoder` 是图像编辑链路必须关注的新增组件。

## 4. EvalMuse 评测

如果需要对 HMONNX 版本的文生图效果做批量评测，可以使用 EvalScope：

```bash
python examples/aigc/flux2_klein/flux2_evalMuse.py \
  --model /data02/datasets/flux-4b \
  --meta work_dirs/flux-4b_XH2a_1024x1024/text_encoder/meta.json \
  --root-meta work_dirs/flux-4b_XH2a_1024x1024/meta.json \
  --components text_encoder,transformer,vae \
  --steps 4 \
  --height 1024 \
  --width 1024 \
  --work-dir outputs/flux2_klein_hmonnx_evalmuse
```

调试时建议先限制样本数：

```bash
python examples/aigc/flux2_klein/flux2_evalMuse.py \
  --model /data02/datasets/flux-4b \
  --meta work_dirs/flux-4b_XH2a_1024x1024/text_encoder/meta.json \
  --limit 5
```

## 5. AutoRound 量化 text_encoder

该脚本用于单独把 FLUX.2-klein 的 `text_encoder` 做 W8 量化，并把其他资源链接或拷贝到新的模型目录，便于继续复用浮点 demo 直接加载。

```bash
python examples/aigc/flux2_klein/flux2_klein_quant_text_encoder_autoround.py \
  --model /data02/datasets/flux-4b \
  --output-dir work_dirs/flux2-klein-4b/autoround-w8-text-encoder \
  --bits 8 \
  --group-size 128 \
  --nsamples 32 \
  --seqlen 512
```

如果希望用自定义校准 prompt：

```bash
python examples/aigc/flux2_klein/flux2_klein_quant_text_encoder_autoround.py \
  --model /data02/datasets/flux-4b \
  --calib-prompt "A cat holding a sign that says hello world" \
  --calib-prompt "A futuristic city skyline at night, ultra detailed"
```

量化完成后，脚本会在输出目录下写入：

- `text_encoder/`：量化后的文本编码器
- `autoround_text_encoder_w8_meta.json`：量化参数记录

若未指定 `--no-assets`，还会把其余 FLUX 资源组织成一个完整模型目录，随后可直接复用浮点 demo：

```bash
python examples/aigc/flux2_klein/flux2_klein_fp_demo.py \
  --model work_dirs/flux2-klein-4b/autoround-w8-text-encoder
```

## 6. 常见问题

### 1) 提示 `diffusers` 不包含 `Flux2KleinPipeline`

升级到支持 FLUX.2-klein 的版本：

```bash
pip install git+https://github.com/huggingface/diffusers.git
```

### 2) `meta.json` 应该传哪个

- 只验证 `text_encoder` 时，传对应组件目录下的 `meta.json` 即可。
- 同时启用 `transformer`、`vae` 或 `vae_encoder` 时，建议同时传根目录 `meta.json` 到 `--root-meta`。

### 3) 图像编辑最少需要哪些组件

完整链路建议包含：

- `text_encoder`
- `transformer`
- `vae_encoder`
- `vae`

### 4) 为什么默认步数只有 4

本目录示例以导出验证和链路连通性为主，因此默认使用较小步数来缩短验证时间；若要观察更稳定的视觉效果，可自行增大 `--steps`。

## 免责声明

您明确了解并同意，相关第三方模型、配置和依赖由对应提供方维护。使用这些第三方软件、数据或模型时，应同时遵守其许可证、使用条款和隐私政策。因使用第三方资源产生的风险，由使用者自行承担。