# LingBot Video Dense 1.3B Merak Workflow

本目录提供 LingBot Video Dense 1.3B 的 Merak 全模型量化、HMONNX 导出和推理示例。所有命令默认从
`xh2modelzoo` 仓库根目录执行。

本流程依赖配套版本的 `xhquanttool`。运行前请同步仓库，并将当前环境重新绑定到该工作区：

```bash
git -C ../xhquanttool pull --ff-only
python -m pip install --no-deps --no-build-isolation -e ../xhquanttool
```

## 上游依赖

适配基于 [Robbyant/lingbot-video](https://github.com/Robbyant/lingbot-video) 的
`a638721cf2271804d02738b69f2ad788c4a559fc` 版本，许可证为 Apache-2.0。

```bash
git clone https://github.com/Robbyant/lingbot-video.git ../lingbot-video
git -C ../lingbot-video checkout a638721cf2271804d02738b69f2ad788c4a559fc
python -m pip install "diffusers==0.37.1" "imageio==2.37.3" "imageio-ffmpeg==0.6.0"
python -m pip install --no-deps --no-build-isolation -e ../lingbot-video
```

## 模型下载

- [ModelScope](https://www.modelscope.cn/models/Robbyant/lingbot-video-dense-1.3b)
- [Hugging Face](https://huggingface.co/robbyant/lingbot-video-dense-1.3b)

ModelScope：

```bash
modelscope download \
  --model Robbyant/lingbot-video-dense-1.3b \
  --local_dir data/models/lingbot-video-dense-1.3b
```

Hugging Face：

```bash
hf download robbyant/lingbot-video-dense-1.3b \
  --local-dir data/models/lingbot-video-dense-1.3b
```

## 配置

- W8A8：`configs_merak/workflows/xh2a/other_models/lingbot_video/dense_1_3b/lingbot_video_dense_1_3b_w8a8.yaml`
- W8A16：`configs_merak/workflows/xh2a/other_models/lingbot_video/dense_1_3b/lingbot_video_dense_1_3b_w8a16.yaml`
- W16A16：`configs_merak/workflows/xh2a/other_models/lingbot_video/dense_1_3b/lingbot_video_dense_1_3b_w16a16.yaml`

量化组件包括 Qwen3-VL 文本编码器、Qwen3-VL 视觉编码器、LingBot Video Transformer、Wan VAE Encoder
和 Wan VAE Decoder。Token Embedding 按 Merak LLM 约定保存为 FP16 `nn.Embedding`。

Qwen3-VL Text Encoder 使用静态长度 2048。HMONNX demo 会将 Pipeline 的 tokenizer 最大长度同步为导出图长度，
超过该长度的结构化 Prompt 会在分词阶段截断；调整 YAML 中的 `sequence_length` 后，推理侧会自动读取新长度。

## HMONNX 导出

### T2I

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/aigc/lingbot_video/lingbot_video_workflow.py \
  --model-dir data/models/lingbot-video-dense-1.3b \
  --export-output-dir work_dirs/lingbot_video_w8a8_t2i_1f \
  --device cuda:0 \
  --mode t2i \
  --height 480 \
  --width 832 \
  --num-frames 1 \
  --steps 40 \
  --overwrite
```

### T2V

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/aigc/lingbot_video/lingbot_video_workflow.py \
  --model-dir data/models/lingbot-video-dense-1.3b \
  --export-output-dir work_dirs/lingbot_video_w8a8_t2v_121f \
  --device cuda:0 \
  --mode t2v \
  --height 480 \
  --width 832 \
  --num-frames 121 \
  --steps 40 \
  --overwrite
```

### TI2V

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/aigc/lingbot_video/lingbot_video_workflow.py \
  --model-dir data/models/lingbot-video-dense-1.3b \
  --export-output-dir work_dirs/lingbot_video_w8a8_ti2v_121f \
  --device cuda:0 \
  --mode ti2v \
  --height 480 \
  --width 832 \
  --num-frames 121 \
  --steps 40 \
  --overwrite
```

导出 W8A16 或 W16A16 时分别增加对应的配置参数，并同步修改输出目录名称：

```text
--config-path configs_merak/workflows/xh2a/other_models/lingbot_video/dense_1_3b/lingbot_video_dense_1_3b_w8a16.yaml

--config-path configs_merak/workflows/xh2a/other_models/lingbot_video/dense_1_3b/lingbot_video_dense_1_3b_w16a16.yaml
```

## HMONNX 推理

官方结构化 Prompt 位于 `../lingbot-video/assets/cases/`，完整清单见
`../lingbot-video/assets/cases/manifest.json`。

### T2I

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/aigc/lingbot_video/hmonnx_demo.py \
  --mode t2i \
  --export-dir work_dirs/lingbot_video_w8a8_t2i_1f \
  --prompt-json ../lingbot-video/assets/cases/t2i/example_1/prompt.json \
  --output work_dirs/lingbot_video_w8a8_t2i_1f/example_1.png \
  --device cuda:0
```

### T2V

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/aigc/lingbot_video/hmonnx_demo.py \
  --mode t2v \
  --export-dir work_dirs/lingbot_video_w8a8_t2v_121f \
  --prompt-json ../lingbot-video/assets/cases/t2v/example_1/prompt.json \
  --output work_dirs/lingbot_video_w8a8_t2v_121f/example_1.mp4 \
  --device cuda:0 \
  --fps 24
```

### TI2V

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/aigc/lingbot_video/hmonnx_demo.py \
  --mode ti2v \
  --export-dir work_dirs/lingbot_video_w8a8_ti2v_121f \
  --prompt-json ../lingbot-video/assets/cases/ti2v/example_1/prompt.json \
  --image ../lingbot-video/assets/cases/ti2v/example_1/first_frame.png \
  --output work_dirs/lingbot_video_w8a8_ti2v_121f/example_1.mp4 \
  --device cuda:0 \
  --fps 24
```

## Golden

在对应静态 profile 的导出命令中增加 `--dump-golden`，产物保存在导出目录的 `golden/` 下。
