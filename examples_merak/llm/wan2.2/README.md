# Wan2.2 A14B 目录盘点示例

本目录是 Wan2.2 A14B 的第一版分析入口，只做本地模型结构盘点和适配前检查。当前仓库尚未提供 Wan/Wan2.2 converter，因此这里不提供 HMONNX 导出、量化或推理脚本，避免把未验证能力包装成可用流程。

## 背景

- T2V 本地目录：`/data01/datasets/Wan2.2-T2V-A14B`
- I2V 官方地址：<https://www.modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B>
- 模型类型：视频扩散模型，目录内包含 high noise / low noise 两套 diffusion transformer 权重，并复用 VAE 与 T5 文本编码器权重。

## 本地 T2V 结构

已知本地 `Wan2.2-T2V-A14B` 目录包含：

```text
Wan2.2-T2V-A14B/
  configuration.json
  high_noise_model/
    config.json
    diffusion_pytorch_model-00001-of-00006.safetensors
    ...
    diffusion_pytorch_model-00006-of-00006.safetensors
    diffusion_pytorch_model.safetensors.index.json
  low_noise_model/
    config.json
    diffusion_pytorch_model-00001-of-00006.safetensors
    ...
    diffusion_pytorch_model-00006-of-00006.safetensors
    diffusion_pytorch_model.safetensors.index.json
  Wan2.1_VAE.pth
  models_t5_umt5-xxl-enc-bf16.pth
```

`configuration.json` 声明任务为 `text-to-video-synthesis`，并列出 high/low noise 两套 diffusion safetensors。`high_noise_model/config.json` 与 `low_noise_model/config.json` 均为 `WanModel`，本地配置显示 `model_type=t2v`、`num_layers=40`、`num_heads=40`、`dim=5120`、`text_len=512`。

## I2V 差异

官方 I2V 模型是 image-to-video 方向，预计会在 diffusion 配置中体现 `model_type=i2v` 或等价图像条件输入字段，并可能包含图像条件分支或不同输入通道配置。当前没有提供本地 I2V 目录时，脚本只输出：

```text
未提供本地 I2V，仅基于 README/ModelScope 待补充
```

拿到本地 I2V 后，可通过 `--i2v-model-dir` 对比 `configuration.json`、high/low noise 配置、safetensors index、权重分片数量和文件大小。

## 运行环境

该脚本只使用 Python 标准库，不加载 PyTorch、Diffusers、safetensors 或大权重，不占用 GPU。

推荐在仓库根目录执行：

```bash
conda run -n xhquant python examples_merak/llm/wan2.2/analyze_wan22.py --help
conda run -n xhquant python examples_merak/llm/wan2.2/analyze_wan22.py
```

如果当前机器没有可用 conda 环境，可直接使用系统 Python：

```bash
python examples_merak/llm/wan2.2/analyze_wan22.py
```

保存完整 JSON：

```bash
python examples_merak/llm/wan2.2/analyze_wan22.py \
  --output work_dirs/wan22_t2v_inventory.json
```

同时分析本地 I2V：

```bash
python examples_merak/llm/wan2.2/analyze_wan22.py \
  --i2v-model-dir /path/to/Wan2.2-I2V-A14B \
  --output work_dirs/wan22_t2v_i2v_inventory.json
```

## GPU 单卡约束

- 当前脚本是 CPU 文件盘点工具，不需要 GPU。
- Wan2.2 A14B 是视频扩散模型，单卡直接加载完整生成链路会受显存强约束；T2V 本地目录中 high/low noise diffusion 权重合计约百 GiB 量级，另有 VAE 与 T5 encoder 权重。
- 第一版适配不应假设单卡能完成导出或推理。后续 converter 设计需要先明确 Host/NPU/GPU 边界、文本编码器和 VAE 的部署位置、high/low noise 切换策略、权重分片读取方式、动态图/视频时序维度约束。

## 后续适配建议

1. 先补齐只读结构对比：用本脚本输出 T2V/I2V JSON，确认 high/low noise 配置差异、输入通道、任务类型、权重索引完整性。
2. 建立最小 graph spec：明确 T5 encoder、Wan diffusion transformer、VAE decode 三段边界，避免直接套用 LLM converter。
3. 单独设计 diffusion transformer converter：复用现有 xh2modelzoo 的配置注册和验证框架，但不要把视频扩散模型伪装成自回归 LLM。
4. 增加无权重单元测试：覆盖 `configuration.json`、`config.json`、safetensors index 解析，以及 I2V 缺失时的提示。
5. 再做小样本验证：使用官方最小输入和固定 seed 对齐 shape、dtype、scheduler 入口，确认 CPU/GPU 原生输出路径后再进入 HMONNX 导出验证。

## 文件说明

- `analyze_wan22.py`：标准库脚本，输出控制台表格摘要，并可用 `--output` 保存完整 JSON。
- `__init__.py`：保持示例目录可被工具识别；目录名包含 `.`，不建议作为常规 Python import 路径使用。
