# Whisper Merak Workflow 示例

本目录提供 Whisper 迁移到 `xhmodel_merak/xh_other_model` 后的导出和 HMONNX 推理示例。所有命令默认从仓库根目录执行。

## 背景

Whisper 导出三个 HMONNX 图：

- `encoder`：`model.model.encoder`。decoder 各层的 cross-attention `k_proj`/`v_proj` 被折叠进 encoder 图，因此 encoder 输出 `key_state_i` / `value_state_i`（`2 * num_decoder_layers` 个张量），decoder 无需每步重算 cross-attn 的 k/v。
- `prefill`：decoder 的 prefill 阶段，输入 prompt `[sot, lang, transcribe, notimestamps]`。
- `decoder`：decoder 的单步 decode 阶段。

多图之间的调度（encoder → prefill → decode 循环）在 `hmonnx_demo.py` 中实现，HMONNX runtime 不会自动理解业务级调度。

Whisper 不使用 `LLMBaseModel`/`MODELS.build()` 框架，`XHWhisperModel` 是一个仅承载 `WORKFLOW_CLS` 的轻量注册类，workflow 内部直接驱动 transformers 的 `WhisperForConditionalGeneration` 与 `_model_opt.py` 中的 forward rewriter。

## 环境安装

环境名称和 Python 版本可按实际工程要求调整。示例安装步骤如下：

```bash
conda create -n <env_name> python=3.12
conda activate <env_name>

pip install -v -e . --no-build-isolation
pip install torchcodec
pip install onnx-ir==0.1.14 onnx==1.16.2 onnxscript==0.5.7
```

`transformers` 版本需与 xhquant 运行时一致（已验证 4.57.6）。若运行环境已有统一的 xhquant/HMONNX 运行时，请以该环境的依赖版本为准。

## 配置

默认 workflow YAML：

```text
configs_merak/workflows/xh2a/other_models/whisper/whisper.yaml
```

`--model-dir` 必须显式传入。默认导出目录：

- `work_dirs/whisper_export`

YAML 默认导出三个组件：`encoder`、`prefill`、`decoder`，每个组件的 `quant_type` 均可通过 YAML 或命令行覆盖。workflow 会在导出前把实际使用的 YAML dump 到导出目录，文件名使用 workflow 配置文件名。`AutoWorkflow` 通过 YAML 中的 `export.model.type = XHWhisperModel` 解析模型，并绑定到 `WhisperWorkflow`。

## 导出

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/asr/whisper/whisper_workflow.py \
  --model-dir <whisper_model_dir> \
  --device cuda:0 \
  --overwrite
```

覆盖量化精度：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/asr/whisper/whisper_workflow.py \
  --model-dir <whisper_model_dir> \
  --quant-type w8a8_sefp \
  --export-output-dir work_dirs/whisper_export_w8a8 \
  --device cuda:0 \
  --overwrite
```

Whisper 没有独立量化阶段。workflow 的 `quant()` 会记录跳过状态并返回 `QuantResult`，随后 `export()` 按 YAML 配置依次导出 encoder、prefill、decoder。导出的 HMONNX 文件名包含 `target_device` 和 `quant_type`。

## Golden

golden 生成与 export 分离，由 workflow 的 `dump_golden(export_result, device)` 接口完成，覆盖 encoder、prefill、decoder 三个图。需要同时导出并生成 golden 时加 `--dump-golden`：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/asr/whisper/whisper_workflow.py \
  --model-dir <whisper_model_dir> \
  --device cuda:0 \
  --dump-golden \
  --overwrite
```

说明：encoder 输入通过 `export.components.encoder.audio_path`（YAML 配置）指定的音频文件生成，与迁移前行为一致——读取音频并重采样到 16 kHz，经 `WhisperProcessor` 构造 mel 特征。export 和 golden 共用同一份音频输入。prefill/decoder 的 golden 沿用 legacy 的 cache 构造（`-65504` 哨兵填充）。

## HMONNX 推理

导出完成后，把导出目录传给 demo：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/asr/whisper/hmonnx_demo.py \
  --hf-model <whisper_model_dir> \
  --work-dir work_dirs/whisper_export \
  --audio <audio_file> \
  --device cuda:0
```

demo 读取导出目录下的 `export_meta_info.json`，按 `encoder -> prefill -> decode` 调度多图，并复用 KV cache。decoder 的 self-attention cache 容量从导出图的输入 shape 自动读取。

> **注意**：`hmonnx_demo.py` 需要 encoder、prefill、decoder 三个组件全部导出才能运行。YAML 中可以通过 `enabled: false` 关闭某个组件（例如只导出 encoder 做单独测试），但此时 demo 会因缺少组件而报错。
