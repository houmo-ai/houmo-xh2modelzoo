# Qwen3-ASR Merak Workflow 示例

本目录提供 Qwen3-ASR 迁移到 `xhmodel_merak/xh_other_model` 后的导出和 HMONNX 推理示例。所有命令默认从仓库根目录执行。

## 环境安装

环境名称和 Python 版本可按实际工程要求调整。示例安装步骤如下：

```bash
conda create -n <env_name> python=3.12
conda activate <env_name>

pip install -v -e . --no-build-isolation
pip install qwen_asr
pip install onnx-ir==0.1.14 onnx==1.16.2 onnxscript==0.5.7
```

Qwen3-ASR 依赖 `qwen_asr`，该包会带入对应的 `transformers` 版本约束。若运行环境已有统一的 xhquant/HMONNX 运行时，请以该环境的依赖版本为准。

## 配置

默认 workflow YAML：

```text
configs_merak/workflows/xh2a/other_models/qwen3_asr/0_6b/qwen3_asr.yaml
```

`--model-dir` 必须显式传入。默认输出目录是相对当前工作目录的字符串路径：

- 量化目录：`work_dirs/qwen3_asr_quant`
- 导出目录：`work_dirs/qwen3_asr_export`

YAML 默认导出两个组件：

- `encoder`
- `prefill_decode`

workflow 会在导出前把实际使用的 YAML dump 到导出目录，文件名使用 workflow 配置文件名。`AutoWorkflow` 会通过 YAML 中的 `export.model.type` 解析模型，并绑定到 `Qwen3ASRWorkflow`。

## 导出

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/asr/qwen3_asr/qwen3_asr_workflow.py \
  --model-dir <qwen3_asr_model_dir> \
  --device cuda:0 \
  --overwrite
```

覆盖音频长度和 prefix token 预算：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/asr/qwen3_asr/qwen3_asr_workflow.py \
  --model-dir <qwen3_asr_model_dir> \
  --max-audio-length 6000 \
  --prefix-token-budget 512 \
  --export-output-dir work_dirs/qwen3_asr_export_60s \
  --device cuda:0 \
  --overwrite
```

Qwen3-ASR 当前的 prefill/decode 量化在导出阶段执行。workflow 的 `quant()` 阶段会记录跳过状态并返回 `QuantResult`，随后 `export()` 按 YAML 配置导出 encoder 和 prefill/decode。

## HMONNX 推理

导出完成后，把导出目录传给 HMONNX demo。以下示例使用默认导出目录：

```text
work_dirs/qwen3_asr_export
```

分段离线推理：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/asr/qwen3_asr/hmonnx_demo.py \
  --work-dir work_dirs/qwen3_asr_export \
  --audio <audio_file> \
  --device cuda:0 \
  --max-audio-length 1500
```

累计音频加文本 prefix 推理：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/asr/qwen3_asr/hmonnx_demo_chunk_prefix.py \
  --work-dir work_dirs/qwen3_asr_export \
  --audio <audio_file> \
  --device cuda:0 \
  --max-audio-length 6000 \
  --chunk-seconds 20
```

`hmonnx_demo.py` 每个音频片段独立推理，只需要 `max-audio-length` 覆盖单个片段。

`hmonnx_demo_chunk_prefix.py` 会基于累计音频重复推理，并在后续片段注入已识别文本作为 prefix。使用该模式时，导出时需要设置更大的 `max-audio-length` 和足够的 `prefix-token-budget`。

## Golden

golden 生成与 quant/export 分离，由 workflow 的 `dump_golden(export_result, device)` 接口完成。需要同时导出并生成 golden 时，在 workflow 脚本中加入 `--dump-golden`：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/asr/qwen3_asr/qwen3_asr_workflow.py \
  --model-dir <qwen3_asr_model_dir> \
  --device cuda:0 \
  --dump-golden \
  --overwrite
```
