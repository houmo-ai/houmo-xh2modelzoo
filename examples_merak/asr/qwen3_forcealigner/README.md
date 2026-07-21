# Qwen3-ForceAligner Merak 示例

本目录提供 Qwen3-ForcedAligner-0.6B 的 Merak workflow、Encoder/Prefill
导出、golden 生成和 HMONNX 对齐推理。命令均从仓库根目录运行。

## 环境安装

环境名称和 Python 版本可按工程要求调整：

```bash
conda create -n <env_name> python=3.12
conda activate <env_name>

pip install -v -e . --no-build-isolation
pip install qwen_asr librosa
```

`qwen_asr` 提供 Qwen3-ASR processor 和 ForcedAligner 文本处理逻辑，
`librosa` 用于读取并重采样音频。若运行环境已有统一的 xhquant/HMONNX
运行时和 Transformers 版本约束，请以工程环境为准。

## 配置

默认配置位于：

```text
configs_merak/workflows/xh2a/other_models/qwen3_forcealigner/0_6b/qwen3_forcealigner.yaml
```

默认导出固定 3000 帧音频输入和 411 token Prefill，Encoder 和 Prefill
的量化精度分别由 `export.audio.quant_type` 和
`export.prefill.quant_type` 控制。

## 导出

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/asr/qwen3_forcealigner/qwen3_forcealigner_workflow.py \
  --model-dir <model_dir> \
  --device cuda:0 \
  --export-output-dir work_dirs/qwen3_forcealigner_export \
  --overwrite
```

同时生成 Encoder 和 Prefill golden：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/asr/qwen3_forcealigner/qwen3_forcealigner_workflow.py \
  --model-dir <model_dir> \
  --device cuda:0 \
  --dump-golden \
  --overwrite
```

`quant()` 没有独立量化阶段，会返回 skipped。`export()` 只导出模型，
`dump_golden()` 会分别刷新 Encoder 和 Prefill 的 golden 目录。

## HMONNX 推理

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/asr/qwen3_forcealigner/hmonnx_demo.py \
  --work-dir work_dirs/qwen3_forcealigner_export \
  --audio <audio_file> \
  --text "HE BEGAN A CONFUSED COMPLAINT AGAINST THE WIZARD" \
  --language English \
  --device cuda:0
```

Demo 会从顶层 `export_meta_info.json` 解析全部产物。音频特征长度不能
超过导出时的 `max_audio_length`，合并后的音频和文本 token 也不能超过
`prefill.sequence_length`。超出限制时 demo 会明确报错；可以缩短输入，
或使用更大的 `--max-audio-length`/`--sequence-length` 重新导出。
