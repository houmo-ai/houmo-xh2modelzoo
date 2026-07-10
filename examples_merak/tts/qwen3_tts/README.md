# Qwen3-TTS Merak Workflow 示例

本目录提供 Qwen3-TTS 迁移到 `xhmodel_merak/xh_other_model` 后的导出、HMONNX demo、流式 demo 和小样本评估入口。所有命令默认从仓库根目录执行。

## 环境安装

环境名称和 Python 版本可按实际工程要求调整。示例安装步骤如下：

```bash
conda create -n <env_name> python=3.12
conda activate <env_name>

pip install -v -e . --no-build-isolation
pip install qwen_tts
pip install soundfile loguru tqdm onnx onnxruntime onnxsim onnxscript
```

Qwen3-TTS 依赖 `qwen_tts`。若运行环境已有统一的 xhquant/HMONNX 运行时，请以该环境的依赖版本为准。

## 配置

脚本按 `--variant` 选择默认 YAML 和默认导出目录：

- `0_6B_base`: `configs_merak/workflows/xh2a/other_models/qwen3_tts/0_6b_base/qwen3_tts_12hz_0_6b_base.yaml`
- `0_6B_customvoice`: `configs_merak/workflows/xh2a/other_models/qwen3_tts/0_6b_customvoice/qwen3_tts_12hz_0_6b_customvoice.yaml`
- `1_7B_voicedesign`: `configs_merak/workflows/xh2a/other_models/qwen3_tts/1_7b_voicedesign/qwen3_tts_12hz_1_7b_voicedesign.yaml`

`--model-dir` 必须显式传入。默认输出目录是相对当前工作目录的字符串路径：

- 量化目录：`work_dirs/qwen3_tts_quant`
- `0_6B_base` 导出目录：`work_dirs/Qwen3-TTS-12Hz-0.6B-Base_XH2a`
- `0_6B_customvoice` 导出目录：`work_dirs/Qwen3-TTS-12Hz-0.6B-CustomVoice_XH2a`
- `1_7B_voicedesign` 导出目录：`work_dirs/Qwen3-TTS-12Hz-1.7B-VoiceDesign_XH2a`

每个 YAML 都把原来传给 `MODELS.build()` 的配置完整放在 `export.model` 下，并固定主模型类型位置为 `export.model.type`。各子模型导出精度由 `export.quant_types.*` 控制，导出文件名会体现 `target_device` 和量化精度。

## 导出

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/tts/qwen3_tts/qwen3_tts_workflow.py \
  --variant 0_6B_customvoice \
  --model-dir <qwen3_tts_model_dir> \
  --device cuda:0 \
  --overwrite
```

## HMONNX 普通 Demo

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/tts/qwen3_tts/hmonnx_demo.py \
  --work-dir work_dirs/Qwen3-TTS-12Hz-0.6B-CustomVoice_XH2a \
  --device cuda:0
```

## 流式 Demo

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/tts/qwen3_tts/hmonnx_streaming_demo.py \
  --work-dir work_dirs/Qwen3-TTS-12Hz-0.6B-CustomVoice_XH2a \
  --device cuda:0
```

## 小样本评估

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/tts/qwen3_tts/eval/qwen3_tts_eval.py \
  --work-dir work_dirs/Qwen3-TTS-12Hz-0.6B-CustomVoice_XH2a \
  --device cuda:0 \
  --max-samples 1
```

HMONNX demo 和评估脚本会从导出目录读取顶层 `export_meta_info.json` 以及各组件原有的 `meta.json`。

## Golden 生成

golden 数据只通过 `dump_golden` 生成。需要同时导出并生成 golden 时，在 workflow 脚本中加入 `--dump-golden`：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/tts/qwen3_tts/qwen3_tts_workflow.py \
  --variant 0_6B_customvoice \
  --model-dir <qwen3_tts_model_dir> \
  --device cuda:0 \
  --dump-golden \
  --overwrite
```
