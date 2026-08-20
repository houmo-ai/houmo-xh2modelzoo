# MiniCPM-o-4.5 Merak

本目录提供 MiniCPM-o-4.5 的 Merak workflow、Hugging Face demo、HMONNX demo 和流式 demo。
模型实现位于 `xhmodel_merak/xh_llm/models/minicpm_o_4_5/`，配置位于
`configs_merak/workflows/xh2a/llm_models/minicpm_o_4_5/`。

## 正式配置

仓库保留两个正式配置：

| 配置 | LLM | GPTQ | TTS |
| --- | --- | --- | --- |
| `minicpm_o_4_5_xh2a_w8a8_gptq.yaml`（默认） | `w8a8_sefp` | 8 bit，group size 64 | `w8a8_sefp` |
| `minicpm_o_4_5_xh2a_w4a8_gptq.yaml` | `w4a8h0_ssfp` | 4 bit，group size 64 | `w8a8_sefp` |

两者使用同一份 80 条混合校准集。GPTQ 只量化 LLM 主干；各组件的实际导出精度以各自的
`quant_type` 为唯一来源：Vision、Audio、Flow、HiFT 和 Speaker 为 `w8a16_sefp`，TTS 为
`w8a8_sefp`。

## 环境

真实导出需要仓库依赖、`transformers==4.57.1`、MiniCPM-o/Token2Wav 依赖、可用 CUDA
设备，以及 XH2a 导出和 HMONNX 运行环境。

## 校准数据

正式配置使用：

```text
xh2modelzoo://data/calib_data/minicpm_o_4_5_qwen_vl_style_mix80.jsonl
```

该数据集参考 Qwen-VL 的校准来源，由 CMMMU、COCO、DocVQA、MMMU 各 18 条和
Wikitext 8 条组成，共 80 条；固定 seed 为 1024，记录按来源轮询交织。构造脚本保留在
`build_vl_calibration_jsonl.py`，可从仓库根目录重新生成：

YAML 中的 `xh2modelzoo://` 前缀表示仓库内资源，不依赖调用命令时的当前目录；若代码安装在仓库
之外，可设置 `XH2MODELZOO_ROOT` 指向仓库根目录。

```bash
python examples_merak/llm/minicpm_o_4_5/build_vl_calibration_jsonl.py \
  --source CMMMU=data/calib_data/Qwen2.5-VL-7B-Instruct_CMMMU_VAL_20250923102519_struct.json \
  --source COCO=data/calib_data/Qwen2.5-VL-7B-Instruct_COCO_VAL_20250923104643_struct.json \
  --source DocVQA=data/calib_data/Qwen2.5-VL-7B-Instruct_DocVQA_VAL_20250923102720_struct.json \
  --source MMMU=data/calib_data/Qwen2.5-VL-7B-Instruct_MMMU_DEV_VAL_20250923102615_struct.json \
  --text-source Wikitext=data/calib_data/wikitext-2-raw-v1.jsonl \
  --samples-per-source 18 \
  --text-samples-per-source 8 \
  --sampling-mode random \
  --seed 1024 \
  --output data/calib_data/minicpm_o_4_5_qwen_vl_style_mix80.jsonl
```

脚本同时写入同名 `.meta.json`，记录来源计数和生成文件的 SHA256。GPTQ 入口还支持显式配置
预计算 `inputs_embeds` 校准文件，或将其与 JSONL 文本混合；正式 YAML 不启用这些实验模式。

## 导出

默认导出 W8A8 LLM + GPTQ G64 + W8A8 TTS：

```bash
PYTHONPATH=$PWD python \
  examples_merak/llm/minicpm_o_4_5/minicpm_o_4_5_workflow.py \
  --model-dir /path/to/MiniCPM-o-4_5 \
  --quant-output-dir work_dirs/minicpm_o_4_5_xh2a_w8a8_gptq_quant \
  --export-output-dir work_dirs/minicpm_o_4_5_xh2a_w8a8_gptq_export \
  --device cuda:0 \
  --overwrite
```

上例显式展示了 `--quant-output-dir` / `--export-output-dir` 可以传入；不传时默认目录按
config stem 生成，结果同样是：
`work_dirs/minicpm_o_4_5_xh2a_w8a8_gptq_quant` 与
`work_dirs/minicpm_o_4_5_xh2a_w8a8_gptq_export`。

后续导出/Golden 示例均省略这两个参数，使用默认目录。

导出 W4A8 LLM + GPTQ G64 + W8A8 TTS：

```bash
PYTHONPATH=$PWD python \
  examples_merak/llm/minicpm_o_4_5/minicpm_o_4_5_workflow.py \
  --model-dir /path/to/MiniCPM-o-4_5 \
  --config-path configs_merak/workflows/xh2a/llm_models/minicpm_o_4_5/minicpm_o_4_5_xh2a_w4a8_gptq.yaml \
  --device cuda:0 \
  --overwrite
```

该配置的默认目录为 `work_dirs/minicpm_o_4_5_xh2a_w4a8_gptq_quant` 与
`work_dirs/minicpm_o_4_5_xh2a_w4a8_gptq_export`。

导出物仍需原始模型目录中的 processor、tokenizer、remote code 和 Token2Wav assets。
Audio Offline 对超过静态 batch 容量的长音频会分块执行并按原顺序拼接；流式 Audio 使用独立
session 图。

## HMONNX Demo

```bash
PYTHONPATH=$PWD python \
  examples_merak/llm/minicpm_o_4_5/minicpm_o_4_5_hmonnx_demo.py \
  --model-dir /path/to/MiniCPM-o-4_5 \
  --work-dir work_dirs/minicpm_o_4_5_xh2a_w8a8_gptq_export \
  --video /path/to/video.mp4 \
  --exec-device cuda:0
```

需要语音回复时增加 `--generate-audio` 和 `--ref-audio /path/to/reference.wav`。输出目录包含
文本和 metadata；启用语音回复后还会写入 WAV 和 speech token。

## 流式 Demo

流式入口支持以下固定用例：

```text
session_audio_text
session_audio_reply
duplex_audio_text
duplex_audio_reply
duplex_omni_reply
```

示例：

```bash
PYTHONPATH=$PWD python \
  examples_merak/llm/minicpm_o_4_5/minicpm_o_4_5_streaming_demo.py \
  --case session_audio_reply \
  --model-dir /path/to/MiniCPM-o-4_5 \
  --work-dir work_dirs/minicpm_o_4_5_xh2a_w8a8_gptq_export \
  --media /path/to/input.wav \
  --exec-device cuda:0 \
  --require-full-hmonnx \
  --output-dir work_dirs/minicpm_o_4_5_streaming_demo
```

Session 与 Duplex API 不能在同一会话中交错使用。流式 Token2Wav 的神经网络阶段只使用
HMONNX 图，不提供 native fallback；固定容量不足时运行时会明确报错。

## Golden

Workflow 支持 `synthetic` 和 `real` 两种 Golden。`synthetic` 逐图检查固定输入输出契约；
`real` 使用真实视频请求记录实际执行到的图。例如：

```bash
PYTHONPATH=$PWD python \
  examples_merak/llm/minicpm_o_4_5/minicpm_o_4_5_workflow.py \
  --model-dir /path/to/MiniCPM-o-4_5 \
  --device cuda:0 \
  --dump-golden \
  --golden-mode synthetic
```
