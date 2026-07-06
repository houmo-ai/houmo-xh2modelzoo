# VoxCPM2 XH2a

## 功能

`VoxCPM2` 是 TTS/语音生成模型，可以支持：

- zero-shot Text-to-Speech：只输入文本生成语音
- 参考音频音色克隆：输入文本和 `reference_wav`
- prompt 续生成：输入 `prompt_wav + prompt_text + text`
- reference + prompt 组合模式
- streaming / non-streaming 两种输出方式

模型以 audio latent patches 作为基本音频分块单位。

## 架构

导出时按 VoxCPM2 架构拆成多个 XH2a HMONNX 子图：

- `AudioVAE Encoder`：`wav -> audio latent patches`
- `AudioVAE Decoder`：`audio latent patches -> wav`
- `LocEnc`：`audio latent patch -> LM embedding`
- `LocDiT`：diffusion/CFM 单步 estimator
- `BaseLM`：拆成 `Prefill` 和 `Decode`
- `ResidualLM`：拆成 `Prefill` 和 `Decode`

推理时 host 侧负责 tokenizer、mask 组装、KV cache 管理、diffusion solver 循环、stop 判断和 streaming 分块；神经网络主体由导出的 HMONNX 子图执行。

## 依赖和环境

基础依赖：

```bash
pip install voxcpm
```

本机示例环境：

```bash
export PY=/data01/home/she.gao/miniconda3/envs/xhquant/bin/python
export REPO=/data01/home/she.gao/xh2modelzoo
export EXAMPLE=$REPO/examples/audio/voxcpm2
export PYTHONPATH=$REPO
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
cd $EXAMPLE
```

如果使用 `voxcpm>=2.0`，`VoxCPM2Model` 可能位于 `voxcpm.model.voxcpm2`。当前脚本已兼容顶层导入和新版路径导入。

## 一键参考参数

下面以共享模型为例：

```bash
export MODEL=/data01/nfs_shared/ASR_TTS/VoxCPM2
export CAL_WAV=/data01/nfs_shared/ASR_TTS/CAM++/examples/speaker1_b_cn_16k.wav
```

导出产物默认写到：

```text
$EXAMPLE/work_dirs/VoxCPM2_XH2a
```

## 导出流程

### 1. 导出 BaseLM / ResidualLM

这个脚本会导出四张 LM 图：

- `BaseLM_Prefill`
- `BaseLM_Decode`
- `ResidualLM_Prefill`
- `ResidualLM_Decode`

同时会保存 host 侧小模块和 tokenizer/config 元信息：

```bash
$PY hmonnx_export_voxcpm2.py \
  --model "$MODEL" \
  --prefill_length 216 \
  --cache_length 1024
```

可选：

```bash
--gen_golden
--skip_verify
--verify_fail_on_mismatch
```

主要产物：

```text
work_dirs/VoxCPM2_XH2a/
  BaseLM_Prefill/voxcpm2_baselm_prefill_xh2a.onnx
  BaseLM_Decode/voxcpm2_baselm_decode_xh2a.onnx
  ResidualLM_Prefill/voxcpm2_residuallm_prefill_xh2a.onnx
  ResidualLM_Decode/voxcpm2_residuallm_decode_xh2a.onnx
  host_modules/*.pt
  ConfigFiles/*
  lm_export_meta_info.json
```

### 2. 导出 LocEnc

建议传入真实 wav 做 calibration。脚本会先通过 AudioVAE encoder 得到真实 latent，再切成 patch 作为校准输入：

```bash
$PY export_locenc.py \
  --model "$MODEL" \
  --cal_wav "$CAL_WAV"
```

主要产物：

```text
work_dirs/VoxCPM2_XH2a/LocEnc/
  voxcpm2_locenc_step.onnx
  hmonnx/voxcpm2_locenc_step_xh2a_w8a8_sefp.onnx
  locenc_meta_info.json
```

### 3. 导出 LocDiT

```bash
$PY export_locdit.py \
  --model "$MODEL"
```

主要产物：

```text
work_dirs/VoxCPM2_XH2a/LocDiT/
  voxcpm2_locdit_step.onnx
  hmonnx/voxcpm2_locdit_step_xh2a_w8a8_sefp.onnx
  locdit_meta_info.json
```

### 4. 导出 AudioVAE Encoder

`num_patches` 决定 encoder 定长图可覆盖的 prompt/reference 音频长度。默认 `128` 约覆盖 20.48 秒 @16k：

```bash
$PY export_audiovae_encoder.py \
  --model "$MODEL" \
  --num_patches 128 \
  --audio "$CAL_WAV"
```

主要产物：

```text
work_dirs/VoxCPM2_XH2a/AudioVAE_Encoder_np128/
  voxcpm2_audiovae_encoder.onnx
  hmonnx/voxcpm2_audiovae_encoder_np128_xh2a_w16a16_sefp.onnx
  audiovae_encoder_meta_info.json
```

### 5. 导出 AudioVAE Decoder

建议至少导出两份：

- `num_patches=3`：overlap/crop 流式近似 decoder
- `num_patches=128`：full/non-streaming decoder

```bash
$PY export_audiovae_decoder.py \
  --model "$MODEL" \
  --num_patches 3

$PY export_audiovae_decoder.py \
  --model "$MODEL" \
  --num_patches 128
```

主要产物：

```text
work_dirs/VoxCPM2_XH2a/AudioVAE_Decoder_np3/
  voxcpm2_audiovae_decoder.onnx
  hmonnx/voxcpm2_audiovae_decoder_np3_xh2a_w8a8_sefp.onnx
  audiovae_decoder_np3_meta_info.json

work_dirs/VoxCPM2_XH2a/AudioVAE_Decoder_np128/
  voxcpm2_audiovae_decoder.onnx
  hmonnx/voxcpm2_audiovae_decoder_np128_xh2a_w8a8_sefp.onnx
  audiovae_decoder_np128_meta_info.json
```

### 6. 导出真流式 AudioVAE Decoder

原生 PyTorch 的 `audio_vae.streaming_decode()` 每步只吃最新一个 latent patch，并在 decoder 内部维护 causal conv / transpose conv cache。对应的 HMONNX 真流式图需要把这些 cache 显式做成输入输出：

```bash
$PY export_audiovae_decoder_streaming_stateful.py \
  --model "$MODEL" \
  --num_patches 1
```

主要产物：

```text
work_dirs/VoxCPM2_XH2a/AudioVAE_Decoder_StreamState_np1/
  voxcpm2_audiovae_decoder_streaming_stateful.onnx
  hmonnx/voxcpm2_audiovae_decoder_streaming_stateful_np1_xh2a_w8a8_sefp.onnx
  audiovae_decoder_streaming_stateful_np1_meta_info.json
```

这个图的输入是 `z`、`sr_idx`、`state_in_0..state_in_25`，输出是 `audio`、`state_out_0..state_out_25`。host 侧每条语音开始时把 state 清零，每个 chunk 后把 `state_out_*` 回填到下一步 `state_in_*`。

## 完整导出命令汇总

```bash
export PY=/data01/home/she.gao/miniconda3/envs/xhquant/bin/python
export REPO=/data01/home/she.gao/xh2modelzoo
export EXAMPLE=$REPO/examples/audio/voxcpm2
export PYTHONPATH=$REPO
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export MODEL=/data01/nfs_shared/ASR_TTS/VoxCPM2
export CAL_WAV=/data01/nfs_shared/ASR_TTS/CAM++/examples/speaker1_b_cn_16k.wav

cd $EXAMPLE

$PY hmonnx_export_voxcpm2.py --model "$MODEL" --prefill_length 216 --cache_length 1024
$PY export_locenc.py --model "$MODEL" --cal_wav "$CAL_WAV"
$PY export_locdit.py --model "$MODEL"
$PY export_audiovae_encoder.py --model "$MODEL" --num_patches 128 --audio "$CAL_WAV"
$PY export_audiovae_decoder.py --model "$MODEL" --num_patches 3
$PY export_audiovae_decoder.py --model "$MODEL" --num_patches 128
$PY export_audiovae_decoder_streaming_stateful.py --model "$MODEL" --num_patches 1
```

## Demo 使用流程

Demo 入口是 `hmonnx_demo.py`，需要传入完整导出目录：

```bash
export WORK_DIR=$EXAMPLE/work_dirs/VoxCPM2_XH2a
```

### 1. Zero-shot TTS

只传目标文本：

```bash
$PY hmonnx_demo.py \
  --work_dir "$WORK_DIR" \
  --text "你好，这是一次 VoxCPM2 HMONNX 语音合成测试。" \
  --output zero_shot.wav
```

### 2. 参考音频音色克隆

传 `reference_wav`：

```bash
$PY hmonnx_demo.py \
  --work_dir "$WORK_DIR" \
  --text "这是一段使用参考音色合成的语音。" \
  --reference_wav "$CAL_WAV" \
  --output reference_clone.wav
```

### 3. Prompt 续生成

`prompt_wav` 和 `prompt_text` 必须成对传入：

```bash
$PY hmonnx_demo.py \
  --work_dir "$WORK_DIR" \
  --prompt_wav "$CAL_WAV" \
  --prompt_text "这是一段提示音频对应的文本。" \
  --text "现在继续生成后面的内容。" \
  --output continuation.wav
```

### 4. Reference + Prompt 组合模式

同时传 `reference_wav`、`prompt_wav` 和 `prompt_text`：

```bash
$PY hmonnx_demo.py \
  --work_dir "$WORK_DIR" \
  --reference_wav "$CAL_WAV" \
  --prompt_wav "$CAL_WAV" \
  --prompt_text "这是一段提示音频对应的文本。" \
  --text "这是目标合成文本。" \
  --output reference_prompt.wav
```

### 5. Streaming

加 `--streaming` 走当前 demo 的流式输出。默认后端是 `stateful`，会使用 `AudioVAE_Decoder_StreamState_np1`，每步只解最新一个 latent patch，并显式传递 `state_in/state_out` cache，对齐 PyTorch 原生 `audio_vae.streaming_decode()`：

```bash
$PY hmonnx_demo.py \
  --work_dir "$WORK_DIR" \
  --text "这是流式语音合成测试。" \
  --streaming \
  --output streaming.wav
```

旧版 overlap/crop 近似流式接口仍然保留，用 `--streaming_backend overlap` 切回 `AudioVAE_Decoder_np3`：

```bash
$PY hmonnx_demo.py \
  --work_dir "$WORK_DIR" \
  --text "这是旧版 overlap 流式语音合成测试。" \
  --streaming \
  --streaming_backend overlap \
  --output streaming_overlap_legacy.wav
```

### 5.1 真流式 stateful 对齐

`voxcpm2_stateful_streaming_align.py` 会用 `AudioVAE_Decoder_StreamState_np1`，每步只输入最新 latent patch，并维护 `state_in/state_out` cache，对齐 PyTorch 原生 `audio_vae.streaming_decode()`：

```bash
$PY voxcpm2_stateful_streaming_align.py \
  --work_dir "$WORK_DIR" \
  --stateful_decoder_dir "$WORK_DIR/AudioVAE_Decoder_StreamState_np1" \
  --text "这是一个真实流式解码对齐测试。" \
  --output_dir streaming_align_results/stateful_streaming_align
```

输出包括：

```text
torch_true_streaming.wav
hmonnx_stateful_streaming.wav
stateful_streaming_alignment_report.json
stateful_streaming_alignment_chunks.jsonl
```

### 6. 音频编码后端

`prompt_wav` / `reference_wav` 编码后端由 `--audio_encoder_backend` 控制：

```text
hmonnx  使用导出的 AudioVAE Encoder
torch   使用 PyTorch AudioVAE
auto    优先 torch，失败回退 hmonnx
```

示例：

```bash
$PY hmonnx_demo.py \
  --work_dir "$WORK_DIR" \
  --text "参考音频后端测试。" \
  --reference_wav "$CAL_WAV" \
  --audio_encoder_backend hmonnx \
  --output backend_hmonnx.wav
```

### 7. 端到端 PyTorch 对齐

可以加 `--align_torch` 生成 HMONNX vs PyTorch 的音频级对齐报告：

```bash
$PY hmonnx_demo.py \
  --work_dir "$WORK_DIR" \
  --text "端到端对齐测试。" \
  --output align.wav \
  --align_torch \
  --model_dir "$MODEL"
```

报告会写到：

```text
align.wav.align.json
```

## CV3-Eval 生成测试

`voxcpm2_cv3_eval.py` 会按 `examples/audio/qwen3_tts/eval/qwen3_tts_eval.py` 的方式读取 CV3-Eval zero-shot 数据并生成音频，不计算指标。默认读取：

- zh：`/data01/home/she.gao/CV3-Eval/data/zero_shot/zh`
- en：`/data01/home/she.gao/CV3-Eval/data/zero_shot/en`

支持分别生成：

- 原始 PyTorch/float VoxCPM2：`--mode float`，模型目录默认 `/data01/nfs_shared/ASR_TTS/VoxCPM2`
- 已导出的 HMONNX VoxCPM2：`--mode hmonnx`，work dir 默认 `work_dirs/VoxCPM2_XH2a`

快速验证 zh/en 各 1 条：

```bash
$PY voxcpm2_cv3_eval.py \
  --mode hmonnx \
  --languages zh,en \
  --gpus 0 \
  --max-samples 1 \
  --min-len 2 \
  --max-len 4 \
  --exp-dir cv3_eval_results/zh_en_smoke_hmonnx
```

生成 HMONNX 的 zh/en 全量数据。默认不传 `--max-samples`，会生成每个语种全部样本，当前 CV3-Eval zero-shot zh/en 各 500 条：

```bash
$PY voxcpm2_cv3_eval.py \
  --mode hmonnx \
  --languages zh,en \
  --gpus 0 \
  --exp-dir cv3_eval_results/voxcpm2_cv3_hmonnx
```

多卡并行时，脚本会启动 `len(gpus)` 个 worker，同一个语种的样本按下标分片到各张卡。比如使用 4 张可见卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 $PY voxcpm2_cv3_eval.py \
  --mode hmonnx \
  --languages zh,en \
  --gpus auto \
  --exp-dir cv3_eval_results/voxcpm2_cv3_hmonnx_4gpu
```

也可以显式指定 PyTorch 可见卡编号：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 $PY voxcpm2_cv3_eval.py \
  --mode hmonnx \
  --languages zh,en \
  --gpus 0,1,2,3 \
  --exp-dir cv3_eval_results/voxcpm2_cv3_hmonnx_4gpu
```

生成 float VoxCPM2 的 zh/en 全量数据：

```bash
$PY voxcpm2_cv3_eval.py \
  --mode float \
  --languages zh,en \
  --gpus 0 \
  --exp-dir cv3_eval_results/voxcpm2_cv3_float
```

float 模式同样支持多卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 $PY voxcpm2_cv3_eval.py \
  --mode float \
  --languages zh,en \
  --gpus auto \
  --exp-dir cv3_eval_results/voxcpm2_cv3_float_4gpu
```

默认使用 CV3-Eval 的 `prompt_wav + prompt_text + text` 模式，对应 `--cv3-mode prompt`。如果要把 CV3 的 prompt wav 当作 reference wav 测音色克隆，可改为：

```bash
--cv3-mode reference
```

结果目录结构：

```text
cv3_eval_results/<run>/
  hmonnx/zh/*.wav                 # --mode hmonnx
  hmonnx/en/*.wav                 # --mode hmonnx
  float/zh/*.wav                  # --mode float
  float/en/*.wav                  # --mode float
  generated.<mode>.zh.jsonl
  generated.<mode>.en.jsonl
  summary.<mode>.json
  run_config.<mode>.json
```

## 当前注意事项

- `prompt_wav` 和 `prompt_text` 必须同时提供或同时不提供。
- Demo/pipeline 不包含原始 VoxCPM2 外层的 text normalizer、denoiser、LoRA 切换。
- `LocEnc` 推荐使用真实 wav calibration；如果不传 `--cal_wav`，脚本会 fallback 到随机 latent。
- 如果 `prefill` 输入长度超过 `--prefill_length`，需要重新用更大的 `--prefill_length` 导出 LM。
- 当前导出脚本会做模块级 parity 检查；若 LM decode parity 不达标，说明产物已生成但数值对齐还需要继续排查，不建议直接作为正式交付版本。
