# FireRedASR xh2a Export Guide

## 1. 环境准备

```bash
# 1) xh2modelzoo
cd /home/jiangyong.yu/xh2_work/xh2modelzoo
pip install -e .

# 2) FireRedASR 依赖
# FireRedASR 仓库建议放在 xh2modelzoo 同级目录: ../FireRedASR
pip install -r ../FireRedASR/requirements.txt
pip install kaldi_native_fbank kaldiio peft onnx onnxruntime onnxsim jiwer

# 3) 可选检查
python -c "import torch, xhquant, onnxruntime; print(torch.__version__)"
```

## 2. 模型目录约定

使用 FireRedASR-LLM-L 自带的 Qwen2 基座：

- `weights/FireRedASR-LLM-L/model.pth.tar`
- `weights/FireRedASR-LLM-L/asr_encoder.pth.tar`
- `weights/FireRedASR-LLM-L/cmvn.ark`
- `weights/FireRedASR-LLM-L/Qwen2-7B-Instruct/`

## 3. Common Quant（仅 w4a8 模式需要）

### 3.1 merge_lora

```bash
python examples/audio/fireredasr/audio_llm_xh2a_common_quant.py \
  --config configs/fireredasr/fireredasr_llm_xh2a_2k_gptq_quarot_4bit_ssfp.py \
  --fireredasr_model_dir weights/FireRedASR-LLM-L \
  --lora_mode merge_lora \
  --gptq_calib_dataset wikitext2 \
  --gptq_calib_samples 128 \
  --gptq_seqlen 2048 \
  --save_rotation_matrix \
  --rotate_audio_projector \
  --valid \
  --valid_asr \
  --use_gpu
```

### 3.2 keep_lora

```bash
python examples/audio/fireredasr/audio_llm_xh2a_common_quant.py \
  --config configs/fireredasr/fireredasr_llm_xh2a_2k_gptq_quarot_4bit_ssfp.py \
  --fireredasr_model_dir weights/FireRedASR-LLM-L \
  --lora_mode keep_lora \
  --gptq_calib_dataset wikitext2 \
  --gptq_calib_samples 128 \
  --gptq_seqlen 2048 \
  --save_rotation_matrix \
  --rotate_audio_projector \
  --valid \
  --valid_asr \
  --use_gpu
```

## 4. 四种导出模式

说明：
- Encoder 固定导出为 `w8w8`。
- LLM 默认导出 `w8a8`；带 `resume_from` 时复用 common quant 结果导出 `w4a8`。
- `audio_encoder_xh2a_export.py` 已支持命名增强：输出目录自动区分 `merge/keep + resume/default`。

### 模式1：w8a8 默认，encoder w8w8，llm w8a8 merge_lora

#### 4.1.1 Encoder

```bash
python examples/audio/fireredasr/audio_encoder_xh2a_export.py \
  --model_dir weights/FireRedASR-LLM-L \
  --lora_mode merge_lora \
  --export_hmonnx \
  --valid \
  --valid_asr \
  --use_gpu \
  --generate_hmonnx_golden
```

#### 4.1.2 LLM

```bash
python examples/audio/fireredasr/audio_llm_xh2a_export.py \
  --config configs/fireredasr/fireredasr_llm_xh2a_4k.py \
  --fireredasr_model_dir weights/FireRedASR-LLM-L \
  --lora_mode merge_lora \
  --valid \
  --valid_asr \
  --use_gpu
```

### 模式2：w8a8 默认，encoder w8w8，llm w8a8 keep_lora

#### 4.2.1 Encoder

```bash
python examples/audio/fireredasr/audio_encoder_xh2a_export.py \
  --model_dir weights/FireRedASR-LLM-L \
  --lora_mode keep_lora \
  --export_hmonnx \
  --valid \
  --valid_asr \
  --use_gpu \
  --generate_hmonnx_golden
```

#### 4.2.2 LLM

```bash
python examples/audio/fireredasr/audio_llm_xh2a_export.py \
  --config configs/fireredasr/fireredasr_llm_xh2a_4k.py \
  --fireredasr_model_dir weights/FireRedASR-LLM-L \
  --lora_mode keep_lora \
  --valid \
  --valid_asr \
  --use_gpu
```

### 模式3：w4a8 resume，encoder w8w8，llm w4a8 merge_lora

#### 4.3.1 Encoder（带 resume）

```bash
python examples/audio/fireredasr/audio_encoder_xh2a_export.py \
  --model_dir weights/FireRedASR-LLM-L \
  --lora_mode merge_lora \
  --resume_from work_dirs/fireredasr_llm_xh2a_2k_gptq_quarot_4bit_ssfp_fireredasr_merge_lora/quarot_gptq-state-dict.safetensors \
  --rotated_adapter_path work_dirs/fireredasr_llm_xh2a_2k_gptq_quarot_4bit_ssfp_fireredasr_merge_lora/audio_projector_rotated.safetensors \
  --export_hmonnx \
  --valid \
  --valid_asr \
  --use_gpu \
  --audio_seconds 15 \
  --generate_hmonnx_golden
```

#### 4.3.2 LLM（带 resume）

```bash
python examples/audio/fireredasr/audio_llm_xh2a_export.py \
  --config configs/fireredasr/fireredasr_llm_xh2a_4k.py \
  --fireredasr_model_dir weights/FireRedASR-LLM-L \
  --lora_mode merge_lora \
  --resume_from work_dirs/fireredasr_llm_xh2a_2k_gptq_quarot_4bit_ssfp_fireredasr_merge_lora/quarot_gptq-state-dict.safetensors \
  --rotated_adapter_path work_dirs/fireredasr_llm_xh2a_2k_gptq_quarot_4bit_ssfp_fireredasr_merge_lora/audio_projector_rotated.safetensors \
  --valid \
  --valid_asr \
  --use_gpu \
  --golden \
  --max_seq_length 512 \
  --input_seq_length 256
```

### 模式4：w4a8 resume，encoder w8w8，llm w4a8 keep_lora

#### 4.4.1 Encoder（带 resume）

```bash
python examples/audio/fireredasr/audio_encoder_xh2a_export.py \
  --model_dir weights/FireRedASR-LLM-L \
  --lora_mode keep_lora \
  --resume_from work_dirs/fireredasr_llm_xh2a_2k_gptq_quarot_4bit_ssfp_fireredasr_keep_lora/quarot_gptq-state-dict.safetensors \
  --rotated_adapter_path work_dirs/fireredasr_llm_xh2a_2k_gptq_quarot_4bit_ssfp_fireredasr_keep_lora/audio_projector_rotated.safetensors \
  --export_hmonnx \
  --valid \
  --valid_asr \
  --use_gpu \
  --audio_seconds 15 \
  --generate_hmonnx_golden  
```

#### 4.4.2 LLM（带 resume）

```bash
python examples/audio/fireredasr/audio_llm_xh2a_export.py \
  --config configs/fireredasr/fireredasr_llm_xh2a_4k.py \
  --fireredasr_model_dir weights/FireRedASR-LLM-L \
  --lora_mode keep_lora \
  --resume_from work_dirs/fireredasr_llm_xh2a_2k_gptq_quarot_4bit_ssfp_fireredasr_keep_lora/quarot_gptq-state-dict.safetensors \
  --rotated_adapter_path work_dirs/fireredasr_llm_xh2a_2k_gptq_quarot_4bit_ssfp_fireredasr_keep_lora/audio_projector_rotated.safetensors \
  --valid \
  --valid_asr \
  --use_gpu \
  --golden \
  --max_seq_length 512 \
  --input_seq_length 256
```

## 5. HMONNX Demo 正确性验证

### 5.0 Golden 单独生成（不重新导出）

当 ONNX 已经导出完成，只想单独验证 golden 生成时：

```bash
python examples/audio/fireredasr/audio_llm_xh2a_export.py \
  --config configs/fireredasr/fireredasr_llm_xh2a_4k.py \
  --fireredasr_model_dir weights/FireRedASR-LLM-L \
  --lora_mode keep_lora \
  --resume_from work_dirs/fireredasr_llm_xh2a_2k_gptq_quarot_4bit_ssfp_fireredasr_keep_lora/quarot_gptq-state-dict.safetensors \
  --golden_only \
  --golden_skip_pack
```

- `--golden_only`：仅基于已存在 prefill/decode ONNX 生成 golden，不走导出流程。
- `--golden_skip_pack`：仅生成 golden 目录，不打包 `.tar.gz`（更快，便于验证）。

### 5.1 端到端 demo

```bash
python examples/audio/fireredasr/fireredasr_xh2a_demo.py \
  --audio_hmonnx_path <encoder_hmonnx_path> \
  --llm_hmonnx_dir <llm_export_work_dir> \
  --cmvn_path weights/FireRedASR-LLM-L/cmvn.ark \
  --wav_path data/wav/*.wav \
  --ref_file data/wav/text \
  --use_gpu
```

### 5.2 结果判定

- Encoder 导出脚本中的 `--valid` + `--valid_asr` 需通过。
- LLM 导出脚本中的 `valid_wrap_vs_hf / valid_quant_vs_hf / valid_asr` 需通过。
- standalone demo 输出与 `ref_file` 对齐（或 CER 在可接受范围内）。

## 6. 输出产物命名

`audio_encoder_xh2a_export.py` 输出目录和文件自动区分模式，示例：

- `work_dirs/fireredasr_audio_encoder/merge_lora_w8a8_default/`
- `work_dirs/fireredasr_audio_encoder/keep_lora_w8a8_default/`
- `work_dirs/fireredasr_audio_encoder/merge_lora_resume_quarot_gptq/`
- `work_dirs/fireredasr_audio_encoder/keep_lora_resume_quarot_gptq/`

目录内文件名示例：

- `audio_encoder_<mode_suffix>.onnx`
- `audio_encoder_<mode_suffix>_hmonnx.onnx`
