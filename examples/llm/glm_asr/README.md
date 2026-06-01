# GLM-ASR Quantization for XH2A

This directory contains scripts for quantizing and exporting GLM-ASR models to XH2A HMONNX format.

## Model Overview

GLM-ASR (General Language Model for Automatic Speech Recognition) is a speech recognition model that combines:
- A Whisper-based audio encoder (`audio_tower`)
- A multi-modal projector for audio feature projection
- A Llama-based language model for text generation

## Supported Models

- `GLM-ASR-Nano-2512` (2B parameters)

## Prerequisites

```bash
pip install transformers==5.1.0
```

## Export Workflow

### 1. Export Audio Encoder

Export the audio encoder (audio_tower) component to HMONNX:

```bash
python examples/llm/glm_asr/hmonnx_export_encoder.py \
    --model glm-asr-nano-2512 \
    --quant-type w8a8_sefp \
    --gen_golden
```

This will create:
- `work_dirs/glm-asr-nano-2512_XH2a/Encoder/hmonnx/` - Quantized encoder HMONNX model

### 2. Export LLM (Prefill + Decode)

Export the language model with prefill and decode phases:

```bash
python examples/llm/glm_asr/hmonnx_export_prefill_decode.py \
    --model /data/gexinyu_workspace/modelzoo/llm/glm-asr-nano-2512 \
    --gen_golden
```

This will create:
- `work_dirs/glm-asr-nano-2512_XH2a/Prefill/` - Prefill phase HMONNX model
- `work_dirs/glm-asr-nano-2512_XH2a/Decoder/` - Decode phase HMONNX model
- `work_dirs/glm-asr-nano-2512_XH2a/ConfigFiles/` - HF model configuration files
- `work_dirs/glm-asr-nano-2512_XH2a/token_embedding.pt` - Token embeddings

## Directory Structure

```
examples/llm/glm_asr/
├── README.md                              # This documentation
├── config/
│   ├── config_glm_asr.py                  # Main model config
│   └── llm/
│       └── glm_asr_decode_xh2a.py       # LLM decode config for XH2A
├── hmonnx_export_encoder.py               # Audio encoder export script
└── hmonnx_export_prefill_decode.py        # LLM prefill/decode export script
```

## Model Architecture Details

### Audio Encoder (audio_tower)
- Input: Mel-spectrogram features (128 bins x 3000 frames)
- Architecture: Whisper-style encoder with conv1d layers + transformer
- Output: Hidden states for audio tokens

### Language Model (language_model)
- Base: Llama architecture
- Hidden size: 2048
- Num layers: 28
- Num attention heads: 16
- Num key-value heads: 4 (GQA)
- Max position embeddings: 8192

## Configuration

The configuration file `config/llm/glm_asr_decode_xh2a.py` specifies:
- Target device: `XH2a`
- Frontend type: `TorchFX`
- Input sequence length: 411
- Max sequence length: 2048
- KV cache: Enabled with cache axis at dimension 2

## Quantization Types

- `w8a8_sefp` (default): 8-bit weights, 8-bit activations with SEFP format
- `w8a16_sefp`: 8-bit weights, 16-bit activations
- `w16a16`: 16-bit weights and activations

## Core Dependencies

The model wrapper is implemented in:
- `xh_model_zoo/xh_llm/models/glm_asr/__init__.py`
- `xh_model_zoo/xh_llm/models/glm_asr/_glm_asr_llm_model.py`
- `xh_model_zoo/xh_llm/models/glm_asr/_llm_model_impl.py`
- `xh_model_zoo/xh_llm/models/glm_asr/_llm_onnx_model.py`

## Troubleshooting

### Issue: Model path not found
Ensure the `hf_model_dir` and `config_dir` in `config/llm/glm_asr_decode_xh2a.py` point to valid model checkpoints.

### Issue: CUDA out of memory
Try reducing the batch size or sequence length in the config, or run on CPU by setting `cfg.device = "cpu"`.

### Issue: Missing tokenizer files
Ensure the model directory contains all required HF files (tokenizer_config.json, vocab.json, etc.)
