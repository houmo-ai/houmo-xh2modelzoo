#!/bin/bash
# QAT 训练启动脚本 — LLM+Flow Pred100 模式
set -e

# ---- 环境 ----
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export COSYVOICE_ROOT=$HOME/workspace/repo/develop/CosyVoice
export MODEL_DIR=/data01/nfs_shared/ASR_TTS/CosyVoice3-0.5B-2512
export HF_MODEL_DIR=$MODEL_DIR/CosyVoice-BlankEN
export YAML_PATH=$MODEL_DIR/cosyvoice3.yaml
export TRAIN_DATA=$HOME/workspace/repo/xh2modelzoo/examples/audio/Cosyvoice3/data_librispeech/train_abs.list
export CV_DATA=$HOME/workspace/repo/xh2modelzoo/examples/audio/Cosyvoice3/data_librispeech/dev_abs.list
export BATCH_SIZE=1
export TRAIN_STEPS=500
export EVAL_INTERVAL=99999
export EVAL_MAX_BATCHES=5
export LEARNING_RATE=1e-5
export W_MAN_BIT=8

# ---- 激活 conda ----
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate xhquant

echo "=== GPU check ==="
python -c "import torch; print(f'GPU: {torch.cuda.current_device()} — {torch.cuda.get_device_name(0)}')"

echo "=== Starting QAT training (LLM+Flow Pred100) ==="
python -u cosyvoice3_qat_e2e_llm_pred100.py
