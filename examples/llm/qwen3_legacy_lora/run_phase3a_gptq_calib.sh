#!/usr/bin/env bash
# Phase 3a: 3 gptq calibrations (quarot+gptq), one per distinct w-bit.
# Each writes to its own out-dir so quarot_gptq-state-dict.safetensors don't collide.
set -u
cd /data01/home/yujy/work/xh2modelzoo
PY=/data01/home/yujy/miniconda3/envs/qwen3vl/bin/python
COMMONQ=examples/llm/qwen3_legacy_lora/qwen3_lora_xh2a_common_quant.py
BASE=weights/qwen3_xf_base
GGUF=weights/qwen3_xf_lora.gguf
LOG=work_dirs/logs/matrix; mkdir -p "$LOG"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1  # cache exists; skip net HEAD retries

# wbit -> gpu
run() { # $1=wbits $2=gpu
  local wb=$1 gpu=$2
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY $COMMONQ \
    --model $BASE --lora $GGUF \
    --w-bits $wb --out-dir work_dirs/gptq_w${wb}/ \
    --datasets-dir data/datasets/ \
    > "$LOG/gptq_calib_w${wb}.log" 2>&1 &
  echo "launched gptq calib w-bits=$wb on GPU $gpu pid $!"
}
run 4 5
run 5 6
run 8 7
wait
echo "PHASE3A_GPTQ_CALIB_DONE"
