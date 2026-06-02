#!/usr/bin/env bash
# Phase 4: 4 QK-protected common cells (QK matmul 2nd operand = 16-bit).
# QK precision is encoded natively in quant_type via a 2nd 'a' field (act_bit_2).
set -u
cd /data01/home/yujy/work/xh2modelzoo
PY=/data01/home/yujy/miniconda3/envs/qwen3vl/bin/python
EXPORT=examples/llm/qwen3_legacy_lora/qwen3_legacy_lora_xh2a_export_hmonnx.py
BASE=weights/qwen3_xf_base; GGUF=weights/qwen3_xf_lora.gguf
CTX=16384; ISL=256
LOG=work_dirs/logs/matrix; mkdir -p "$LOG"

# w{W}a{A}a16 => QK protected at 16-bit
QK=(w4a8a16h1_sefp w5a8a16h1_sefp w4a16a16h1_sefp w5a16a16h1_sefp)
GPU=0
for qt in "${QK[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU nohup $PY $EXPORT \
    --model $BASE --lora-checkpoint $GGUF \
    --quant-type "$qt" --quant-weight none --method-tag commonQK \
    --context-length $CTX --input-sequence-length $ISL \
    > "$LOG/commonQK_${qt}.log" 2>&1 &
  echo "launched commonQK $qt on GPU $GPU pid $!"
  GPU=$((GPU+1))
done
wait
echo "PHASE4_COMMONQK_DONE"
