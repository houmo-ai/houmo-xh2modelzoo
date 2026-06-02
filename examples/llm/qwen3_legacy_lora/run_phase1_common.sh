#!/usr/bin/env bash
# Phase 1: 5 "common" quant cells (no quarot/gptq) — fully proven by smoke.
# Each pins one physical GPU (CUDA_VISIBLE_DEVICES=1 is global; we override).
set -u
cd /data01/home/yujy/work/xh2modelzoo
PY=/data01/home/yujy/miniconda3/envs/qwen3vl/bin/python
EXPORT=examples/llm/qwen3_legacy_lora/qwen3_legacy_lora_xh2a_export_hmonnx.py
BASE=weights/qwen3_xf_base
GGUF=weights/qwen3_xf_lora.gguf
CTX=16384; ISL=256
LOG=work_dirs/logs/matrix; mkdir -p "$LOG"

BITS=(w8a16h1_sefp w4a8h1_sefp w5a8h1_sefp w4a16h1_sefp w5a16h1_sefp)
GPU=0
for qt in "${BITS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU nohup $PY $EXPORT \
    --model $BASE --lora-checkpoint $GGUF \
    --quant-type "$qt" --quant-weight none --method-tag common \
    --context-length $CTX --input-sequence-length $ISL \
    > "$LOG/common_${qt}.log" 2>&1 &
  echo "launched common $qt on GPU $GPU pid $!"
  GPU=$((GPU+1))
done
wait
echo "PHASE1_COMMON_DONE"
