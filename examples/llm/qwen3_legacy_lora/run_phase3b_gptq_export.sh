#!/usr/bin/env bash
# Phase 3b: 9 gptq export cells. Consumes per-w-bit quarot_gptq state-dicts.
# w4 -> {w4a8,w4a16}; w5 -> {w5a8,w5a16}; w8 -> {w8a16}; (+QK variants added separately)
set -u
cd /data01/home/yujy/work/xh2modelzoo
PY=/data01/home/yujy/miniconda3/envs/qwen3vl/bin/python
EXPORT=examples/llm/qwen3_legacy_lora/qwen3_legacy_lora_xh2a_export_hmonnx.py
BASE=weights/qwen3_xf_base; GGUF=weights/qwen3_xf_lora.gguf
CTX=16384; ISL=256
LOG=work_dirs/logs/matrix; mkdir -p "$LOG"

# locate the gptq state-dict for a given w-bit dir
sd_for() { ls work_dirs/gptq_w$1/*_quarot_gptq/quarot_gptq-state-dict.safetensors 2>/dev/null | head -1; }

# GPU-contention guard: phase3b shares GPUs 0-4 with phase4 QK cells. Wait until
# no other export process is running before claiming the GPUs.
guard_gpus_free() {
  local waited=0
  while :; do
    # count export processes NOT tagged gptq (i.e. commonQK/quarot still on 0-4)
    local others
    others=$(ps -eo cmd | grep qwen3_legacy_lora_xh2a_export | grep -v grep \
             | grep -v "method-tag gptq" | wc -l)
    [ "$others" = 0 ] && break
    sleep 20; waited=$((waited+20))
    [ "$waited" -ge 1800 ] && { echo "guard timeout after ${waited}s, proceeding"; break; }
  done
}
guard_gpus_free

# cell: $1=quant_type $2=wbit $3=gpu
cell() {
  local qt=$1 wb=$2 gpu=$3
  local SD; SD=$(sd_for "$wb")
  if [ -z "$SD" ]; then echo "MISSING gptq state-dict for w$wb, skip $qt"; return; fi
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY $EXPORT \
    --model $BASE --lora-checkpoint $GGUF \
    --quant-type "$qt" --quant-weight "$SD" --method-tag gptq \
    --context-length $CTX --input-sequence-length $ISL \
    > "$LOG/gptq_${qt}.log" 2>&1 &
  echo "launched gptq $qt (w$wb, SD=$SD) on GPU $gpu pid $!"
}

cell w4a8h1_sefp  4 0
cell w4a16h1_sefp 4 1
cell w5a8h1_sefp  5 2
cell w5a16h1_sefp 5 3
cell w8a16h1_sefp 8 4
wait
echo "PHASE3B_GPTQ_EXPORT_DONE"
