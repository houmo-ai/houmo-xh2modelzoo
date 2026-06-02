#!/usr/bin/env bash
# Detached waiter: launch phase4 QK cells once quarot exports free GPUs 0-4.
# Runs as nohup so it may sleep freely (not in a response channel).
set -u
cd /data01/home/yujy/work/xh2modelzoo
DIR=examples/llm/qwen3_legacy_lora
LOG=work_dirs/logs/matrix; mkdir -p "$LOG"
W="$LOG/phase4_waiter.log"
echo "[p4wait $(date +%T)] start, waiting on quarot exports" >> "$W"

cell_done() {
  [ -f "$1/meta.json" ] && \
  ls "$1"/hmonnx/prefill/*.onnx >/dev/null 2>&1 && \
  ls "$1"/hmonnx/decode/*.onnx  >/dev/null 2>&1
}

# wait for all 5 quarot exports to complete (frees GPUs 0-4)
while :; do
  all=1
  for qt in w8a16h1_sefp w4a8h1_sefp w5a8h1_sefp w4a16h1_sefp w5a16h1_sefp; do
    cell_done "work_dirs/qwen3_xf_base-XH2a-16k-${qt}-lora-quarot" || all=0
  done
  [ "$all" = 1 ] && break
  sleep 30
done
echo "[p4wait $(date +%T)] quarot done -> launching phase4 QK" >> "$W"
bash "$DIR/run_phase4_commonqk.sh" >> "$LOG/phase4.log" 2>&1
echo "[p4wait $(date +%T)] phase4 returned" >> "$W"
