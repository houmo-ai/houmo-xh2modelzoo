#!/usr/bin/env bash
# Detached orchestrator: chains quant matrix phases with file-based gates.
# Runs as nohup (NOT in a response channel) so it may sleep freely.
set -u
cd /data01/home/yujy/work/xh2modelzoo
DIR=examples/llm/qwen3_legacy_lora
LOG=work_dirs/logs/matrix; mkdir -p "$LOG"
ORCH="$LOG/orchestrator.log"
echo "[orch $(date +%T)] start" >> "$ORCH"

cell_done() { # $1=full work_dir
  [ -f "$1/meta.json" ] && \
  ls "$1"/hmonnx/prefill/*.onnx >/dev/null 2>&1 && \
  ls "$1"/hmonnx/decode/*.onnx  >/dev/null 2>&1
}

wait_common() {
  local all
  while :; do
    all=1
    for qt in w8a16h1_sefp w4a8h1_sefp w5a8h1_sefp w4a16h1_sefp w5a16h1_sefp; do
      cell_done "work_dirs/qwen3_xf_base-XH2a-16k-${qt}-lora-common" || all=0
    done
    [ "$all" = 1 ] && break
    sleep 30
  done
  echo "[orch $(date +%T)] common all done" >> "$ORCH"
}

wait_gptq_calib() {
  local all
  while :; do
    all=1
    for wb in 4 5 8; do
      ls work_dirs/gptq_w${wb}/*_quarot_gptq/quarot_gptq-state-dict.safetensors >/dev/null 2>&1 || all=0
    done
    [ "$all" = 1 ] && break
    sleep 60
  done
  echo "[orch $(date +%T)] gptq calib all done" >> "$ORCH"
}

# GATE 1: common cells free GPUs 0-4 -> launch quarot exports
wait_common
echo "[orch $(date +%T)] launching phase2 quarot" >> "$ORCH"
bash "$DIR/run_phase2_quarot.sh" >> "$LOG/phase2.log" 2>&1
echo "[orch $(date +%T)] phase2 quarot returned" >> "$ORCH"

# GATE 2: gptq calibrations produced state-dicts -> launch gptq exports
wait_gptq_calib
echo "[orch $(date +%T)] launching phase3b gptq export" >> "$ORCH"
bash "$DIR/run_phase3b_gptq_export.sh" >> "$LOG/phase3b.log" 2>&1
echo "[orch $(date +%T)] phase3b gptq export returned" >> "$ORCH"

echo "[orch $(date +%T)] ALL_PHASES_DONE" >> "$ORCH"
