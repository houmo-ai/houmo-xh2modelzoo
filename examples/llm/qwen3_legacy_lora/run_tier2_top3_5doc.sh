#!/usr/bin/env bash
# Tier-2 confirmation: top-3 variants × all 5 docx, CUDA graph ON, 8-GPU pool.
set -u
cd /data01/home/yujy/work/xh2modelzoo
PY=/data01/home/yujy/miniconda3/envs/qwen3vl/bin/python
EVAL=examples/llm/qwen3_legacy_lora/eval_customer_docx.py
DOCX_DIR=/data01/home/yujy/work/xh2modelzoo/data/../data01/home/yujy/work/xunfei/xf_data
[ -d /data01/home/yujy/work/xunfei/xf_data ] && DOCX_DIR=/data01/home/yujy/work/xunfei/xf_data
LOG=work_dirs/logs/matrix; mkdir -p "$LOG"

# top-3 变体 + meta 路径
VARS=(
  "w8a16-common|work_dirs/qwen3_xf_base-XH2a-16k-w8a16h1_sefp-lora-common/meta.json"
  "w8a16-gptq|work_dirs/qwen3_xf_base-XH2a-16k-w8a16h1_sefp-lora-gptq/meta.json"
  "w5a16-gptq|work_dirs/qwen3_xf_base-XH2a-16k-w5a16h1_sefp-lora-gptq/meta.json"
)

GPU=0
for entry in "${VARS[@]}"; do
  tag="${entry%%|*}"
  meta="${entry##*|}"
  if [ ! -f "$meta" ]; then echo "MISSING: $meta"; continue; fi
  out="tier2_${tag}"
  echo "  launching $tag on GPU $GPU"
  CUDA_VISIBLE_DEVICES=$GPU XH2A_CUDA_GRAPH=1 nohup $PY $EVAL \
    --model "$meta" --backend xh2a \
    --out-tag "$out" \
    --device cuda:0 \
    --max-new-tokens 8192 \
    --docx-dir "$DOCX_DIR" \
    --prompt-md "$DOCX_DIR/prompt.md" \
    > "$LOG/tier2_${tag}.log" 2>&1 &
  echo "    pid=$!  log=$LOG/tier2_${tag}.log"
  GPU=$((GPU+1))
done
wait
echo "TIER2_TOP3_5DOC_DONE"
