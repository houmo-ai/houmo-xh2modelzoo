#!/usr/bin/env bash
# Batch export Qwen3.5 family for xh2a with kv=8192, MTP k=4, DFlash input=9 (draft=8).
#
# Covers 5 base models × {w8a8, w4a8} × {MTP, DFlash} = 20 exports.
# w4a8 uses the auto-round GPTQ-packed output dirs.
#
# GPUs: pass CUDA_VISIBLE_DEVICES via env to pin; default 0.
# Example:  bash examples/llm/qwen3_5/batch_export_8k.sh 2>&1 | tee /tmp/batch.log
# Set MTP_HEAD_WEIGHT_BITS=8 to export MTP draft lm_head with W8 instead of default W4.
# Set DFLASH_HEAD_WEIGHT_BITS=8 to export DFlash draft lm_head with W8 instead of default W4.
#
# Feel free to comment out rows you don't want to re-run.

set -u  # intentionally no -e: keep running on single failure; check the log.

cd "$(dirname "$0")/../../.."
source /data01/home/yujy/miniconda3/etc/profile.d/conda.sh
conda activate xhquant

OUT_ROOT="${OUT_ROOT:-work_dirs}"
SEQ=8192
MTP_K=4
MTP_HEAD_WEIGHT_BITS="${MTP_HEAD_WEIGHT_BITS:-4}"
DFLASH_DRAFT_TOKENS=9
DFLASH_HEAD_WEIGHT_BITS="${DFLASH_HEAD_WEIGHT_BITS:-4}"
DFLASH_INPUT_SIZE=$((DFLASH_DRAFT_TOKENS + 1))
LOG_DIR="${LOG_DIR:-output/qwen35_exports}"
ONLY_TAG_REGEX="${ONLY_TAG_REGEX:-}"
FORCE_REEXPORT="${FORCE_REEXPORT:-0}"
for head_bits_var in MTP_HEAD_WEIGHT_BITS DFLASH_HEAD_WEIGHT_BITS; do
  case "${!head_bits_var}" in
    4|8) ;;
    *)
      echo "$head_bits_var must be 4 or 8, got: ${!head_bits_var}" >&2
      exit 2
      ;;
  esac
done
MTP_HEAD_TAG="headw${MTP_HEAD_WEIGHT_BITS}"
DFLASH_HEAD_TAG="headw${DFLASH_HEAD_WEIGHT_BITS}"
mkdir -p "$LOG_DIR"

# model_id | config | float_dir | dflash_dir | autoround_dir
ROWS=(
  "qwen3_5_4b|configs/qwen3_5/qwen3_5_4b_xh2a.py|weights/Qwen3.5-4B|weights/Qwen3.5-4B-DFlash|weights/Qwen3.5-4B-mode1-llm-only"
  "qwen3_5_9b|configs/qwen3_5/qwen3_5_9b_xh2a.py|weights/Qwen3.5-9B|weights/Qwen3.5-9B-DFlash|weights/Qwen3.5-9B-mode1-llm-only"
  "qwen3_5_27b|configs/qwen3_5/qwen3_5_27b_xh2a.py|weights/Qwen3.5-27B|weights/Qwen3.5-27B-DFlash|/data01/home/yujy/work/auto-round/output/sym-mode1"
  # "qwen3_5_35b_a3b|configs/qwen3_5/qwen3_5_35b_a3b_xh2a.py|weights/Qwen3.5-35B-A3B|weights/Qwen3.5-35B-A3B-DFlash|/data01/home/yujy/work/auto-round/output/Qwen3.5-35B-A3B-mode1-llm-only"
  # "qwen3_6_35b_a3b|configs/qwen3_5/qwen3_6_35b_a3b_xh2a.py|weights/Qwen3.6-35B-A3B|weights/Qwen3.6-35B-A3B-DFlash|/data01/home/yujy/work/auto-round/output/Qwen3.6-35B-A3B-mode1-llm-only"
)

run_one () {
  local tag="$1"; shift
  local work_dir="$OUT_ROOT/$tag"
  local log="$LOG_DIR/${tag}.log"
  if [[ -n "$ONLY_TAG_REGEX" ]] && ! [[ "$tag" =~ $ONLY_TAG_REGEX ]]; then
    echo ">>> FILTER $tag"
    return 0
  fi
  if [[ -f "$work_dir/meta.json" ]]; then
    if [[ "$FORCE_REEXPORT" == "1" ]]; then
      echo ">>> CLEAN $tag (FORCE_REEXPORT=1)"
      rm -rf "$work_dir"
    else
    echo ">>> SKIP $tag (meta.json exists)"
    return 0
    fi
  fi
  echo ">>> RUN $tag -> $work_dir (log: $log)"
  python -u examples/llm/qwen3_5/qwen3_5_xh2a_export_hmonnx.py "$@" \
      --work_dir "$work_dir" --max_sequence_length "$SEQ" \
      > "$log" 2>&1
  local rc=$?
  echo "<<< DONE $tag rc=$rc"
  return $rc
}

run_one_moe () {
  local tag="$1"; shift
  local model_dir="$1"; shift
  local quant_type="$1"; shift
  local spec_mode="$1"; shift
  local draft_tokens="$1"; shift
  local optional_args=("$@")
  local quant_weight=""
  local dflash_dir=""
  local spec_head_args=()
  if (( ${#optional_args[@]} >= 1 )); then
    quant_weight="${optional_args[0]}"
  fi
  if (( ${#optional_args[@]} >= 2 )); then
    dflash_dir="${optional_args[1]}"
  fi
  case "$spec_mode" in
    mtp) spec_head_args=(--spec-draft-head-weight-bits "$MTP_HEAD_WEIGHT_BITS") ;;
    dflash) spec_head_args=(--spec-draft-head-weight-bits "$DFLASH_HEAD_WEIGHT_BITS") ;;
  esac
  local work_dir="$OUT_ROOT/$tag"
  local log="$LOG_DIR/${tag}.log"
  if [[ -n "$ONLY_TAG_REGEX" ]] && ! [[ "$tag" =~ $ONLY_TAG_REGEX ]]; then
    echo ">>> FILTER $tag"
    return 0
  fi
  if [[ -f "$work_dir/meta.json" ]]; then
    if [[ "$FORCE_REEXPORT" == "1" ]]; then
      echo ">>> CLEAN $tag (FORCE_REEXPORT=1)"
      rm -rf "$work_dir"
    else
    echo ">>> SKIP $tag (meta.json exists)"
    return 0
    fi
  fi
  echo ">>> RUN $tag -> $work_dir (log: $log)"
  if [[ -n "$quant_weight" && -n "$dflash_dir" ]]; then
    python -u examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_hmonnx.py \
        --model "$model_dir" \
        --context-length "$SEQ" \
        --input-sequence-length 256 \
        --quant-type "$quant_type" \
        --spec-decode-mode "$spec_mode" \
        --num-draft-tokens "$draft_tokens" \
        "${spec_head_args[@]}" \
        --quant-weight "$quant_weight" \
        --dflash-model-dir "$dflash_dir" \
        > "$log" 2>&1
  elif [[ -n "$quant_weight" ]]; then
    python -u examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_hmonnx.py \
        --model "$model_dir" \
        --context-length "$SEQ" \
        --input-sequence-length 256 \
        --quant-type "$quant_type" \
        --spec-decode-mode "$spec_mode" \
        --num-draft-tokens "$draft_tokens" \
        "${spec_head_args[@]}" \
        --quant-weight "$quant_weight" \
        > "$log" 2>&1
  elif [[ -n "$dflash_dir" ]]; then
    python -u examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_hmonnx.py \
        --model "$model_dir" \
        --context-length "$SEQ" \
        --input-sequence-length 256 \
        --quant-type "$quant_type" \
        --spec-decode-mode "$spec_mode" \
        --num-draft-tokens "$draft_tokens" \
        "${spec_head_args[@]}" \
        --dflash-model-dir "$dflash_dir" \
        > "$log" 2>&1
  else
    python -u examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_hmonnx.py \
        --model "$model_dir" \
        --context-length "$SEQ" \
        --input-sequence-length 256 \
        --quant-type "$quant_type" \
        --spec-decode-mode "$spec_mode" \
        --num-draft-tokens "$draft_tokens" \
        "${spec_head_args[@]}" \
        > "$log" 2>&1
  fi
  local rc=$?
  echo "<<< DONE $tag rc=$rc"
  return $rc
}

for row in "${ROWS[@]}"; do
  IFS='|' read -r MID CFG FLOAT_DIR DFLASH_DIR AR_DIR <<<"$row"

  if [[ "$MID" == "qwen3_5_35b_a3b" || "$MID" == "qwen3_6_35b_a3b" ]]; then
    local_model_name="$(basename "$FLOAT_DIR")"
    # run_one_moe "${local_model_name}-XH2a-8k-w8a8h0_sefp-spec_mtp_${MTP_HEAD_TAG}" \
    #   "$FLOAT_DIR" "w8a8h0_sefp" "mtp" "$MTP_K"

    # run_one_moe "${local_model_name}-XH2a-8k-w8a8h0_sefp-spec_dflash_input${DFLASH_INPUT_SIZE}_${DFLASH_HEAD_TAG}" \
    #   "$FLOAT_DIR" "w8a8h0_sefp" "dflash" "$DFLASH_DRAFT_TOKENS" "" "$DFLASH_DIR"

    run_one_moe "${local_model_name}-XH2a-8k-w4a8h1_ssfp-gptq-spec_mtp_${MTP_HEAD_TAG}" \
      "$FLOAT_DIR" "w4a8h1_ssfp" "mtp" "$MTP_K" "$AR_DIR"

    run_one_moe "${local_model_name}-XH2a-8k-w4a8h1_ssfp-gptq-spec_dflash_input${DFLASH_INPUT_SIZE}_${DFLASH_HEAD_TAG}" \
      "$FLOAT_DIR" "w4a8h1_ssfp" "dflash" "$DFLASH_DRAFT_TOKENS" "$AR_DIR" "$DFLASH_DIR"
    continue
  fi

  # # w8a8 + MTP
  # run_one "${MID}_mtp_k${MTP_K}_${MTP_HEAD_TAG}_w8a8_8k" \
  #     --config "$CFG" --hf_model_dir "$FLOAT_DIR" \
  #     --spec_decode_mode mtp --num_draft_tokens "$MTP_K" \
  #     --spec-draft-head-weight-bits "$MTP_HEAD_WEIGHT_BITS"

  # # w8a8 + DFlash
  # run_one "${MID}_dflash_input${DFLASH_INPUT_SIZE}_${DFLASH_HEAD_TAG}_w8a8_8k" \
  #     --config "$CFG" --hf_model_dir "$FLOAT_DIR" \
  #     --spec_decode_mode dflash --num_draft_tokens "$DFLASH_DRAFT_TOKENS" \
  #     --spec-draft-head-weight-bits "$DFLASH_HEAD_WEIGHT_BITS" \
  #     --dflash_model_dir "$DFLASH_DIR"

  # w4a8 + MTP
  run_one "${MID}_mtp_k${MTP_K}_${MTP_HEAD_TAG}_w4a8_8k" \
      --config "$CFG" --hf_model_dir "$AR_DIR" \
      --spec_decode_mode mtp --num_draft_tokens "$MTP_K" \
      --spec-draft-head-weight-bits "$MTP_HEAD_WEIGHT_BITS"

  # w4a8 + DFlash
  run_one "${MID}_dflash_input${DFLASH_INPUT_SIZE}_${DFLASH_HEAD_TAG}_w4a8_8k" \
      --config "$CFG" --hf_model_dir "$AR_DIR" \
      --spec_decode_mode dflash --num_draft_tokens "$DFLASH_DRAFT_TOKENS" \
      --spec-draft-head-weight-bits "$DFLASH_HEAD_WEIGHT_BITS" \
      --dflash_model_dir "$DFLASH_DIR"
done

echo "ALL DONE"
