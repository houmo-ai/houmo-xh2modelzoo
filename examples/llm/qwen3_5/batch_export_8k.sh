#!/usr/bin/env bash
# Batch export Qwen3.5 family for xh2a with kv=8192, MTP k=4, DFlash block=16.
#
# Covers 5 base models × {w8a8, w4a8} × {MTP, DFlash} = 20 exports.
# w4a8 uses the auto-round GPTQ-packed output dirs.
#
# GPUs: pass CUDA_VISIBLE_DEVICES via env to pin; default 0.
# Example:  bash examples/llm/qwen3_5/batch_export_8k.sh 2>&1 | tee /tmp/batch.log
#
# Feel free to comment out rows you don't want to re-run.

set -u  # intentionally no -e: keep running on single failure; check the log.

cd "$(dirname "$0")/../../.."
source /data01/home/yujy/miniconda3/etc/profile.d/conda.sh
conda activate xhquant

OUT_ROOT="${OUT_ROOT:-work_dirs}"
SEQ=8192
MTP_K=4
DFLASH_BS=9
LOG_DIR="${LOG_DIR:-output/qwen35_exports}"
ONLY_TAG_REGEX="${ONLY_TAG_REGEX:-}"
FORCE_REEXPORT="${FORCE_REEXPORT:-0}"
mkdir -p "$LOG_DIR"

# model_id | config | float_dir | dflash_dir | autoround_dir
ROWS=(
  "qwen3_5_4b|configs/qwen3_5/qwen3_5_4b_xh2a.py|weights/Qwen3.5-4B|weights/Qwen3.5-4B-DFlash|/data01/home/yujy/work/auto-round/output/Qwen3.5-4B-mode1-llm-only"
  "qwen3_5_9b|configs/qwen3_5/qwen3_5_9b_xh2a.py|weights/Qwen3.5-9B|weights/Qwen3.5-9B-DFlash|/data01/home/yujy/work/auto-round/output/Qwen3.5-9B-mode1-llm-only"
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
  if (( ${#optional_args[@]} >= 1 )); then
    quant_weight="${optional_args[0]}"
  fi
  if (( ${#optional_args[@]} >= 2 )); then
    dflash_dir="${optional_args[1]}"
  fi
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
    # run_one_moe "${local_model_name}-XH2a-8k-w8a8h0_sefp-spec_mtp" \
    #   "$FLOAT_DIR" "w8a8h0_sefp" "mtp" "$MTP_K"

    # run_one_moe "${local_model_name}-XH2a-8k-w8a8h0_sefp-spec_dflash" \
    #   "$FLOAT_DIR" "w8a8h0_sefp" "dflash" "$DFLASH_BS" "" "$DFLASH_DIR"

    run_one_moe "${local_model_name}-XH2a-8k-w4a8h1_ssfp-gptq-spec_mtp" \
      "$FLOAT_DIR" "w4a8h1_ssfp" "mtp" "$MTP_K" "$AR_DIR"

    run_one_moe "${local_model_name}-XH2a-8k-w4a8h1_ssfp-gptq-spec_dflash" \
      "$FLOAT_DIR" "w4a8h1_ssfp" "dflash" "$DFLASH_BS" "$AR_DIR" "$DFLASH_DIR"
    continue
  fi

  # # w8a8 + MTP
  # run_one "${MID}_mtp_k${MTP_K}_w8a8_8k" \
  #     --config "$CFG" --hf_model_dir "$FLOAT_DIR" \
  #     --spec_decode_mode mtp --num_draft_tokens "$MTP_K"

  # # w8a8 + DFlash
  # run_one "${MID}_dflash_bs${DFLASH_BS}_w8a8_8k" \
  #     --config "$CFG" --hf_model_dir "$FLOAT_DIR" \
  #     --spec_decode_mode dflash --num_draft_tokens "$DFLASH_BS" \
  #     --dflash_model_dir "$DFLASH_DIR"

  # w4a8 + MTP
  run_one "${MID}_mtp_k${MTP_K}_w4a8_8k" \
      --config "$CFG" --hf_model_dir "$AR_DIR" \
      --spec_decode_mode mtp --num_draft_tokens "$MTP_K"

  # w4a8 + DFlash
  run_one "${MID}_dflash_bs${DFLASH_BS}_w4a8_8k" \
      --config "$CFG" --hf_model_dir "$AR_DIR" \
      --spec_decode_mode dflash --num_draft_tokens "$DFLASH_BS" \
      --dflash_model_dir "$DFLASH_DIR"
done

echo "ALL DONE"
