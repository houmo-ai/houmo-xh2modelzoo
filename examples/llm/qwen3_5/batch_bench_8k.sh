#!/usr/bin/env bash
# Batch bench all exported Qwen3.5 xh2a variants against the 256-case dataset.
# Runs think=off and think=on (or both) for each exported work_dir.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash examples/llm/qwen3_5/batch_bench_8k.sh
#
# Tunables via env:
#   OUT_ROOT       (default work_dirs) — where exports live
#   BENCH_OUT      (default /tmp/qwen35_bench) — per-model MD/JSON dir
#   DATASET        (default examples/llm/qwen3_5/spec_decode_eval_prompts.jsonl)
#   THINK_MODE     on|off|both (default both)
#   MAX_NEW_TOKENS (default 8192)
#   LIMIT          (default 0 = all 256)
#   DTYPE          (default fp16)
#   SHARD_INDEX    (default 0)
#   NUM_SHARDS     (default 1)
#   AUTO_OFFLOAD_MAX_MEMORY          optional, forwarded to dense bench
#   PREFILL_AUTO_OFFLOAD_MAX_MEMORY  optional, forwarded to dense bench
#   DECODE_AUTO_OFFLOAD_MAX_MEMORY   optional, forwarded to dense bench

set -u
cd "$(dirname "$0")/../../.."
source /data01/home/yujy/miniconda3/etc/profile.d/conda.sh
conda activate xhquant

OUT_ROOT="${OUT_ROOT:-work_dirs}"
BENCH_OUT="${BENCH_OUT:-output/qwen35_bench}"
DATASET="${DATASET:-examples/llm/qwen3_5/spec_decode_eval_prompts.jsonl}"
THINK_MODE="${THINK_MODE:-both}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
LIMIT="${LIMIT:-0}"
DTYPE="${DTYPE:-fp16}"
ONLY_TAG_REGEX="${ONLY_TAG_REGEX:-}"
SHARD_INDEX="${SHARD_INDEX:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
AUTO_OFFLOAD_MAX_MEMORY="${AUTO_OFFLOAD_MAX_MEMORY:-}"
PREFILL_AUTO_OFFLOAD_MAX_MEMORY="${PREFILL_AUTO_OFFLOAD_MAX_MEMORY:-}"
DECODE_AUTO_OFFLOAD_MAX_MEMORY="${DECODE_AUTO_OFFLOAD_MAX_MEMORY:-}"

mkdir -p "$BENCH_OUT"

LIMIT_ARG=()
if [[ "$LIMIT" != "0" ]]; then
  LIMIT_ARG=(--limit "$LIMIT")
fi

SHARD_ARG=()
SUFFIX=""
if [[ "$NUM_SHARDS" != "1" ]]; then
  SHARD_ARG=(--shard-index "$SHARD_INDEX" --num-shards "$NUM_SHARDS")
  SUFFIX=".shard${SHARD_INDEX}of${NUM_SHARDS}"
fi

DENSE_OFFLOAD_ARG=()
if [[ -n "$AUTO_OFFLOAD_MAX_MEMORY" ]]; then
  DENSE_OFFLOAD_ARG+=(--auto-offload-max-memory "$AUTO_OFFLOAD_MAX_MEMORY")
fi
if [[ -n "$PREFILL_AUTO_OFFLOAD_MAX_MEMORY" ]]; then
  DENSE_OFFLOAD_ARG+=(--prefill-auto-offload-max-memory "$PREFILL_AUTO_OFFLOAD_MAX_MEMORY")
fi
if [[ -n "$DECODE_AUTO_OFFLOAD_MAX_MEMORY" ]]; then
  DENSE_OFFLOAD_ARG+=(--decode-auto-offload-max-memory "$DECODE_AUTO_OFFLOAD_MAX_MEMORY")
fi

mapfile -t metas < <(find "$OUT_ROOT" -mindepth 2 -maxdepth 2 -type f -name meta.json | sort)
for meta in "${metas[@]}"; do
  tag="$(basename "$(dirname "$meta")")"
  if [[ "$tag" != *8k* ]]; then
    echo ">>> FILTER $tag (non-8k)"
    continue
  fi
  if [[ -n "$ONLY_TAG_REGEX" ]] && ! [[ "$tag" =~ $ONLY_TAG_REGEX ]]; then
    echo ">>> FILTER $tag"
    continue
  fi
  md="$BENCH_OUT/${tag}${SUFFIX}.md"
  js="$BENCH_OUT/${tag}${SUFFIX}.json"
  if [[ -f "$md" && -f "$js" ]]; then
    echo ">>> SKIP $tag (result exists)"
    continue
  fi
  echo ">>> BENCH $tag"
  if [[ "$tag" == Qwen3.* ]]; then
    python -u examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_spec_decode_bench.py \
        --meta "$meta" --dataset "$DATASET" \
        --output-md "$md" --output-json "$js" \
        --think-mode "$THINK_MODE" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        "${LIMIT_ARG[@]}" \
        "${SHARD_ARG[@]}" \
        2>&1 | tee "$BENCH_OUT/${tag}${SUFFIX}.log"
  else
    python -u examples/llm/qwen3_5/qwen3_5_xh2a_spec_decode_bench.py \
        --meta "$meta" --dataset "$DATASET" \
        --output-md "$md" --output-json "$js" \
        --think-mode "$THINK_MODE" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --dtype "$DTYPE" \
        "${DENSE_OFFLOAD_ARG[@]}" \
        "${LIMIT_ARG[@]}" \
        "${SHARD_ARG[@]}" \
        2>&1 | tee "$BENCH_OUT/${tag}${SUFFIX}.log"
  fi
  echo "<<< DONE $tag"
done

echo "ALL BENCHES DONE -> $BENCH_OUT"
