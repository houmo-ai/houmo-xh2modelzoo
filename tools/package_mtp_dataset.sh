#!/usr/bin/env bash
# Package Qwen3.5 MTP-head pruning dataset for upload to artifactory.
#
# Two artefacts are produced (selectable via flags):
#
#   1. Dataset (default, ~13 MB) -- precomputed token frequencies and hot-vocab
#      selections, sufficient to drive `rerank_model_for_mtp()` and export with
#      `--mtp-head-k`. Source layout:
#        analysis/mtp_head_longtail/v2_reranked/
#          ├── freqs/      (~9 MB; per-corpus token .counter.pt)
#          └── selection/  (~35 MB; hot_ids_{K}.pt, id_map_{K}.pt, stats)
#
#   2. Raw corpus (--corpus, ~2.5 GB gz / ~8.2 GB raw) -- the moss-003-sft-data
#      JSONL used by `build_mtp_hot_vocab.py` to rebuild `freqs/` from scratch.
#      Public HF corpora (BELLE, C4-en, COIG-PC, ...) are HF-streamed at
#      runtime and not bundled. Openclaw raw TSV is internal-only and not
#      shipped.
#
# The reranked HF repos (analysis/.../output, ~1.6 GB) are not bundled — they
# are reproducible from selection/ via examples/llm/qwen3_5/utils/model_utils.py
# `rerank_model_for_mtp()`.
#
# Usage:
#   bash tools/package_mtp_dataset.sh                # dataset tarball under /tmp
#   bash tools/package_mtp_dataset.sh --upload       # dataset + upload
#   bash tools/package_mtp_dataset.sh --corpus       # corpus tarball under /tmp
#   bash tools/package_mtp_dataset.sh --corpus --upload
#
# After upload the artefacts are available at:
#   dataset: http://10.10.1.53:8081/artifactory/model_zoo2/qwen3_5_mtp_head_pruning/qwen3_5_mtp_head_pruning_dataset_v1.tar.gz
#   corpus:  http://10.10.1.53:8081/artifactory/model_zoo2/qwen3_5_mtp_head_pruning/qwen3_5_mtp_head_pruning_corpus_v1.tar.gz
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="${REPO_ROOT}/analysis/mtp_head_longtail/v2_reranked"
DATASET_NAME="qwen3_5_mtp_head_pruning_dataset_v1.tar.gz"
CORPUS_NAME="qwen3_5_mtp_head_pruning_corpus_v1.tar.gz"
UPLOAD=0
MODE=dataset

for arg in "$@"; do
  case "$arg" in
    --upload) UPLOAD=1 ;;
    --corpus) MODE=corpus ;;
    --dataset) MODE=dataset ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

if [[ "${MODE}" == "dataset" ]]; then
  ARCHIVE_NAME="${DATASET_NAME}"
  ARCHIVE_PATH="/tmp/${ARCHIVE_NAME}"
  if [[ ! -d "${SRC_DIR}/freqs" || ! -d "${SRC_DIR}/selection" ]]; then
    echo "ERROR: expected ${SRC_DIR}/{freqs,selection} to exist." >&2
    echo "Run the build pipeline first (see examples/llm/qwen3_5/MTP_HEAD_PRUNING.md)." >&2
    exit 1
  fi
  echo "[package] creating ${ARCHIVE_PATH} (freqs + selection)"
  tar --exclude='*.json.lock' \
      --exclude='__pycache__' \
      -czf "${ARCHIVE_PATH}" \
      -C "${SRC_DIR}" \
      freqs selection
else
  ARCHIVE_NAME="${CORPUS_NAME}"
  ARCHIVE_PATH="/tmp/${ARCHIVE_NAME}"
  if [[ ! -d "${SRC_DIR}/corpus" ]]; then
    echo "ERROR: expected ${SRC_DIR}/corpus to exist (raw JSONL corpus)." >&2
    exit 1
  fi
  echo "[package] creating ${ARCHIVE_PATH} (raw corpus ~8 GB, will take several minutes)"
  tar -czf "${ARCHIVE_PATH}" -C "${SRC_DIR}" corpus
fi

ls -lh "${ARCHIVE_PATH}"

if [[ "${UPLOAD}" == "1" ]]; then
  if [[ ! -x /data01/modelzoo2/upload.sh ]]; then
    echo "ERROR: /data01/modelzoo2/upload.sh not available on this host." >&2
    exit 1
  fi
  # upload.sh embeds the local path into the artifactory key, so cd into the
  # tarball directory and pass only the basename to get a clean download URL.
  echo "[upload] bash /data01/modelzoo2/upload.sh ${ARCHIVE_NAME} qwen3_5_mtp_head_pruning/"
  (cd "$(dirname "${ARCHIVE_PATH}")" && bash /data01/modelzoo2/upload.sh "${ARCHIVE_NAME}" qwen3_5_mtp_head_pruning/)
  echo "[done] http://10.10.1.53:8081/artifactory/model_zoo2/qwen3_5_mtp_head_pruning/${ARCHIVE_NAME}"
fi
