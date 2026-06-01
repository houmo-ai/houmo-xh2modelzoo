#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

cd "${REPO_ROOT}"

conda run -n xhquant python "${SCRIPT_DIR}/export_yolo26_artifacts.py" "$@"

ONNX_PATH="${SCRIPT_DIR}/yolo26m.onnx"
QUANT_TYPE="w8a8h1_sefp"
for ((i = 1; i <= $#; i++)); do
  if [[ "${!i}" == "--onnx" ]]; then
    j=$((i + 1))
    ONNX_PATH="${!j}"
  elif [[ "${!i}" == "--quant-type" ]]; then
    j=$((i + 1))
    QUANT_TYPE="${!j}"
  fi
done

ONNX_NAME=$(basename "${ONNX_PATH}")
ONNX_NAME="${ONNX_NAME%.onnx}"
HMONNX_PATH="${SCRIPT_DIR}/work_dirs/${ONNX_NAME}/hmonnx/${ONNX_NAME}_${QUANT_TYPE}_XH2a.onnx"
REPORT_PATH="${SCRIPT_DIR}/work_dirs/${ONNX_NAME}/gather_index_report.json"

conda run -n xhquant python "${SCRIPT_DIR}/analyze_gather_indices.py" \
  --model "${HMONNX_PATH}" \
  --json-out "${REPORT_PATH}"
