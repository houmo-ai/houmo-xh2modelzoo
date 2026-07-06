#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TRANSFORMER55_CI_ARTIFACT_CACHE="${TRANSFORMER55_CI_ARTIFACT_CACHE:-${REPO_ROOT}/work_dirs/transformer55_ci_quant_export_cache}"
export TRANSFORMER55_CI_OUTPUT_ROOT="${TRANSFORMER55_CI_OUTPUT_ROOT:-${REPO_ROOT}/work_dirs/transformer55_ci_quant_export_outputs}"
export TRANSFORMER55_CI_DEVICE="${TRANSFORMER55_CI_DEVICE:-cuda}"
export TRANSFORMER55_CI_TIMEOUT_SECONDS="${TRANSFORMER55_CI_TIMEOUT_SECONDS:-3600}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export NO_PROXY="${NO_PROXY:-10.10.1.53,localhost,127.0.0.1}"
export no_proxy="${no_proxy:-${NO_PROXY}}"

cd "${REPO_ROOT}"


python -m pytest \
  --confcutdir="${SCRIPT_DIR}" \
  "$@" \
  "${SCRIPT_DIR}"
