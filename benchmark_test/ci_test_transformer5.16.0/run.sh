#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${GPTQMODEL_SOURCE_DIR:-}" ]]; then
  export PYTHONPATH="${GPTQMODEL_SOURCE_DIR}:${GPTQMODEL_SOURCE_DIR}/third_party/auto-round:${PYTHONPATH}"
fi

selection_file="$(mktemp)"
trap 'rm -f "${selection_file}"' EXIT

selector_args=(--repo-root "${REPO_ROOT}" --format paths)
if [[ -n "${CI_TEST_CHANGED_FILES_FILE:-}" ]]; then
  selector_args+=(--changed-files-file "${CI_TEST_CHANGED_FILES_FILE}")
fi
if [[ "${CI_TEST_FORCE_ALL:-0}" == "1" ]]; then
  selector_args+=(--all)
fi

python "${SCRIPT_DIR}/select_tests.py" "${selector_args[@]}" > "${selection_file}"
mapfile -t selected_tests < "${selection_file}"
if [[ ${#selected_tests[@]} -eq 0 ]]; then
  echo "CI impact selector returned no tests" >&2
  exit 2
fi

cd "${REPO_ROOT}"
python -m pytest \
  -c "${SCRIPT_DIR}/pytest.ini" \
  --confcutdir="${SCRIPT_DIR}" \
  "$@" \
  "${selected_tests[@]}"
