#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

PROFILE_TOOL=${PROFILE_TOOL:-auto}
PROFILE_MODE=${PROFILE_MODE:-node_profile_only}
REQUESTED_PROFILE_TOOL=${PROFILE_TOOL}
PYTHON_BIN=${PYTHON_BIN:-/opt/extdata/.conda/envs/xhquant_torch28/bin/python}
PROFILER_SCRIPT=${PROFILER_SCRIPT:-${SCRIPT_DIR}/graph_module_nsight_profiler.py}
CONFIG=${CONFIG:-configs_merak/xh2a/llm_models/qwen3_legacy/8b/qwen3_8b_legacy_xh2a_2k.py}
EVAL_TYPE=${EVAL_TYPE:-quanted_disable}
WARMUP_RUNS=${WARMUP_RUNS:-2}
PROFILE_RUNS=${PROFILE_RUNS:-1}
TOPK=${TOPK:-20}
REPORT_SUFFIX=${REPORT_SUFFIX:-}
NCU_SET=${NCU_SET:-full}
TARGET_PROCESSES=${TARGET_PROCESSES:-all}
NVTX_INCLUDE=${NVTX_INCLUDE:-}
NSIGHT_BIN_DIR=${NSIGHT_BIN_DIR:-}
NSYS_BIN=${NSYS_BIN:-}
NCU_BIN=${NCU_BIN:-}
AUTO_FALLBACK_TO_NSYS_ON_NCU_PERMISSION_ERROR=${AUTO_FALLBACK_TO_NSYS_ON_NCU_PERMISSION_ERROR:-1}
NCU_PERMISSION_ERROR_SEEN=0
NCU_PERMISSION_LOG=
FELL_BACK_TO_NSYS=0


default_nvtx_include() {
  case "${PROFILE_MODE}" in
    node_profile_only)
      printf '%s\n' 'regex:graph_module_node::.*/'
      ;;
    total_latency_only)
      printf '%s\n' 'regex:(graph_module|fx_interpreter)::total_latency/'
      ;;
    *)
      printf '%s\n' ''
      ;;
  esac
}


resolve_profiler_bin() {
  local tool_name="$1"
  local explicit_bin=""
  local candidate=""
  local -a candidates=()

  case "${tool_name}" in
    nsys)
      explicit_bin="${NSYS_BIN}"
      ;;
    ncu)
      explicit_bin="${NCU_BIN}"
      ;;
    *)
      return 1
      ;;
  esac

  if [[ -n "${explicit_bin}" && -x "${explicit_bin}" ]]; then
    printf '%s\n' "${explicit_bin}"
    return 0
  fi

  if candidate=$(command -v "${tool_name}" 2>/dev/null); then
    printf '%s\n' "${candidate}"
    return 0
  fi

  if [[ -n "${NSIGHT_BIN_DIR}" ]]; then
    candidates+=("${NSIGHT_BIN_DIR}/${tool_name}")
  fi
  if [[ -n "${CUDA_HOME:-}" ]]; then
    candidates+=("${CUDA_HOME}/bin/${tool_name}")
  fi
  if [[ -n "${CUDA_PATH:-}" ]]; then
    candidates+=("${CUDA_PATH}/bin/${tool_name}")
  fi
  candidates+=("/usr/local/cuda/bin/${tool_name}")

  case "${tool_name}" in
    nsys)
      candidates+=(
        "/opt/nvidia/nsight-systems/bin/nsys"
        /opt/nvidia/nsight-systems/*/bin/nsys
        /opt/nvidia/nsight-systems/*/target-linux-x64/nsys
        /usr/local/NVIDIA-Nsight-Systems/*/target-linux-x64/nsys
        /opt/NVIDIA-Nsight-Systems/*/target-linux-x64/nsys
      )
      ;;
    ncu)
      candidates+=(
        "/opt/nvidia/nsight-compute/ncu"
        /opt/nvidia/nsight-compute/*/ncu
        /usr/local/cuda/NsightCompute-*/ncu
        /usr/local/NVIDIA-Nsight-Compute/*/ncu
        /opt/NVIDIA-Nsight-Compute/*/ncu
      )
      ;;
  esac

  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}


select_profile_tool() {
  local -a auto_tool_order=()

  if [[ "${PROFILE_TOOL}" == "auto" ]]; then
    if [[ "${PROFILE_MODE}" == "node_profile_only" ]]; then
      auto_tool_order=(ncu nsys)
    else
      auto_tool_order=(nsys ncu)
    fi

    for tool_name in "${auto_tool_order[@]}"; do
      if PROFILE_BIN=$(resolve_profiler_bin "${tool_name}"); then
        PROFILE_TOOL="${tool_name}"
        return 0
      fi
    done

    echo "Neither nsys nor ncu was found." >&2
    echo "Searched PATH, NSIGHT_BIN_DIR, NSYS_BIN/NCU_BIN, CUDA_HOME/CUDA_PATH, and common Nsight install directories." >&2
    echo "Set NSIGHT_BIN_DIR, NSYS_BIN, or NCU_BIN explicitly if Nsight is installed outside PATH." >&2
    exit 127
  fi

  if PROFILE_BIN=$(resolve_profiler_bin "${PROFILE_TOOL}"); then
    return 0
  fi

  echo "${PROFILE_TOOL} not found." >&2
  echo "Searched PATH, NSIGHT_BIN_DIR, NSYS_BIN/NCU_BIN, CUDA_HOME/CUDA_PATH, and common Nsight install directories." >&2
  echo "Set NSIGHT_BIN_DIR, NSYS_BIN, or NCU_BIN explicitly if Nsight is installed outside PATH." >&2
  exit 127
}


build_profiler_command() {
  local tool_name="$1"
  local report_base="$2"

  CMD=()
  case "${tool_name}" in
    nsys)
      CMD=(
        "${PROFILE_BIN}" profile
        --trace=cuda,nvtx,osrt
        --capture-range=cudaProfilerApi
        --capture-range-end=stop
        --sample=none
        --force-overwrite=true
        -o "${report_base}"
        "${PYTHON_BIN}"
        "${PROFILER_ARGS[@]}"
        --cuda-profiler-api
      )
      ;;

    ncu)
      CMD=(
        "${PROFILE_BIN}"
        --set "${NCU_SET}"
        --target-processes "${TARGET_PROCESSES}"
        --force-overwrite
        -o "${report_base}"
        --nvtx
      )

      if [[ -n "${NVTX_INCLUDE}" ]]; then
        CMD+=(--nvtx-include "${NVTX_INCLUDE}")
      fi

      CMD+=("${PYTHON_BIN}" "${PROFILER_ARGS[@]}")
      ;;

    *)
      echo "Unsupported PROFILE_TOOL: ${tool_name}. Expected one of: nsys, ncu" >&2
      exit 2
      ;;
  esac
}


execute_profiler_command() {
  local log_path="$1"

  printf 'Running command:'
  for arg in "${CMD[@]}"; do
    printf ' %q' "${arg}"
  done
  printf '\n'

  set +e
  "${CMD[@]}" 2>&1 | tee "${log_path}"
  local cmd_status=${PIPESTATUS[0]}
  set -e
  return "${cmd_status}"
}


handle_ncu_permission_error() {
  local log_path="$1"

  if ! grep -q 'ERR_NVGPUCTRPERM' "${log_path}"; then
    return 1
  fi

  NCU_PERMISSION_ERROR_SEEN=1
  NCU_PERMISSION_LOG="${log_path}"

  echo "NCU failed because this user cannot access NVIDIA GPU performance counters." >&2
  echo "Fix the permission first, or use Nsight Systems as a timeline-only fallback." >&2
  echo "Reference: https://developer.nvidia.com/ERR_NVGPUCTRPERM" >&2

  if [[ "${REQUESTED_PROFILE_TOOL}" == "auto" && "${AUTO_FALLBACK_TO_NSYS_ON_NCU_PERMISSION_ERROR}" == "1" ]]; then
    if PROFILE_BIN=$(resolve_profiler_bin nsys); then
      FELL_BACK_TO_NSYS=1
      PROFILE_TOOL="nsys"
      REPORT_BASE="${REPORT_DIR}/${PROFILE_TOOL}_${EVAL_TYPE}${REPORT_SUFFIX}"
      RUN_LOG="${REPORT_BASE}.launcher.log"
      echo "Retrying with Nsight Systems to at least capture a timeline report." >&2
      build_profiler_command "${PROFILE_TOOL}" "${REPORT_BASE}"
      if execute_profiler_command "${RUN_LOG}"; then
        return 0
      fi
    fi
  fi

  return 1
}


report_path_for_tool() {
  local tool_name="$1"
  local report_base="$2"

  case "${tool_name}" in
    nsys)
      if [[ -f "${report_base}.nsys-rep" ]]; then
        printf '%s\n' "${report_base}.nsys-rep"
        return 0
      fi
      if [[ -f "${report_base}.qdrep" ]]; then
        printf '%s\n' "${report_base}.qdrep"
        return 0
      fi
      printf '%s\n' "${report_base}.nsys-rep"
      return 0
      ;;
    ncu)
      printf '%s\n' "${report_base}.ncu-rep"
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}


print_result_summary() {
  local report_path=""

  report_path=$(report_path_for_tool "${PROFILE_TOOL}" "${REPORT_BASE}")

  echo "Launcher log: ${RUN_LOG}"
  if [[ "${NCU_PERMISSION_ERROR_SEEN}" == "1" && -n "${NCU_PERMISSION_LOG}" ]]; then
    echo "NCU permission-error log: ${NCU_PERMISSION_LOG}"
  fi
  if [[ "${FELL_BACK_TO_NSYS}" == "1" ]]; then
    echo "Profiler fallback: switched from ncu to nsys because GPU performance counters were not accessible."
  fi

  if [[ -f "${report_path}" ]]; then
    echo "Profiler result: generated ${PROFILE_TOOL} report at ${report_path}"
    if [[ "${PROFILE_TOOL}" == "nsys" ]]; then
      echo "Open report with: nsys-ui ${report_path}"
    else
      echo "Open report with: ncu-ui ${report_path}"
    fi
    return 0
  fi

  if [[ "${NCU_PERMISSION_ERROR_SEEN}" == "1" ]]; then
    echo "Profiler result: no ncu report was generated because GPU performance-counter access was denied."
    echo "Reference: https://developer.nvidia.com/ERR_NVGPUCTRPERM"
    return 0
  fi

  if grep -q 'No kernels were profiled\.' "${RUN_LOG}" 2>/dev/null; then
    echo "Profiler result: no report was generated because no kernels matched the selected NVTX filter."
    return 0
  fi

  echo "Profiler result: profiler finished without creating the expected report file ${report_path}."
  return 0
}

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN=$(command -v python || true)
fi

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "No runnable Python interpreter found. Set PYTHON_BIN explicitly." >&2
  exit 127
fi

if [[ ! -f "${PROFILER_SCRIPT}" ]]; then
  echo "Profiler script not found: ${PROFILER_SCRIPT}" >&2
  exit 2
fi

CONFIG_STEM=$(basename -- "${CONFIG}")
CONFIG_STEM=${CONFIG_STEM%.*}
CFG_NAME="${CONFIG_STEM}_graph_profiler"
select_profile_tool

if [[ -z "${NVTX_INCLUDE}" ]]; then
  NVTX_INCLUDE=$(default_nvtx_include)
fi

REPORT_DIR=${REPORT_DIR:-${REPO_ROOT}/work_dirs/${CFG_NAME}/${EVAL_TYPE}/nsight/${PROFILE_MODE}}
REPORT_BASE=${REPORT_BASE:-${REPORT_DIR}/${PROFILE_TOOL}_${EVAL_TYPE}${REPORT_SUFFIX}}

mkdir -p "${REPORT_DIR}"
cd "${REPO_ROOT}"

declare -a PROFILER_ARGS=(
  "${PROFILER_SCRIPT}"
  --config "${CONFIG}"
  --eval-type "${EVAL_TYPE}"
  --warmup-runs "${WARMUP_RUNS}"
  --profile-runs "${PROFILE_RUNS}"
  --topk "${TOPK}"
  --profile-mode "${PROFILE_MODE}"
  --emit-nvtx
)

RUN_LOG="${REPORT_BASE}.launcher.log"
build_profiler_command "${PROFILE_TOOL}" "${REPORT_BASE}"

if execute_profiler_command "${RUN_LOG}"; then
  :
else
  status=$?
  if [[ "${PROFILE_TOOL}" == "ncu" ]] && handle_ncu_permission_error "${RUN_LOG}"; then
    status=0
  fi

  if [[ "${status}" -ne 0 ]]; then
    echo "Profiler failed. See launcher log: ${RUN_LOG}" >&2
    exit "${status}"
  fi
fi

print_result_summary
