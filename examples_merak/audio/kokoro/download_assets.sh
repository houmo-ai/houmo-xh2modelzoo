#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <kokoro-model-dir>" >&2
  exit 2
fi

MODEL_ROOT=$1
SOURCE_ROOT="${MODEL_ROOT}/source/kokoro"
PYTORCH_ROOT="${MODEL_ROOT}/pytorch"
KOKORO_COMMIT=dfb907a02bba8152ca444717ca5d78747ccb4bec

if ! command -v hf >/dev/null 2>&1; then
  echo "missing 'hf' CLI; install it with: pip install huggingface_hub" >&2
  exit 1
fi

mkdir -p "${MODEL_ROOT}/source" "${PYTORCH_ROOT}" "${MODEL_ROOT}/onnx"

if [[ ! -d "${SOURCE_ROOT}/.git" ]]; then
  git clone --filter=blob:none https://github.com/hexgrad/kokoro.git "${SOURCE_ROOT}"
fi
git -C "${SOURCE_ROOT}" fetch origin "${KOKORO_COMMIT}"
git -C "${SOURCE_ROOT}" checkout --detach "${KOKORO_COMMIT}"

hf download \
  hexgrad/Kokoro-82M-v1.1-zh \
  config.json \
  kokoro-v1_1-zh.pth \
  voices/zf_001.pt \
  --local-dir "${PYTORCH_ROOT}"

hf download \
  xun/kokoro-v1.1-zh-onnx \
  onnx/config.json \
  onnx/kokoro-v1.1-zh.onnx \
  onnx/voices-v1.1-zh.bin \
  --local-dir "${MODEL_ROOT}"

verify_sha256() {
  local expected=$1
  local path=$2
  local actual
  actual=$(sha256sum "${path}" | awk '{print $1}')
  if [[ "${actual}" != "${expected}" ]]; then
    echo "SHA256 mismatch: ${path}" >&2
    echo "expected: ${expected}" >&2
    echo "actual:   ${actual}" >&2
    exit 1
  fi
}

verify_sha256 \
  bc333efa5ce4ceff433c8c8e5d027a1eca0166001e4e4a62bea2d26ff7a46890 \
  "${PYTORCH_ROOT}/config.json"
verify_sha256 \
  b1d8410fa44dfb5c15471fd6c4225ea6b4e9ac7fa03c98e8bea47a9928476e2b \
  "${PYTORCH_ROOT}/kokoro-v1_1-zh.pth"
verify_sha256 \
  9bdc9a87e13e9bb1ea3e7803259c2ecbfebaeeb2ff80b5d0c76df1a464c1c962 \
  "${PYTORCH_ROOT}/voices/zf_001.pt"
verify_sha256 \
  eefec708cbc7aba8e8129b5c2f7cb92e1fe7d281af1e1dd451592d9ff0714a0d \
  "${MODEL_ROOT}/onnx/kokoro-v1.1-zh.onnx"

echo "Kokoro assets are ready: ${MODEL_ROOT}"
echo "Install the pinned model package before export:"
echo "  pip install -e ${SOURCE_ROOT}"
