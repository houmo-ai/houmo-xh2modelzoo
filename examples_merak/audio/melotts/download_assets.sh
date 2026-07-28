#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <melotts-model-dir>" >&2
  exit 2
fi

MODEL_ROOT=$1
SOURCE_ROOT="${MODEL_ROOT}/source"
WEIGHT_ROOT="${MODEL_ROOT}/melotts_weights"
PACKAGE_ROOT="${MODEL_ROOT}/packages"
DOWNLOAD_ROOT="${MODEL_ROOT}/downloads"
MELO_COMMIT=209145371cff8fc3bd60d7be902ea69cbdb7965a
MELO_ARCHIVE="${DOWNLOAD_ROOT}/vits-melo-tts-zh_en.tar.bz2"

mkdir -p \
  "${SOURCE_ROOT}" \
  "${WEIGHT_ROOT}" \
  "${PACKAGE_ROOT}" \
  "${DOWNLOAD_ROOT}"

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

if [[ ! -d "${SOURCE_ROOT}/MeloTTS/.git" ]]; then
  git clone \
    --filter=blob:none \
    https://github.com/myshell-ai/MeloTTS.git \
    "${SOURCE_ROOT}/MeloTTS"
fi
git -C "${SOURCE_ROOT}/MeloTTS" fetch origin "${MELO_COMMIT}"
git -C "${SOURCE_ROOT}/MeloTTS" checkout --detach "${MELO_COMMIT}"

curl -L --fail --retry 20 --retry-delay 2 \
  -o "${WEIGHT_ROOT}/config.json" \
  https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/ZH/config.json
curl -L --fail --retry 20 --retry-delay 2 \
  -o "${WEIGHT_ROOT}/checkpoint.pth" \
  https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/ZH/checkpoint.pth
curl -L --fail --retry 20 --retry-delay 2 \
  -o "${MELO_ARCHIVE}" \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-melo-tts-zh_en.tar.bz2

verify_sha256 \
  d58b5acdab89ad2bbd65325affab309ae3cb964834b02f9a60587474e81c8bb9 \
  "${WEIGHT_ROOT}/config.json"
verify_sha256 \
  a74e9eadffff065c75eb6dfa040efa72cad23e72cfea70d39190bc174fb97093 \
  "${WEIGHT_ROOT}/checkpoint.pth"
verify_sha256 \
  e58351ed7149f290a54534538badd4077cdbe6fddc964b24d0bee870415d1514 \
  "${MELO_ARCHIVE}"

if [[ ! -f "${PACKAGE_ROOT}/vits-melo-tts-zh_en/model.onnx" ]]; then
  tar -xjf "${MELO_ARCHIVE}" -C "${PACKAGE_ROOT}"
fi
verify_sha256 \
  bf30582eb1b012250a35b1a4a80e7dfbcf8485e7bb9de0d95efbbeef0e4ad86d \
  "${PACKAGE_ROOT}/vits-melo-tts-zh_en/model.onnx"

echo "MeloTTS assets are ready: ${MODEL_ROOT}"
