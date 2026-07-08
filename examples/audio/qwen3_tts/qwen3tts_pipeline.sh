#!/bin/bash
# Qwen3-TTS one-click export and evaluation pipeline
# Supports export, test and accuracy eval for 1.7B-VoiceDesign / 1.7B-CustomVoice / 0.6B-CustomVoice / 0.6B-Base

set -e  # exit immediately on error

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHONPATH="/data01/home/she.gao/xh2modelzoo"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# default test text (data values, kept in Chinese)
TEST_TEXT_1P7B="基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈。"
TEST_INSTRUCT_1P7B="体现温柔甜美的女声，音调适中，语速平稳。"
TEST_TEXT_1P7B_CUSTOM="基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈。"
TEST_SPEAKER_1P7B_CUSTOM="vivian"
TEST_TEXT_0P6B="基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈。"
TEST_SPEAKER_0P6B="vivian"

# 0.6B-Base voice-clone reference audio + text (follows the README clone_1.wav convention)
TEST_TEXT_0P6B_BASE="基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈。"
REF_AUDIO_0P6B_BASE="/tmp/clone_1.wav"
REF_TEXT_0P6B_BASE="甚至出现交易几乎停滞的情况。"
REF_AUDIO_URL="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_1.wav"

# accuracy eval config (README defaults)
EVAL_MAX_SAMPLES=20
EVAL_GPUS="0,1,2,3"
EVAL_SPEAKER_MODE="round-robin"

# golden export flag (empty = off; set to "--golden" by --golden)
GOLDEN_FLAG=""
HF_MODEL_DIR_OVERRIDE=""
HF_MODEL_DIR_ARGS=()

# ============================================================================
# Helper functions
# ============================================================================

# print a timestamped log line
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[${timestamp}] [${level}] ${message}" | tee -a "${LOG_FILE}"
}

log_info() {
    log "INFO" "$@"
}

log_error() {
    log "ERROR" "$@"
}

log_success() {
    log "SUCCESS" "$@"
}

# check whether the last command succeeded
check_status() {
    if [ $? -ne 0 ]; then
        log_error "$1 failed, stopping"
        exit 1
    fi
}

resolve_model_dir() {
    local default_dir="$1"
    if [ -n "${HF_MODEL_DIR_OVERRIDE}" ]; then
        echo "${HF_MODEL_DIR_OVERRIDE}"
    else
        echo "${default_dir}"
    fi
}

# ensure the voice-clone reference audio exists (download from OSS if missing)
ensure_ref_audio() {
    if [ ! -f "${REF_AUDIO_0P6B_BASE}" ]; then
        log_info "reference audio missing, downloading: ${REF_AUDIO_URL}"
        curl -sSL -o "${REF_AUDIO_0P6B_BASE}" "${REF_AUDIO_URL}"
        check_status "reference audio download"
    fi
    log_info "reference audio ready: ${REF_AUDIO_0P6B_BASE}"
}

# show usage help
show_usage() {
    cat << EOF
Usage: $0 [options]

Options:
    --model MODEL           model(s): 1_7B_voicedesign, 1_7B_customvoice, 0_6B_customvoice, 0_6B_base
                            (short aliases 1.7b / 1.7b-custom / 0.6b / 0.6b-base also accepted);
                            comma-joined for multiple, e.g. 1_7B_voicedesign,0_6B_customvoice
    --hf-model-dir PATH     override HF model directory for export/native test
    --export                export HMONNX models
    --test-native           test the original float model
    --test-hmonnx           test the HMONNX model
    --eval                  run accuracy evaluation
    --eval-samples N        eval sample count (default: 20)
    --eval-gpus GPUS        GPUs for eval (default: 0,1,2,3)
    --golden                also dump golden tensors during export (for hardware comparison)
    --help                  show this help

Examples:
    # export the 1.7B-VoiceDesign model
    $0 --model 1_7B_voicedesign --export

    # export the 1.7B-CustomVoice model from a HF cache snapshot
    $0 --model 1_7B_customvoice --export \\
       --hf-model-dir /data01/home/she.gao/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots/0c0e3051f131929182e2c023b9537f8b1c68adfe

    # test the 0.6B-CustomVoice float model
    $0 --model 0_6B_customvoice --test-native

    # full flow: export + test + eval
    $0 --model 1_7B_voicedesign --export --test-native --test-hmonnx --eval

    # export with golden dump (for hardware comparison)
    $0 --model 0_6B_customvoice --export --golden

    # process multiple models at once
    $0 --model 1_7B_voicedesign,1_7B_customvoice,0_6B_customvoice --export --test-hmonnx

EOF
}

# ============================================================================
# 1.7B-VoiceDesign export
# ============================================================================

export_1p7b() {
    log_info "=========================================="
    log_info "Exporting 1.7B-VoiceDesign model"
    log_info "=========================================="

    # 1. Talker
    log_info "[1/4] export Talker..."
    PYTHONPATH=${PYTHONPATH} python qwen3_tts_talker_export.py \
        --config ./config/llm/qwen3_tts_12hz_talker_2k_xh2a.py \
        --variant 1_7B_voicedesign \
        --name qwen3_tts_12hz_1_7B_voicedesign_talker_2k_xh2a \
        "${HF_MODEL_DIR_ARGS[@]}" \
        ${GOLDEN_FLAG} \
        >> "${LOG_FILE}" 2>&1
    check_status "Talker export"
    log_success "Talker export done"

    # 2. CodePredictor
    log_info "[2/4] export CodePredictor..."
    PYTHONPATH=${PYTHONPATH} python qwen3_tts_code_predictor_export.py \
        --config ./config/llm/qwen3_tts_12hz_code_predictor_2k_xh2a.py \
        --variant 1_7B_voicedesign \
        --name qwen3_tts_12hz_1_7B_voicedesign_code_predictor_2k_xh2a \
        "${HF_MODEL_DIR_ARGS[@]}" \
        ${GOLDEN_FLAG} \
        >> "${LOG_FILE}" 2>&1
    check_status "CodePredictor export"
    log_success "CodePredictor export done"

    # 3. TextProjection
    log_info "[3/4] export TextProjection..."
    PYTHONPATH=${PYTHONPATH} python qwen3_tts_text_projection_export.py \
        --config ./config/llm/qwen3_tts_12hz_text_projection_xh2a.py \
        --variant 1_7B_voicedesign \
        --name qwen3_tts_12hz_1_7B_text_projection_xh2a \
        "${HF_MODEL_DIR_ARGS[@]}" \
        ${GOLDEN_FLAG} \
        >> "${LOG_FILE}" 2>&1
    check_status "TextProjection export"
    log_success "TextProjection export done"

    # 4. SpeechTokenizer
    log_info "[4/4] export SpeechTokenizer..."
    PYTHONPATH=${PYTHONPATH} python qwen3_tts_speech_tokenizer_export.py \
        --config ./config/llm/qwen3_tts_12hz_speech_tokenizer_xh2a.py \
        --variant 1_7B_voicedesign \
        --name qwen3_tts_12hz_1_7B_speech_tokenizer_xh2a \
        "${HF_MODEL_DIR_ARGS[@]}" \
        ${GOLDEN_FLAG} \
        >> "${LOG_FILE}" 2>&1
    check_status "SpeechTokenizer export"
    log_success "SpeechTokenizer export done"

    log_success "1.7B-VoiceDesign export complete"
}

# ============================================================================
# 1.7B-CustomVoice export
# ============================================================================

export_1p7b_customvoice() {
    log_info "=========================================="
    log_info "Exporting 1.7B-CustomVoice model"
    log_info "=========================================="

    # 1. Talker
    log_info "[1/4] export Talker..."
    PYTHONPATH=${PYTHONPATH} python qwen3_tts_talker_export.py \
        --config ./config/llm/qwen3_tts_12hz_talker_2k_xh2a.py \
        --variant 1_7B_customvoice \
        --name qwen3_tts_12hz_1_7B_customvoice_talker_2k_xh2a \
        "${HF_MODEL_DIR_ARGS[@]}" \
        ${GOLDEN_FLAG} \
        >> "${LOG_FILE}" 2>&1
    check_status "Talker export"
    log_success "Talker export done"

    # 2. CodePredictor
    log_info "[2/4] export CodePredictor..."
    PYTHONPATH=${PYTHONPATH} python qwen3_tts_code_predictor_export.py \
        --config ./config/llm/qwen3_tts_12hz_code_predictor_2k_xh2a.py \
        --variant 1_7B_customvoice \
        --name qwen3_tts_12hz_1_7B_customvoice_code_predictor_2k_xh2a \
        "${HF_MODEL_DIR_ARGS[@]}" \
        ${GOLDEN_FLAG} \
        >> "${LOG_FILE}" 2>&1
    check_status "CodePredictor export"
    log_success "CodePredictor export done"

    # 3. TextProjection
    log_info "[3/4] export TextProjection..."
    PYTHONPATH=${PYTHONPATH} python qwen3_tts_text_projection_export.py \
        --config ./config/llm/qwen3_tts_12hz_text_projection_xh2a.py \
        --variant 1_7B_customvoice \
        --name qwen3_tts_12hz_1_7B_customvoice_text_projection_xh2a \
        "${HF_MODEL_DIR_ARGS[@]}" \
        ${GOLDEN_FLAG} \
        >> "${LOG_FILE}" 2>&1
    check_status "TextProjection export"
    log_success "TextProjection export done"

    # 4. SpeechTokenizer
    log_info "[4/4] export SpeechTokenizer..."
    PYTHONPATH=${PYTHONPATH} python qwen3_tts_speech_tokenizer_export.py \
        --config ./config/llm/qwen3_tts_12hz_speech_tokenizer_xh2a.py \
        --variant 1_7B_customvoice \
        --name qwen3_tts_12hz_1_7B_customvoice_speech_tokenizer_xh2a \
        "${HF_MODEL_DIR_ARGS[@]}" \
        ${GOLDEN_FLAG} \
        >> "${LOG_FILE}" 2>&1
    check_status "SpeechTokenizer export"
    log_success "SpeechTokenizer export done"

    log_success "1.7B-CustomVoice export complete"
}

# ============================================================================
# 0.6B-CustomVoice export
# ============================================================================

export_0p6b() {
    log_info "=========================================="
    log_info "Exporting 0.6B-CustomVoice model"
    log_info "=========================================="

    # 1. Talker
    log_info "[1/4] export Talker..."
    PYTHONPATH=${PYTHONPATH} python qwen3_tts_talker_export.py \
        --config ./config/llm/qwen3_tts_12hz_talker_2k_xh2a.py \
        --variant 0_6B_customvoice \
        --name qwen3_tts_12hz_0_6B_customvoice_talker_2k_xh2a \
        "${HF_MODEL_DIR_ARGS[@]}" \
        ${GOLDEN_FLAG} \
        >> "${LOG_FILE}" 2>&1
    check_status "Talker export"
    log_success "Talker export done"

    # 2. CodePredictor
    log_info "[2/4] export CodePredictor..."
    PYTHONPATH=${PYTHONPATH} python qwen3_tts_code_predictor_export.py \
        --config ./config/llm/qwen3_tts_12hz_code_predictor_2k_xh2a.py \
        --variant 0_6B_customvoice \
        --name qwen3_tts_12hz_0_6B_customvoice_code_predictor_2k_xh2a \
        "${HF_MODEL_DIR_ARGS[@]}" \
        ${GOLDEN_FLAG} \
        >> "${LOG_FILE}" 2>&1
    check_status "CodePredictor export"
    log_success "CodePredictor export done"

    # 3. TextProjection
    log_info "[3/4] export TextProjection..."
    PYTHONPATH=${PYTHONPATH} python qwen3_tts_text_projection_export.py \
        --config ./config/llm/qwen3_tts_12hz_text_projection_xh2a.py \
        --variant 0_6B_customvoice \
        --name qwen3_tts_12hz_0_6B_customvoice_text_projection_xh2a \
        "${HF_MODEL_DIR_ARGS[@]}" \
        ${GOLDEN_FLAG} \
        >> "${LOG_FILE}" 2>&1
    check_status "TextProjection export"
    log_success "TextProjection export done"

    # 4. SpeechTokenizer
    log_info "[4/4] export SpeechTokenizer..."
    PYTHONPATH=${PYTHONPATH} python qwen3_tts_speech_tokenizer_export.py \
        --config ./config/llm/qwen3_tts_12hz_speech_tokenizer_xh2a.py \
        --variant 0_6B_customvoice \
        --name qwen3_tts_12hz_0_6B_customvoice_speech_tokenizer_xh2a \
        "${HF_MODEL_DIR_ARGS[@]}" \
        ${GOLDEN_FLAG} \
        >> "${LOG_FILE}" 2>&1
    check_status "SpeechTokenizer export"
    log_success "SpeechTokenizer export done"

    log_success "0.6B-CustomVoice export complete"
}

# ============================================================================
# 0.6B-Base export (voice-clone)
# ============================================================================

export_0p6b_base() {
    log_info "=========================================="
    log_info "Exporting 0.6B-Base model (voice-clone)"
    log_info "=========================================="

    ensure_ref_audio

    # 1. Talker
    log_info "[1/5] export Talker..."
    PYTHONPATH=${PYTHONPATH} python qwen3_tts_talker_export.py \
        --config ./config/llm/qwen3_tts_12hz_talker_2k_xh2a.py \
        --variant 0_6B_base \
        --name qwen3_tts_12hz_0_6B_base_talker_2k_xh2a \
        "${HF_MODEL_DIR_ARGS[@]}" \
        ${GOLDEN_FLAG} \
        >> "${LOG_FILE}" 2>&1
    check_status "Talker export"
    log_success "Talker export done"

    # 2. CodePredictor
    log_info "[2/5] export CodePredictor..."
    PYTHONPATH=${PYTHONPATH} python qwen3_tts_code_predictor_export.py \
        --config ./config/llm/qwen3_tts_12hz_code_predictor_2k_xh2a.py \
        --variant 0_6B_base \
        --name qwen3_tts_12hz_0_6B_base_code_predictor_2k_xh2a \
        "${HF_MODEL_DIR_ARGS[@]}" \
        ${GOLDEN_FLAG} \
        >> "${LOG_FILE}" 2>&1
    check_status "CodePredictor export"
    log_success "CodePredictor export done"

    # 3. TextProjection
    log_info "[3/5] export TextProjection..."
    PYTHONPATH=${PYTHONPATH} python qwen3_tts_text_projection_export.py \
        --config ./config/llm/qwen3_tts_12hz_text_projection_xh2a.py \
        --variant 0_6B_base \
        --name qwen3_tts_12hz_0_6B_base_text_projection_xh2a \
        "${HF_MODEL_DIR_ARGS[@]}" \
        ${GOLDEN_FLAG} \
        >> "${LOG_FILE}" 2>&1
    check_status "TextProjection export"
    log_success "TextProjection export done"

    # 4. SpeechTokenizer
    log_info "[4/5] export SpeechTokenizer..."
    PYTHONPATH=${PYTHONPATH} python qwen3_tts_speech_tokenizer_export.py \
        --config ./config/llm/qwen3_tts_12hz_speech_tokenizer_xh2a.py \
        --variant 0_6B_base \
        --name qwen3_tts_12hz_0_6B_base_speech_tokenizer_xh2a \
        "${HF_MODEL_DIR_ARGS[@]}" \
        ${GOLDEN_FLAG} \
        >> "${LOG_FILE}" 2>&1
    check_status "SpeechTokenizer export"
    log_success "SpeechTokenizer export done"

    # 5. Voice-clone frontend: speech_tokenizer.encode + speaker_encoder
    log_info "[5/5] export voice-clone frontend (speech_tokenizer.encode + speaker_encoder)..."
    PYTHONPATH=${PYTHONPATH} python qwen3_tts_base_frontend_export.py \
        --variant 0_6B_base \
        --name qwen3_tts_12hz_0_6B_base_frontend_xh2a \
        "${HF_MODEL_DIR_ARGS[@]}" \
        --force \
        ${GOLDEN_FLAG} \
        >> "${LOG_FILE}" 2>&1
    check_status "Voice-clone frontend export"
    log_success "Voice-clone frontend export done"

    log_success "0.6B-Base export complete"
}

# ============================================================================
# Test the original float model
# ============================================================================

test_native_1p7b() {
    log_info "=========================================="
    log_info "Testing 1.7B-VoiceDesign float model"
    log_info "=========================================="

    local output_file="${SCRIPT_DIR}/test_native_1p7b_${TIMESTAMP}.wav"

    PYTHONPATH=${PYTHONPATH} python native_demo.py \
        --mode voice-design \
        --model "$(resolve_model_dir ./data/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign/)" \
        --text "${TEST_TEXT_1P7B}" \
        --instruct "${TEST_INSTRUCT_1P7B}" \
        --dtype bf16 \
        --out "${output_file}" \
        >> "${LOG_FILE}" 2>&1
    check_status "1.7B native test"

    if [ -f "${output_file}" ]; then
        log_success "1.7B native test done, output: ${output_file}"
    else
        log_error "1.7B native test failed, no audio produced"
        exit 1
    fi
}

test_native_1p7b_customvoice() {
    log_info "=========================================="
    log_info "Testing 1.7B-CustomVoice float model"
    log_info "=========================================="

    local output_file="${SCRIPT_DIR}/test_native_1p7b_customvoice_${TIMESTAMP}.wav"

    PYTHONPATH=${PYTHONPATH} python native_demo.py \
        --mode custom-voice \
        --model "$(resolve_model_dir ./data/models/Qwen3-TTS-12Hz-1.7B-CustomVoice/)" \
        --text "${TEST_TEXT_1P7B_CUSTOM}" \
        --speaker "${TEST_SPEAKER_1P7B_CUSTOM}" \
        --dtype bf16 \
        --out "${output_file}" \
        >> "${LOG_FILE}" 2>&1
    check_status "1.7B-CustomVoice native test"

    if [ -f "${output_file}" ]; then
        log_success "1.7B-CustomVoice native test done, output: ${output_file}"
    else
        log_error "1.7B-CustomVoice native test failed, no audio produced"
        exit 1
    fi
}

test_native_0p6b() {
    log_info "=========================================="
    log_info "Testing 0.6B-CustomVoice float model"
    log_info "=========================================="

    local output_file="${SCRIPT_DIR}/test_native_0p6b_${TIMESTAMP}.wav"

    PYTHONPATH=${PYTHONPATH} python native_demo.py \
        --mode custom-voice \
        --model "$(resolve_model_dir ./data/models/Qwen3-TTS-12Hz-0.6B-CustomVoice/)" \
        --text "${TEST_TEXT_0P6B}" \
        --speaker "${TEST_SPEAKER_0P6B}" \
        --dtype bf16 \
        --out "${output_file}" \
        >> "${LOG_FILE}" 2>&1
    check_status "0.6B native test"

    if [ -f "${output_file}" ]; then
        log_success "0.6B native test done, output: ${output_file}"
    else
        log_error "0.6B native test failed, no audio produced"
        exit 1
    fi
}

test_native_0p6b_base() {
    log_info "=========================================="
    log_info "Testing 0.6B-Base float model (voice-clone)"
    log_info "=========================================="

    ensure_ref_audio

    local output_file="${SCRIPT_DIR}/test_native_0p6b_base_${TIMESTAMP}.wav"

    # use fp32+sdpa: README notes fp16+sdpa hits multinomial NaN during native sampling
    PYTHONPATH=${PYTHONPATH} python native_demo.py \
        --mode voice-clone \
        --model "$(resolve_model_dir ./data/models/Qwen3-TTS-12Hz-0.6B-Base/)" \
        --dtype fp32 \
        --text "${TEST_TEXT_0P6B_BASE}" \
        --ref_audio "${REF_AUDIO_0P6B_BASE}" \
        --ref_text "${REF_TEXT_0P6B_BASE}" \
        --out "${output_file}" \
        >> "${LOG_FILE}" 2>&1
    check_status "0.6B-Base native test"

    if [ -f "${output_file}" ]; then
        log_success "0.6B-Base native test done, output: ${output_file}"
    else
        log_error "0.6B-Base native test failed, no audio produced"
        exit 1
    fi
}

# ============================================================================
# Test the HMONNX model
# ============================================================================

test_hmonnx_1p7b() {
    log_info "=========================================="
    log_info "Testing 1.7B-VoiceDesign HMONNX model"
    log_info "=========================================="

    PYTHONPATH=${PYTHONPATH} python qwen3_tts_demo.py \
        --config ./config/llm/qwen3_tts_12hz_xh2a_hmonnx.py \
        --variant 1_7B_voicedesign \
        >> "${LOG_FILE}" 2>&1
    check_status "1.7B HMONNX test"

    log_success "1.7B HMONNX test done"
}

test_hmonnx_1p7b_customvoice() {
    log_info "=========================================="
    log_info "Testing 1.7B-CustomVoice HMONNX model"
    log_info "=========================================="

    PYTHONPATH=${PYTHONPATH} python qwen3_tts_demo.py \
        --config ./config/llm/qwen3_tts_12hz_xh2a_hmonnx.py \
        --variant 1_7B_customvoice \
        >> "${LOG_FILE}" 2>&1
    check_status "1.7B-CustomVoice HMONNX test"

    log_success "1.7B-CustomVoice HMONNX test done"
}

test_hmonnx_0p6b() {
    log_info "=========================================="
    log_info "Testing 0.6B-CustomVoice HMONNX model"
    log_info "=========================================="

    PYTHONPATH=${PYTHONPATH} python qwen3_tts_demo.py \
        --config ./config/llm/qwen3_tts_12hz_xh2a_hmonnx.py \
        --variant 0_6B_customvoice \
        >> "${LOG_FILE}" 2>&1
    check_status "0.6B HMONNX test"

    log_success "0.6B HMONNX test done"
}

test_hmonnx_0p6b_base() {
    log_info "=========================================="
    log_info "Testing 0.6B-Base HMONNX model (voice-clone)"
    log_info "=========================================="

    ensure_ref_audio

    PYTHONPATH=${PYTHONPATH} python qwen3_tts_demo.py \
        --config ./config/llm/qwen3_tts_12hz_xh2a_hmonnx.py \
        --variant 0_6B_base \
        >> "${LOG_FILE}" 2>&1
    check_status "0.6B-Base HMONNX test"

    log_success "0.6B-Base HMONNX test done"
}

# ============================================================================
# Accuracy evaluation
# ============================================================================

run_eval() {
    local variant="$1"
    local exp_dir="qwen3tts_eval_zh"
    local eval_desc="custom voice"
    local hf_model

    if [ "${variant}" = "0_6B_base" ]; then
        exp_dir="qwen3tts_eval_zh_voice_clone"
        eval_desc="voice clone"
    fi
    case "${variant}" in
        0_6B_base)
            hf_model="$(resolve_model_dir ./data/models/Qwen3-TTS-12Hz-0.6B-Base)"
            ;;
        1_7B_customvoice)
            hf_model="$(resolve_model_dir ./data/models/Qwen3-TTS-12Hz-1.7B-CustomVoice)"
            ;;
        *)
            hf_model="$(resolve_model_dir ./data/models/Qwen3-TTS-12Hz-0.6B-CustomVoice)"
            ;;
    esac

    log_info "=========================================="
    log_info "Running ${eval_desc} accuracy evaluation for ${variant}"
    log_info "config: samples=${EVAL_MAX_SAMPLES}, GPUs=${EVAL_GPUS}, speaker-mode=${EVAL_SPEAKER_MODE}, exp-dir=${exp_dir}, hf-model=${hf_model}"
    log_info "=========================================="

    # native mode eval
    log_info "Running native-mode accuracy eval (${variant})..."
    PYTHONPATH=${PYTHONPATH} python eval/qwen3_tts_eval.py \
        --mode native \
        --variant ${variant} \
        --hf-model "${hf_model}" \
        --gpus ${EVAL_GPUS} \
        --max-samples ${EVAL_MAX_SAMPLES} \
        --speaker-mode ${EVAL_SPEAKER_MODE} \
        --exp-dir ${exp_dir} \
        >> "${LOG_FILE}" 2>&1
    check_status "native-mode eval (${variant})"
    log_success "native-mode eval done (${variant})"

    # hmonnx mode eval
    log_info "Running HMONNX-mode accuracy eval (${variant})..."
    PYTHONPATH=${PYTHONPATH} python eval/qwen3_tts_eval.py \
        --mode hmonnx \
        --variant ${variant} \
        --gpus ${EVAL_GPUS} \
        --max-samples ${EVAL_MAX_SAMPLES} \
        --speaker-mode ${EVAL_SPEAKER_MODE} \
        --exp-dir ${exp_dir} \
        >> "${LOG_FILE}" 2>&1
    check_status "HMONNX-mode eval (${variant})"
    log_success "HMONNX-mode eval done (${variant})"

    log_success "accuracy eval done (${variant}), results in ${exp_dir}/"
}

# ============================================================================
# Main
# ============================================================================

main() {
    # parse command-line args
    local models=""
    local do_export=false
    local do_test_native=false
    local do_test_hmonnx=false
    local do_eval=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --model)
                models="$2"
                shift 2
                ;;
            --export)
                do_export=true
                shift
                ;;
            --test-native)
                do_test_native=true
                shift
                ;;
            --test-hmonnx)
                do_test_hmonnx=true
                shift
                ;;
            --eval)
                do_eval=true
                shift
                ;;
            --golden)
                GOLDEN_FLAG="--golden"
                shift
                ;;
            --hf-model-dir)
                HF_MODEL_DIR_OVERRIDE="$2"
                HF_MODEL_DIR_ARGS=(--hf-model-dir "$2")
                shift 2
                ;;
            --eval-samples)
                EVAL_MAX_SAMPLES="$2"
                shift 2
                ;;
            --eval-gpus)
                EVAL_GPUS="$2"
                shift 2
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                echo "unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    # require --model
    if [ -z "${models}" ]; then
        echo "error: --model is required"
        show_usage
        exit 1
    fi

    # require at least one action
    if [ "${do_export}" = false ] && [ "${do_test_native}" = false ] && \
       [ "${do_test_hmonnx}" = false ] && [ "${do_eval}" = false ]; then
        echo "error: at least one action is required (--export, --test-native, --test-hmonnx, --eval)"
        show_usage
        exit 1
    fi

    # create log dir
    mkdir -p "${LOG_DIR}"

    # set log file
    LOG_FILE="${LOG_DIR}/qwen3tts_pipeline_${TIMESTAMP}.log"

    log_info "=========================================="
    log_info "Qwen3-TTS pipeline starting"
    log_info "time: $(date)"
    log_info "models: ${models}"
    if [ -n "${HF_MODEL_DIR_OVERRIDE}" ]; then
        log_info "HF model dir override: ${HF_MODEL_DIR_OVERRIDE}"
    fi
    log_info "log file: ${LOG_FILE}"
    log_info "=========================================="

    # cd to script dir
    cd "${SCRIPT_DIR}"

    # process each model
    IFS=',' read -ra MODEL_ARRAY <<< "${models}"
    for model in "${MODEL_ARRAY[@]}"; do
        model=$(echo "${model}" | xargs)  # trim spaces

        case "${model}" in
            1_7B_voicedesign|1.7b)
                # export
                if [ "${do_export}" = true ]; then
                    export_1p7b
                fi

                # test native
                if [ "${do_test_native}" = true ]; then
                    test_native_1p7b
                fi

                # test hmonnx
                if [ "${do_test_hmonnx}" = true ]; then
                    test_hmonnx_1p7b
                fi

                if [ "${do_eval}" = true ]; then
                    log_info "Skipping --eval for 1_7B_voicedesign: eval/qwen3_tts_eval.py covers custom voice and base voice-clone."
                fi
                ;;

            1_7B_customvoice|1.7b-custom|1.7b-customvoice)
                # export
                if [ "${do_export}" = true ]; then
                    export_1p7b_customvoice
                fi

                # test native
                if [ "${do_test_native}" = true ]; then
                    test_native_1p7b_customvoice
                fi

                # test hmonnx
                if [ "${do_test_hmonnx}" = true ]; then
                    test_hmonnx_1p7b_customvoice
                fi

                if [ "${do_eval}" = true ]; then
                    run_eval 1_7B_customvoice
                fi
                ;;

            0_6B_customvoice|0.6b)
                # export
                if [ "${do_export}" = true ]; then
                    export_0p6b
                fi

                # test native
                if [ "${do_test_native}" = true ]; then
                    test_native_0p6b
                fi

                # test hmonnx
                if [ "${do_test_hmonnx}" = true ]; then
                    test_hmonnx_0p6b
                fi

                if [ "${do_eval}" = true ]; then
                    run_eval 0_6B_customvoice
                fi
                ;;

            0_6B_base|0.6b-base)
                # export
                if [ "${do_export}" = true ]; then
                    export_0p6b_base
                fi

                # test native
                if [ "${do_test_native}" = true ]; then
                    test_native_0p6b_base
                fi

                # test hmonnx
                if [ "${do_test_hmonnx}" = true ]; then
                    test_hmonnx_0p6b_base
                fi

                if [ "${do_eval}" = true ]; then
                    run_eval 0_6B_base
                fi
                ;;

            *)
                log_error "unknown model: ${model}; supported: 1_7B_voicedesign, 1_7B_customvoice, 0_6B_customvoice, 0_6B_base (aliases 1.7b / 1.7b-custom / 0.6b / 0.6b-base)"
                exit 1
                ;;
        esac
    done

    log_info "=========================================="
    log_success "all tasks complete!"
    log_info "log file: ${LOG_FILE}"
    log_info "=========================================="
}

# run main
main "$@"
