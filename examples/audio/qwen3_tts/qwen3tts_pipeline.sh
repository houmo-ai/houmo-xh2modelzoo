#!/bin/bash
# Qwen3-TTS one-click export and evaluation pipeline
# Supports export, test and accuracy eval for 1.7B-VoiceDesign / 0.6B-CustomVoice / 0.6B-Base

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
    --model MODEL           model(s): 1_7B_voicedesign, 0_6B_customvoice, 0_6B_base
                            (short aliases 1.7b / 0.6b / 0.6b-base also accepted);
                            comma-joined for multiple, e.g. 1_7B_voicedesign,0_6B_customvoice
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

    # test the 0.6B-CustomVoice float model
    $0 --model 0_6B_customvoice --test-native

    # full flow: export + test + eval
    $0 --model 1_7B_voicedesign --export --test-native --test-hmonnx --eval

    # export with golden dump (for hardware comparison)
    $0 --model 0_6B_customvoice --export --golden

    # process multiple models at once
    $0 --model 1_7B_voicedesign,0_6B_customvoice --export --test-hmonnx

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
        ${GOLDEN_FLAG} \
        >> "${LOG_FILE}" 2>&1
    check_status "SpeechTokenizer export"
    log_success "SpeechTokenizer export done"

    log_success "1.7B-VoiceDesign export complete"
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
    log_info "[1/4] export Talker..."
    PYTHONPATH=${PYTHONPATH} python qwen3_tts_talker_export.py \
        --config ./config/llm/qwen3_tts_12hz_talker_2k_xh2a.py \
        --variant 0_6B_base \
        --name qwen3_tts_12hz_0_6B_base_talker_2k_xh2a \
        ${GOLDEN_FLAG} \
        >> "${LOG_FILE}" 2>&1
    check_status "Talker export"
    log_success "Talker export done"

    # 2. CodePredictor
    log_info "[2/4] export CodePredictor..."
    PYTHONPATH=${PYTHONPATH} python qwen3_tts_code_predictor_export.py \
        --config ./config/llm/qwen3_tts_12hz_code_predictor_2k_xh2a.py \
        --variant 0_6B_base \
        --name qwen3_tts_12hz_0_6B_base_code_predictor_2k_xh2a \
        ${GOLDEN_FLAG} \
        >> "${LOG_FILE}" 2>&1
    check_status "CodePredictor export"
    log_success "CodePredictor export done"

    # 3. TextProjection
    log_info "[3/4] export TextProjection..."
    PYTHONPATH=${PYTHONPATH} python qwen3_tts_text_projection_export.py \
        --config ./config/llm/qwen3_tts_12hz_text_projection_xh2a.py \
        --variant 0_6B_base \
        --name qwen3_tts_12hz_0_6B_base_text_projection_xh2a \
        ${GOLDEN_FLAG} \
        >> "${LOG_FILE}" 2>&1
    check_status "TextProjection export"
    log_success "TextProjection export done"

    # 4. SpeechTokenizer
    log_info "[4/4] export SpeechTokenizer..."
    PYTHONPATH=${PYTHONPATH} python qwen3_tts_speech_tokenizer_export.py \
        --config ./config/llm/qwen3_tts_12hz_speech_tokenizer_xh2a.py \
        --variant 0_6B_base \
        --name qwen3_tts_12hz_0_6B_base_speech_tokenizer_xh2a \
        ${GOLDEN_FLAG} \
        >> "${LOG_FILE}" 2>&1
    check_status "SpeechTokenizer export"
    log_success "SpeechTokenizer export done"

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
        --model ./data/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign/ \
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

test_native_0p6b() {
    log_info "=========================================="
    log_info "Testing 0.6B-CustomVoice float model"
    log_info "=========================================="

    local output_file="${SCRIPT_DIR}/test_native_0p6b_${TIMESTAMP}.wav"

    PYTHONPATH=${PYTHONPATH} python native_demo.py \
        --mode custom-voice \
        --model ./data/models/Qwen3-TTS-12Hz-0.6B-CustomVoice/ \
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
        --model ./data/models/Qwen3-TTS-12Hz-0.6B-Base/ \
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
    log_info "=========================================="
    log_info "Running accuracy evaluation"
    log_info "config: samples=${EVAL_MAX_SAMPLES}, GPUs=${EVAL_GPUS}, speaker-mode=${EVAL_SPEAKER_MODE}"
    log_info "=========================================="

    # native mode eval
    log_info "Running native-mode accuracy eval..."
    PYTHONPATH=${PYTHONPATH} python eval/qwen3_tts_eval.py \
        --mode native \
        --gpus ${EVAL_GPUS} \
        --max-samples ${EVAL_MAX_SAMPLES} \
        --speaker-mode ${EVAL_SPEAKER_MODE} \
        >> "${LOG_FILE}" 2>&1
    check_status "native-mode eval"
    log_success "native-mode eval done"

    # hmonnx mode eval
    log_info "Running HMONNX-mode accuracy eval..."
    PYTHONPATH=${PYTHONPATH} python eval/qwen3_tts_eval.py \
        --mode hmonnx \
        --gpus ${EVAL_GPUS} \
        --max-samples ${EVAL_MAX_SAMPLES} \
        --speaker-mode ${EVAL_SPEAKER_MODE} \
        >> "${LOG_FILE}" 2>&1
    check_status "HMONNX-mode eval"
    log_success "HMONNX-mode eval done"

    log_success "accuracy eval done, results in qwen3tts_eval_zh/"
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
                ;;

            *)
                log_error "unknown model: ${model}; supported: 1_7B_voicedesign, 0_6B_customvoice, 0_6B_base (aliases 1.7b / 0.6b / 0.6b-base)"
                exit 1
                ;;
        esac
    done

    # Accuracy evaluation (run once, model-agnostic)
    if [ "${do_eval}" = true ]; then
        run_eval
    fi

    log_info "=========================================="
    log_success "all tasks complete!"
    log_info "log file: ${LOG_FILE}"
    log_info "=========================================="
}

# run main
main "$@"
