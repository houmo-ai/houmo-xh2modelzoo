#!/bin/bash
# Qwen3-TTS 一键导出和评估流程脚本
# 支持 1.7B-VoiceDesign 和 0.6B-CustomVoice 模型的导出、测试和精度评估

set -e  # 遇到错误立即退出

# ============================================================================
# 配置区域
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHONPATH="/data01/home/she.gao/xh2modelzoo_new"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 默认测试文本
TEST_TEXT_1P7B="基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈。"
TEST_INSTRUCT_1P7B="体现温柔甜美的女声，音调适中，语速平稳。"
TEST_TEXT_0P6B="基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈。"
TEST_SPEAKER_0P6B="vivian"

# 精度评估配置（使用 README 默认值）
EVAL_MAX_SAMPLES=20
EVAL_GPUS="0,1,2,3"
EVAL_SPEAKER_MODE="round-robin"

# ============================================================================
# 辅助函数
# ============================================================================

# 打印带时间戳的日志
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

# 检查命令是否成功
check_status() {
    if [ $? -ne 0 ]; then
        log_error "$1 失败，停止执行"
        exit 1
    fi
}

# 显示使用帮助
show_usage() {
    cat << EOF
用法: $0 [选项]

选项:
    --model MODEL           指定模型 (1.7b, 0.6b, 或 1.7b,0.6b)
    --export                导出 HMONNX 模型
    --test-native           测试原始浮点模型
    --test-hmonnx           测试 HMONNX 模型
    --eval                  运行精度评估
    --eval-samples N        精度评估样本数 (默认: 20)
    --eval-gpus GPUS        精度评估使用的 GPU (默认: 0,1,2,3)
    --help                  显示此帮助信息

示例:
    # 导出 1.7B 模型
    $0 --model 1.7b --export

    # 测试 0.6B 浮点模型
    $0 --model 0.6b --test-native

    # 完整流程：导出 + 测试 + 评估
    $0 --model 1.7b --export --test-native --test-hmonnx --eval

    # 同时处理两个模型
    $0 --model 1.7b,0.6b --export --test-hmonnx

EOF
}

# ============================================================================
# 1.7B-VoiceDesign 导出函数
# ============================================================================

export_1p7b() {
    log_info "=========================================="
    log_info "开始导出 1.7B-VoiceDesign 模型"
    log_info "=========================================="

    # 1. Talker
    log_info "[1/4] 导出 Talker..."
    PYTHONPATH=${PYTHONPATH} python qwen3_tts_talker_xh2a_export.py \
        --config ./config/llm/qwen3_tts_12hz_1_7B_voicedesign_talker_2k_xh2a.py \
        >> "${LOG_FILE}" 2>&1
    check_status "Talker 导出"
    log_success "Talker 导出完成"

    # 2. CodePredictor
    log_info "[2/4] 导出 CodePredictor..."
    PYTHONPATH=${PYTHONPATH} python qwen3_tts_code_predictor_xh2a_export.py \
        --config ./config/llm/qwen3_tts_12hz_1_7B_voicedesign_code_predictor_2k_xh2a.py \
        >> "${LOG_FILE}" 2>&1
    check_status "CodePredictor 导出"
    log_success "CodePredictor 导出完成"

    # 3. TextProjection
    log_info "[3/4] 导出 TextProjection..."
    PYTHONPATH=${PYTHONPATH} python qwen3_tts_text_projection_xh2a_export.py \
        --config ./config/llm/qwen3_tts_12hz_1_7B_text_projection_xh2a.py \
        >> "${LOG_FILE}" 2>&1
    check_status "TextProjection 导出"
    log_success "TextProjection 导出完成"

    # 4. SpeechTokenizer
    log_info "[4/4] 导出 SpeechTokenizer..."
    PYTHONPATH=${PYTHONPATH} python qwen3_tts_speech_tokenizer_xh2a_export.py \
        --config ./config/llm/qwen3_tts_12hz_1_7B_speech_tokenizer_xh2a.py \
        >> "${LOG_FILE}" 2>&1
    check_status "SpeechTokenizer 导出"
    log_success "SpeechTokenizer 导出完成"

    log_success "1.7B-VoiceDesign 模型导出完成"
}

# ============================================================================
# 0.6B-CustomVoice 导出函数
# ============================================================================

export_0p6b() {
    log_info "=========================================="
    log_info "开始导出 0.6B-CustomVoice 模型"
    log_info "=========================================="

    # 1. Talker
    log_info "[1/4] 导出 Talker..."
    PYTHONPATH=${PYTHONPATH} python qwen3_tts_0p6b_cv_talker_xh2a_export.py \
        --config ./config/llm/qwen3_tts_12hz_0_6B_customvoice_talker_2k_xh2a.py \
        >> "${LOG_FILE}" 2>&1
    check_status "Talker 导出"
    log_success "Talker 导出完成"

    # 2. CodePredictor
    log_info "[2/4] 导出 CodePredictor..."
    PYTHONPATH=${PYTHONPATH} python qwen3_tts_0p6b_cv_code_predictor_xh2a_export.py \
        --config ./config/llm/qwen3_tts_12hz_0_6B_customvoice_code_predictor_2k_xh2a.py \
        >> "${LOG_FILE}" 2>&1
    check_status "CodePredictor 导出"
    log_success "CodePredictor 导出完成"

    # 3. TextProjection
    log_info "[3/4] 导出 TextProjection..."
    PYTHONPATH=${PYTHONPATH} python qwen3_tts_0p6b_cv_text_projection_xh2a_export.py \
        --config ./config/llm/qwen3_tts_12hz_0_6B_customvoice_text_projection_xh2a.py \
        >> "${LOG_FILE}" 2>&1
    check_status "TextProjection 导出"
    log_success "TextProjection 导出完成"

    # 4. SpeechTokenizer
    log_info "[4/4] 导出 SpeechTokenizer..."
    PYTHONPATH=${PYTHONPATH} python qwen3_tts_0p6b_cv_speech_tokenizer_xh2a_export.py \
        --config ./config/llm/qwen3_tts_12hz_0_6B_customvoice_speech_tokenizer_xh2a.py \
        >> "${LOG_FILE}" 2>&1
    check_status "SpeechTokenizer 导出"
    log_success "SpeechTokenizer 导出完成"

    log_success "0.6B-CustomVoice 模型导出完成"
}

# ============================================================================
# 测试原始浮点模型
# ============================================================================

test_native_1p7b() {
    log_info "=========================================="
    log_info "测试 1.7B-VoiceDesign 原始浮点模型"
    log_info "=========================================="

    local output_file="${SCRIPT_DIR}/test_native_1p7b_${TIMESTAMP}.wav"

    PYTHONPATH=${PYTHONPATH} python native_demo.py \
        --mode voice-design \
        --model ./data/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign/ \
        --text "${TEST_TEXT_1P7B}" \
        --instruct "${TEST_INSTRUCT_1P7B}" \
        --out "${output_file}" \
        >> "${LOG_FILE}" 2>&1
    check_status "1.7B Native 测试"

    if [ -f "${output_file}" ]; then
        log_success "1.7B Native 测试完成，输出: ${output_file}"
    else
        log_error "1.7B Native 测试失败，未生成音频文件"
        exit 1
    fi
}

test_native_0p6b() {
    log_info "=========================================="
    log_info "测试 0.6B-CustomVoice 原始浮点模型"
    log_info "=========================================="

    local output_file="${SCRIPT_DIR}/test_native_0p6b_${TIMESTAMP}.wav"

    PYTHONPATH=${PYTHONPATH} python native_demo.py \
        --mode custom-voice \
        --model ./data/models/Qwen3-TTS-12Hz-0.6B-CustomVoice/ \
        --text "${TEST_TEXT_0P6B}" \
        --speaker "${TEST_SPEAKER_0P6B}" \
        --out "${output_file}" \
        >> "${LOG_FILE}" 2>&1
    check_status "0.6B Native 测试"

    if [ -f "${output_file}" ]; then
        log_success "0.6B Native 测试完成，输出: ${output_file}"
    else
        log_error "0.6B Native 测试失败，未生成音频文件"
        exit 1
    fi
}

# ============================================================================
# 测试 HMONNX 模型
# ============================================================================

test_hmonnx_1p7b() {
    log_info "=========================================="
    log_info "测试 1.7B-VoiceDesign HMONNX 模型"
    log_info "=========================================="

    PYTHONPATH=${PYTHONPATH} python qwen3_tts_xh2a_demo.py \
        --config ./config/llm/qwen3_tts_12hz_1_7b_voicedesign_xh2a_hmonnx.py \
        >> "${LOG_FILE}" 2>&1
    check_status "1.7B HMONNX 测试"

    log_success "1.7B HMONNX 测试完成"
}

test_hmonnx_0p6b() {
    log_info "=========================================="
    log_info "测试 0.6B-CustomVoice HMONNX 模型"
    log_info "=========================================="

    PYTHONPATH=${PYTHONPATH} python qwen3_tts_0p6b_cv_xh2a_demo.py \
        --config ./config/llm/qwen3_tts_12hz_0_6B_customvoice_xh2a_hmonnx.py \
        >> "${LOG_FILE}" 2>&1
    check_status "0.6B HMONNX 测试"

    log_success "0.6B HMONNX 测试完成"
}

# ============================================================================
# 精度评估
# ============================================================================

run_eval() {
    log_info "=========================================="
    log_info "运行精度评估"
    log_info "配置: 样本数=${EVAL_MAX_SAMPLES}, GPU=${EVAL_GPUS}, Speaker模式=${EVAL_SPEAKER_MODE}"
    log_info "=========================================="

    # Native 模式评估
    log_info "运行 Native 模式精度评估..."
    PYTHONPATH=${PYTHONPATH} python eval/qwen3_tts_eval.py \
        --mode native \
        --gpus ${EVAL_GPUS} \
        --max-samples ${EVAL_MAX_SAMPLES} \
        --speaker-mode ${EVAL_SPEAKER_MODE} \
        >> "${LOG_FILE}" 2>&1
    check_status "Native 模式精度评估"
    log_success "Native 模式精度评估完成"

    # HMONNX 模式评估
    log_info "运行 HMONNX 模式精度评估..."
    PYTHONPATH=${PYTHONPATH} python eval/qwen3_tts_eval.py \
        --mode hmonnx \
        --gpus ${EVAL_GPUS} \
        --max-samples ${EVAL_MAX_SAMPLES} \
        --speaker-mode ${EVAL_SPEAKER_MODE} \
        >> "${LOG_FILE}" 2>&1
    check_status "HMONNX 模式精度评估"
    log_success "HMONNX 模式精度评估完成"

    log_success "精度评估完成，结果保存在 qwen3tts_eval_zh/ 目录"
}

# ============================================================================
# 主流程
# ============================================================================

main() {
    # 解析命令行参数
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
                echo "未知选项: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    # 检查是否指定了模型
    if [ -z "${models}" ]; then
        echo "错误: 必须指定 --model 参数"
        show_usage
        exit 1
    fi

    # 检查是否指定了至少一个操作
    if [ "${do_export}" = false ] && [ "${do_test_native}" = false ] && \
       [ "${do_test_hmonnx}" = false ] && [ "${do_eval}" = false ]; then
        echo "错误: 必须指定至少一个操作 (--export, --test-native, --test-hmonnx, --eval)"
        show_usage
        exit 1
    fi

    # 创建日志目录
    mkdir -p "${LOG_DIR}"

    # 设置日志文件
    LOG_FILE="${LOG_DIR}/qwen3tts_pipeline_${TIMESTAMP}.log"

    log_info "=========================================="
    log_info "Qwen3-TTS Pipeline 开始执行"
    log_info "时间: $(date)"
    log_info "模型: ${models}"
    log_info "日志文件: ${LOG_FILE}"
    log_info "=========================================="

    # 切换到脚本目录
    cd "${SCRIPT_DIR}"

    # 处理每个模型
    IFS=',' read -ra MODEL_ARRAY <<< "${models}"
    for model in "${MODEL_ARRAY[@]}"; do
        model=$(echo "${model}" | xargs)  # 去除空格

        case "${model}" in
            1.7b)
                # 导出
                if [ "${do_export}" = true ]; then
                    export_1p7b
                fi

                # 测试 Native
                if [ "${do_test_native}" = true ]; then
                    test_native_1p7b
                fi

                # 测试 HMONNX
                if [ "${do_test_hmonnx}" = true ]; then
                    test_hmonnx_1p7b
                fi
                ;;

            0.6b)
                # 导出
                if [ "${do_export}" = true ]; then
                    export_0p6b
                fi

                # 测试 Native
                if [ "${do_test_native}" = true ]; then
                    test_native_0p6b
                fi

                # 测试 HMONNX
                if [ "${do_test_hmonnx}" = true ]; then
                    test_hmonnx_0p6b
                fi
                ;;

            *)
                log_error "未知模型: ${model}，支持的模型: 1.7b, 0.6b"
                exit 1
                ;;
        esac
    done

    # 精度评估（只运行一次，不区分模型）
    if [ "${do_eval}" = true ]; then
        run_eval
    fi

    log_info "=========================================="
    log_success "所有任务执行完成！"
    log_info "日志文件: ${LOG_FILE}"
    log_info "=========================================="
}

# 执行主函数
main "$@"
