#!/bin/bash

# ==============================================================================
# YOLOv7 自动量化导出与COCO并行评测脚本 (最终优化版 - 修复进度识别及mkdir问题)
# 功能:
#   1. 确定项目根目录。
#   2. 并行调用 xhquant 脚本对预存的 ONNX 模型进行量化导出 (导出到项目根目录的 work_dirs/yolo7)。
#      - 在导出前检查目标 HMONNX 文件是否存在，若存在则跳过。
#   3. 启动 YOLOv7 量化模型的 COCO 并行评测任务，使用指定 GPU (5, 6, 7)。
#   4. 在终端实时显示每个并行评估任务的完整整体 tqdm 进度条，避免刷屏和残余。
#   5. 将所有测试结果保存到 'quantization_coco_evaluation_results' 文件夹 (位于脚本同级目录)。
#   6. 脚本中断时自动终止所有后台子进程，并干净恢复终端。
#   7. 等待所有任务完成，然后统一报告结果。
# ==============================================================================

# --- 关键修复: 确保脚本所在目录的路径正确，不在此处切换工作目录 ---
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# --- 确定项目根目录 ---
PROJECT_ROOT=""
CURRENT_DIR=$(pwd) # 从当前执行脚本的目录开始向上查找
while [[ "$CURRENT_DIR" != "/" ]]; do
    if [[ -d "$CURRENT_DIR/.git" ]]; then
        PROJECT_ROOT="$CURRENT_DIR"
        break
    fi
    if [[ -d "$CURRENT_DIR/examples/cv/yolo" ]]; then
        PROJECT_ROOT="$CURRENT_DIR"
        break
    fi
    CURRENT_DIR=$(dirname "$CURRENT_DIR")
done

if [[ -z "$PROJECT_ROOT" ]]; then
    echo "错误: 无法找到项目根目录 (未找到 .git 目录或顶层 examples/cv/yolo 目录)。"
    exit 1
fi
echo "项目根目录: $PROJECT_ROOT"

# --- 配置 ---
MODELS=("yolov7") # 仅处理 YOLOv7
QUANT_TYPES=("w8a8-sefp" "w8a16-sefp" "w4a8-ssfp")

# 定义原始 ONNX 模型路径
YOLO_ONNX_MODEL_INPUT_PATH="/data01/home/tianjian.yang/xh2_model_zoo/data/models/yolo/yolov7.onnx"

# 定义量化后 HMONNX 模型的输出根目录 (由 export.py 硬编码决定，位于 PROJECT_ROOT 下)
HMONNX_OUTPUT_BASE_DIR="$PROJECT_ROOT/work_dirs"

# 定义所有测试结果（日志和JSON）的统一保存目录 (相对于当前脚本目录)
EVAL_RESULTS_DIR="quantization_coco_evaluation_results"
EVAL_RESULTS_FULL_PATH="$SCRIPT_DIR/$EVAL_RESULTS_DIR"
mkdir -p "$EVAL_RESULTS_FULL_PATH"

# 指定要使用的GPU列表
GPUS_TO_USE=(5 6 7)
NUM_GPUS=${#GPUS_TO_USE[@]}
TASK_COUNT=0 # 用于 GPU 轮询

# --- 存储所有后台进程的PID (用于全局清理) ---
ALL_BACKGROUND_PIDS=()

# --- 存储评估任务信息 (PID;LOG_FILE;TASK_NAME) ---
EVAL_TASK_INFO=()

# --- 清理函数: 终止所有后台进程并恢复终端 ---
cleanup() {
    # 确保光标可见并移动到新行
    tput cnorm # 恢复光标可见
    printf "\r\n" # 确保输出在新的行开始，避免覆盖

    # 清理进度条区域，确保终端干净
    local num_monitored_tasks=${#EVAL_TASK_INFO[@]}
    if [ "$num_monitored_tasks" -gt 0 ]; then
        tput cuu "$num_monitored_tasks" 2>/dev/null # 尝试上移光标
        for ((i=0; i<num_monitored_tasks; i++)); do
            tput el 2>/dev/null # 清除当前行
            tput cud1 2>/dev/null # 下移一行 (如果不是最后一行)
        done
        tput cuu "$num_monitored_tasks" 2>/dev/null # 移回起始位置，再次确保清除
        tput el 2>/dev/null # 清除第一行
    fi

    echo "接收到中断信号 (Ctrl+C) 或脚本退出。正在终止所有后台进程..."
    for pid in "${ALL_BACKGROUND_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then # 检查进程是否存在
            echo "终止进程: $pid"
            kill "$pid" 2>/dev/null # 发送 SIGTERM
            sleep 0.1 # 给一点时间让进程响应
            if kill -0 "$pid" 2>/dev/null; then # 再次检查是否终止
                echo "进程 $pid 未响应 SIGTERM, 发送 SIGKILL."
                kill -9 "$pid" 2>/dev/null # 强制终止
            fi
        fi
    done
    echo "清理完成。"
    exit 1 # 退出脚本，因为是中断信号
}

# --- 设置 trap: 捕获 SIGINT (Ctrl+C) 和 EXIT (脚本退出) 信号 ---
trap cleanup SIGINT EXIT

# --- 函数: 从日志文件中获取整体测试的完整 tqdm 进度行 ---
# 最终逻辑: 查找包含 "/5000" 的行，并验证它不包含来自内部编译/加载的进度条特征 (如 "/214")
get_overall_tqdm_line() {
    local log_file=$1
    local default_tqdm_line="--- Waiting for progress ---"

    if [ ! -f "$log_file" ]; then
        echo "$default_tqdm_line"
        return
    fi

    # 1. 过滤出包含 "/5000" 的行（总进度标志），并取最后一行
    local target_line=$(grep "/5000" "$log_file" 2>/dev/null | tail -n 1)

    # 2. 如果没找到行，直接返回
    if [ -z "$target_line" ]; then
        echo "$default_tqdm_line"
        return
    fi
    
    # 3. 验证行是否“干净”：如果行内同时包含其他进度条的特征 (根据日志是 /214)，则判定为“脏”数据并忽略
    #   ======= 关键修复: 将yolo6的/284修改为yolo7日志中观察到的/214 =======
    if echo "$target_line" | grep -q "/214"; then
        echo "$default_tqdm_line"
        return
    fi

    # 4. 如果是“干净”行，则进行格式化并返回
    # 4a. 移除可能存在的 "could not decide..." 后缀
    local progress_part=$(echo "$target_line" | sed 's/could not decide.*//')
    # 4b. 移除所有回车符 `\r`，防止覆盖终端前缀
    local clean_progress=$(echo "$progress_part" | tr -d '\r')
    
    echo "$clean_progress"
}


echo "开始YOLOv7自动量化导出与并行评测流程..."
echo "==========================================================="

# --- 第1部分: 并行调用 xhquant 脚本导出所有 YOLOv7 量化模型 ---
echo "--- 第1部分: 正在并行对 yolov7.onnx 进行量化导出 ---"

# 检查原始 ONNX 输入模型是否存在
if [ ! -f "$YOLO_ONNX_MODEL_INPUT_PATH" ]; then
    echo "错误: 原始 ONNX 模型文件未找到: $YOLO_ONNX_MODEL_INPUT_PATH。"
    echo "脚本将终止。"
    exit 1
fi

EXPORT_PIDS_FOR_WAIT=() # 存储导出进程的PID，用于内部wait
ALL_EXPORTS_SKIPPED=true # 标记是否所有导出任务都被跳过

# 在这里临时切换工作目录到 PROJECT_ROOT，并在这个子shell中执行所有导出任务
(
    cd "$PROJECT_ROOT" || { echo "错误: 无法切换到项目根目录！脚本将终止。"; exit 1; }
    
    for model_prefix in "${MODELS[@]}"; do
        MODEL_PATH_NAME_LOOP="yolov7" # 匹配 work_dirs/ 结构中的目录名
        for qtype in "${QUANT_TYPES[@]}"; do
            # HMONNX 模型的预期输出路径
            CURRENT_HMONNX_FILE="$HMONNX_OUTPUT_BASE_DIR/$MODEL_PATH_NAME_LOOP/hmonnx/${MODEL_PATH_NAME_LOOP}_${qtype}_XH2a.onnx"
            
            # 检查当前量化模型是否已存在
            if [ -f "$CURRENT_HMONNX_FILE" ]; then
                echo "量化模型 $MODEL_PATH_NAME_LOOP ($qtype) 已存在 ($CURRENT_HMONNX_FILE)，跳过导出。"
                continue # 跳过当前 qtype 的导出任务
            fi

            ALL_EXPORTS_SKIPPED=false # 至少有一个导出任务需要运行
            echo "启动导出任务: $model_prefix ($qtype) ..."
            mkdir -p "$(dirname "$CURRENT_HMONNX_FILE")"

            python "/data01/home/tianjian.yang/xh2_model_zoo/examples/cv/yolo/yolov7/yolov7_export.py" \
                --quant-type "$qtype" \
                --onnx "$YOLO_ONNX_MODEL_INPUT_PATH" \
                &> "/dev/null" & # 将所有输出重定向到 /dev/null
            
            pid=$!
            EXPORT_PIDS_FOR_WAIT+=("$pid")
            ALL_BACKGROUND_PIDS+=("$pid")
        done
    done

    if $ALL_EXPORTS_SKIPPED; then
        echo "所有必需的量化 HMONNX 模型都已存在，无需执行新的导出任务。"
    else
        echo "所有导出任务已启动，正在等待它们完成..."
        for pid in "${EXPORT_PIDS_FOR_WAIT[@]}"; do
            wait "$pid"
            if [ $? -ne 0 ]; then
                echo "警告: PID $pid 的导出任务失败。请检查日志。"
            fi
        done
        echo "所有导出任务已完成。"
    fi

) # 子shell结束

echo "--- 第1部分完成: 所有YOLOv7量化模型已处理。 ---"
echo ""

# --- 第2部分: 并行启动所有YOLOv7 COCO评测任务 ---
echo "--- 第2部分: 正在并行启动所有YOLOv7 COCO评测任务 ---"

# 启动评估任务
for model_prefix in "${MODELS[@]}"; do
    MODEL_PATH_NAME="yolov7"
    EVAL_SCRIPT="/data01/home/tianjian.yang/xh2_model_zoo/yolo6_yolo7_test/evaluate_yolov7_on_coco.py"

    for qtype in "${QUANT_TYPES[@]}"; do
        GPU_ID=${GPUS_TO_USE[$((TASK_COUNT % NUM_GPUS))]}
        HMONNX_FILE="$HMONNX_OUTPUT_BASE_DIR/$MODEL_PATH_NAME/hmonnx/${MODEL_PATH_NAME}_${qtype}_XH2a.onnx"
        LOG_FILE="$EVAL_RESULTS_FULL_PATH/${model_prefix}_${qtype}_eval.log"
        DEST_JSON_FILE="$EVAL_RESULTS_FULL_PATH/${model_prefix}_${qtype}_coco_results.json"

        echo "启动评测任务: $model_prefix ($qtype) 在 GPU $GPU_ID 上"
        
        if [ ! -f "$HMONNX_FILE" ]; then
            echo "   -> 错误: 量化 HMONNX 模型未找到，已跳过。路径: $HMONNX_FILE"
            TASK_COUNT=$((TASK_COUNT + 1))
            continue
        fi

        CUDA_VISIBLE_DEVICES=$GPU_ID python "$EVAL_SCRIPT" \
            --hmonnx "$HMONNX_FILE" \
            --batch-size 1 \
            --output-json "$DEST_JSON_FILE" \
            > "$LOG_FILE" 2>&1 &

        pid=$!
        ALL_BACKGROUND_PIDS+=("$pid")
        EVAL_TASK_INFO+=("$pid;$LOG_FILE;${model_prefix} (${qtype})")
        TASK_COUNT=$((TASK_COUNT + 1))
    done
done

# --- 实时进度监控 (第3部分) ---
echo ""
echo "-----------------------------------------------------------"
echo "正在实时监控COCO评测进度 (按 Ctrl+C 终止并清理)..."
echo "-----------------------------------------------------------"

tput civis
NUM_MONITORED_TASKS=${#EVAL_TASK_INFO[@]}

if [ "$NUM_MONITORED_TASKS" -eq 0 ]; then
    echo "没有评估任务需要监控。"
else
    for ((i=0; i<NUM_MONITORED_TASKS; i++)); do echo ""; done
    tput cuu "$NUM_MONITORED_TASKS" 2>/dev/null 

    MONITOR_INTERVAL=2 # 缩短刷新间隔以获得更及时的更新

    while true; do
        all_eval_tasks_done=true
        tput cuu "$NUM_MONITORED_TASKS" 2>/dev/null 

        temp_task_info=()
        for item in "${EVAL_TASK_INFO[@]}"; do
            IFS=';' read -r pid log_file task_name <<< "$item"
            
            if kill -0 "$pid" 2>/dev/null; then
                all_eval_tasks_done=false
                current_tqdm_line=$(get_overall_tqdm_line "$log_file")
                printf "\r[%-25s] %s" "$task_name" "$current_tqdm_line"
                tput el 2>/dev/null
                temp_task_info+=("$item")
            else
                if [ -f "$log_file" ] && grep -q -i "error\|traceback" "$log_file"; then
                    printf "\r[%-25s] 状态: 失败 (查看 %s)" "$task_name" "$(basename "$log_file")"
                else
                    printf "\r[%-25s] 状态: 完成" "$task_name"
                fi
                tput el 2>/dev/null
            fi
            echo
        done
        EVAL_TASK_INFO=("${temp_task_info[@]}")
        NUM_MONITORED_TASKS=${#EVAL_TASK_INFO[@]}

        if $all_eval_tasks_done; then break; fi
        sleep "$MONITOR_INTERVAL"
    done
fi

tput cnorm

echo ""
echo "-----------------------------------------------------------"
echo "所有COCO评测任务已完成！"
echo "-----------------------------------------------------------"
echo ""

# --- 第4部分: 汇总并报告结果 ---
echo "--- 第4部分: 最终YOLOv7评测结果摘要 ---"
for log_file in "$EVAL_RESULTS_FULL_PATH"/*_eval.log; do
    if [ ! -f "$log_file" ]; then continue; fi
    echo "-----------------------------------------------------------"
    filename=$(basename "$log_file")
    echo "结果: ${filename%_eval.log}"

    json_file="${log_file%_eval.log}_coco_results.json"
    if [ -f "$json_file" ] && ! grep -q -i "error\|traceback" "$log_file"; then
        echo "   - 状态: 成功"
        echo "   - mAP 摘要:"
        tail -n 12 "$log_file" | grep 'Average Precision'
    else
        echo "   - 状态: 失败"
        echo "   - 错误日志摘要 (最后10行):"
        tail -n 10 "$log_file"
    fi
done
echo "==========================================================="
echo "所有流程已完成！"

# --- 移除 EXIT 信号的 trap，确保脚本自然退出时不重复调用 cleanup ---
trap - EXIT