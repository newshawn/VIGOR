#!/bin/bash
set -euo pipefail

export NCCL_P2P_DISABLE=1
source /home/wenxuexiang/projects/Intuitor/open-r1-intuitor/openr1_intuitor/bin/activate
which python
cd /home/wenxuexiang/projects/Intuitor/open-r1-intuitor
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH}
export PATH=/usr/local/cuda-12.4/bin:${PATH}
export CUDA_HOME=/usr/local/cuda-12.4
export WANDB_BASE_URL=https://api.bandw.top
export WANDB_API_KEY=4117ed9c927aaa675b1e5c34fe7aebf892ed2009
export WANDB_MODE=online
export ACCELERATE_LOG_LEVEL=info
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${RESUME_MODE:=false}" # : 是空操作；:= 表示“若变量未定义或为空，就赋默认值”。所以这行意思是：如果外面没传 RESUME_MODE，就把它设为 false
: "${RESUME_TIMESTAMP:=}"
: "${WANDB_RUN_ID:=}"

num_generations=2
EXP_TYPE=intuitor
MAX_STEPS=-1
START_VLLM=false  # 调试脚本默认关闭 vLLM
SAVE_TOTAL_LIMIT=16

VLLM_DEVICE=""
CUDA_DEVICES="0,1"
NUM_PROCESSES=2
BATCH_SIZE=1
GRAD_ACCUM=2
lr=3.0e-06

if [ "$EXP_TYPE" = "grpo" ]; then
    SCRIPT_PATH="src/open_r1/grpo.py"
    CONFIG_FILE="recipes/Qwen2.5-3B/grpo/config_demo.yaml"
    WANDB_PROJECT="open-r1-grpo"
    RUN_NAME="Qwen2.5-GRPO-3B"
    LOG_PREFIX="grpo"
    BASE_OUTPUT_DIR="/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-GRPO-3B"
else
    SCRIPT_PATH="src/open_r1/intuitor.py"
    CONFIG_FILE="recipes/Qwen2.5-3B/intuitor/config_demo_lora.yaml"
    WANDB_PROJECT="open-r1-intuitor-debug"
    RUN_NAME="Qwen2.5-Intuitor-3B"
    LOG_PREFIX="intuitor"
    BASE_OUTPUT_DIR="/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B"
fi

if [ "$RESUME_MODE" = true ]; then
    : "${RESUME_TIMESTAMP:?RESUME_MODE=true 但 RESUME_TIMESTAMP 未设置}"
    PROJECT_DIR="${BASE_OUTPUT_DIR}_${RESUME_TIMESTAMP}"
    LOG_DIR="${PROJECT_DIR}/logs"
    OUTPUT_DIR="${PROJECT_DIR}/ckpt"
    echo "[RESUME] PROJECT_DIR=$PROJECT_DIR"
    echo "[RESUME] LOG_DIR=$LOG_DIR"
    echo "[RESUME] OUTPUT_DIR=$OUTPUT_DIR"
else
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    PROJECT_DIR="${BASE_OUTPUT_DIR}_${TIMESTAMP}"
    LOG_DIR="${PROJECT_DIR}/logs"
    OUTPUT_DIR="${PROJECT_DIR}/ckpt"
    RUN_NAME="${RUN_NAME}_${TIMESTAMP}"
fi
mkdir -p "$PROJECT_DIR" "$LOG_DIR" "$OUTPUT_DIR"   # 创建项目根目录及日志/ckpt 子目录
RUN_LOG_FILE="${LOG_DIR}/run_${LOG_PREFIX}.log"

# 汇总当前 run 的关键参数，既打印也写入日志文件
read -r -d '' RUN_CONFIG <<EOF || true
===== 当前配置参数 =====
NUM_PROCESSES: $NUM_PROCESSES
EXP_TYPE: $EXP_TYPE
CUDA_DEVICES: $CUDA_DEVICES
BATCH_SIZE: $BATCH_SIZE
GRAD_ACCUM: $GRAD_ACCUM
lr: $lr
SCRIPT_PATH: $SCRIPT_PATH
CONFIG_FILE: $CONFIG_FILE
WANDB_PROJECT: $WANDB_PROJECT
RUN_NAME: $RUN_NAME
LOG_PREFIX: $LOG_PREFIX
LOG_DIR: $LOG_DIR
num_generations: $num_generations
OUTPUT_DIR: $OUTPUT_DIR
PROJECT_DIR: $PROJECT_DIR
START_VLLM: $START_VLLM
SAVE_TOTAL_LIMIT: $SAVE_TOTAL_LIMIT
RESUME_MODE: $RESUME_MODE
WANDB_RUN_ID: $WANDB_RUN_ID
RESUME_TIMESTAMP: $RESUME_TIMESTAMP
========================
EOF
printf "%s\n" "$RUN_CONFIG"

# 调试模式也保持 GPU 监控，便于现场查看显存/利用率
nohup bash scripts/gpu_monitor.sh 5 "gpu_usage" "${LOG_DIR}" > /dev/null 2>&1 &
MONITOR_PID=$!
trap 'kill $MONITOR_PID >/dev/null 2>&1 || true' EXIT

echo "GPU监控已启动 PID: $MONITOR_PID"

PARAM_LOG="${LOG_DIR}/config_params.log"
printf "%s\n" "$RUN_CONFIG" > "$PARAM_LOG"

EXTRA_ARGS="--use_vllm false"
TRAIN_ENV=(env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" ACCELERATE_LOG_LEVEL=info)
# 将训练命令拆成数组，方便后续通过 tee 同时写日志和输出到前台
TRAIN_CMD=(
  accelerate launch --config_file recipes/accelerate_configs/zero3.yaml --num_processes="$NUM_PROCESSES"
  "$SCRIPT_PATH"
  --per_device_eval_batch_size "$BATCH_SIZE"
  --per_device_train_batch_size "$BATCH_SIZE"
  --gradient_accumulation_steps "$GRAD_ACCUM"
  --learning_rate "$lr"
  --max_steps "$MAX_STEPS"
  --max_completion_length 256
  --num_generations "$num_generations"
  --output_dir "$OUTPUT_DIR"
  --config "$CONFIG_FILE"
  --wandb_project "$WANDB_PROJECT"
  --run_name "$RUN_NAME"
  --save_total_limit "$SAVE_TOTAL_LIMIT"
  $EXTRA_ARGS
)

echo ">>> 调试模式：日志实时输出并写入 $RUN_LOG_FILE"
# 使用 tee 保留后台日志，同时让 stdout/stderr 直接展示在终端
if [ "$RESUME_MODE" = true ]; then
  ("${TRAIN_ENV[@]}" "${TRAIN_CMD[@]}") 2>&1 | tee -a "$RUN_LOG_FILE"
else
  ("${TRAIN_ENV[@]}" "${TRAIN_CMD[@]}") 2>&1 | tee "$RUN_LOG_FILE"
fi

echo "Training finished. Logs: $RUN_LOG_FILE"

echo "停止 GPU 监控..."
kill "$MONITOR_PID"
trap - EXIT
echo "GPU监控已停止"
