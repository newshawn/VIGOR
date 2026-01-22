#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
source "$VENV_DIR/bin/activate"
which python
cd "$REPO_ROOT"
# unset http_proxy
# unset https_proxy
# clash on
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH}
export PATH=/usr/local/cuda-12.4/bin:${PATH}
export CUDA_HOME=/usr/local/cuda-12.4
export WANDB_API_KEY=4117ed9c927aaa675b1e5c34fe7aebf892ed2009
export WANDB_MODE=offline
export WANDB_DISABLED=True
export http_proxy=http://10.130.130.5:7891
export https_proxy=http://10.130.130.5:7891
export no_proxy="127.0.0.1,localhost,0.0.0.0,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,.local"
export NO_PROXY="$no_proxy"
export ACCELERATE_LOG_LEVEL=info
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 设置中国时区
# export TZ='Asia/Shanghai'

# Debug 配置（沿用原脚本的 debug 分支）
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -r -a __DEV_ARR <<< "$CUDA_DEVICES"
NUM_PROCESSES=${#__DEV_ARR[@]}
if [ "$NUM_PROCESSES" -le 0 ]; then
    NUM_PROCESSES=1
fi
BATCH_SIZE=3
GRAD_ACCUM=1
lr=3.0e-06
num_generations=3
MAX_STEPS=${MAX_STEPS:--1}
if [ "$MAX_STEPS" = "-1" ]; then
    MAX_STEPS=50
fi
START_VLLM=false
EXP_TYPE=intuitor
ACCELERATE_CONFIG="recipes/accelerate_configs/zero3.yaml"
DEBUG_TTY_MODE=true  # 设为 true 可让日志同步输出到终端，便于 pdb 交互,false则是打印到run_${LOG_PREFIX}.log

# 根据实验类型设置脚本和配置
if [ "$EXP_TYPE" = "grpo" ]; then
    SCRIPT_PATH="src/open_r1/grpo.py"
    CONFIG_FILE="recipes/Qwen2.5-1.5B/grpo/config_demo.yaml"
    WANDB_PROJECT="open-r1-grpo_qwen2.5-1.5B_debug"
    RUN_NAME="Qwen2.5-GRPO-1.5B-debug"
    LOG_PREFIX="grpo"
else
    SCRIPT_PATH="src/open_r1/intuitor.py"
    CONFIG_FILE="recipes/Qwen2.5-1.5B/intuitor/config_demo.yaml"
    WANDB_PROJECT="open-r1-intuitor_qwen2.5-1.5B_debug"
    RUN_NAME="Qwen2.5-Intuitor-1.5B-debug"
    LOG_PREFIX="intuitor"
fi
BASE_OUTPUT_DIR="/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-1.5B"
LOG_DIR="logs/${LOG_PREFIX}_${NUM_PROCESSES}_1.5B/debug"
OUTPUT_DIR="${BASE_OUTPUT_DIR}_debug"
RUN_LOG_FILE="${LOG_DIR}/run_${LOG_PREFIX}.log"
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

# 显示当前配置参数
echo "===== 当前配置参数（DEBUG） ====="
echo "NUM_PROCESSES: $NUM_PROCESSES"
echo "CUDA_DEVICES: $CUDA_DEVICES"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "GRAD_ACCUM: $GRAD_ACCUM"
echo "lr: $lr"
echo "SCRIPT_PATH: $SCRIPT_PATH"
echo "CONFIG_FILE: $CONFIG_FILE"
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo "RUN_NAME: $RUN_NAME"
echo "LOG_DIR: $LOG_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "MAX_STEPS: $MAX_STEPS"
echo "=============================="

PARAM_LOG="${LOG_DIR}/config_params.log"
echo "DEBUG MODE" > "$PARAM_LOG"
echo "NUM_PROCESSES: $NUM_PROCESSES" >> "$PARAM_LOG"
echo "CUDA_DEVICES: $CUDA_DEVICES" >> "$PARAM_LOG"
echo "BATCH_SIZE: $BATCH_SIZE" >> "$PARAM_LOG"
echo "GRAD_ACCUM: $GRAD_ACCUM" >> "$PARAM_LOG"
echo "lr: $lr" >> "$PARAM_LOG"
echo "SCRIPT_PATH: $SCRIPT_PATH" >> "$PARAM_LOG"
echo "CONFIG_FILE: $CONFIG_FILE" >> "$PARAM_LOG"
echo "WANDB_PROJECT: $WANDB_PROJECT" >> "$PARAM_LOG"
echo "RUN_NAME: $RUN_NAME" >> "$PARAM_LOG"
echo "LOG_DIR: $LOG_DIR" >> "$PARAM_LOG"
echo "OUTPUT_DIR: $OUTPUT_DIR" >> "$PARAM_LOG"
echo "MAX_STEPS: $MAX_STEPS" >> "$PARAM_LOG"

EXTRA_ARGS="--max_completion_length 256 --report_to none --use_vllm false"

echo "[DEBUG] 通过 accelerate launch 前台启动，便于多卡调试"
echo "Command: CUDA_VISIBLE_DEVICES=$CUDA_DEVICES accelerate launch --config_file $ACCELERATE_CONFIG --num_processes $NUM_PROCESSES $SCRIPT_PATH ..."

RUN_CMD=(
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
  accelerate launch --config_file "$ACCELERATE_CONFIG" --num_processes "$NUM_PROCESSES"
  "$SCRIPT_PATH"
    --per_device_eval_batch_size "$BATCH_SIZE"
    --per_device_train_batch_size "$BATCH_SIZE"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --learning_rate "$lr"
    --max_steps "$MAX_STEPS"
    --num_generations "$num_generations"
    --output_dir "$OUTPUT_DIR"
    --config "$CONFIG_FILE"
    --wandb_project "$WANDB_PROJECT"
    --run_name "$RUN_NAME"
    $EXTRA_ARGS
)

if [ "$DEBUG_TTY_MODE" = true ]; then
  echo "[DEBUG] DEBUG_TTY_MODE=true，日志会同时输出到终端和 $RUN_LOG_FILE，便于 pdb 交互。"
  "${RUN_CMD[@]}" 2>&1 | tee "$RUN_LOG_FILE"
else
  "${RUN_CMD[@]}" > "$RUN_LOG_FILE" 2>&1
fi

echo "[DEBUG] 日志输出至: $RUN_LOG_FILE"
