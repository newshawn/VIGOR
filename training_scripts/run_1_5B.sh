#!/bin/bash

source /home/wenxuexiang/projects/Intuitor/open-r1-intuitor/openr1_intuitor/bin/activate
which python
cd /home/wenxuexiang/projects/Intuitor/open-r1-intuitor
# unset http_proxy
# unset https_proxy
# clash on
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH}
export PATH=/usr/local/cuda-12.4/bin:${PATH}
export CUDA_HOME=/usr/local/cuda-12.4
export WANDB_API_KEY=4117ed9c927aaa675b1e5c34fe7aebf892ed2009
export WANDB_MODE=online
export ACCELERATE_LOG_LEVEL=info
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 设置中国时区
# export TZ='Asia/Shanghai'

# === 手动/环境可配置的续跑参数（默认关闭） ===
# export RESUME_MODE=false
export RESUME_MODE=true
export RESUME_TIMESTAMP=20251113_073223
export WANDB_RUN_ID=ihmq57k3

START_VLLM=true   # 是否使用单独的 vLLM 进程
NUM_PROCESSES=4   # 仅支持 4 或 8（会结合 START_VLLM 自动调整）
EXP_TYPE=intuitor
MAX_STEPS=-1
num_generations=7
SAVE_TOTAL_LIMIT=10

# GPU 分配策略（与此前 train 分支一致）
REQUESTED_PROCESSES=$NUM_PROCESSES
if [ "$NUM_PROCESSES" -eq 8 ]; then
    if [ "$START_VLLM" = true ]; then
        CUDA_DEVICES="1,2,3,4,5,6,7"
        num_generations=7
        NUM_PROCESSES=7  # 预留 GPU0 给 vLLM
    else
        CUDA_DEVICES="0,1,2,3,4,5,6,7"
        num_generations=8
    fi
    BATCH_SIZE=3
    GRAD_ACCUM=32
    lr=3.0e-06
elif [ "$NUM_PROCESSES" -eq 4 ]; then
    if [ "$START_VLLM" = true ]; then
        CUDA_DEVICES="1,2,3"
        num_generations=3
        NUM_PROCESSES=3  # 预留 GPU0 给 vLLM
    else
        CUDA_DEVICES="0,1,2,3"
    fi
    BATCH_SIZE=3
    GRAD_ACCUM=1
    lr=3.0e-06
else
    echo "[ERROR] 当前脚本仅支持 NUM_PROCESSES=4 或 8，实际为 ${NUM_PROCESSES}." >&2
    exit 1
fi

if [ "$START_VLLM" = true ] && [ "$REQUESTED_PROCESSES" -ne "$NUM_PROCESSES" ]; then
    echo "[INFO] vLLM 占用 1 张卡，训练进程数从 ${REQUESTED_PROCESSES} 调整为 ${NUM_PROCESSES}"
fi

# 根据实验类型设置脚本和配置
if [ "$EXP_TYPE" = "grpo" ]; then
    SCRIPT_PATH="src/open_r1/grpo.py"
    CONFIG_FILE="recipes/Qwen2.5-1.5B/grpo/config_demo.yaml"
    WANDB_PROJECT="open-r1-grpo_qwen2.5-1.5B"
    RUN_NAME="Qwen2.5-GRPO-1.5B"
    LOG_PREFIX="grpo"
else
    SCRIPT_PATH="src/open_r1/intuitor.py"
    CONFIG_FILE="recipes/Qwen2.5-1.5B/intuitor/config_demo.yaml"
    WANDB_PROJECT="open-r1-intuitor-qwen2.5-1.5B"
    RUN_NAME="Qwen2.5-Intuitor-1.5B"
    LOG_PREFIX="intuitor"
fi
BASE_OUTPUT_DIR="/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-1.5B"

# 创建日志/输出目录
if [ "$RESUME_MODE" = true ]; then
    : "${RESUME_TIMESTAMP:?RESUME_MODE=true 但 RESUME_TIMESTAMP 未设置}"

    PROJECT_DIR="${BASE_OUTPUT_DIR}_${RESUME_TIMESTAMP}"
    LOG_DIR="${PROJECT_DIR}/logs"
    OUTPUT_DIR="${PROJECT_DIR}/ckpt"
    echo "[RESUME] 自动推导 PROJECT_DIR=$PROJECT_DIR"
    echo "[RESUME] 自动推导 LOG_DIR=$LOG_DIR"
    echo "[RESUME] 自动推导 OUTPUT_DIR=$OUTPUT_DIR"
else
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    PROJECT_DIR="${BASE_OUTPUT_DIR}_${TIMESTAMP}"
    LOG_DIR="${PROJECT_DIR}/logs"
    OUTPUT_DIR="${PROJECT_DIR}/ckpt"
    RUN_NAME="${RUN_NAME}_${TIMESTAMP}"
fi
if [ -n "$PROJECT_DIR" ]; then
    mkdir -p "$PROJECT_DIR"
fi
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"
RUN_LOG_FILE="${LOG_DIR}/run_${LOG_PREFIX}.log"
VLLM_LOG_FILE="${LOG_DIR}/vllm-server.log"

# 显示当前配置参数
echo "===== 当前配置参数 ====="
echo "NUM_PROCESSES: $NUM_PROCESSES"
echo "EXP_TYPE: $EXP_TYPE"
echo "CUDA_DEVICES: $CUDA_DEVICES"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "GRAD_ACCUM: $GRAD_ACCUM"
echo "lr: $lr"
echo "SCRIPT_PATH: $SCRIPT_PATH"
echo "CONFIG_FILE: $CONFIG_FILE"
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo "RUN_NAME: $RUN_NAME"
echo "LOG_PREFIX: $LOG_PREFIX"
echo "LOG_DIR: $LOG_DIR"
echo "num_generations: $num_generations"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "PROJECT_DIR: $PROJECT_DIR"
echo "START_VLLM: $START_VLLM"
echo "SAVE_TOTAL_LIMIT: $SAVE_TOTAL_LIMIT"
echo "RESUME_MODE: $RESUME_MODE"
echo "WANDB_RUN_ID: $WANDB_RUN_ID"
echo "RESUME_TIMESTAMP: $RESUME_TIMESTAMP"
echo "========================"

# 启动GPU监控（每5秒记录一次）
nohup bash scripts/gpu_monitor.sh 5 "gpu_usage" "${LOG_DIR}" > /dev/null 2>&1 &
MONITOR_PID=$!
echo "GPU监控已启动 PID: $MONITOR_PID"

# 记录配置参数到日志文件
PARAM_LOG="${LOG_DIR}/config_params.log"
echo "NUM_PROCESSES: $NUM_PROCESSES" > "$PARAM_LOG"
echo "EXP_TYPE: $EXP_TYPE" >> "$PARAM_LOG"
echo "CUDA_DEVICES: $CUDA_DEVICES" >> "$PARAM_LOG"
echo "BATCH_SIZE: $BATCH_SIZE" >> "$PARAM_LOG"
echo "GRAD_ACCUM: $GRAD_ACCUM" >> "$PARAM_LOG"
echo "lr: $lr" >> "$PARAM_LOG"
echo "SCRIPT_PATH: $SCRIPT_PATH" >> "$PARAM_LOG"
echo "CONFIG_FILE: $CONFIG_FILE" >> "$PARAM_LOG"
echo "WANDB_PROJECT: $WANDB_PROJECT" >> "$PARAM_LOG"
echo "RUN_NAME: $RUN_NAME" >> "$PARAM_LOG"
echo "LOG_PREFIX: $LOG_PREFIX" >> "$PARAM_LOG"
echo "LOG_DIR: $LOG_DIR" >> "$PARAM_LOG"
echo "num_generations: $num_generations" >> "$PARAM_LOG"
echo "OUTPUT_DIR: $OUTPUT_DIR" >> "$PARAM_LOG"
echo "PROJECT_DIR: $PROJECT_DIR" >> "$PARAM_LOG"
echo "START_VLLM: $START_VLLM" >> "$PARAM_LOG"
echo "SAVE_TOTAL_LIMIT: $SAVE_TOTAL_LIMIT" >> "$PARAM_LOG"
echo "RESUME_MODE: $RESUME_MODE" >> "$PARAM_LOG"
echo "WANDB_RUN_ID: $WANDB_RUN_ID" >> "$PARAM_LOG"
echo "RESUME_TIMESTAMP: $RESUME_TIMESTAMP" >> "$PARAM_LOG"

# 启动 vLLM（可选）
if [ "$START_VLLM" = true ]; then
  if [ "$RESUME_MODE" = true ]; then
    nohup env CUDA_VISIBLE_DEVICES=0 \
        trl vllm-serve \
        --model /run/determined/NAS1/public/HuggingFace/Qwen/Qwen2.5-1.5B \
        >> "$VLLM_LOG_FILE" 2>&1 &
  else
    nohup env CUDA_VISIBLE_DEVICES=0 \
        trl vllm-serve \
        --model /run/determined/NAS1/public/HuggingFace/Qwen/Qwen2.5-1.5B \
        > "$VLLM_LOG_FILE" 2>&1 &
  fi
  VLLM_PID=$!
  echo "vLLM server started with PID: $VLLM_PID"
else
  echo "vLLM server skipped (START_VLLM=false)"
fi

# CLI 追加参数
EXTRA_ARGS=""
if [ "$START_VLLM" != true ]; then
  EXTRA_ARGS="--use_vllm false"
fi

# 启动训练脚本（线上 runs 均走 nohup 后台）
if [ "$RESUME_MODE" = true ]; then
  nohup env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES ACCELERATE_LOG_LEVEL=info \
      accelerate launch --config_file recipes/accelerate_configs/zero3.yaml --num_processes=$NUM_PROCESSES \
      $SCRIPT_PATH \
      --per_device_eval_batch_size $BATCH_SIZE --per_device_train_batch_size $BATCH_SIZE --gradient_accumulation_steps $GRAD_ACCUM --learning_rate $lr --max_steps $MAX_STEPS \
      --num_generations $num_generations --output_dir $OUTPUT_DIR \
      --config $CONFIG_FILE --wandb_project $WANDB_PROJECT --run_name $RUN_NAME --save_total_limit $SAVE_TOTAL_LIMIT $EXTRA_ARGS \
      >> "$RUN_LOG_FILE" 2>&1 &
else
  nohup env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES ACCELERATE_LOG_LEVEL=info \
      accelerate launch --config_file recipes/accelerate_configs/zero3.yaml --num_processes=$NUM_PROCESSES \
      $SCRIPT_PATH \
      --per_device_eval_batch_size $BATCH_SIZE --per_device_train_batch_size $BATCH_SIZE --gradient_accumulation_steps $GRAD_ACCUM --learning_rate $lr --max_steps $MAX_STEPS \
      --num_generations $num_generations --output_dir $OUTPUT_DIR \
      --config $CONFIG_FILE --wandb_project $WANDB_PROJECT --run_name $RUN_NAME --save_total_limit $SAVE_TOTAL_LIMIT $EXTRA_ARGS \
      > "$RUN_LOG_FILE" 2>&1 &
fi
TRAINING_PID=$!
wait $TRAINING_PID
echo "Training process started with PID: $TRAINING_PID"

echo "进程已启动。查看 ${LOG_DIR}/run_${LOG_PREFIX}.log 与 ${LOG_DIR}/vllm-server.log 了解详情。"

# 结束监控
kill $MONITOR_PID
echo "GPU监控已停止"
