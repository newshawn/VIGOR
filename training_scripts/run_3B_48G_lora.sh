#!/bin/bash
export NCCL_P2P_DISABLE=1
source /home/wenxuexiang/projects/Intuitor/open-r1-intuitor/openr1_intuitor/bin/activate
which python
cd /home/wenxuexiang/projects/Intuitor/open-r1-intuitor
# unset http_proxy
# unset https_proxy
# clash on
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH}
export PATH=/usr/local/cuda-12.4/bin:${PATH} 
export CUDA_HOME=/usr/local/cuda-12.4
export WANDB_BASE_URL=https://api.bandw.top
export WANDB_API_KEY=4117ed9c927aaa675b1e5c34fe7aebf892ed2009
export WANDB_MODE=online
export ACCELERATE_LOG_LEVEL=info
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 设置中国时区
# export TZ='Asia/Shanghai'

# === 手动/环境可配置的续跑参数（默认关闭） ===
export RESUME_MODE=false
# export RESUME_MODE=true
# export RESUME_TIMESTAMP=20251113_073223
# export WANDB_RUN_ID=ihmq57k3
num_generations=8 # 作者使用7，但是3卡时候用7会犯错
EXP_TYPE=intuitor  # 可选值: intuitor 或 grpo
MAX_STEPS=-1    # 可选 -1 或者具体步数
START_VLLM=true # true: 1卡 vLLM + 3卡训练；false: 4卡训练
SAVE_TOTAL_LIMIT=16 # 控制最多保留的 checkpoint 数量

# GPU 分配策略
# - START_VLLM=true: vLLM 使用 GPU 0；训练用 1,2,3 共 3 卡
# - START_VLLM=false: 训练用 0,1,2,3 共 4 卡
if [ "$START_VLLM" = true ]; then
    VLLM_DEVICE="0"
    CUDA_DEVICES="1,2,3"
    NUM_PROCESSES=3
    BATCH_SIZE=4
    GRAD_ACCUM=32
    lr=3.0e-06
else
    VLLM_DEVICE=""
    CUDA_DEVICES="0,1,2,3"
    NUM_PROCESSES=4
    BATCH_SIZE=3
    GRAD_ACCUM=32
    lr=3.0e-06
fi

# 根据实验类型设置脚本和配置
if [ "$EXP_TYPE" = "grpo" ]; then
    SCRIPT_PATH="src/open_r1/grpo.py"
    CONFIG_FILE="recipes/Qwen2.5-3B/grpo/config_demo.yaml"
    WANDB_PROJECT="open-r1-grpo_qwen2.5-3B"
    RUN_NAME="Qwen2.5-GRPO-3B"
    LOG_PREFIX="grpo"
    BASE_OUTPUT_DIR="/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-GRPO-3B"
else
    SCRIPT_PATH="src/open_r1/intuitor.py"
    CONFIG_FILE="recipes/Qwen2.5-3B/intuitor/config_demo_lora.yaml"     
    WANDB_PROJECT="open-r1-intuitor_qwen2.5-3B"
    RUN_NAME="Qwen2.5-Intuitor-3B"
    LOG_PREFIX="intuitor"
    BASE_OUTPUT_DIR="/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B"
fi


# 创建日志/输出目录
if [ "$RESUME_MODE" = true ]; then
    : "${RESUME_TIMESTAMP:?RESUME_MODE=true 但 RESUME_TIMESTAMP 未设置}"  # 如果没有传 RESUME_TIMESTAMP，脚本会立刻报错并退出，提示用户传参
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
    RUN_NAME="${RUN_NAME}_${TIMESTAMP}"   # wandb 的 run_name
fi
if [ -n "$PROJECT_DIR" ]; then
    mkdir -p "$PROJECT_DIR"
fi
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"
RUN_LOG_FILE="${LOG_DIR}/run_${LOG_PREFIX}.log"
VLLM_LOG_FILE="${LOG_DIR}/vllm-server.log"

# 显示/记录配置参数
read -r -d '' RUN_CONFIG <<EOF
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
printf "%s\n" "$RUN_CONFIG" > "${LOG_DIR}/config_params.log"

# 启动GPU监控（每5秒记录一次）
nohup bash scripts/gpu_monitor.sh 5 "gpu_usage" "${LOG_DIR}" > /dev/null 2>&1 &
MONITOR_PID=$!
echo "GPU监控已启动 PID: $MONITOR_PID"


# 启动 vLLM（可选）
# 启动 vLLM 时，根据 RESUME_MODE 选择追加日志还是覆盖 >> 是追加， > 是覆盖
if [ "$START_VLLM" = true ]; then
  if [ "$RESUME_MODE" = true ]; then
    nohup env CUDA_VISIBLE_DEVICES=$VLLM_DEVICE \
        trl vllm-serve \
        --model /run/determined/NAS1/public/HuggingFace/Qwen/Qwen2.5-3B \
        >> "$VLLM_LOG_FILE" 2>&1 &
  else
    nohup env CUDA_VISIBLE_DEVICES=$VLLM_DEVICE \
        trl vllm-serve \
        --model /run/determined/NAS1/public/HuggingFace/Qwen/Qwen2.5-3B \
        > "$VLLM_LOG_FILE" 2>&1 &
  fi
  VLLM_PID=$!
  echo "vLLM server started with PID: $VLLM_PID on GPU $VLLM_DEVICE"
else
  echo "vLLM server skipped (START_VLLM=false)"
fi

# 如果不启动 vLLM，则通过 CLI 覆盖 YAML，关闭 use_vllm
EXTRA_ARGS=""
if [ "$START_VLLM" != true ]; then
  EXTRA_ARGS="--use_vllm false"
fi

# 启动训练脚本，默认的是24g版本，num_processes=7；为3的时候就使用48g显存
# 唯一差别在重定向符号：RESUME_MODE=true 时用 >> "$RUN_LOG_FILE"（追加），保留旧日志并把新的 stdout/stderr 接在文件末尾；RESUME_MODE=false 时用 > "$RUN_LOG_FILE"（覆盖），启动前会清空同名文件再写入
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

echo "Both processes started in the background. Check ${LOG_DIR}/vllm-server.log and ${LOG_DIR}/run_${LOG_PREFIX}.log for output."

# 结束监控
kill $MONITOR_PID
echo "GPU监控已停止"
