#!/bin/bash
export NCCL_P2P_DISABLE=1
source /home/wenxuexiang/projects/Intuitor/open-r1-intuitor/openr1_intuitor/bin/activate
which python
cd /home/wenxuexiang/projects/Intuitor/open-r1-intuitor
# 保证源码优先被 import
export PYTHONPATH=/home/wenxuexiang/projects/Intuitor/open-r1-intuitor/src:${PYTHONPATH}
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
# clash off
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH}
export PATH=/usr/local/cuda-12.4/bin:${PATH} 
export CUDA_HOME=/usr/local/cuda-12.4
export http_proxy=http://127.0.0.1:18093
export https_proxy=http://127.0.0.1:18093
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
export no_proxy="127.0.0.1,localhost,0.0.0.0,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,.local"
export NO_PROXY="$no_proxy"
export WANDB_API_KEY=4117ed9c927aaa675b1e5c34fe7aebf892ed2009
export WANDB_MODE=online
export ACCELERATE_LOG_LEVEL=info
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export INTUITOR_SKIP_GIT_CHECK=1
export UPLOAD_WANDB_ARTIFACTS=true        # true: 上传日志等文件到 wandb artifact

# === 手动/环境可配置的续跑参数（默认关闭） ===
RESUME_MODE=false              # true: 续跑；false: 新跑
RESUME_TIMESTAMP=20251129_080256   # 续跑时填入要复用的时间戳
export WANDB_RESUME=allow
export WANDB_RUN_ID=rtzejar0       # 续跑时填入要续写的 wandb run id

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
    GRAD_ACCUM=8
    lr=6.0e-7
else
    VLLM_DEVICE=""
    CUDA_DEVICES="0,1,2,3"
    NUM_PROCESSES=4
    BATCH_SIZE=8
    GRAD_ACCUM=16
    lr=1.0e-06
fi

# 根据实验类型设置脚本和配置
if [ "$EXP_TYPE" = "grpo" ]; then
    SCRIPT_PATH="src/open_r1/grpo.py"
    CONFIG_FILE="recipes/Qwen2.5-3B/grpo/3B_grpo_lora.yaml"
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


# 创建日志/输出目录 + 续跑逻辑
if [ "$RESUME_MODE" = true ]; then
    : "${RESUME_TIMESTAMP:?RESUME_MODE=true 但 RESUME_TIMESTAMP 未设置}"
    TIMESTAMP="$RESUME_TIMESTAMP"
    PROJECT_DIR="${BASE_OUTPUT_DIR}_${TIMESTAMP}"
    LOG_DIR="${PROJECT_DIR}/logs"
    OUTPUT_DIR="${PROJECT_DIR}/ckpt"
    if [ -n "$WANDB_RUN_ID" ]; then
        echo "[RESUME] WANDB_RUN_ID=$WANDB_RUN_ID, WANDB_RESUME=$WANDB_RESUME"
    else
        echo "[RESUME] 未设置 WANDB_RUN_ID，将不会在 wandb 上续写同一个 run（仅本地续跑）"
    fi
    echo "[RESUME] 复用目录: $PROJECT_DIR"
else
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    PROJECT_DIR="${BASE_OUTPUT_DIR}_${TIMESTAMP}"
    LOG_DIR="${PROJECT_DIR}/logs"
    OUTPUT_DIR="${PROJECT_DIR}/ckpt"
    unset WANDB_RESUME
    unset WANDB_RUN_ID
    unset RESUME_TIMESTAMP
fi
if [ -n "$PROJECT_DIR" ]; then
    if [ "$RESUME_MODE" = true ] && [ ! -d "$PROJECT_DIR" ]; then
        echo "[ERROR] RESUME_MODE=true 但目录不存在: $PROJECT_DIR" >&2
        exit 1
    fi
    mkdir -p "$PROJECT_DIR"
fi
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"
RUN_NAME="${RUN_NAME}_${TIMESTAMP}"   # wandb 的 run_name
RUN_LAUNCH_TS=$(date +"%Y%m%d_%H%M%S")
if [ "$RESUME_MODE" = true ]; then
    RUN_LOG_FILE="${LOG_DIR}/run_${LOG_PREFIX}_resume_${RUN_LAUNCH_TS}.log"
else
    RUN_LOG_FILE="${LOG_DIR}/run_${LOG_PREFIX}_${RUN_LAUNCH_TS}.log"
fi
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
RUN_LOG_FILE: $RUN_LOG_FILE
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
if [ "$START_VLLM" = true ]; then
  nohup env CUDA_VISIBLE_DEVICES=$VLLM_DEVICE \
      trl vllm-serve \
      --model /run/determined/NAS1/public/HuggingFace/Qwen/Qwen2.5-3B \
      > "$VLLM_LOG_FILE" 2>&1 &
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
nohup env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes=$NUM_PROCESSES \
    $SCRIPT_PATH \
    --per_device_eval_batch_size $BATCH_SIZE --per_device_train_batch_size $BATCH_SIZE --gradient_accumulation_steps $GRAD_ACCUM --learning_rate $lr --max_steps $MAX_STEPS \
    --num_generations $num_generations --output_dir $OUTPUT_DIR \
    --config $CONFIG_FILE --wandb_project $WANDB_PROJECT --run_name $RUN_NAME --save_total_limit $SAVE_TOTAL_LIMIT $EXTRA_ARGS \
    > "$RUN_LOG_FILE" 2>&1 &
TRAINING_PID=$!
wait $TRAINING_PID
echo "Training process started with PID: $TRAINING_PID"

echo "Both processes started in the background. Check ${LOG_DIR}/vllm-server.log and ${RUN_LOG_FILE} for output."

# 结束监控
kill $MONITOR_PID
echo "GPU监控已停止"
