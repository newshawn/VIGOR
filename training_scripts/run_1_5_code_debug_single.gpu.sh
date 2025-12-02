#!/bin/bash

source /home/wenxuexiang/projects/Intuitor/open-r1-intuitor/openr1_intuitor/bin/activate
# pip install -e /home/wenxuexiang/projects/Intuitor/open-r1-intuitor
which python
cd /home/wenxuexiang/projects/Intuitor/open-r1-intuitor
# unset http_proxy
# unset https_proxy
# unset HTTP_PROXY
# unset HTTPS_PROXY
# clash on
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH}
export PATH=/usr/local/cuda-12.4/bin:${PATH} 
export CUDA_HOME=/usr/local/cuda-12.4
export http_proxy=http://127.0.0.1:18093
export https_proxy=http://127.0.0.1:18093
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
export no_proxy="127.0.0.1,localhost,0.0.0.0,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,.local"
export NO_PROXY="$no_proxy"
# export WANDB_BASE_URL="https://api.bandw.top"
export WANDB_API_KEY=4117ed9c927aaa675b1e5c34fe7aebf892ed2009
export WANDB_ENTITY=w597744907-zhejiang-university
export WANDB_MODE=online
export WANDB_DISABLED=false
export INTUITOR_ENABLE_WANDB_GIT_PATCH=1  # upload current git diff via wandb.save_git_patch()
export UPLOAD_WANDB_ARTIFACTS=false        # true: 训练结束后上传日志等文件到 wandb artifact
export ACCELERATE_LOG_LEVEL=info
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3
export INTUITOR_SKIP_GIT_CHECK=1  # 调试模式下，设置为 1 跳过 Git 检查
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODE=debug                     # 固定为 debug 模式
START_VLLM=false               # debug 强制关闭 vLLM
NUM_PROCESSES=1
BATCH_SIZE=6
GRAD_ACCUM=1
lr=3.0e-06
num_generations=3
EXP_TYPE=intuitor              # 可选值: intuitor 或 grpo
MAX_STEPS=50                   # debug 下默认 50 步

# 续跑时手动填：把上次 run 的时间戳/ID 填到下一行和 WANDB_RUN_ID，再把 RESUME_MODE 设为 true
RESUME_MODE=false              # true: 续跑；false: 新跑
RESUME_TIMESTAMP=20251128-043845
export WANDB_RUN_ID=8txj3olh
export WANDB_RESUME=allow

# 若续跑则复用传入的时间戳，否则生成新的
if [ "$RESUME_MODE" = "true" ]; then
  : "${RESUME_TIMESTAMP:?RESUME_MODE=true 但 RESUME_TIMESTAMP 未设置}"
  TIMESTAMP="$RESUME_TIMESTAMP"
else
  TIMESTAMP=$(date +%Y%m%d-%H%M%S)
  unset WANDB_RESUME
  unset WANDB_RUN_ID
  unset RESUME_TIMESTAMP
fi


# 只使用intuitor
SCRIPT_PATH="src/open_r1/intuitor.py"
CONFIG_FILE="recipes/Qwen2.5-1.5B/intuitor/config_code_demo_debug.yaml"
WANDB_PROJECT="open-r1-debug"
RUN_NAME="Intuitor-1.5B-Code-${TIMESTAMP}"
LOG_PREFIX="intuitor_code"


# 日志和模型输出目录（仅 debug）
DEBUG_BASE="/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-1.5B-Code_debug_${TIMESTAMP}"
LOG_DIR="${DEBUG_BASE}/logs"
OUTPUT_DIR="${DEBUG_BASE}/ckpt"
# 每次启动都生成独立日志文件，避免多次续跑覆盖
RUN_LAUNCH_TS=$(date +%Y%m%d-%H%M%S)
if [ "$RESUME_MODE" = "true" ]; then
  RUN_LOG_FILE="${LOG_DIR}/run_${LOG_PREFIX}_resume_${RUN_LAUNCH_TS}.log"
else
  RUN_LOG_FILE="${LOG_DIR}/run_${LOG_PREFIX}_${RUN_LAUNCH_TS}.log"
fi
# 新跑清空目录；续跑保留目录并打开 wandb resume
if [ "$RESUME_MODE" = "true" ]; then
  if [ ! -d "$DEBUG_BASE" ]; then
    echo "[ERROR] RESUME_MODE=true 但目录不存在: $DEBUG_BASE" >&2
    exit 1
  fi

  if [ -n "$WANDB_RUN_ID" ]; then
    echo "[RESUME] WANDB_RUN_ID=$WANDB_RUN_ID, WANDB_RESUME=$WANDB_RESUME"
  else
    echo "[RESUME] 未设置 WANDB_RUN_ID，将不会在 wandb 上续写同一个 run（仅本地续跑）"
  fi
  echo "[RESUME] 复用目录: $DEBUG_BASE"
else
  if [ -n "$DEBUG_BASE" ]; then
    rm -rf "$DEBUG_BASE"
  fi
fi
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# 将后续输出同时写入终端与日志文件
exec > >(tee "$RUN_LOG_FILE") 2>&1


# 显示/记录当前配置参数
PARAM_LOG="${LOG_DIR}/config_params.log"
read -r -d '' RUN_CONFIG <<EOF2
===== 当前配置参数 =====
MODE: $MODE
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
WANDB_RUN_ID: ${WANDB_RUN_ID:-}
LOG_PREFIX: $LOG_PREFIX
RUN_LOG_FILE: $RUN_LOG_FILE
LOG_DIR: $LOG_DIR
RESUME_MODE: $RESUME_MODE
RESUME_TIMESTAMP: ${RESUME_TIMESTAMP:-}
num_generations: $num_generations
OUTPUT_DIR: $OUTPUT_DIR
START_VLLM: $START_VLLM
========================
EOF2
printf "%s\n" "$RUN_CONFIG"
printf "%s\n" "$RUN_CONFIG" > "$PARAM_LOG"

# vLLM 关闭，强制通过 CLI 覆盖 YAML
EXTRA_ARGS="--use_vllm false"

# 仅在 debug 模式下，缩短生成长度（覆盖 YAML 的 max_completion_length: 2048）
EXTRA_ARGS="$EXTRA_ARGS --max_completion_length 128"

# 启动训练脚本：直接前台运行，便于调试
echo "[DEBUG] 直接以 python 前台运行，便于 pdb 调试（不使用 accelerate）"
echo "Command: CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -u $SCRIPT_PATH ..."
env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
  python -u "$SCRIPT_PATH" \
    --per_device_eval_batch_size "$BATCH_SIZE" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$lr" \
    --max_steps "$MAX_STEPS" \
    --num_generations "$num_generations" \
    --output_dir "$OUTPUT_DIR" \
    --config "$CONFIG_FILE" \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "$RUN_NAME" \
    $EXTRA_ARGS
TRAIN_EXIT=$?
exit $TRAIN_EXIT
