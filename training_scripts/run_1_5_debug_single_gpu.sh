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
export http_proxy=http://10.130.130.5:7891
export https_proxy=http://10.130.130.5:7891
export no_proxy="127.0.0.1,localhost,0.0.0.0,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,.local"
export NO_PROXY="$no_proxy"
export WANDB_API_KEY=4117ed9c927aaa675b1e5c34fe7aebf892ed2009
export ACCELERATE_LOG_LEVEL=info
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=3
# 设置中国时区
# export TZ='Asia/Shanghai'
USE_WANDB=true               # true 开启 wandb，上报到 WANDB_PROJECT；false 彻底关闭
MODE=debug                     # 固定为 debug 模式
START_VLLM=false               # debug 强制关闭 vLLM
NUM_PROCESSES=1
BATCH_SIZE=6
GRAD_ACCUM=1
lr=3.0e-06
num_generations=3
EXP_TYPE=intuitor              # 可选值: intuitor 或 grpo
MAX_STEPS=50                   # debug 下默认 50 步
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# wandb 开关
if [ "$USE_WANDB" = "true" ]; then
  export WANDB_MODE=online
  unset WANDB_DISABLED
else
  export WANDB_MODE=offline
  export WANDB_DISABLED=true
fi

# 根据实验类型设置脚本和配置
if [ "$EXP_TYPE" = "grpo" ]; then
    SCRIPT_PATH="src/open_r1/grpo.py"
    CONFIG_FILE="recipes/Qwen2.5-1.5B/grpo/config_demo.yaml"
    WANDB_PROJECT="open-r1-debug"
    RUN_NAME="GRPO-1.5B-$(date +%Y%m%d-%H%M%S)"
    LOG_PREFIX="grpo"
else
    SCRIPT_PATH="src/open_r1/intuitor.py"
    CONFIG_FILE="recipes/Qwen2.5-1.5B/intuitor/config_demo.yaml"
    WANDB_PROJECT="open-r1-debug"
    RUN_NAME="Intuitor-1.5B-$(date +%Y%m%d-%H%M%S)"
    LOG_PREFIX="intuitor"
fi


# 日志和模型输出目录（仅 debug）
DEBUG_BASE="/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-1.5B_debug"
LOG_DIR="${DEBUG_BASE}/logs"
OUTPUT_DIR="${DEBUG_BASE}/ckpt"
# 清空上次运行的输出/日志目录
if [ -n "$DEBUG_BASE" ]; then
  rm -rf "$DEBUG_BASE"
fi
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# 显示当前配置参数
echo "===== 当前配置参数 ====="
echo "MODE: $MODE"
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
echo "START_VLLM: $START_VLLM"
echo "========================"

# 记录配置参数到日志文件
PARAM_LOG="${LOG_DIR}/config_params.log"
echo "MODE: $MODE" > "$PARAM_LOG"
echo "NUM_PROCESSES: $NUM_PROCESSES" >> "$PARAM_LOG"
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
echo "START_VLLM: $START_VLLM" >> "$PARAM_LOG"

# vLLM 关闭，强制通过 CLI 覆盖 YAML
EXTRA_ARGS="--use_vllm false"

# 仅在 debug 模式下，缩短生成长度（覆盖 YAML 的 max_completion_length: 2048）
EXTRA_ARGS="$EXTRA_ARGS --max_completion_length 256"

if [ "$USE_WANDB" != "true" ]; then
  # 关闭所有报告集成，覆盖 YAML 的 report_to: [wandb]
  EXTRA_ARGS="$EXTRA_ARGS --report_to none"
fi

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
