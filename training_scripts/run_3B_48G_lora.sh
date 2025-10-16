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
export WANDB_MODE=offline
export ACCELERATE_LOG_LEVEL=info
# 设置中国时区
# export TZ='Asia/Shanghai'
num_generations=8 # 作者使用7，但是3卡时候用7会犯错
EXP_TYPE=intuitor  # 可选值: intuitor 或 grpo
MAX_STEPS=-1    # 可选 -1 或者具体步数
START_VLLM=true # true: 1卡 vLLM + 3卡训练；false: 4卡训练

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
    WANDB_PROJECT="open-r1-grpo"
    RUN_NAME="Qwen2.5-GRPO-3B"
    LOG_PREFIX="grpo"
else
    SCRIPT_PATH="src/open_r1/intuitor.py"
    CONFIG_FILE="recipes/Qwen2.5-3B/intuitor/config_demo_lora.yaml"     
    WANDB_PROJECT="open-r1-intuitor"
    RUN_NAME="Qwen2.5-Intuitor-3B"
    LOG_PREFIX="intuitor"
fi


# 创建日志目录
# TIMESTAMP=$(date -d "+8 hour" +"%Y%m%d_%H%M%S")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/${LOG_PREFIX}_${NUM_PROCESSES}_3B/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# 模型 checkpoint 输出目录（带时间戳，避免覆盖）
BASE_OUTPUT_DIR="/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B"
OUTPUT_DIR="${BASE_OUTPUT_DIR}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

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
echo "START_VLLM: $START_VLLM"
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
echo "START_VLLM: $START_VLLM" >> "$PARAM_LOG"


# 启动 vLLM（可选）
if [ "$START_VLLM" = true ]; then
  nohup env CUDA_VISIBLE_DEVICES=$VLLM_DEVICE \
      trl vllm-serve \
      --model /run/determined/NAS1/public/HuggingFace/Qwen/Qwen2.5-3B \
      > "${LOG_DIR}/vllm-server.log" 2>&1 &
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
    accelerate launch --config_file recipes/accelerate_configs/zero3.yaml --num_processes=$NUM_PROCESSES \
    $SCRIPT_PATH \
    --per_device_eval_batch_size $BATCH_SIZE --per_device_train_batch_size $BATCH_SIZE --gradient_accumulation_steps $GRAD_ACCUM --learning_rate $lr --max_steps $MAX_STEPS \
    --num_generations $num_generations --output_dir $OUTPUT_DIR \
    --config $CONFIG_FILE --wandb_project $WANDB_PROJECT --run_name $RUN_NAME $EXTRA_ARGS > "${LOG_DIR}/run_${LOG_PREFIX}.log" 2>&1 &
TRAINING_PID=$!
wait $TRAINING_PID
echo "Training process started with PID: $TRAINING_PID"

echo "Both processes started in the background. Check ${LOG_DIR}/vllm-server.log and ${LOG_DIR}/run_${LOG_PREFIX}.log for output."

# 结束监控
kill $MONITOR_PID
echo "GPU监控已停止"
