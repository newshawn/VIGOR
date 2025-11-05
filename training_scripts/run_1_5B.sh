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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=3
# 设置中国时区
# export TZ='Asia/Shanghai'
## 手动切换模式（只需改这里）
MODE=debug        # train | debug
START_VLLM=true   # 是否启动 vLLM（debug 下会强制关闭）

num_generations=7   # 作者使用7，但是3卡时候用7会犯错
NUM_PROCESSES=8     # 训练时的并行进程数（train 模式使用）
EXP_TYPE=intuitor    # 可选值: intuitor 或 grpo
MAX_STEPS=-1        # 可选 -1 或者具体步数

# 根据模式/进程数设置 CUDA 设备与 batch 参数
if [ "$MODE" = "debug" ]; then
    # 单卡调试：沿用 CUDA_VISIBLE_DEVICES（默认0），小 batch，小步数，并强制关闭 vLLM
    CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    NUM_PROCESSES=1
    BATCH_SIZE=3
    GRAD_ACCUM=1
    lr=3.0e-06
    num_generations=3
    START_VLLM=false
    # 彻底关闭 wandb（即使 YAML 里有 report_to: wandb）
    export WANDB_DISABLED=true
    if [ "$MAX_STEPS" = "-1" ]; then
        MAX_STEPS=50
    fi
else
    # 目前仅使用4090，仅支持 8卡 或 4卡 训练
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
            NUM_PROCESSES=3  # 预留 GPU0 给 vLLM
        else
            CUDA_DEVICES="0,1,2,3"
        fi
        BATCH_SIZE=2
        GRAD_ACCUM=1
        lr=3.0e-06
    else
        echo "[ERROR] 当前脚本仅支持 NUM_PROCESSES=4 或 8，实际为 ${NUM_PROCESSES}." >&2
        exit 1
    fi

    if [ "$START_VLLM" = true ] && [ "$REQUESTED_PROCESSES" -ne "$NUM_PROCESSES" ]; then
        echo "[INFO] vLLM 占用 1 张卡，训练进程数从 ${REQUESTED_PROCESSES} 调整为 ${NUM_PROCESSES}"
    fi
fi

# 根据实验类型设置脚本和配置
if [ "$EXP_TYPE" = "grpo" ]; then
    SCRIPT_PATH="src/open_r1/grpo.py"
    CONFIG_FILE="recipes/Qwen2.5-1.5B/grpo/config_demo.yaml"
    WANDB_PROJECT="open-r1-grpo"
    RUN_NAME="Qwen2.5-GRPO-1.5B"
    LOG_PREFIX="grpo"
else
    SCRIPT_PATH="src/open_r1/intuitor.py"
    CONFIG_FILE="recipes/Qwen2.5-1.5B/intuitor/config_demo.yaml"
    WANDB_PROJECT="open-r1-intuitor"
    RUN_NAME="Qwen2.5-Intuitor-1.5B"
    LOG_PREFIX="intuitor"
fi


# 创建日志目录
# TIMESTAMP=$(date -d "+8 hour" +"%Y%m%d_%H%M%S")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR="/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-1.5B"
if [ "$MODE" = "debug" ]; then
    LOG_DIR="logs/${LOG_PREFIX}_${NUM_PROCESSES}_1.5B/debug"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}_debug"
else
    LOG_DIR="logs/${LOG_PREFIX}_${NUM_PROCESSES}_1.5B/${TIMESTAMP}"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}_${TIMESTAMP}"
fi
mkdir -p "$LOG_DIR"

# 模型 checkpoint 输出目录（train 用时间戳，debug 用固定后缀）
mkdir -p "$OUTPUT_DIR"

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

# 启动GPU监控（每5秒记录一次）- debug 模式默认不启用
if [ "$MODE" != "debug" ]; then
  nohup bash scripts/gpu_monitor.sh 5 "gpu_usage" "${LOG_DIR}" > /dev/null 2>&1 &
  MONITOR_PID=$!
  echo "GPU监控已启动 PID: $MONITOR_PID"
fi

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

# 启动 vLLM（可选）
if [ "$START_VLLM" = true ]; then
  nohup env CUDA_VISIBLE_DEVICES=0 \
      trl vllm-serve \
      --model /run/determined/NAS1/public/HuggingFace/Qwen/Qwen2.5-1.5B \
      > "${LOG_DIR}/vllm-server.log" 2>&1 &
  VLLM_PID=$!
  echo "vLLM server started with PID: $VLLM_PID"
else
  echo "vLLM server skipped (START_VLLM=false)"
fi

# 如果不启动 vLLM，则通过 CLI 覆盖 YAML，关闭 use_vllm
EXTRA_ARGS=""
if [ "$START_VLLM" != true ]; then
  EXTRA_ARGS="--use_vllm false"
fi

# 仅在 debug 模式下，缩短生成长度（覆盖 YAML 的 max_completion_length: 2048）
if [ "$MODE" = "debug" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --max_completion_length 256"
fi

# debug 关闭所有报告集成，覆盖 YAML 的 report_to: [wandb]
if [ "$MODE" = "debug" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --report_to none"
fi

# 启动训练脚本：debug 模式不加载 zero3 配置；train 模式使用 zero3 配置
if [ "$MODE" = "debug" ]; then
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
  TRAINING_PID=$!
else
  nohup env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES ACCELERATE_LOG_LEVEL=info \
      accelerate launch --config_file recipes/accelerate_configs/zero3.yaml --num_processes=$NUM_PROCESSES \
      $SCRIPT_PATH \
      --per_device_eval_batch_size $BATCH_SIZE --per_device_train_batch_size $BATCH_SIZE --gradient_accumulation_steps $GRAD_ACCUM --learning_rate $lr --max_steps $MAX_STEPS \
      --num_generations $num_generations --output_dir $OUTPUT_DIR \
      --config $CONFIG_FILE --wandb_project $WANDB_PROJECT --run_name $RUN_NAME $EXTRA_ARGS \
      > "${LOG_DIR}/run_${LOG_PREFIX}.log" 2>&1 &
fi
if [ "$MODE" != "debug" ]; then
  TRAINING_PID=$!
  wait $TRAINING_PID
  echo "Training process started with PID: $TRAINING_PID"
fi

if [ "$MODE" != "debug" ]; then
  if [ "$START_VLLM" = true ]; then
    echo "Both processes started in the background. Check ${LOG_DIR}/vllm-server.log and ${LOG_DIR}/run_${LOG_PREFIX}.log for output."
  else
    echo "Training process started in the background. Check ${LOG_DIR}/run_${LOG_PREFIX}.log for output."
  fi
fi

# 结束监控
if [ "$MODE" != "debug" ]; then
  kill $MONITOR_PID
  echo "GPU监控已停止"
fi
