#!/bin/bash
export NCCL_P2P_DISABLE=1
# 强制使用容器内共享盘上的 venv，避免落到 /home 挂载的本地 venv
cd /home/wenxuexiang/projects/Intuitor/open-r1-intuitor
which python
# 保证源码优先被 import
export PYTHONPATH=/home/wenxuexiang/projects/Intuitor/open-r1-intuitor/src:${PYTHONPATH:-}
export HF_HOME=/run/determined/localcq1/xuexiang/dataset/MATH-lighteval/.hf
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
export USER=wenxuexiang
export LOGNAME=wenxuexiang
export USERNAME=wenxuexiang
export TORCHINDUCTOR_CACHE_DIR=/run/determined/workdir/.cache/torchinductor
export TRITON_CACHE_DIR=/run/determined/workdir/.triton
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# unset http_proxy
# unset https_proxy
# unset HTTP_PROXY
# unset HTTPS_PROXY
clash on
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH}
export PATH=/usr/local/cuda-12.4/bin:${PATH} 
export CUDA_HOME=/usr/local/cuda-12.4
export http_proxy=http://172.17.0.1:7899
export https_proxy=http://172.17.0.1:7899
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
export no_proxy="127.0.0.1,localhost,0.0.0.0,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,.local"
export NO_PROXY="$no_proxy"
export WANDB_API_KEY=4117ed9c927aaa675b1e5c34fe7aebf892ed2009
export WANDB_MODE=online
export WANDB_BASE_URL=https://api.bandw.top
export ACCELERATE_LOG_LEVEL=info
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export INTUITOR_SKIP_GIT_CHECK=1
export UPLOAD_WANDB_ARTIFACTS=true        # true: 上传日志等文件到 wandb artifact

# === 手动/环境可配置的续跑参数（默认关闭） ===
RESUME_MODE=false              # true: 续跑；false: 新跑
RESUME_TIMESTAMP=20251217_053932   # 续跑时填入要复用的时间戳
export WANDB_RESUME=allow
export WANDB_RUN_ID=c78lq7y8       # 续跑时填入要续写的 wandb run id

num_generations=8 # 作者使用7，但是3卡时候用7会犯错
EXP_TYPE=intuitor  # 可选值: intuitor 或 grpo
ACCELERATE_CONFIG_FILE="recipes/accelerate_configs/zero2.yaml"
MAX_STEPS=-1    # 可选 -1 或者具体步数
START_VLLM=true # true: 1卡 vLLM + 3卡训练；false: 4卡训练
### 传统的保存方式，如果SAVE_STRATEGY="yes"，则会按照SAVE_STEPS保存checkpoint，否则只保存top-K的accuracy_reward的ckpt
SAVE_TOTAL_LIMIT=20 # 控制最多保留的 checkpoint 数量
SAVE_STEPS=20      # checkpoint 间隔步数
SAVE_STRATEGY="no" # 使用 top-k 保存时关闭定期保存
### 保存top-K的ckpt，默认保存accuracy_reward/mean最大的ckpt，并且不保存优化器，每LOGGING_STEPS更新一次top-K
SAVE_ONLY_MODEL=true
SAVE_TOP_K=5
SAVE_TOP_K_METRIC="rewards/accuracy_reward/mean"
SAVE_TOP_K_GREATER_IS_BETTER=true
LOGGING_STEPS=5
MAX_COMPLETION_LENGTH=3072
KL_REWARD_SQRT_LEN_SCALING_ENABLED=true
KL_REWARD_RANK_NORMALIZATION_ENABLED=true
KL_ENTROPY_WEIGHTING_ENABLED=false
KL_ENTROPY_FOCAL_LAMBDA=0.1

# GPU 分配策略
# - START_VLLM=true: vLLM 使用 GPU 0；训练用 1,2,3 共 3 卡
# - START_VLLM=false: 训练用 0,1,2,3 共 4 卡
if [ "$START_VLLM" = true ]; then
    VLLM_DEVICE="0"
    CUDA_DEVICES="1,2,3,4,5,6,7"  # 1-7 共 7 卡
    NUM_PROCESSES=7
    num_generations=8
    BATCH_SIZE=4
    GRAD_ACCUM=32
    lr=2e-6
    beta=0.01   # kl penalty
else
    VLLM_DEVICE=""
    CUDA_DEVICES="0,1,2,3"
    NUM_PROCESSES=4
    BATCH_SIZE=3
    GRAD_ACCUM=16
    lr=1.0e-06
    beta=0.01   # kl penalty
fi

# 如果不启动 vLLM，则通过 CLI 覆盖 YAML，关闭 use_vllm
EXTRA_ARGS=""
if [ "$START_VLLM" != true ]; then
  EXTRA_ARGS="--use_vllm false"
fi

# 根据实验类型设置脚本和配置
if [ "$EXP_TYPE" = "grpo" ]; then
    SCRIPT_PATH="src/open_r1/grpo.py"
    CONFIG_FILE="recipes/Qwen2.5-7B/grpo/config_demo_h800.yaml"
    WANDB_PROJECT="open-r1-grpo_qwen2.5-7B"
    RUN_NAME="Qwen2.5-GRPO-7B"
    LOG_PREFIX="grpo"
    BASE_OUTPUT_DIR="/run/determined/localcq1/xuexiang/Intuitor_ckpt/Qwen2.5-GRPO-7B"
else
    SCRIPT_PATH="src/open_r1/intuitor.py"
    CONFIG_FILE="recipes/Qwen2.5-7B/intuitor/config_demo_h800.yaml"     
    WANDB_PROJECT="open-r1-intuitor_qwen2.5-7B"
    RUN_NAME="Qwen2.5-Intuitor-7B"
    LOG_PREFIX="intuitor"
    BASE_OUTPUT_DIR="/run/determined/localcq1/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-7B"
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
mkdir -p /run/determined/localcq1/xuexiang/Intuitor_ckpt

# 让每次运行的 wandb 产物落在 PROJECT_DIR/wandb（而不是默认的 ./wandb 或 $HOME）
export WANDB_DIR="${PROJECT_DIR}/wandb"
mkdir -p "$WANDB_DIR"

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
ACCELERATE_CONFIG_FILE: $ACCELERATE_CONFIG_FILE
NUM_PROCESSES: $NUM_PROCESSES
EXP_TYPE: $EXP_TYPE
CUDA_DEVICES: $CUDA_DEVICES
BATCH_SIZE: $BATCH_SIZE
PER_DEVICE_TRAIN_BATCH_SIZE: $BATCH_SIZE
PER_DEVICE_EVAL_BATCH_SIZE: $BATCH_SIZE
GRAD_ACCUM: $GRAD_ACCUM
lr: $lr
beta: $beta
MAX_STEPS: $MAX_STEPS
LOGGING_STEPS: $LOGGING_STEPS
SCRIPT_PATH: $SCRIPT_PATH
CONFIG_FILE: $CONFIG_FILE
WANDB_PROJECT: $WANDB_PROJECT
RUN_NAME: $RUN_NAME
LOG_PREFIX: $LOG_PREFIX
LOG_DIR: $LOG_DIR
RUN_LOG_FILE: $RUN_LOG_FILE
num_generations: $num_generations
MAX_COMPLETION_LENGTH: $MAX_COMPLETION_LENGTH
OUTPUT_DIR: $OUTPUT_DIR
PROJECT_DIR: $PROJECT_DIR
START_VLLM: $START_VLLM
SAVE_TOTAL_LIMIT: $SAVE_TOTAL_LIMIT
SAVE_STEPS: $SAVE_STEPS
SAVE_STRATEGY: $SAVE_STRATEGY
SAVE_ONLY_MODEL: $SAVE_ONLY_MODEL
SAVE_TOP_K: $SAVE_TOP_K
SAVE_TOP_K_METRIC: $SAVE_TOP_K_METRIC
SAVE_TOP_K_GREATER_IS_BETTER: $SAVE_TOP_K_GREATER_IS_BETTER
EXTRA_ARGS: ${EXTRA_ARGS:-}
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

echo "===== DIAG: vLLM/trl env ====="
echo "[diag] hostname: $(hostname || true)"
set +e
echo "[diag] VIRTUAL_ENV: ${VIRTUAL_ENV:-}"
echo "[diag] PATH: ${PATH:-}"
echo "[diag] which python:"
which python || true
echo "[diag] python -V:"
python -V || true
echo "[diag] python exe:"
python -c 'import sys; print(sys.executable)' || true
echo "[diag] ls -l python:"
ls -l "$(command -v python 2>/dev/null)" 2>/dev/null || true
echo "[diag] readlink -f python:"
readlink -f "$(command -v python 2>/dev/null)" 2>/dev/null || true
echo "[diag] which trl:"
which trl || true
echo "[diag] trl --version (timeout 60s):"
timeout 60s trl --version; echo "[diag] trl exit=$?"
echo "[diag] import vllm,trl (timeout 20s):"
if command -v timeout >/dev/null 2>&1; then timeout 20s python -c "import vllm, trl; print('ok')"; else python -c "import vllm, trl; print('ok')"; fi
echo "[diag] model dir:"
ls -ld /run/determined/localcq1/xuexiang/Qwen2.5-7B || true
echo "[diag] nvidia-smi -L (timeout 60s):"
timeout 60s nvidia-smi -L; echo "[diag] nvidia-smi exit=$?"
set -e
echo "===== DIAG END ====="

# 启动 vllm
if [ "$START_VLLM" = true ]; then
  echo "[vLLM] starting on GPU $VLLM_DEVICE..."
  nohup env CUDA_VISIBLE_DEVICES=$VLLM_DEVICE TRITON_CACHE_DIR=$TRITON_CACHE_DIR \
    trl vllm-serve \
    --model /run/determined/localcq1/xuexiang/Qwen2.5-7B \
    --host 127.0.0.1 --port 8000 \
    > "$VLLM_LOG_FILE" 2>&1 &

  VLLM_PID=$!
  echo "vLLM server started with PID: $VLLM_PID on GPU $VLLM_DEVICE"
else
  echo "vLLM server skipped (START_VLLM=false)"
fi

# 启动训练脚本：同时输出到控制台 + 写入 RUN_LOG_FILE
if command -v stdbuf >/dev/null 2>&1; then
  stdbuf -oL -eL env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES ACCELERATE_LOG_LEVEL=info \
      accelerate launch --config_file $ACCELERATE_CONFIG_FILE --num_processes=$NUM_PROCESSES \
      $SCRIPT_PATH \
      --per_device_eval_batch_size $BATCH_SIZE --per_device_train_batch_size $BATCH_SIZE --gradient_accumulation_steps $GRAD_ACCUM --learning_rate $lr --beta $beta --max_steps $MAX_STEPS --logging_steps $LOGGING_STEPS --max_completion_length $MAX_COMPLETION_LENGTH \
      --num_generations $num_generations --output_dir $OUTPUT_DIR \
      --config $CONFIG_FILE --wandb_project $WANDB_PROJECT --run_name $RUN_NAME --save_total_limit $SAVE_TOTAL_LIMIT --save_only_model $SAVE_ONLY_MODEL --save_steps $SAVE_STEPS \
      --save_strategy $SAVE_STRATEGY --save_top_k $SAVE_TOP_K --save_top_k_metric "$SAVE_TOP_K_METRIC" --save_top_k_greater_is_better $SAVE_TOP_K_GREATER_IS_BETTER $EXTRA_ARGS \
      --kl_reward_sqrt_len_scaling_enabled $KL_REWARD_SQRT_LEN_SCALING_ENABLED \
      --kl_reward_rank_normalization_enabled $KL_REWARD_RANK_NORMALIZATION_ENABLED \
      --kl_entropy_weighting_enabled $KL_ENTROPY_WEIGHTING_ENABLED \
      --kl_entropy_focal_lambda $KL_ENTROPY_FOCAL_LAMBDA \
      2>&1 | tee "$RUN_LOG_FILE"
else
  env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES ACCELERATE_LOG_LEVEL=info \
      accelerate launch --config_file $ACCELERATE_CONFIG_FILE --num_processes=$NUM_PROCESSES \
      $SCRIPT_PATH \
      --per_device_eval_batch_size $BATCH_SIZE --per_device_train_batch_size $BATCH_SIZE --gradient_accumulation_steps $GRAD_ACCUM --learning_rate $lr --beta $beta --max_steps $MAX_STEPS --logging_steps $LOGGING_STEPS --max_completion_length $MAX_COMPLETION_LENGTH \
      --num_generations $num_generations --output_dir $OUTPUT_DIR \
      --config $CONFIG_FILE --wandb_project $WANDB_PROJECT --run_name $RUN_NAME --save_total_limit $SAVE_TOTAL_LIMIT --save_only_model $SAVE_ONLY_MODEL --save_steps $SAVE_STEPS \
      --save_strategy $SAVE_STRATEGY --save_top_k $SAVE_TOP_K --save_top_k_metric "$SAVE_TOP_K_METRIC" --save_top_k_greater_is_better $SAVE_TOP_K_GREATER_IS_BETTER $EXTRA_ARGS \
      --kl_reward_sqrt_len_scaling_enabled $KL_REWARD_SQRT_LEN_SCALING_ENABLED \
      --kl_reward_rank_normalization_enabled $KL_REWARD_RANK_NORMALIZATION_ENABLED \
      --kl_entropy_weighting_enabled $KL_ENTROPY_WEIGHTING_ENABLED \
      --kl_entropy_focal_lambda $KL_ENTROPY_FOCAL_LAMBDA \
      2>&1 | tee "$RUN_LOG_FILE"
fi
TRAIN_EXIT_CODE=${PIPESTATUS[0]}

echo "Training finished with exit code: $TRAIN_EXIT_CODE"
echo "Logs: ${VLLM_LOG_FILE} and ${RUN_LOG_FILE}"

# 结束监控
kill $MONITOR_PID
echo "GPU监控已停止"
exit $TRAIN_EXIT_CODE
