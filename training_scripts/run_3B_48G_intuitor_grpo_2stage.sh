#!/bin/bash
export NCCL_P2P_DISABLE=1
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
source "$VENV_DIR/bin/activate"
which python
cd "$REPO_ROOT"
# 保证源码优先被 import
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
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
export UPLOAD_WANDB_ARTIFACTS=true        # true: 上传日志等文件到 wandb artifact

# === 手动/环境可配置的续跑参数（默认关闭） ===
RESUME_MODE=false              # true: 续跑；false: 新跑
RESUME_TIMESTAMP=20251210_085759   # 续跑时填入要复用的时间戳
export WANDB_RESUME=allow
export WANDB_RUN_ID=                 # 续跑时填入 wandb run id（可留空，若留空则不续写同一个 run）

# === 两阶段训练设置 ===
EXP_ORDER=intuitor_then_grpo   # 可选值: intuitor_then_grpo 或 grpo_then_intuitor
# EXP_ORDER=grpo_then_intuitor
RUN_STAGE1=true                # true: 跑第一阶段；false: 跳过第一阶段
RUN_STAGE2=true                # true: 跑第二阶段；false: 跳过第二阶段

CONFIG_INTUITOR="recipes/Qwen2.5-3B/intuitor/config_demo.yaml"
CONFIG_GRPO="recipes/Qwen2.5-3B/grpo/config_demo.yaml"

# ===== 手动设置参数（尽量与 run_3B_48G.sh 一致） =====
WANDB_PROJECT="open-r1-qwen2.5-3B-2stage" # 两阶段共用一个 wandb project，方便对比/分析
MAX_STEPS=-1    # 可选 -1 或者具体步数
START_VLLM=true # true: 1卡 vLLM + 3卡训练；false: 4卡训练
SAVE_TOTAL_LIMIT=15 # 控制最多保留的 checkpoint 数量，当SAVE_STRATEGY="no"时无效
SAVE_STRATEGY="steps" # "no" top-k; "steps" “每 N 步保存”的逻辑
SAVE_STEPS=30 # 保存 checkpoint 的步数间隔（覆盖 YAML 里的 save_steps）
SAVE_TOP_K=0
SAVE_TOP_K_METRIC="rewards/accuracy_reward/mean"
SAVE_TOP_K_GREATER_IS_BETTER=true
SAVE_RESUME_STEPS=30 # 每 N 步覆盖保存 checkpoint-last（包含训练状态），0 表示关闭
LOGGING_STEPS=3 # 训练日志记录步数间隔（配合 logging_strategy=steps）
MAX_COMPLETION_LENGTH=1024 # completion 最大长度（覆盖 YAML 里的 max_completion_length）
KL_REWARD_SQRT_LEN_SCALING_ENABLED=true
KL_REWARD_RANK_NORMALIZATION_ENABLED=true

# GPU 分配策略
# - START_VLLM=true: vLLM 使用 GPU 0；训练用 1,2,3共 3 卡
num_generations=8
VLLM_DEVICE="0"
CUDA_DEVICES="1,2,3"
NUM_PROCESSES=3
BATCH_SIZE=4
GRAD_ACCUM=32

# 阶段超参（默认沿用 run_3B_48G.sh）
lr_intuitor=1e-6
beta_intuitor=0.01
lr_grpo=1e-6
beta_grpo=0.01

# 两阶段输出目录（统一放一个 project 目录下）
BASE_OUTPUT_DIR="/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-3B-Intuitor-GRPO-2stage"

# 根据 EXP_ORDER 决定 stage1/stage2 类型
if [ "$EXP_ORDER" = "intuitor_then_grpo" ]; then
  STAGE1_TYPE="intuitor"
  STAGE2_TYPE="grpo"
elif [ "$EXP_ORDER" = "grpo_then_intuitor" ]; then
  STAGE1_TYPE="grpo"
  STAGE2_TYPE="intuitor"
else
  echo "[ERROR] EXP_ORDER=$EXP_ORDER 不支持。可选: intuitor_then_grpo / grpo_then_intuitor" >&2
  exit 1
fi

stage_to_script() {
  if [ "$1" = "grpo" ]; then
    echo "src/open_r1/grpo.py"
  else
    echo "src/open_r1/intuitor.py"
  fi
}

stage_to_config() {
  if [ "$1" = "grpo" ]; then
    echo "$CONFIG_GRPO"
  else
    echo "$CONFIG_INTUITOR"
  fi
}

stage_to_project() {
  echo "$WANDB_PROJECT"
}

stage_to_run_name() {
  if [ "$1" = "grpo" ]; then
    echo "Qwen2.5-GRPO-3B"
  else
    echo "Qwen2.5-Intuitor-3B"
  fi
}

stage_to_lr() {
  if [ "$1" = "grpo" ]; then
    echo "$lr_grpo"
  else
    echo "$lr_intuitor"
  fi
}

stage_to_beta() {
  if [ "$1" = "grpo" ]; then
    echo "$beta_grpo"
  else
    echo "$beta_intuitor"
  fi
}

# 续跑/新跑：统一复用 project 目录
if [ "$RESUME_MODE" = true ]; then
    : "${RESUME_TIMESTAMP:?RESUME_MODE=true 但 RESUME_TIMESTAMP 未设置}"
    TIMESTAMP="$RESUME_TIMESTAMP"
    PROJECT_DIR="${BASE_OUTPUT_DIR}_${TIMESTAMP}"
else
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    PROJECT_DIR="${BASE_OUTPUT_DIR}_${TIMESTAMP}"
    unset WANDB_RESUME
    unset WANDB_RUN_ID
    unset RESUME_TIMESTAMP
fi

WANDB_RUN_GROUP="Qwen2.5-3B-2stage_${TIMESTAMP}" # 两个 stage 归到同一个 group 里

LOG_DIR="${PROJECT_DIR}/logs"
STAGE1_DIR="${PROJECT_DIR}/stage1"
STAGE2_DIR="${PROJECT_DIR}/stage2"
STAGE1_OUT="${STAGE1_DIR}/ckpt"
STAGE2_OUT="${STAGE2_DIR}/ckpt"

mkdir -p "$LOG_DIR" "$STAGE1_OUT" "$STAGE2_OUT"

VLLM_LOG_FILE="${LOG_DIR}/vllm-server.log"

STAGE1_SCRIPT_PATH="$(stage_to_script "$STAGE1_TYPE")"
STAGE1_CONFIG_FILE="$(stage_to_config "$STAGE1_TYPE")"
STAGE1_WANDB_PROJECT="$(stage_to_project "$STAGE1_TYPE")"
STAGE1_RUN_NAME="$(stage_to_run_name "$STAGE1_TYPE")"
STAGE1_LR="$(stage_to_lr "$STAGE1_TYPE")"
STAGE1_BETA="$(stage_to_beta "$STAGE1_TYPE")"

STAGE2_SCRIPT_PATH="$(stage_to_script "$STAGE2_TYPE")"
STAGE2_CONFIG_FILE="$(stage_to_config "$STAGE2_TYPE")"
STAGE2_WANDB_PROJECT="$(stage_to_project "$STAGE2_TYPE")"
STAGE2_RUN_NAME="$(stage_to_run_name "$STAGE2_TYPE")"
STAGE2_LR="$(stage_to_lr "$STAGE2_TYPE")"
STAGE2_BETA="$(stage_to_beta "$STAGE2_TYPE")"

if [ ! -f "$STAGE1_CONFIG_FILE" ]; then
  echo "[ERROR] stage1 config 不存在: $STAGE1_CONFIG_FILE" >&2
  exit 1
fi
if [ ! -f "$STAGE2_CONFIG_FILE" ]; then
  echo "[ERROR] stage2 config 不存在: $STAGE2_CONFIG_FILE" >&2
  exit 1
fi

# stage2 初始化权重：默认从 stage1 输出目录加载（可手动改成某个 checkpoint）
STAGE2_INIT_MODEL_PATH="${STAGE2_INIT_MODEL_PATH:-$STAGE1_OUT}"

# resume checkpoint（每个 stage 各自独立）
STAGE1_RESUME_FROM_CHECKPOINT="${STAGE1_OUT}/checkpoint-last"
STAGE2_RESUME_FROM_CHECKPOINT="${STAGE2_OUT}/checkpoint-last"

# wandb 运行名称
RUN_NAME_STAGE1="${STAGE1_RUN_NAME}_${TIMESTAMP}"
RUN_NAME_STAGE2="${STAGE2_RUN_NAME}_${TIMESTAMP}"

# 显示/记录配置参数
read -r -d '' RUN_CONFIG <<EOF
===== 当前配置参数 =====
EXP_ORDER: $EXP_ORDER
RUN_STAGE1: $RUN_STAGE1
RUN_STAGE2: $RUN_STAGE2
WANDB_PROJECT: $WANDB_PROJECT
WANDB_RUN_GROUP: $WANDB_RUN_GROUP

NUM_PROCESSES: $NUM_PROCESSES
CUDA_DEVICES: $CUDA_DEVICES
BATCH_SIZE: $BATCH_SIZE
GRAD_ACCUM: $GRAD_ACCUM
num_generations: $num_generations

MAX_STEPS: $MAX_STEPS
START_VLLM: $START_VLLM
SAVE_TOTAL_LIMIT: $SAVE_TOTAL_LIMIT
SAVE_STRATEGY: $SAVE_STRATEGY
SAVE_STEPS: $SAVE_STEPS
SAVE_TOP_K: $SAVE_TOP_K
SAVE_TOP_K_METRIC: $SAVE_TOP_K_METRIC
SAVE_TOP_K_GREATER_IS_BETTER: $SAVE_TOP_K_GREATER_IS_BETTER
SAVE_RESUME_STEPS: $SAVE_RESUME_STEPS
LOGGING_STEPS: $LOGGING_STEPS
MAX_COMPLETION_LENGTH: $MAX_COMPLETION_LENGTH
KL_REWARD_SQRT_LEN_SCALING_ENABLED: $KL_REWARD_SQRT_LEN_SCALING_ENABLED
KL_REWARD_RANK_NORMALIZATION_ENABLED: $KL_REWARD_RANK_NORMALIZATION_ENABLED

PROJECT_DIR: $PROJECT_DIR
LOG_DIR: $LOG_DIR
STAGE1_TYPE: $STAGE1_TYPE
STAGE1_SCRIPT_PATH: $STAGE1_SCRIPT_PATH
STAGE1_CONFIG_FILE: $STAGE1_CONFIG_FILE
STAGE1_WANDB_PROJECT: $STAGE1_WANDB_PROJECT
RUN_NAME_STAGE1: $RUN_NAME_STAGE1
STAGE1_LR: $STAGE1_LR
STAGE1_BETA: $STAGE1_BETA
STAGE1_OUT: $STAGE1_OUT
STAGE1_RESUME_FROM_CHECKPOINT: $STAGE1_RESUME_FROM_CHECKPOINT
STAGE2_TYPE: $STAGE2_TYPE
STAGE2_SCRIPT_PATH: $STAGE2_SCRIPT_PATH
STAGE2_CONFIG_FILE: $STAGE2_CONFIG_FILE
STAGE2_WANDB_PROJECT: $STAGE2_WANDB_PROJECT
RUN_NAME_STAGE2: $RUN_NAME_STAGE2
STAGE2_LR: $STAGE2_LR
STAGE2_BETA: $STAGE2_BETA
STAGE2_INIT_MODEL_PATH: $STAGE2_INIT_MODEL_PATH
STAGE2_OUT: $STAGE2_OUT
STAGE2_RESUME_FROM_CHECKPOINT: $STAGE2_RESUME_FROM_CHECKPOINT
WANDB_RUN_ID: ${WANDB_RUN_ID:-}

RESUME_MODE: $RESUME_MODE
RESUME_TIMESTAMP: $RESUME_TIMESTAMP
========================
EOF
printf "%s\n" "$RUN_CONFIG"
printf "%s\n" "$RUN_CONFIG" > "${LOG_DIR}/config_params.log"

# 启动GPU监控（每5秒记录一次）
nohup bash scripts/gpu_monitor.sh 5 "gpu_usage" "${LOG_DIR}" > /dev/null 2>&1 &
MONITOR_PID=$!
echo "GPU监控已启动 PID: $MONITOR_PID"

cleanup() {
  if [ -n "${VLLM_PID:-}" ]; then
    kill "$VLLM_PID" 2>/dev/null || true
  fi
  if [ -n "${MONITOR_PID:-}" ]; then
    kill "$MONITOR_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# 启动 vLLM（可选）
if [ "$START_VLLM" = true ]; then
  nohup env CUDA_VISIBLE_DEVICES=$VLLM_DEVICE \
      trl vllm-serve \
      --model /run/determined/NAS1/public/HuggingFace/Qwen/Qwen2.5-3B \
      > "$VLLM_LOG_FILE" 2>&1 &
  VLLM_PID=$!
  echo "vLLM server started with PID: $VLLM_PID on GPU $VLLM_DEVICE"
else
  VLLM_PID=""
  echo "vLLM server skipped (START_VLLM=false)"
fi

# 如果不启动 vLLM，则通过 CLI 覆盖 YAML，关闭 use_vllm
EXTRA_ARGS=""
if [ "$START_VLLM" != true ]; then
  EXTRA_ARGS="--use_vllm false"
fi

run_one_stage() {
  local stage_idx="$1"        # 1 or 2
  local stage_type="$2"       # intuitor or grpo
  local script_path="$3"
  local config_file="$4"
  local wandb_project="$5"
  local run_name="$6"
  local output_dir="$7"
  local lr="$8"
  local beta="$9"
  local init_model_path="${10:-}"
  local resume_from_checkpoint="${11:-}"

  local stage_tag
  stage_tag="stage${stage_idx}_${stage_type}"
  local run_launch_ts
  run_launch_ts=$(date +"%Y%m%d_%H%M%S")
  local run_log_file="${LOG_DIR}/run_${stage_tag}_${run_launch_ts}.log"

  RESUME_ARGS=""
  if [ "$RESUME_MODE" = true ] && [ -n "$resume_from_checkpoint" ] && [ -d "$resume_from_checkpoint" ]; then
    RESUME_ARGS="--resume_from_checkpoint $resume_from_checkpoint"
  fi

  MODEL_ARGS=""
  if [ -n "$init_model_path" ]; then
    MODEL_ARGS="--model_name_or_path $init_model_path"
  fi

  if [ "$RESUME_MODE" = true ] && [ -n "${WANDB_RUN_ID:-}" ]; then
    export WANDB_RUN_ID="$WANDB_RUN_ID"
    export WANDB_RESUME=allow
  else
    unset WANDB_RUN_ID
    unset WANDB_RESUME
  fi

  # Log: 同时写入文件 + 输出到终端（方便 det 看日志）
  touch "$run_log_file"

  echo "========================"
  echo "[RUN] $stage_tag"
  echo "SCRIPT_PATH: $script_path"
  echo "CONFIG_FILE: $config_file"
  echo "WANDB_PROJECT: $wandb_project"
  echo "RUN_NAME: $run_name"
  echo "OUTPUT_DIR: $output_dir"
  echo "lr: $lr"
  echo "beta: $beta"
  echo "MODEL_ARGS: ${MODEL_ARGS:-<none>}"
  echo "RESUME_ARGS: ${RESUME_ARGS:-<none>}"
  echo "RUN_LOG_FILE: $run_log_file"
  echo "========================"

  env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES ACCELERATE_LOG_LEVEL=info \
      accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes=$NUM_PROCESSES \
      $script_path \
      --per_device_eval_batch_size $BATCH_SIZE --per_device_train_batch_size $BATCH_SIZE --gradient_accumulation_steps $GRAD_ACCUM --learning_rate $lr --beta $beta --max_steps $MAX_STEPS --logging_steps $LOGGING_STEPS --max_completion_length $MAX_COMPLETION_LENGTH \
      --num_generations $num_generations --output_dir $output_dir \
      --config $config_file --wandb_project $wandb_project --wandb_run_group "$WANDB_RUN_GROUP" --run_name $run_name --save_only_model true --save_total_limit $SAVE_TOTAL_LIMIT \
      --save_strategy $SAVE_STRATEGY --save_steps $SAVE_STEPS --save_top_k $SAVE_TOP_K --save_top_k_metric "$SAVE_TOP_K_METRIC" --save_top_k_greater_is_better $SAVE_TOP_K_GREATER_IS_BETTER --save_resume_steps $SAVE_RESUME_STEPS \
      --kl_reward_sqrt_len_scaling_enabled $KL_REWARD_SQRT_LEN_SCALING_ENABLED \
      --kl_reward_rank_normalization_enabled $KL_REWARD_RANK_NORMALIZATION_ENABLED \
      $MODEL_ARGS $RESUME_ARGS $EXTRA_ARGS \
      2>&1 | tee -a "$run_log_file"
  local train_exit=${PIPESTATUS[0]}
  echo "Training process finished. exit_code: $train_exit"
  if [ "$train_exit" -ne 0 ]; then
    echo "[ERROR] stage${stage_idx} (${stage_type}) failed. Stop before next stage. See log: $run_log_file" >&2
    exit "$train_exit"
  fi
}

# ===== Stage 1 =====
# 分别传入 script.py / config.yaml / wandb_project / run_name / output_dir / lr / beta / init_model_path / resume_from_checkpoint
if [ "$RUN_STAGE1" = true ]; then
  run_one_stage \
    1 "$STAGE1_TYPE" \
    "$STAGE1_SCRIPT_PATH" "$STAGE1_CONFIG_FILE" "$STAGE1_WANDB_PROJECT" "$RUN_NAME_STAGE1" "$STAGE1_OUT" \
    "$STAGE1_LR" "$STAGE1_BETA" \
    "" "$STAGE1_RESUME_FROM_CHECKPOINT"
else
  echo "[SKIP] RUN_STAGE1=false"
fi

# ===== Stage 2 =====
# 分别传入 script.py / config.yaml / wandb_project / run_name / output_dir / lr / beta / init_model_path / resume_from_checkpoint
# 为什么 stage2 一定要从 stage1 输出目录 init，而不是从 stage1 的 checkpoint-last 续跑？因为 resume 只能恢复同一个阶段自己产出的 checkpoint
if [ "$RUN_STAGE2" = true ]; then
  if [ ! -d "$STAGE2_INIT_MODEL_PATH" ]; then
    echo "[ERROR] STAGE2_INIT_MODEL_PATH 不存在: $STAGE2_INIT_MODEL_PATH" >&2
    exit 1
  fi
  run_one_stage \
    2 "$STAGE2_TYPE" \
    "$STAGE2_SCRIPT_PATH" "$STAGE2_CONFIG_FILE" "$STAGE2_WANDB_PROJECT" "$RUN_NAME_STAGE2" "$STAGE2_OUT" \
    "$STAGE2_LR" "$STAGE2_BETA" \
    "$STAGE2_INIT_MODEL_PATH" "$STAGE2_RESUME_FROM_CHECKPOINT"
else
  echo "[SKIP] RUN_STAGE2=false"
fi

echo "Both stages finished. Check ${VLLM_LOG_FILE} and logs under ${LOG_DIR} for output."

# 结束监控（trap 会兜底）
kill $MONITOR_PID 2>/dev/null || true
echo "GPU监控已停止"
