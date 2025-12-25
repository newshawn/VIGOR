#!/bin/bash

# 用法：
#   ./eval_local.sh <path1> [path2 ...]
# path 可以是具体 checkpoint 目录，也可以是父目录；父目录下凡是包含
# "checkpoint-" 的子目录都会自动加入评测列表。若不传参数，则使用脚本内
# 的默认 MODEL 路径（或 MODELS 数组）。

cd /home/wenxuexiang/projects/Intuitor/open-r1-intuitor
source /home/wenxuexiang/projects/Intuitor/open-r1-intuitor/.venv_lighteval/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 确保 HF 本地缓存与离线模式（集群/无网环境使用本地缓存）
# 指定数据集缓存目录到共享盘，便于复用/持久化
# export http_proxy=http://10.130.130.5:7891
# export https_proxy=http://10.130.130.5:7891
# export no_proxy="127.0.0.1,localhost,0.0.0.0,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,.local"
# export NO_PROXY="$no_proxy"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="/run/determined/NAS1/public/xuexiang/light_eval"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_DISABLE_XET=1  # 避免 xet 后端在无代理/内网环境访问 cas-server 失败
mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

# 在线/离线开关（默认在线以便必要时下载到上面路径；设置 EVAL_ONLINE=0 可强制离线）
EVAL_ONLINE=0
if [[ "$EVAL_ONLINE" == "1" ]]; then
  export TRANSFORMERS_OFFLINE=0
  export HF_HUB_OFFLINE=0
  export HF_DATASETS_OFFLINE=0
  export HF_DATASETS_DISABLE_DOWNLOAD=0
else
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export HF_DATASETS_DISABLE_DOWNLOAD=1
fi
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "HF_HOME=$HF_HOME"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "EVAL_ONLINE=$EVAL_ONLINE (0=offline,1=online)"

# 默认模型（当未传入参数/未配置数组时使用）
DEFAULT_MODEL=/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251028_124723/checkpoint-10/
# 也可切换为以下任一模型作为默认：
# DEFAULT_MODEL=/run/determined/NAS1/public/HuggingFace/Qwen/Qwen2.5-3B
# DEFAULT_MODEL=/home/wenxuexiang/projects/Intuitor/open-r1-intuitor/data/Qwen2.5-1.5B-Intuitor/checkpoint-58

# 在此处填写多个模型路径；留空则使用外部传参或 DEFAULT_MODEL
# MODELS=(
#   "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-grpo-3B_20251014_152032/checkpoint-30/"
#   "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-grpo-3B_20251014_152032/checkpoint-40/"
# )
MODELS=(
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251127_022509/ckpt/checkpoint-130"    # 会自动加入该目录下所有 checkpoint-*
  # # "/run/determined/NAS1/public/HuggingFace/Qwen/Qwen2.5-3B" # 也可填具体 checkpoint
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251127_022509/ckpt/checkpoint-140"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251127_022509/ckpt/checkpoint-150"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251127_022509/ckpt/checkpoint-156"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251221_082957/ckpt"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251221_083515/ckpt"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B-code_20251222_185707/ckpt"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B-code_20251222_112548/ckpt"
  "/run/determined/NAS1/public/xuexiang/model/models--sunblaze-ucb--Qwen2.5-3B-Intuitor-MATH-1EPOCH/snapshots/bdb0b2fc48a80ab8906521ac55a4ce278a0538e5"
  "/run/determined/NAS1/public/xuexiang/model/Qwen2.5-7B-Intuitor-MATH-1EPOCH"
)


# 允许通过环境变量传参（便于集群作业/自动化脚本）
#   MODEL_PATH=<单个模型>
#   MODEL_PATHS="path1 path2" (或逗号分隔)
declare -a INPUT_MODELS=()
if [[ -n "${MODEL_PATH:-}" ]]; then
  INPUT_MODELS+=("$MODEL_PATH")
fi
if [[ -n "${MODEL_PATHS:-}" ]]; then
  # 支持逗号/空白分隔
  IFS=', ' read -r -a __MODEL_PATHS_ARRAY <<< "$MODEL_PATHS"
  for path in "${__MODEL_PATHS_ARRAY[@]}"; do
    [[ -z "$path" ]] && continue
    INPUT_MODELS+=("$path")
  done
fi
if [[ "$#" -gt 0 ]]; then
  INPUT_MODELS+=("$@")
fi

# 收集原始候选路径：优先使用外部传参；否则使用 MODELS；再否则用 DEFAULT_MODEL
declare -a RAW_MODEL_PATHS=()
if [ ${#INPUT_MODELS[@]} -gt 0 ]; then
  RAW_MODEL_PATHS=("${INPUT_MODELS[@]}")
elif [ ${#MODELS[@]} -gt 0 ]; then
  RAW_MODEL_PATHS=("${MODELS[@]}")
else
  RAW_MODEL_PATHS=("$DEFAULT_MODEL")
fi

# 将父目录展开为包含 "checkpoint-" 的子目录；如果自身就是 checkpoint 目录也保留
expand_checkpoints() {
  local path="$1"
  local -n _out_ref="$2"
  [[ -z "$path" ]] && return

  local trimmed="${path%/}"
  if [[ -d "$trimmed" ]]; then
    local added=0
    # 如果目录本身就是 checkpoint 目录，直接加入
    if [[ "$(basename "$trimmed")" == *checkpoint-* ]]; then
      _out_ref+=("$trimmed")
      added=1
    fi
    # 搜索一层子目录中包含 checkpoint- 的目录
    while IFS= read -r -d '' subdir; do
      _out_ref+=("${subdir%/}")
      added=1
    done < <(find "$trimmed" -maxdepth 1 -mindepth 1 -type d -name '*checkpoint-*' -print0)
    # 如果没找到 checkpoint-* 子目录，把原始目录也加入（兼容 HF 根目录）
    if [[ $added -eq 0 ]]; then
      _out_ref+=("$trimmed")
    fi
  else
    # 非目录：按原样加入（可能是具体模型路径或将被后续过滤）
    _out_ref+=("$trimmed")
  fi
}

declare -a EXPANDED_MODELS=()
for cand in "${RAW_MODEL_PATHS[@]}"; do
  expand_checkpoints "$cand" EXPANDED_MODELS
done

# 去重并过滤不存在的路径
declare -A __SEEN_MODELS=()
MODELS_TO_RUN=()
for model in "${EXPANDED_MODELS[@]}"; do
  [[ -z "$model" ]] && continue
  if [[ -n "${__SEEN_MODELS[$model]:-}" ]]; then
    continue
  fi
  __SEEN_MODELS[$model]=1
  if [[ -d "$model" ]]; then
    MODELS_TO_RUN+=("$model")
  else
    echo "⚠️ 模型路径不存在，已跳过：$model" >&2
  fi
done

# 调试：打印本次将要评测的模型列表
echo "=== 将评测以下模型（${#MODELS_TO_RUN[@]}） ==="
for __m in "${MODELS_TO_RUN[@]}"; do
  echo " - ${__m}"
done
echo "==============================="

# 日志目录按时间戳（统一时间戳，便于多模型同批次检索）
# TIMESTAMP=$(TZ='Asia/Shanghai' date +"%Y%m%d_%H%M%S")
TIMESTAMP=$(date -d "+8 hour" +"%Y%m%d_%H%M%S")

# 需要评测的任务列表
TASKS=(
  # "extended|lcb:codegeneration_release_v6"
  # "mmlu_pro"
  "gsm8k"
  "math_500"
  # 如需开启其它任务，取消注释并添加到数组：
  # "gpqa:diamond"
  # "mmlu_pro"
  # "extended|lcb:codegeneration"  # LiveCodeBench（用于扩展语法示例）
)

# 获取可用 GPU 列表：优先读取 CUDA_VISIBLE_DEVICES，否则默认使用 GPU 0
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
  GPU_LIST=(0)
fi
GPU_COUNT=${#GPU_LIST[@]}
echo "可用 GPU 槽位（共 $GPU_COUNT 个）：${GPU_LIST[*]}"

run_single_task() {
  local MODEL="$1"
  local TASK="$2"
  local GPU_LABEL="$3"

  # 去除模型路径末尾的斜杠，避免 basename/dirname 解析偏差
  local MODEL_TRIMMED="${MODEL%/}"

  # 生成输出目录名（上上级/上级/当前）
  local GRANDPARENT
  local PARENT
  local CHILD
  GRANDPARENT=$(basename "$(dirname "$(dirname "$MODEL_TRIMMED")")")
  PARENT=$(basename "$(dirname "$MODEL_TRIMMED")")
  CHILD=$(basename "$MODEL_TRIMMED")
  local MODEL_NAME="$GRANDPARENT/$PARENT/$CHILD"

  # vLLM 推理参数（保持与单模型版本一致）
  local MODEL_ARGS="model_name=$MODEL_TRIMMED,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:3072,temperature:0,top_p:1}"
  if [[ "$MODEL_ARGS" == *"temperature:0"* && "$MODEL_ARGS" == *"top_p:1"* ]]; then
    export EVAL_GREEDY_DECODE=1
  else
    export EVAL_GREEDY_DECODE=0
  fi

  local OUTPUT_DIR="$PWD/data/evals/$MODEL_NAME"
  local LOG_DIR="$OUTPUT_DIR/logs"
  mkdir -p "$LOG_DIR"

  echo "[GPU $GPU_LABEL] === 评估参数 ==="
  echo "[GPU $GPU_LABEL] MODEL=$MODEL_TRIMMED"
  echo "[GPU $GPU_LABEL] TASK=$TASK"
  echo "[GPU $GPU_LABEL] MODEL_ARGS=\"$MODEL_ARGS\""
  echo "[GPU $GPU_LABEL] OUTPUT_DIR=$OUTPUT_DIR"
  echo "[GPU $GPU_LABEL] 日志目录=$LOG_DIR"
  echo "[GPU $GPU_LABEL] ================="

  # 数据集缓存快速检查（避免离线环境报错定位困难）
  case "$TASK" in
    aime24)
      local AIME_PATH_GLOB="$HF_DATASETS_CACHE/HuggingFaceH4___aime_2024/default"
      if ! ls -d "$AIME_PATH_GLOB" 1>/dev/null 2>&1; then
        echo "[GPU $GPU_LABEL] ⚠️ 未检测到 aime_2024 本地缓存目录：$AIME_PATH_GLOB"
        echo "[GPU $GPU_LABEL]    若在离线环境，请先在有网环境预下载该数据集（保持相同用户缓存路径）。"
      fi
      ;;
    math_500)
      local MATH_PATH_GLOB="$HF_DATASETS_CACHE/HuggingFaceH4___math-500/default"
      if ! ls -d "$MATH_PATH_GLOB" 1>/dev/null 2>&1; then
        echo "[GPU $GPU_LABEL] ⚠️ 未检测到 math_500 本地缓存目录：$MATH_PATH_GLOB"
      fi
      ;;
  esac

  echo "[GPU $GPU_LABEL] === 开始 $TASK ==="

  # 默认 0-shot（mmlu_pro 也使用 0-shot）
  local FEW_SHOT=0

  local logfile_task="${TASK//\//_}"
  local logfile="$LOG_DIR/${logfile_task}_${TIMESTAMP}.log"
  # 对于 LiveCodeBench 扩展语法示例（以 extended| 前缀区分），否则默认使用 lighteval 套件名
  if [[ "$TASK" == extended\|* ]]; then
    lighteval vllm "$MODEL_ARGS" "$TASK|$FEW_SHOT" \
      --output-dir "$OUTPUT_DIR" > "$logfile" 2>&1
  else
    lighteval vllm "$MODEL_ARGS" "lighteval|$TASK|$FEW_SHOT" \
      --output-dir "$OUTPUT_DIR" > "$logfile" 2>&1
  fi
  local status=$?
  if [ $status -ne 0 ]; then
    echo "[GPU $GPU_LABEL] ⚠️ 任务 $TASK 失败，退出码 $status（详见日志 $logfile）"
  else
    echo "[GPU $GPU_LABEL] === $TASK 完成 ==="
  fi
}

declare -a GPU_PIDS=()
declare -a GPU_JOBS=()

start_task_on_gpu() {
  local MODEL="$1"
  local TASK="$2"
  local SLOT="$3"
  local GPU_LABEL="${GPU_LIST[$SLOT]}"

  (
    export CUDA_VISIBLE_DEVICES="$GPU_LABEL"
    run_single_task "$MODEL" "$TASK" "$GPU_LABEL"
  ) &
  local PID=$!
  GPU_PIDS[$SLOT]=$PID
  GPU_JOBS[$SLOT]="$MODEL::$TASK"
  echo ">>> TASK=$TASK MODEL=$MODEL assigned to GPU $GPU_LABEL (slot $SLOT), PID=$PID"
}

wait_for_slot() {
  local SLOT
  while true; do
    for SLOT in "${!GPU_LIST[@]}"; do
      local PID="${GPU_PIDS[$SLOT]}"
      if [[ -z "$PID" ]]; then
        AVAILABLE_SLOT=$SLOT
        return
      fi
      if ! kill -0 "$PID" 2>/dev/null; then
        wait "$PID" 2>/dev/null
        GPU_PIDS[$SLOT]=""
        GPU_JOBS[$SLOT]=""
        AVAILABLE_SLOT=$SLOT
        return
      fi
    done
    sleep 5
  done
}

declare -a JOB_MODELS=()
declare -a JOB_TASKS=()
for MODEL in "${MODELS_TO_RUN[@]}"; do
  for TASK in "${TASKS[@]}"; do
    JOB_MODELS+=("$MODEL")
    JOB_TASKS+=("$TASK")
  done
done

echo "Jobs to run: ${#JOB_TASKS[@]}"

for idx in "${!JOB_MODELS[@]}"; do
  wait_for_slot
  start_task_on_gpu "${JOB_MODELS[$idx]}" "${JOB_TASKS[$idx]}" "$AVAILABLE_SLOT"
done

# 等待所有后台任务完成
for PID in "${GPU_PIDS[@]}"; do
  if [[ -n "$PID" ]]; then
    wait "$PID"
  fi
done
