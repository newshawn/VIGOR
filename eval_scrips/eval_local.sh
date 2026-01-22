#!/bin/bash

# 用法：
#   ./eval_local.sh <path1> [path2 ...]
# path 可以是具体 checkpoint 目录，也可以是父目录；父目录下凡是包含
# "checkpoint-" 的子目录都会自动加入评测列表。若不传参数，则使用脚本内
# 的默认 MODEL 路径（或 MODELS 数组）。

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv_lighteval}"
source "$VENV_DIR/bin/activate"
cd "$REPO_ROOT"
which python
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=0   # 新增这一行
# 确保 HF 本地缓存与离线模式（集群/无网环境使用本地缓存）
# 指定数据集缓存目录到共享盘，便于复用/持久化
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="/run/determined/NAS1/public/xuexiang/light_eval"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_DISABLE_XET=1  # 避免 xet 后端在无代理/内网环境访问 cas-server 失败
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0

# 默认模型（当未传入参数/未配置数组时使用）
DEFAULT_MODEL=/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251028_124723/checkpoint-10/
MODELS=(
  "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-GRPO-3B_20251111_160918/checkpoint-40"
  ""/run/determined/NAS1/public/xuexiang/model/Qwen2.5-3B""
  # 基本盘
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-GRPO-7B_20251212_051822/merged/checkpoint-50"
  # "/run/determined/NAS1/public/xuexiang/SFT_ckpt/Qwen2.5-7B-SFT-20251213-134744/merged"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-7B_20251205_142916/merged"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251210_083219/ckpt/checkpoint-150"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251210_090136/ckpt/checkpoint-20"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251210_085759/ckpt/checkpoint-90"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251210_083219/ckpt/checkpoint-80"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251209_172035/ckpt/checkpoint-150"
  # "/run/determined/NAS1/public/xuexiang/model/Qwen2.5-3B"
  # "/run/determined/NAS1/public/xuexiang/model/Qwen2.5-1.5B"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251129_080256/ckpt/checkpoint-90"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-7B_20251203_155525/merged/checkpoint-400"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251210_085759/ckpt/checkpoint-90"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251210_085759/ckpt/checkpoint-156"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251210_083219/ckpt/checkpoint-150"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251210_083219/ckpt/checkpoint-80"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251210_090136/ckpt/checkpoint-20"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251210_090136/ckpt/checkpoint-30"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251209_172035/ckpt/checkpoint-150"
  # "/run/determined/NAS1/public/xuexiang/model/Co-rewarding-I-Qwen2.5-7B-MATH"
  # "/run/determined/NAS1/public/xuexiang/model/Co-rewarding-II-Qwen2.5-7B-MATH"
  # "/run/determined/NAS1/public/xuexiang/model/Co-rewarding-I-Qwen2.5-3B-MATH"
  # "/run/determined/NAS1/public/xuexiang/model/Co-rewarding-II-Qwen2.5-3B-MATH"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251209_172035/ckpt"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B-open-data_20251209_150025/ckpt"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251210_085759/ckpt/checkpoint-110"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251210_085759/ckpt/checkpoint-120"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251210_085759/ckpt/checkpoint-130"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251210_085759/ckpt/checkpoint-140"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251210_085759/ckpt/checkpoint-150"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251210_085759/ckpt/checkpoint-156"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-7B_20251210_173753/ckpt"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-7B_20251210_190542/ckpt"
  # "/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-GRPO-7B_20251209_111617/merged"
  # "/run/determined/NAS1/public/xuexiang/model/Co-rewarding-Qwen2.5-7B-MATH-GT"
)
export CUDA_VISIBLE_DEVICES=0,1
# 需要评测的任务列表
TASKS=(
  # "extended|lcb:codegeneration_release_v6"
  # "mmlu_pro"
  # "gsm8k"
  # "aime24"
  # "aime25"
  # "aime24_gpassk"
  # "aime25_gpassk"
  "ifeval"
  # "math_500"
  # 如需开启其它任务，取消注释并添加到数组：
  # "gpqa:diamond"
  # "mmlu_pro"
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

# 获取可用 GPU 列表：优先读取 CUDA_VISIBLE_DEVICES，否则默认使用 GPU 0
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
  GPU_LIST=(0)
fi
GPU_COUNT=${#GPU_LIST[@]}
echo "可用 GPU 槽位（共 $GPU_COUNT 个）：${GPU_LIST[*]}"

run_single_model() {
  local MODEL="$1"
  local GPU_LABEL="$2"

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
  local MODEL_ARGS="model_name=$MODEL_TRIMMED,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,seed=0,generation_parameters={max_new_tokens:3072,temperature:0,top_p:1}"
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
  echo "[GPU $GPU_LABEL] MODEL_ARGS=\"$MODEL_ARGS\""
  echo "[GPU $GPU_LABEL] OUTPUT_DIR=$OUTPUT_DIR"
  echo "[GPU $GPU_LABEL] 日志目录=$LOG_DIR"
  echo "[GPU $GPU_LABEL] ================="

  # 数据集缓存快速检查（避免离线环境报错定位困难）
  local TASK
  for TASK in "${TASKS[@]}"; do
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
  done

  for TASK in "${TASKS[@]}"; do
    echo "[GPU $GPU_LABEL] === 开始 $TASK ==="

    # 默认 0-shot（mmlu_pro 也使用 0-shot）；gsm8k 用 4-shot
    local FEW_SHOT=0
    if [[ "$TASK" == "gsm8k" ]]; then
      FEW_SHOT=0
    fi

    local logfile_task="${TASK//\//_}"
    local logfile="$LOG_DIR/${logfile_task}_${TIMESTAMP}.log"
    # 对于 LiveCodeBench 扩展语法示例（以 extended| 前缀区分），否则默认使用 lighteval 套件名
    # if [[ "$TASK" == extended\|* ]]; then
    lighteval vllm "$MODEL_ARGS" "$TASK|$FEW_SHOT" \
      --output-dir "$OUTPUT_DIR" > "$logfile" 2>&1 \
      --save-details
    local status=$?
    if [ $status -ne 0 ]; then
      echo "[GPU $GPU_LABEL] ⚠️ 任务 $TASK 失败，退出码 $status（详见日志 $logfile）"
    else
      echo "[GPU $GPU_LABEL] === $TASK 完成 ==="
    fi
  done

  echo "[GPU $GPU_LABEL] === $MODEL_TRIMMED 全部任务完成 ==="
}

declare -a GPU_PIDS=()
declare -a GPU_MODELS=()

start_model_on_gpu() {
  local MODEL="$1"
  local SLOT="$2"
  local GPU_LABEL="${GPU_LIST[$SLOT]}"

  (
    export CUDA_VISIBLE_DEVICES="$GPU_LABEL"
    run_single_model "$MODEL" "$GPU_LABEL"
  ) &
  local PID=$!
  GPU_PIDS[$SLOT]=$PID
  GPU_MODELS[$SLOT]=$MODEL
  echo ">>> 模型 $MODEL 分配到 GPU $GPU_LABEL (槽位 $SLOT)，PID=$PID"
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
        GPU_MODELS[$SLOT]=""
        AVAILABLE_SLOT=$SLOT
        return
      fi
    done
    sleep 5
  done
}

for MODEL in "${MODELS_TO_RUN[@]}"; do
  wait_for_slot
  start_model_on_gpu "$MODEL" "$AVAILABLE_SLOT"
done

# 等待所有后台任务完成
for PID in "${GPU_PIDS[@]}"; do
  if [[ -n "$PID" ]]; then
    wait "$PID"
  fi
done
