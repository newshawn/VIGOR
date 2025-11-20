#!/bin/bash

# 用法：
#   ./eval.sh <model_path1> [model_path2 ...]
# 若不传参数，则使用脚本内的默认 MODEL 路径。

cd /home/wenxuexiang/projects/Intuitor/open-r1-intuitor
source /home/wenxuexiang/projects/Intuitor/open-r1-intuitor/openr1_intuitor/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,4
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 确保 HF 本地缓存与离线模式（集群/无网环境使用本地缓存）
# 指定数据集缓存目录到共享盘，便于复用/持久化
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="/run/determined/NAS1/public/xuexiang/light_eval"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

# 在线/离线开关（默认在线以便必要时下载到上面路径；设置 EVAL_ONLINE=0 可强制离线）
EVAL_ONLINE=${EVAL_ONLINE:-1}
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
"/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251110_141709/checkpoint-10"
"/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251110_141709/checkpoint-20"
"/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251110_141709/checkpoint-30"
"/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251110_141709/checkpoint-40"
"/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251110_141709/checkpoint-50"
"/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251110_141709/checkpoint-60"
"/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251110_141709/checkpoint-90"
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

# 收集待评测的模型列表：优先使用外部传参；否则使用上方 MODELS；再否则用 DEFAULT_MODEL
if [ ${#INPUT_MODELS[@]} -gt 0 ]; then
  MODELS_TO_RUN=("${INPUT_MODELS[@]}")
elif [ ${#MODELS[@]} -gt 0 ]; then
  MODELS_TO_RUN=("${MODELS[@]}")
else
  MODELS_TO_RUN=("$DEFAULT_MODEL")
fi

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

run_single_model() {
  local MODEL="$1"
  local GPU_LABEL="$2"

  # 去除模型路径末尾的斜杠，避免 basename/dirname 解析偏差
  local MODEL_TRIMMED="${MODEL%/}"

  # 生成输出目录名（父目录/子目录）
  local PARENT
  local CHILD
  # 取倒数第三层作为父目录（如 .../Qwen2.5-XXX/ckpt/checkpoint-20 -> Qwen2.5-XXX）
  PARENT=$(basename "$(dirname "$(dirname "$MODEL_TRIMMED")")")
  CHILD=$(basename "$MODEL_TRIMMED")
  local MODEL_NAME="$PARENT/$CHILD"

  # vLLM 推理参数（保持与单模型版本一致）
  local MODEL_ARGS="model_name=$MODEL_TRIMMED,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:3072,temperature:0.6,top_p:0.95}"

  local OUTPUT_DIR="data/evals/$MODEL_NAME"
  local LOG_DIR="$OUTPUT_DIR/logs/$TIMESTAMP"
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

    local logfile
    # 对于 LiveCodeBench 扩展语法示例（以 extended| 前缀区分），否则默认使用 lighteval 套件名
    if [[ "$TASK" == extended\|* ]]; then
      # 扩展示例：TASK 字符串已经包含完整的套件标识
      logfile="$LOG_DIR/${TASK//\//_}.log"
      lighteval vllm "$MODEL_ARGS" "$TASK|0|0" \
        --use-chat-template \
        --output-dir "$OUTPUT_DIR" > "$logfile" 2>&1
    else
      logfile="$LOG_DIR/$TASK.log"
      lighteval vllm "$MODEL_ARGS" "lighteval|$TASK|0|0" \
        --use-chat-template \
        --output-dir "$OUTPUT_DIR" > "$logfile" 2>&1
    fi
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
