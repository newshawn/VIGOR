
#!/usr/bin/env bash
set -euo pipefail
cd ~/projects/Intuitor/open-r1-intuitor
# 仅使用本地 Hugging Face 缓存，禁止联网拉取。
export HF_HOME=/home/wenxuexiang/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
PYTHONPATH="${PYTHONPATH:-}"
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
source /home/wenxuexiang/projects/Intuitor/open-r1-intuitor/openr1_intuitor/bin/activate

MODEL_NAME="/run/determined/NAS1/public/xuexiang/model/Qwen2.5-3B"
DATASET_NAME="databricks/databricks-dolly-15k"
DATASET_TEXT_FIELD="response"
LEARNING_RATE="2.0e-5"
NUM_TRAIN_EPOCHS=1
MAX_SEQ_LENGTH=1024
PER_DEVICE_TRAIN_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=8
USE_PEFT=true
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET_MODULES=(q_proj k_proj v_proj o_proj)
RUN_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_ROOT="/run/determined/NAS1/public/xuexiang/SFT_ckpt/Qwen2.5-3B-SFT-${RUN_TIMESTAMP}"
CKPT_DIR="${RUN_ROOT}/ckpt"
LOG_DIR="${RUN_ROOT}/logs"
LOG_FILE="${LOG_DIR}/train.log"
RUN_LOG_FILE="${LOG_DIR}/run.log"
CONFIG_LOG_FILE="${LOG_DIR}/config_params.log"
NUM_PROCESSES=4
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

mkdir -p "$CKPT_DIR" "$LOG_DIR"

OUTPUT_DIR="$CKPT_DIR"

{
echo "===== Run start $(date) ====="
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME"
echo "Output: $OUTPUT_DIR"
echo "Logs: $LOG_DIR"
} | tee "$RUN_LOG_FILE"

# 记录主要参数
cat > "$CONFIG_LOG_FILE" <<EOF
MODEL_NAME: $MODEL_NAME
DATASET_NAME: $DATASET_NAME
DATASET_TEXT_FIELD: $DATASET_TEXT_FIELD
LEARNING_RATE: $LEARNING_RATE
NUM_TRAIN_EPOCHS: $NUM_TRAIN_EPOCHS
MAX_SEQ_LENGTH: $MAX_SEQ_LENGTH
PER_DEVICE_TRAIN_BATCH_SIZE: $PER_DEVICE_TRAIN_BATCH_SIZE
GRADIENT_ACCUMULATION_STEPS: $GRADIENT_ACCUMULATION_STEPS
USE_PEFT: $USE_PEFT
LORA_R: $LORA_R
LORA_ALPHA: $LORA_ALPHA
LORA_DROPOUT: $LORA_DROPOUT
LORA_TARGET_MODULES: ${LORA_TARGET_MODULES[*]}
NUM_PROCESSES: $NUM_PROCESSES
CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
RUN_TIMESTAMP: $RUN_TIMESTAMP
EOF
echo "Config saved to $CONFIG_LOG_FILE" | tee -a "$RUN_LOG_FILE"

# 启动 GPU 监控
nohup bash scripts/gpu_monitor.sh 5 "gpu_usage" "${LOG_DIR}" >/dev/null 2>&1 &
MONITOR_PID=$!
echo "GPU monitor started, pid=$MONITOR_PID" | tee -a "$RUN_LOG_FILE"
trap 'kill $MONITOR_PID >/dev/null 2>&1 || true' EXIT

OUTPUT_DIR="$CKPT_DIR"
echo "Logging to: $LOG_FILE"
accelerate launch --config_file=recipes/accelerate_configs/zero2.yaml --num_processes "$NUM_PROCESSES" src/open_r1/sft.py \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --dataset_text_field "$DATASET_TEXT_FIELD" \
    --attn_implementation flash_attention_2 \
    --use_peft "$USE_PEFT" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --lora_target_modules "${LORA_TARGET_MODULES[@]}" \
    --learning_rate "$LEARNING_RATE" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --gradient_checkpointing \
    --bf16 \
    --report_to none \
    --logging_steps 5 \
    --eval_strategy no \
    --output_dir "$OUTPUT_DIR" >"$LOG_FILE" 2>&1

echo "Training finished at $(date)" | tee -a "$RUN_LOG_FILE"
