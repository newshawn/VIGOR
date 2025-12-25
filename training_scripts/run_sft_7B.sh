#!/usr/bin/env bash
set -euo pipefail

# 基于 run_sft_3B.sh，为 Qwen2.5-7B 的 LoRA SFT（默认 2x40G，bs=2，accum=16，lr=1e-6）
cd ~/projects/Intuitor/open-r1-intuitor
export http_proxy=http://127.0.0.1:18093
export https_proxy=http://127.0.0.1:18093
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
export no_proxy="127.0.0.1,localhost,0.0.0.0,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,.local"
export NO_PROXY="$no_proxy"
# 仅使用本地 Hugging Face 缓存，禁止联网拉取。
export HF_HOME=/home/wenxuexiang/.cache/huggingface
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0

# wandb 配置
export WANDB_MODE=online
export WANDB_PROJECT=SFT_7B
# 如需不同 entity 或 key，提前 export WANDB_ENTITY / WANDB_API_KEY

PYTHONPATH="${PYTHONPATH:-}"
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
source /home/wenxuexiang/projects/Intuitor/open-r1-intuitor/openr1_intuitor/bin/activate

MODEL_NAME="/run/determined/NAS1/public/xuexiang/model/Qwen2.5-7B"
DATASET_NAME="databricks/databricks-dolly-15k"
DATASET_TEXT_FIELD="response"
LEARNING_RATE="1e-5"
NUM_TRAIN_EPOCHS=20
MAX_SEQ_LENGTH=1024
PER_DEVICE_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=4
USE_PEFT=true
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET_MODULES=(all-linear)
RUN_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_ROOT="/run/determined/NAS1/public/xuexiang/SFT_ckpt/Qwen2.5-7B-SFT-${RUN_TIMESTAMP}"
CKPT_DIR="${RUN_ROOT}/ckpt"
LOG_DIR="${RUN_ROOT}/logs"
LOG_FILE="${LOG_DIR}/train.log"
RUN_LOG_FILE="${LOG_DIR}/run.log"
CONFIG_LOG_FILE="${LOG_DIR}/config_params.log"
# 设备数量
NUM_PROCESSES=1
export CUDA_VISIBLE_DEVICES=0
# export NCCL_P2P_DISABLE=1
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
WANDB_PROJECT: $WANDB_PROJECT
WANDB_MODE: $WANDB_MODE
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
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --gradient_checkpointing \
    --bf16 \
    --report_to wandb \
    --run_name "Qwen2.5-7B-SFT-${RUN_TIMESTAMP}" \
    --logging_steps 50 \
    --save_strategy epoch \
    --eval_strategy no \
    --output_dir "$OUTPUT_DIR" >"$LOG_FILE" 2>&1

echo "Training finished at $(date)" | tee -a "$RUN_LOG_FILE"
