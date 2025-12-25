#!/usr/bin/env bash
set -euo pipefail

cd ~/projects/Intuitor/open-r1-intuitor
PYTHONPATH="${PYTHONPATH:-}"
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
source /home/wenxuexiang/projects/Intuitor/open-r1-intuitor/openr1_intuitor/bin/activate
export CUDA_VISIBLE_DEVICES=6
MODEL_NAME="/run/determined/NAS1/public/xuexiang/model/Qwen2.5-1.5B"
DATASET_NAME="databricks/databricks-dolly-15k"
DATASET_TEXT_FIELD="response"
LEARNING_RATE="2.0e-5"
NUM_TRAIN_EPOCHS=1
MAX_SEQ_LENGTH=1024
MAX_STEPS=10
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
USE_PEFT=true
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_TARGET_MODULES=(q_proj k_proj v_proj o_proj)
RUN_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_ROOT="/run/determined/NAS1/public/xuexiang/SFT_ckpt/Qwen2.5-1.5B-SFT-debug-${RUN_TIMESTAMP}"
CKPT_DIR="${RUN_ROOT}/ckpt"
LOG_DIR="${RUN_ROOT}/logs"
LOG_FILE="${LOG_DIR}/train.log"
NUM_PROCESSES=1

mkdir -p "$CKPT_DIR" "$LOG_DIR"

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
    --max_steps "$MAX_STEPS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --gradient_checkpointing \
    --bf16 \
    --report_to none \
    --logging_steps 1 \
    --eval_strategy no \
    --output_dir "$OUTPUT_DIR" >"$LOG_FILE" 2>&1
