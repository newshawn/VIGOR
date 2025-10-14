#!/bin/bash
cd /home/wenxuexiang/projects/Intuitor/open-r1-intuitor
source /home/wenxuexiang/projects/Intuitor/open-r1-intuitor/openr1_intuitor/bin/activate
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# MODEL=/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251014_072137/checkpoint-10/
MODEL=/run/determined/NAS1/public/HuggingFace/Qwen/Qwen2.5-3B
# MODEL=/home/wenxuexiang/projects/Intuitor/open-r1-intuitor/data/Qwen2.5-1.5B-Intuitor/checkpoint-58
# MODEL_NAME=$(basename $MODEL)
PARENT=$(basename $(dirname $MODEL))   # Qwen2.5-1.5B-GRPO
CHILD=$(basename $MODEL)               # checkpoint-58
MODEL_NAME=$PARENT/$CHILD              # Qwen2.5-1.5B-GRPO/checkpoint-58
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:3072,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL_NAME

# 日志目录按时间戳
# TIMESTAMP=$(TZ='Asia/Shanghai' date +"%Y%m%d_%H%M%S")
TIMESTAMP=$(date -d "+8 hour" +"%Y%m%d_%H%M%S")
LOG_DIR=$OUTPUT_DIR/logs/$TIMESTAMP
mkdir -p $LOG_DIR

echo "=== 评估参数 ==="
echo "MODEL=$MODEL"
echo "MODEL_ARGS=\"$MODEL_ARGS\""
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "日志目录=$LOG_DIR"
echo "================"

echo "=== 开始 AIME24 ==="
TASK=aime24
lighteval vllm "$MODEL_ARGS" "lighteval|$TASK|0|0" \
  --use-chat-template \
  --output-dir $OUTPUT_DIR > $LOG_DIR/$TASK.log 2>&1

echo "=== AIME24 结束，开始 MATH-500 ==="
TASK=math_500
lighteval vllm "$MODEL_ARGS" "lighteval|$TASK|0|0" \
  --use-chat-template \
  --output-dir $OUTPUT_DIR > $LOG_DIR/$TASK.log 2>&1

# GPQA Diamond
# TASK=gpqa:diamond
# lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR > $LOG_DIR/$TASK.log 2>&1

# TASK=mmlu_pro
# lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR > $LOG_DIR/$TASK.log 2>&1

# LiveCodeBench
# TASK=lcb:codegeneration
# lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR > $LOG_DIR/$TASK.log 2>&1