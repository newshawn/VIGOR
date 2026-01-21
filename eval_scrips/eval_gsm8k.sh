export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM
export CUDA_VISIBLE_DEVICES=6
MODEL=/run/determined/NAS1/public/HuggingFace/Qwen/Qwen2.5-7B
source /home/wenxuexiang/projects/Intuitor/open-r1-intuitor/openr1_intuitor/bin/activate
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:3072,temperature:0,top_p:1}"
# 提取末三级路径作为子目录（如 HuggingFace/Qwen/Qwen2.5-7B）
MODEL_SUBPATH=$(echo "$MODEL" | awk -F'/' '{n=NF; print $(n-2)"/"$(n-1)"/"$n}')
OUTPUT_DIR=data/evals/$MODEL_SUBPATH


TASK=gsm8k
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
