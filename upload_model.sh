#!/usr/bin/env bash
set -euo pipefail

# 1) 必填：token（不要写进代码仓库；建议用 read -s 输入）
if [[ -z "${HF_TOKEN:-}" ]]; then
  read -s -p "HF_TOKEN: " HF_TOKEN
  echo
  export HF_TOKEN
fi

# 2) 可选：你的 HF 用户名/组织
export HF_OWNER="newshawn"

# 3) 可选：目标仓库名（远端显示名）
export HF_REPO="Qwen2.5-7B-MATH-1EPOCH"

# 4) 可选：本地要上传的目录
export HF_LOCAL_PATH="/run/determined/NAS1/public/xuexiang/H800/best_7B/checkpoint-20"

# 5) 可选：如果你环境需要代理（按需取消注释并填写）
# export HTTPS_PROXY="http://127.0.0.1:7890"
# export HTTP_PROXY="http://127.0.0.1:7890"
# export ALL_PROXY="socks5://127.0.0.1:7890"

python /home/wenxuexiang/projects/VIGOR/upload_model.py \
  --repo-type model \
  --enable-hf-transfer
