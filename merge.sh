#!/usr/bin/env bash
# 在仓库根目录，已激活 .venv_lighteval
set -euo pipefail
# 串行合并，每次只处理一个 checkpoint；merge.py 强制使用 CPU（device-map=cpu）
#
# 全局互斥（排队）：
# - 默认启用（MERGE_GLOBAL_LOCK=1）
# - 多个不同的 merge.sh 只要使用同一个 MERGE_GLOBAL_LOCK_FILE，就会全局最多 1 个在跑
MERGE_GLOBAL_LOCK="${MERGE_GLOBAL_LOCK:-1}"
MERGE_GLOBAL_LOCK_FILE="${MERGE_GLOBAL_LOCK_FILE:-/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/.merge_global.lock}"
BASE="${BASE:-/run/determined/NAS1/public/xuexiang/model/Qwen2.5-7B}"
RUN_ROOT="${RUN_ROOT:-/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-GRPO-7B_20251211_165630}"
CKPT_ROOT="${CKPT_ROOT:-${RUN_ROOT}/ckpt}"
OUT_ROOT="${OUT_ROOT:-${RUN_ROOT}/merged}"
PATTERN="${PATTERN:-checkpoint}"
cd /home/wenxuexiang/projects/Intuitor/open-r1-intuitor
source /home/wenxuexiang/projects/Intuitor/open-r1-intuitor/openr1_intuitor/bin/activate

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${OUT_ROOT}"

if [[ "${MERGE_GLOBAL_LOCK}" == "1" ]]; then
    if command -v flock >/dev/null 2>&1; then
        echo "Waiting for global merge lock: ${MERGE_GLOBAL_LOCK_FILE}"
        exec 9>"${MERGE_GLOBAL_LOCK_FILE}"
        flock -x 9
        echo "Acquired global merge lock: ${MERGE_GLOBAL_LOCK_FILE}"
    else
        echo "Error: 'flock' not found; cannot enforce global merge lock." >&2
        exit 2
    fi
fi

merged_any=false

for ckpt in "${CKPT_ROOT}"/*; do
    [[ -d "${ckpt}" ]] || continue
    name="$(basename "${ckpt}")"
    [[ "${name}" == *"${PATTERN}"* ]] || continue
    merged_any=true
    out="${OUT_ROOT}/${name}"
    if [[ -d "${out}" ]]; then
        echo "Skip ${ckpt}, output already exists at ${out}"
        continue
    fi
    echo "Merging ${ckpt} -> ${out} on CPU"
    python "${SCRIPT_DIR}/merge.py" --device-map cpu --base "${BASE}" --ckpt "${ckpt}" --out "${out}"
done

if [[ "${merged_any}" == false ]]; then
    echo "No checkpoint folders matching pattern '${PATTERN}' found under ${CKPT_ROOT}"
fi
