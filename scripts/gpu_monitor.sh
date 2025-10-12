#!/bin/bash

# GPU监控脚本
# 用法: ./gpu_monitor.sh [间隔秒数] [日志文件名] [日志目录]
# 默认值: 间隔5秒，日志文件名为gpu_usage.log，日志目录为当前目录

# 设置默认值
INTERVAL=${1:-5}
LOGFILE_PREFIX=${2:-"gpu_usage"}
LOG_DIR=${3:-"."}

# 生成带时间戳的日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="${LOG_DIR}/${LOGFILE_PREFIX}_${TIMESTAMP}.log"

# GPU监控函数
monitor_gpu() {
    echo "时间, GPU, 显存使用(MiB), 总显存(MiB), GPU利用率(%)" > "${LOGFILE}"
    echo "GPU监控已启动，间隔: ${INTERVAL}秒，日志文件: ${LOGFILE}"
    
    while true; do
        TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
        echo "------ ${TIMESTAMP} ------" >> "${LOGFILE}"
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
                   --format=csv,noheader,nounits | \
        while read line; do
            echo "${TIMESTAMP},${line}" >> "${LOGFILE}"
        done
        # 在两次记录之间加分隔线
        echo "--------------------------------------------" >> "${LOGFILE}"
        echo "" >> "${LOGFILE}"
        sleep "${INTERVAL}"
    done
}

# 启动监控
monitor_gpu
