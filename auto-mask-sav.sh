#!/bin/bash

# 获取 GPU 数量
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $gpu_count GPU(s)."

# 输入文件路径
video_list="/mnt/juicefs/datasets/re10k/paths/paths.txt"

# 设置日志文件路径（用户指定目录）
log_dir="/mnt/juicefs/datasets/re10k/semantic"
mkdir -p "$log_dir"  # 确保目录存在
LOG_FILE="${log_dir}/processing.log"
echo "Log file will be saved to: $LOG_FILE"

# 读取文件列表到数组
mapfile -t video_paths < "$video_list"
total_tasks=${#video_paths[@]}
echo "Found $total_tasks video files to process"

# 并发处理函数
process_video() {
    local video_path="$1"
    local gpu_id="$2"
    
    # 解析输出路径
    local output_dir="/mnt/juicefs/datasets/re10k/semantic"
    
    export CUDA_VISIBLE_DEVICES=$gpu_id
    mkdir -p "$output_dir"
    
    echo "[GPU $gpu_id] Processing: $video_path"
    
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export OPENBLAS_NUM_THREADS=4
    # 运行处理命令并捕获返回状态
    python ../auto-mask-align-sav.py \
        --video_path "$video_path" \
        --output_dir "$output_dir" \
        --level large \
        
    local ret=$?
    
    # 获取时间戳
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    
    # 记录处理结果到日志文件
    if [ $ret -eq 0 ]; then
        status="Success"
    else
        status="Failed"
    fi
    echo "[$timestamp] [GPU $gpu_id] $status: $video_path" >> "$LOG_FILE"
    
    # 控制台输出
    if [ $ret -eq 0 ]; then
        echo "[GPU $gpu_id] Success: $video_path"
    else
        echo "[GPU $gpu_id] Failed: $video_path"
    fi
}

# 并行任务调度
task_index=0
while [ $task_index -lt $total_tasks ]; do
    # 为每个GPU分配任务
    for ((gpu=0; gpu<gpu_count && task_index<total_tasks; gpu++)); do
        video_path="${video_paths[$task_index]}"
        process_video "$video_path" $gpu &
        ((task_index++))
    done
    wait  # 等待当前批次任务完成
done

echo "All video processing completed!"
echo "Log file saved to: $LOG_FILE"
