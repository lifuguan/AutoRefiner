#!/bin/bash

# 批量SAM2 Mask匹配处理启动脚本

# 设置环境
export PATH="/home/sankuai/conda/envs/vggt/bin:$PATH"
source activate vggt

# 设置CUDA环境（如果需要）
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 数据根目录
DATA_ROOT="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/mm_datasets/lihao_3d_data/processed_scannetpp_fix/"

# 运行批量处理
echo "=== 开始批量SAM2 Mask匹配处理 ==="
echo "数据目录: $DATA_ROOT"
echo "使用环境: vggt"
echo "开始时间: $(date)"

python batch_mask_refining.py \
    --data_root "$DATA_ROOT" \
    --model_cfg "sam2_hiera_l.yaml" \
    --checkpoint "model_zoo/sam/sam2_hiera_large.pt"

echo "结束时间: $(date)"
echo "=== 批量处理完成 ==="
