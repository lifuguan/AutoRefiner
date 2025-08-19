#!/usr/bin/env python3
"""
演示改进的可视化功能
"""

import os
import numpy as np
from PIL import Image
from mask_refiner_matching import SAM2MaskMatcher


def demo_visualization():
    """
    演示同时展示真值mask和匹配后的mask的可视化功能
    """

    # 配置参数
    model_cfg = "sam2_hiera_l.yaml"

    # 尝试不同的模型路径
    possible_checkpoints = [
        "/data/model_zoo/sam/sam2_hiera_large.pt",
        "checkpoints/sam2/sam2_hiera_large.pt",
        "model_zoo/sam/sam2_hiera_large.pt"
    ]

    checkpoint = None
    for ckpt_path in possible_checkpoints:
        if os.path.exists(ckpt_path):
            checkpoint = ckpt_path
            break

    if checkpoint is None:
        print("错误：找不到SAM2模型文件")
        return

    print(f"使用模型文件: {checkpoint}")

    # 示例文件路径
    img_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/mm_datasets/lihao_3d_data/processed_scannetpp_fix/0cf2e9402d/images/frame_000130.jpg"
    mask_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/mm_datasets/lihao_3d_data/processed_scannetpp_fix/0cf2e9402d/obj_ids/frame_000130.jpg.pth"

    # 初始化匹配器
    matcher = SAM2MaskMatcher(
        model_cfg=model_cfg,
        checkpoint=checkpoint,
        points_per_side=32,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.85,
        box_nms_thresh=0.7,
        iou_threshold=0.3,
    )

    # 读取图像
    image = Image.open(img_path).convert('RGB')
    image = np.array(image)

    # 处理图像并匹配mask
    matched_masks = matcher.process_image_with_ground_truth(image, mask_path)

    # 加载真值mask数据
    gt_mask_data, valid_ids = matcher.load_ground_truth_masks(mask_path)

    print(f"\n=== 可视化功能演示 ===")
    print("生成的可视化包含三列：")
    print("1. 原始图像 - 输入的RGB图像")
    print("2. 真值mask - 从mask_path加载的标注")
    print("3. 匹配后的mask - SAM2生成并匹配到真值ID的mask")
    print()
    print("颜色编码：")
    print("- 每个对象ID使用一致的颜色")
    print("- 真值mask和匹配mask使用相同颜色便于对比")
    print("- 标签显示：ID:XX (IoU:0.XX) 或 ID:XX (GT) 表示fallback")
    print("- 低置信度匹配标记为 ID:XX (IoU:0.XX*)")

    # 生成可视化
    matcher.visualize_results(image, matched_masks, gt_mask_data, "demo_visualization.png")

    # 计算统计信息
    invalid_ratio = matcher.calculate_invalid_area_ratio(matched_masks, gt_mask_data)
    sam_matched = sum(1 for m in matched_masks if not m.get('is_fallback', False))
    fallback_count = len(matched_masks) - sam_matched

    print(f"\n=== 匹配统计 ===")
    print(f"总对象数: {len(matched_masks)}")
    print(f"SAM2成功匹配: {sam_matched} ({sam_matched/len(matched_masks)*100:.1f}%)")
    print(f"真值填充: {fallback_count} ({fallback_count/len(matched_masks)*100:.1f}%)")
    print(f"Invalid区域比例: {invalid_ratio:.3f} ({invalid_ratio*100:.1f}%)")

    print(f"\n可视化结果已保存到: demo_visualization.png")
    print("请查看图像以比较真值mask和匹配后的mask的效果！")


if __name__ == "__main__":
    demo_visualization()
