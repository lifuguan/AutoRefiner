#!/usr/bin/env python3
"""
演示mask合并策略的效果
"""

import os
import numpy as np
from PIL import Image
from mask_refiner_matching import SAM2MaskMatcher


def demo_mask_merging():
    """
    演示mask合并策略如何处理割裂的mask
    """

    print("=== SAM2 Mask合并策略演示 ===")
    print()
    print("问题描述：")
    print("- 墙壁等大型连续区域可能被SAM2分割成多个部分")
    print("- 某些部分的IoU可能低于阈值而被忽略")
    print("- 这导致invalid区域增加，匹配效果下降")
    print()
    print("解决方案：")
    print("1. 多轮匹配策略")
    print("2. 合并多个SAM mask来提高IoU")
    print("3. 将invalid区域的mask与已匹配mask合并")
    print("4. 基于空间邻接性的智能合并")
    print()

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
    
    # img_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/mm_datasets/lihao_3d_data/processed_scannetpp_fix/0cf2e9402d/images/frame_000000.jpg"
    # mask_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/mm_datasets/lihao_3d_data/processed_scannetpp_fix/0cf2e9402d/obj_ids/frame_000000.jpg.pth"
    img_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/mm_datasets/lihao_3d_data/processed_scannetpp_fix/036bce3393/images/frame_000000.jpg"
    mask_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/mm_datasets/lihao_3d_data/processed_scannetpp_fix/036bce3393/obj_ids/frame_000000.jpg.pth"

    # 初始化匹配器
    matcher = SAM2MaskMatcher(
        model_cfg=model_cfg,
        checkpoint=checkpoint,
        points_per_side=32,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.5,
        box_nms_thresh=0.7,
        iou_threshold=0.3,  # 匹配阈值
    )

    # 读取图像
    image = Image.open(img_path).convert('RGB')
    image = np.array(image)

    print("\n=== 开始处理 ===")

    # 自动应用所有合并策略
    matched_masks = matcher.process_image_with_ground_truth(image, mask_path)

    # 查看合并结果
    for mask in matched_masks:
        if mask.get('is_merged', False):
            print(f"ID {mask['ground_truth_id']}: 合并了 {mask['num_merged_masks']} 个mask")

    # 加载真值mask数据
    gt_mask_data, valid_ids = matcher.load_ground_truth_masks(mask_path)

    # 计算统计信息
    invalid_ratio = matcher.calculate_invalid_area_ratio(matched_masks, gt_mask_data)
    sam_matched = sum(1 for m in matched_masks if not m.get('is_fallback', False))
    fallback_count = len(matched_masks) - sam_matched
    merged_count = sum(1 for m in matched_masks if m.get('is_merged', False))

    print(f"\n=== 合并策略效果 ===")
    print(f"总对象数: {len(matched_masks)}")
    print(f"SAM2成功匹配: {sam_matched} ({sam_matched/len(matched_masks)*100:.1f}%)")
    print(f"其中合并mask: {merged_count} ({merged_count/len(matched_masks)*100:.1f}%)")
    print(f"真值填充: {fallback_count} ({fallback_count/len(matched_masks)*100:.1f}%)")
    print(f"Invalid区域比例: {invalid_ratio:.3f} ({invalid_ratio*100:.1f}%)")

    # 计算平均IoU
    sam_ious = [m['matching_iou'] for m in matched_masks if not m.get('is_fallback', False)]
    if sam_ious:
        avg_iou = np.mean(sam_ious)
        print(f"SAM2匹配的平均IoU: {avg_iou:.3f}")

    print(f"\n=== 详细合并信息 ===")
    for mask_info in matched_masks:
        gt_id = mask_info['ground_truth_id']
        matching_iou = mask_info['matching_iou']
        is_fallback = mask_info.get('is_fallback', False)
        is_merged = mask_info.get('is_merged', False)
        num_merged = mask_info.get('num_merged_masks', 1)

        if is_merged:
            print(f"  ID {gt_id}: 合并mask (IoU: {matching_iou:.3f}, 合并了 {num_merged} 个SAM mask)")
        elif is_fallback:
            print(f"  ID {gt_id}: 使用真值填充")
        else:
            print(f"  ID {gt_id}: 单个SAM mask匹配 (IoU: {matching_iou:.3f})")

    # 生成可视化
    matcher.visualize_results(image, matched_masks, gt_mask_data, "demo_mask_merging.png")

    print(f"\n=== 合并策略优势 ===")
    print("1. 减少invalid区域：通过合并分割的mask部分")
    print("2. 提高IoU：组合多个低IoU的mask获得更高的整体IoU")
    print("3. 空间连续性：基于邻接性的智能合并")
    print("4. 自适应阈值：对不同情况使用不同的合并策略")
    print()
    print("可视化结果已保存到: demo_mask_merging.png")
    print("查看图像中的 '+N' 标记表示合并了N个SAM mask的结果！")


if __name__ == "__main__":
    demo_mask_merging()
