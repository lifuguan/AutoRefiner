#!/usr/bin/env python3
"""
优化mask匹配参数以减少invalid区域
"""

import os
import numpy as np
from PIL import Image
from mask_refiner_matching import SAM2MaskMatcher


def test_parameters(img_path, mask_path, model_cfg, checkpoint):
    """
    测试不同参数组合的效果
    """

    # 读取图像
    image = Image.open(img_path).convert('RGB')
    image = np.array(image)

    # 参数组合
    param_combinations = [
        # (iou_threshold, pred_iou_thresh, stability_score_thresh, points_per_side)
        (0.2, 0.6, 0.8, 32),   # 更宽松的匹配
        (0.3, 0.7, 0.85, 32),  # 当前参数
        (0.1, 0.5, 0.75, 32),  # 非常宽松
        (0.2, 0.6, 0.8, 64),   # 更多采样点
        (0.1, 0.4, 0.7, 64),   # 最宽松 + 更多点
    ]

    best_invalid_ratio = float('inf')
    best_params = None
    best_results = None

    print("=== 参数优化测试 ===")

    for i, (iou_thresh, pred_iou_thresh, stability_thresh, points_per_side) in enumerate(param_combinations):
        print(f"\n测试组合 {i+1}: IoU阈值={iou_thresh}, 预测IoU阈值={pred_iou_thresh}, "
              f"稳定性阈值={stability_thresh}, 采样点={points_per_side}")

        try:
            # 初始化匹配器
            matcher = SAM2MaskMatcher(
                model_cfg=model_cfg,
                checkpoint=checkpoint,
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_thresh,
                box_nms_thresh=0.7,
                iou_threshold=iou_thresh,
                min_mask_region_area=50,  # 降低最小区域面积
            )

            # 处理图像
            matched_masks = matcher.process_image_with_ground_truth(image, mask_path)

            # 计算统计信息
            gt_mask_data, valid_ids = matcher.load_ground_truth_masks(mask_path)
            invalid_ratio = matcher.calculate_invalid_area_ratio(matched_masks, gt_mask_data)

            sam_matched = sum(1 for m in matched_masks if not m.get('is_fallback', False))
            fallback_count = len(matched_masks) - sam_matched

            sam_ious = [m['matching_iou'] for m in matched_masks if not m.get('is_fallback', False)]
            avg_iou = np.mean(sam_ious) if sam_ious else 0.0

            print(f"  结果: SAM2匹配={sam_matched}, 真值填充={fallback_count}, "
                  f"Invalid比例={invalid_ratio:.3f} ({invalid_ratio*100:.1f}%), 平均IoU={avg_iou:.3f}")

            # 更新最佳结果
            if invalid_ratio < best_invalid_ratio:
                best_invalid_ratio = invalid_ratio
                best_params = (iou_thresh, pred_iou_thresh, stability_thresh, points_per_side)
                best_results = {
                    'sam_matched': sam_matched,
                    'fallback_count': fallback_count,
                    'invalid_ratio': invalid_ratio,
                    'avg_iou': avg_iou,
                    'matched_masks': matched_masks
                }

        except Exception as e:
            print(f"  错误: {e}")
            continue

    print(f"\n=== 最佳参数组合 ===")
    if best_params:
        iou_thresh, pred_iou_thresh, stability_thresh, points_per_side = best_params
        print(f"IoU阈值: {iou_thresh}")
        print(f"预测IoU阈值: {pred_iou_thresh}")
        print(f"稳定性阈值: {stability_thresh}")
        print(f"采样点数: {points_per_side}")
        print(f"SAM2匹配: {best_results['sam_matched']} 个")
        print(f"真值填充: {best_results['fallback_count']} 个")
        print(f"Invalid区域比例: {best_results['invalid_ratio']:.3f} ({best_results['invalid_ratio']*100:.1f}%)")
        print(f"平均IoU: {best_results['avg_iou']:.3f}")

        # 保存最佳结果
        matcher = SAM2MaskMatcher(
            model_cfg=model_cfg,
            checkpoint=checkpoint,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_thresh,
            box_nms_thresh=0.7,
            iou_threshold=iou_thresh,
            min_mask_region_area=50,
        )

        matcher.visualize_results(image, best_results['matched_masks'], "optimized_mask_matching_results.png")
        matcher.save_matched_masks(best_results['matched_masks'], "optimized_matched_masks_output.npz")
        print("\n最佳结果已保存到 optimized_mask_matching_results.png 和 optimized_matched_masks_output.npz")

    else:
        print("未找到有效的参数组合")


def main():
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

    # 运行参数优化
    test_parameters(img_path, mask_path, model_cfg, checkpoint)


if __name__ == "__main__":
    main()
