#!/usr/bin/env python3
"""
调试单张图像处理
"""

import torch
import numpy as np
from PIL import Image
import os
import traceback
from mask_refiner_matching import SAM2MaskMatcher

def debug_single_image():
    """调试单张图像处理"""

    # 测试文件路径
    img_path = '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/mm_datasets/lihao_3d_data/processed_scannetpp_fix/036bce3393/images/frame_000000.jpg'
    mask_path = '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/mm_datasets/lihao_3d_data/processed_scannetpp_fix/036bce3393/obj_ids/frame_000000.jpg.pth'

    print("=== 调试单张图像处理 ===")

    # 1. 检查文件是否存在
    print(f"图像文件存在: {os.path.exists(img_path)}")
    print(f"Mask文件存在: {os.path.exists(mask_path)}")

    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        print("文件不存在，退出调试")
        return

    try:
        # 2. 测试图像读取
        print("\n--- 测试图像读取 ---")
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        print(f"图像尺寸: {image.shape}")
        print(f"图像数据类型: {image.dtype}")

        # 3. 测试mask读取
        print("\n--- 测试mask读取 ---")
        mask_data = torch.load(mask_path, weights_only=False)
        print(f"Mask原始类型: {type(mask_data)}")

        if isinstance(mask_data, torch.Tensor):
            mask_data = mask_data.cpu().numpy()

        print(f"Mask尺寸: {mask_data.shape}")
        print(f"Mask数据类型: {mask_data.dtype}")
        unique_ids = np.unique(mask_data)
        print(f"Mask唯一值数量: {len(unique_ids)}")
        print(f"Mask唯一值前10个: {unique_ids[:10]}")

        # 4. 测试SAM2MaskMatcher初始化
        print("\n--- 测试SAM2MaskMatcher初始化 ---")

        # 检查模型文件
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

        # 初始化匹配器
        matcher = SAM2MaskMatcher(
            model_cfg="sam2_hiera_l.yaml",
            checkpoint=checkpoint,
            points_per_side=32,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.5,
            box_nms_thresh=0.7,
            iou_threshold=0.3,
        )
        print("SAM2MaskMatcher初始化成功")

        # 5. 测试处理单张图像
        print("\n--- 测试处理单张图像 ---")
        matched_masks = matcher.process_image_with_ground_truth(image, mask_path)

        print(f"匹配结果数量: {len(matched_masks)}")

        # 6. 测试保存
        print("\n--- 测试保存 ---")
        gt_mask_data, valid_ids = matcher.load_ground_truth_masks(mask_path)

        # 创建输出mask（uint16格式）
        output_mask = np.zeros_like(gt_mask_data, dtype=np.uint16)

        for mask_info in matched_masks:
            gt_id = mask_info['ground_truth_id']
            mask = mask_info['segmentation'].astype(bool)
            output_mask[mask] = gt_id

        # 保存测试（转换为int32以兼容PyTorch保存）
        output_path = "debug_output_mask.pth"
        torch.save(torch.from_numpy(output_mask.astype(np.int32)), output_path)
        print(f"保存成功: {output_path}")

        # 统计信息
        invalid_ratio = matcher.calculate_invalid_area_ratio(matched_masks, gt_mask_data)
        sam_matched = sum(1 for m in matched_masks if not m.get('is_fallback', False))
        merged_count = sum(1 for m in matched_masks if m.get('is_merged', False))

        print(f"\n--- 处理统计 ---")
        print(f"总对象数: {len(matched_masks)}")
        print(f"SAM2匹配: {sam_matched}")
        print(f"合并mask: {merged_count}")
        print(f"Invalid比例: {invalid_ratio:.3f}")

        print("\n✅ 单张图像处理成功！")

    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_single_image()
