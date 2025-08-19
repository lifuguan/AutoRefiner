import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple
from scipy.optimize import linear_sum_assignment
import cv2
import os


class SAM2MaskMatcher:
    """
    基于SAM2的mask匹配器
    从mask_path读取真值标注，使用SAM2AutomaticMaskGenerator生成mask，
    并将SAM2的mask匹配到真值ID上，确保最终得到的mask数量和ID与真值匹配
    """

    def __init__(
        self,
        model_cfg: str,
        checkpoint: str,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.95,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        min_mask_region_area: int = 100,
        iou_threshold: float = 0.5,
        **kwargs,
    ) -> None:
        """
        初始化SAM2 Mask Matcher

        Arguments:
          model_cfg (str): SAM2模型配置文件路径
          checkpoint (str): SAM2模型权重文件路径
          points_per_side (int): 每边采样点数
          points_per_batch (int): 每批处理的点数
          pred_iou_thresh (float): IoU阈值，用于过滤低质量mask
          stability_score_thresh (float): 稳定性分数阈值
          box_nms_thresh (float): NMS的IoU阈值
          crop_n_layers (int): 裁剪层数
          min_mask_region_area (int): 最小mask区域面积
          iou_threshold (float): 匹配时的IoU阈值
        """

        # 初始化SAM2模型
        sam_model = build_sam2(model_cfg, checkpoint)

        # 初始化SAM2AutomaticMaskGenerator
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=sam_model,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            box_nms_thresh=box_nms_thresh,
            crop_n_layers=crop_n_layers,
            min_mask_region_area=min_mask_region_area,
            output_mode="binary_mask",
            **kwargs,
        )

        self.iou_threshold = iou_threshold

    def load_ground_truth_masks(self, mask_path: str) -> Tuple[np.ndarray, List[int]]:
        """
        从mask_path加载真值标注

        Arguments:
          mask_path (str): mask文件路径，支持.pth, .png, .jpg等格式

        Returns:
          mask_data (np.ndarray): mask标注数据，每个像素值代表不同的对象ID
          valid_ids (List[int]): 有效的对象ID列表
        """

        if mask_path.endswith('.pth'):
            # 加载PyTorch tensor格式的mask
            mask_data = torch.load(mask_path, weights_only=False)
            if isinstance(mask_data, torch.Tensor):
                mask_data = mask_data.cpu().numpy()
        elif mask_path.endswith(('.png', '.jpg', '.jpeg')):
            # 加载图像格式的mask
            mask_img = Image.open(mask_path)
            mask_data = np.array(mask_img)
            # 如果是RGB图像，转换为单通道
            if len(mask_data.shape) == 3:
                mask_data = mask_data[:, :, 0]  # 取第一个通道
        else:
            raise ValueError(f"Unsupported mask format: {mask_path}")

        # 提取有效的对象ID
        unique_ids = np.unique(mask_data)
        valid_ids = unique_ids[unique_ids != 0].tolist()  # 跳过背景ID (0)

        return mask_data, valid_ids

    def calculate_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        计算两个mask之间的IoU

        Arguments:
          mask1, mask2 (np.ndarray): 二值mask

        Returns:
          iou (float): IoU值
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        if union == 0:
            return 0.0

        return intersection / union

    def match_masks_to_ground_truth(
        self,
        sam_masks: List[Dict[str, Any]],
        gt_mask_data: np.ndarray,
        valid_ids: List[int]
    ) -> List[Dict[str, Any]]:
        """
        将SAM2生成的mask匹配到真值ID上，优化匹配策略以减少invalid区域

        Arguments:
          sam_masks (List[Dict]): SAM2生成的mask列表
          gt_mask_data (np.ndarray): 真值mask数据
          valid_ids (List[int]): 有效的真值ID列表

        Returns:
          matched_masks (List[Dict]): 匹配后的mask列表，包含真值ID
        """

        if len(sam_masks) == 0 or len(valid_ids) == 0:
            return []

        # 提取真值mask
        gt_masks = []
        for obj_id in valid_ids:
            gt_mask = (gt_mask_data == obj_id).astype(np.uint8)
            gt_masks.append(gt_mask)

        # 提取SAM2 mask
        sam_binary_masks = []
        for sam_mask in sam_masks:
            sam_binary_masks.append(sam_mask['segmentation'].astype(np.uint8))

        # 计算IoU矩阵和覆盖率矩阵
        iou_matrix = np.zeros((len(sam_binary_masks), len(gt_masks)))
        coverage_matrix = np.zeros((len(sam_binary_masks), len(gt_masks)))  # SAM mask对GT mask的覆盖率

        for i, sam_mask in enumerate(sam_binary_masks):
            for j, gt_mask in enumerate(gt_masks):
                iou_score = self.calculate_iou(sam_mask, gt_mask)
                iou_matrix[i, j] = iou_score

                # 计算覆盖率：SAM mask覆盖GT mask的比例
                if gt_mask.sum() > 0:
                    intersection = np.logical_and(sam_mask, gt_mask).sum()
                    coverage_matrix[i, j] = intersection / gt_mask.sum()
                else:
                    coverage_matrix[i, j] = 0.0

        # 多轮匹配策略：先高IoU匹配，再低IoU但高覆盖率匹配
        matched_masks = []
        used_sam_indices = set()
        used_gt_ids = set()

        # 第一轮：高IoU匹配（IoU >= self.iou_threshold）
        high_iou_pairs = []
        for i in range(len(sam_binary_masks)):
            for j in range(len(gt_masks)):
                if iou_matrix[i, j] >= self.iou_threshold:
                    high_iou_pairs.append((i, j, iou_matrix[i, j]))

        # 按IoU降序排序
        high_iou_pairs.sort(key=lambda x: x[2], reverse=True)

        for sam_idx, gt_idx, iou_score in high_iou_pairs:
            if sam_idx not in used_sam_indices and valid_ids[gt_idx] not in used_gt_ids:
                matched_mask = sam_masks[sam_idx].copy()
                matched_mask['ground_truth_id'] = valid_ids[gt_idx]
                matched_mask['matching_iou'] = iou_score
                matched_mask['coverage_ratio'] = coverage_matrix[sam_idx, gt_idx]
                matched_masks.append(matched_mask)
                used_sam_indices.add(sam_idx)
                used_gt_ids.add(valid_ids[gt_idx])

        # 第二轮：对于未匹配的GT，寻找覆盖率较高的SAM mask（即使IoU较低）
        low_iou_threshold = max(0.1, self.iou_threshold * 0.3)  # 降低的IoU阈值
        min_coverage = 0.3  # 最小覆盖率要求

        for gt_idx, obj_id in enumerate(valid_ids):
            if obj_id not in used_gt_ids:
                # 寻找最佳的未使用SAM mask
                best_sam_idx = -1
                best_score = 0.0

                for sam_idx in range(len(sam_binary_masks)):
                    if sam_idx not in used_sam_indices:
                        iou_score = iou_matrix[sam_idx, gt_idx]
                        coverage_score = coverage_matrix[sam_idx, gt_idx]

                        # 综合评分：IoU + 覆盖率权重
                        combined_score = iou_score * 0.6 + coverage_score * 0.4

                        if (iou_score >= low_iou_threshold and coverage_score >= min_coverage and
                            combined_score > best_score):
                            best_score = combined_score
                            best_sam_idx = sam_idx

                if best_sam_idx != -1:
                    matched_mask = sam_masks[best_sam_idx].copy()
                    matched_mask['ground_truth_id'] = obj_id
                    matched_mask['matching_iou'] = iou_matrix[best_sam_idx, gt_idx]
                    matched_mask['coverage_ratio'] = coverage_matrix[best_sam_idx, gt_idx]
                    matched_mask['is_low_confidence'] = True  # 标记为低置信度匹配
                    matched_masks.append(matched_mask)
                    used_sam_indices.add(best_sam_idx)
                    used_gt_ids.add(obj_id)

        # 第三轮：对于仍未匹配的GT，尝试组合多个SAM mask
        for gt_idx, obj_id in enumerate(valid_ids):
            if obj_id not in used_gt_ids:
                gt_mask = gt_masks[gt_idx]

                # 寻找所有与该GT有重叠的未使用SAM mask
                candidate_masks = []
                for sam_idx in range(len(sam_binary_masks)):
                    if sam_idx not in used_sam_indices:
                        if iou_matrix[sam_idx, gt_idx] > 0.05:  # 有一定重叠
                            candidate_masks.append((sam_idx, iou_matrix[sam_idx, gt_idx],
                                                  coverage_matrix[sam_idx, gt_idx]))

                if candidate_masks:
                    # 选择覆盖率最高的mask
                    candidate_masks.sort(key=lambda x: x[2], reverse=True)
                    best_sam_idx, best_iou, best_coverage = candidate_masks[0]

                    if best_coverage >= 0.2:  # 至少20%覆盖率
                        matched_mask = sam_masks[best_sam_idx].copy()
                        matched_mask['ground_truth_id'] = obj_id
                        matched_mask['matching_iou'] = best_iou
                        matched_mask['coverage_ratio'] = best_coverage
                        matched_mask['is_low_confidence'] = True
                        matched_masks.append(matched_mask)
                        used_sam_indices.add(best_sam_idx)
                        used_gt_ids.add(obj_id)

        # 第四轮：对于仍未匹配的GT，尝试合并多个SAM mask来提高IoU
        for gt_idx, obj_id in enumerate(valid_ids):
            if obj_id not in used_gt_ids:
                gt_mask = gt_masks[gt_idx]

                # 寻找所有与该GT有重叠的未使用SAM mask
                candidate_masks = []
                for sam_idx in range(len(sam_binary_masks)):
                    if sam_idx not in used_sam_indices:
                        if iou_matrix[sam_idx, gt_idx] > 0.01:  # 有任何重叠
                            candidate_masks.append((sam_idx, iou_matrix[sam_idx, gt_idx],
                                                  coverage_matrix[sam_idx, gt_idx]))

                if len(candidate_masks) >= 2:  # 至少有两个候选mask可以合并
                    # 尝试不同的mask组合
                    best_merged_mask = None
                    best_merged_iou = 0.0
                    best_used_indices = []

                    # 按覆盖率排序，优先考虑覆盖率高的mask
                    candidate_masks.sort(key=lambda x: x[2], reverse=True)

                    # 尝试逐步添加mask进行合并
                    for num_masks in range(2, min(len(candidate_masks) + 1, 5)):  # 最多合并5个mask
                        for start_idx in range(len(candidate_masks) - num_masks + 1):
                            selected_candidates = candidate_masks[start_idx:start_idx + num_masks]

                            # 合并这些mask
                            merged_mask = np.zeros_like(gt_mask, dtype=bool)
                            used_indices = []

                            for sam_idx, _, _ in selected_candidates:
                                merged_mask |= sam_binary_masks[sam_idx].astype(bool)
                                used_indices.append(sam_idx)

                            # 计算合并后的IoU
                            merged_iou = self.calculate_iou(merged_mask.astype(np.uint8), gt_mask)

                            # 如果合并后的IoU更好，更新最佳结果
                            if merged_iou > best_merged_iou and merged_iou >= max(0.2, self.iou_threshold * 0.5):
                                best_merged_iou = merged_iou
                                best_merged_mask = merged_mask.astype(np.uint8)
                                best_used_indices = used_indices.copy()

                    # 如果找到了好的合并结果
                    if best_merged_mask is not None and best_merged_iou > 0.2:
                        # 计算合并后的覆盖率
                        merged_coverage = np.logical_and(best_merged_mask, gt_mask).sum() / gt_mask.sum() if gt_mask.sum() > 0 else 0.0

                        # 创建合并后的mask信息
                        merged_mask_info = {
                            'segmentation': best_merged_mask,
                            'area': int(best_merged_mask.sum()),
                            'bbox': self._mask_to_bbox(best_merged_mask),
                            'predicted_iou': best_merged_iou,  # 使用合并后的IoU
                            'stability_score': 0.5,  # 合并mask的稳定性分数设为中等
                            'point_coords': [[0, 0]],
                            'crop_box': [0, 0, best_merged_mask.shape[1], best_merged_mask.shape[0]],
                            'ground_truth_id': obj_id,
                            'matching_iou': best_merged_iou,
                            'coverage_ratio': merged_coverage,
                            'is_merged': True,  # 标记为合并mask
                            'merged_from_indices': best_used_indices,  # 记录合并来源
                            'num_merged_masks': len(best_used_indices)
                        }

                        matched_masks.append(merged_mask_info)

                        # 标记使用的SAM mask索引
                        for idx in best_used_indices:
                            used_sam_indices.add(idx)
                        used_gt_ids.add(obj_id)

                        print(f"  合并策略: ID {obj_id} 通过合并 {len(best_used_indices)} 个SAM mask获得IoU {best_merged_iou:.3f}")

        # 第五轮：尝试将invalid区域的mask与已匹配的mask合并来提高IoU
        print("  尝试invalid区域mask合并...")
        self._try_merge_invalid_masks(matched_masks, sam_binary_masks, gt_masks, valid_ids,
                                    iou_matrix, coverage_matrix, used_sam_indices, gt_mask_data)

        # 最后：对于仍然没有匹配到的真值ID，使用真值mask作为fallback
        for obj_id in valid_ids:
            if obj_id not in used_gt_ids:
                gt_mask = (gt_mask_data == obj_id).astype(np.uint8)

                fallback_mask = {
                    'segmentation': gt_mask,
                    'area': int(gt_mask.sum()),
                    'bbox': self._mask_to_bbox(gt_mask),
                    'predicted_iou': 0.0,
                    'stability_score': 0.0,
                    'point_coords': [[0, 0]],
                    'crop_box': [0, 0, gt_mask.shape[1], gt_mask.shape[0]],
                    'ground_truth_id': obj_id,
                    'matching_iou': 0.0,
                    'coverage_ratio': 1.0,  # 真值mask完全覆盖自己
                    'is_fallback': True
                }
                matched_masks.append(fallback_mask)

        # 按照真值ID排序，确保输出顺序一致
        matched_masks.sort(key=lambda x: x['ground_truth_id'])

        return matched_masks

    def _mask_to_bbox(self, mask: np.ndarray) -> List[float]:
        """
        从mask计算边界框 (XYWH格式)

        Arguments:
          mask (np.ndarray): 二值mask

        Returns:
          bbox (List[float]): [x, y, width, height]
        """
        if mask.sum() == 0:
            return [0.0, 0.0, 0.0, 0.0]

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        return [float(xmin), float(ymin), float(xmax - xmin + 1), float(ymax - ymin + 1)]

    def _try_merge_invalid_masks(
        self,
        matched_masks: List[Dict[str, Any]],
        sam_binary_masks: List[np.ndarray],
        gt_masks: List[np.ndarray],
        valid_ids: List[int],
        iou_matrix: np.ndarray,
        coverage_matrix: np.ndarray,
        used_sam_indices: set,
        gt_mask_data: np.ndarray
    ) -> None:
        """
        尝试将invalid区域的SAM mask与已匹配的mask合并来提高IoU

        Arguments:
          matched_masks: 已匹配的mask列表（会被就地修改）
          sam_binary_masks: SAM2生成的二值mask列表
          gt_masks: 真值mask列表
          valid_ids: 有效的真值ID列表
          iou_matrix: IoU矩阵
          coverage_matrix: 覆盖率矩阵
          used_sam_indices: 已使用的SAM mask索引集合
          gt_mask_data: 真值mask数据
        """

        # 找到所有未使用的SAM mask（invalid区域的mask）
        unused_sam_indices = []
        for i in range(len(sam_binary_masks)):
            if i not in used_sam_indices:
                unused_sam_indices.append(i)

        if not unused_sam_indices:
            print("    没有未使用的SAM mask可以合并")
            return

        print(f"    发现 {len(unused_sam_indices)} 个未使用的SAM mask，尝试合并...")

        # 对每个已匹配的mask，尝试与未使用的mask合并
        improved_count = 0
        for i, matched_mask in enumerate(matched_masks):
            if matched_mask.get('is_fallback', False):
                continue  # 跳过fallback mask

            gt_id = matched_mask['ground_truth_id']
            gt_idx = valid_ids.index(gt_id)
            gt_mask = gt_masks[gt_idx]
            current_mask = matched_mask['segmentation'].astype(np.uint8)
            current_iou = matched_mask['matching_iou']

            # 寻找与该真值mask有重叠的未使用SAM mask
            candidate_masks = []
            for sam_idx in unused_sam_indices:
                sam_mask = sam_binary_masks[sam_idx]

                # 检查这个未使用的mask是否与当前真值有重叠
                overlap_with_gt = self.calculate_iou(sam_mask, gt_mask)
                if overlap_with_gt > 0.05:  # 至少5%重叠
                    # 检查与当前匹配mask的空间邻接性
                    adjacency_score = self._calculate_adjacency(current_mask, sam_mask)
                    candidate_masks.append((sam_idx, overlap_with_gt, adjacency_score))

            if not candidate_masks:
                continue

            # 按重叠度和邻接性排序
            candidate_masks.sort(key=lambda x: x[1] * 0.7 + x[2] * 0.3, reverse=True)

            # 尝试逐步添加候选mask
            best_merged_mask = current_mask.copy()
            best_merged_iou = current_iou
            merged_indices = []

            for sam_idx, overlap, adjacency in candidate_masks:
                # 尝试合并这个mask
                test_merged_mask = np.logical_or(best_merged_mask, sam_binary_masks[sam_idx]).astype(np.uint8)
                test_merged_iou = self.calculate_iou(test_merged_mask, gt_mask)

                # 如果合并后IoU提高，则接受这个合并
                if test_merged_iou > best_merged_iou + 0.01:  # 至少提高1%
                    best_merged_mask = test_merged_mask
                    best_merged_iou = test_merged_iou
                    merged_indices.append(sam_idx)
                    print(f"    ID {gt_id}: 合并SAM mask {sam_idx}, IoU: {current_iou:.3f} -> {test_merged_iou:.3f}")

            # 如果找到了更好的合并结果，更新matched_mask
            if best_merged_iou > current_iou + 0.01:
                # 更新mask信息
                matched_masks[i]['segmentation'] = best_merged_mask
                matched_masks[i]['matching_iou'] = best_merged_iou
                matched_masks[i]['area'] = int(best_merged_mask.sum())
                matched_masks[i]['bbox'] = self._mask_to_bbox(best_merged_mask)

                # 计算新的覆盖率
                new_coverage = np.logical_and(best_merged_mask, gt_mask).sum() / gt_mask.sum() if gt_mask.sum() > 0 else 0.0
                matched_masks[i]['coverage_ratio'] = new_coverage

                # 标记为合并mask
                if not matched_masks[i].get('is_merged', False):
                    matched_masks[i]['is_merged'] = True
                    matched_masks[i]['merged_from_indices'] = merged_indices
                    matched_masks[i]['num_merged_masks'] = len(merged_indices) + 1
                else:
                    # 如果已经是合并mask，更新合并信息
                    existing_indices = matched_masks[i].get('merged_from_indices', [])
                    matched_masks[i]['merged_from_indices'] = existing_indices + merged_indices
                    matched_masks[i]['num_merged_masks'] = len(existing_indices) + len(merged_indices) + 1

                # 将合并的mask索引标记为已使用
                for idx in merged_indices:
                    used_sam_indices.add(idx)

                improved_count += 1
                print(f"    成功改进 ID {gt_id}: IoU {current_iou:.3f} -> {best_merged_iou:.3f} (合并了 {len(merged_indices)} 个mask)")

        print(f"    Invalid区域合并完成，改进了 {improved_count} 个mask")

    def _calculate_adjacency(self, mask1: np.ndarray, mask2: np.ndarray, max_distance: int = 5) -> float:
        """
        计算两个mask之间的邻接性分数

        Arguments:
          mask1, mask2: 二值mask
          max_distance: 最大距离阈值

        Returns:
          adjacency_score: 邻接性分数 (0-1)
        """
        # 获取mask边界
        kernel = np.ones((3, 3), np.uint8)

        # 计算mask1的边界
        boundary1 = cv2.morphologyEx(mask1.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
        # 计算mask2的边界
        boundary2 = cv2.morphologyEx(mask2.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)

        if boundary1.sum() == 0 or boundary2.sum() == 0:
            return 0.0

        # 计算边界之间的最小距离
        boundary1_coords = np.column_stack(np.where(boundary1 > 0))
        boundary2_coords = np.column_stack(np.where(boundary2 > 0))

        if len(boundary1_coords) == 0 or len(boundary2_coords) == 0:
            return 0.0

        # 计算最小距离（采样以提高效率）
        sample_size = min(100, len(boundary1_coords))
        sampled_coords1 = boundary1_coords[np.random.choice(len(boundary1_coords), sample_size, replace=False)]

        min_distances = []
        for coord1 in sampled_coords1:
            distances = np.sqrt(np.sum((boundary2_coords - coord1) ** 2, axis=1))
            min_distances.append(np.min(distances))

        avg_min_distance = np.mean(min_distances)

        # 将距离转换为邻接性分数
        adjacency_score = max(0.0, 1.0 - avg_min_distance / max_distance)

        return adjacency_score

    def process_image_with_ground_truth(
        self,
        image: np.ndarray,
        mask_path: str
    ) -> List[Dict[str, Any]]:
        """
        处理图像，生成与真值匹配的mask

        Arguments:
          image (np.ndarray): 输入图像，HWC uint8格式
          mask_path (str): 真值mask文件路径

        Returns:
          matched_masks (List[Dict]): 匹配后的mask列表
        """

        # 加载真值mask
        gt_mask_data, valid_ids = self.load_ground_truth_masks(mask_path)

        # 检查图像和mask尺寸是否匹配
        img_h, img_w = image.shape[:2]
        mask_h, mask_w = gt_mask_data.shape

        if (img_h, img_w) != (mask_h, mask_w):
            print(f"Warning: Image size ({img_h}, {img_w}) != Mask size ({mask_h}, {mask_w})")
            print("Resizing image to match mask size...")
            # 将图像resize到mask的尺寸
            image_pil = Image.fromarray(image)
            image_resized = image_pil.resize((mask_w, mask_h), Image.LANCZOS)
            image = np.array(image_resized)
            print(f"Image resized to: {image.shape}")

        # 使用SAM2AutomaticMaskGenerator生成mask
        print("Generating masks with SAM2...")
        sam_masks = self.mask_generator.generate(image)
        print(f"Generated {len(sam_masks)} masks")

        # 将SAM2 mask匹配到真值ID
        print("Matching masks to ground truth...")
        matched_masks = self.match_masks_to_ground_truth(sam_masks, gt_mask_data, valid_ids)
        print(f"Matched {len(matched_masks)} masks to {len(valid_ids)} ground truth objects")

        return matched_masks

    def visualize_results(
        self,
        image: np.ndarray,
        matched_masks: List[Dict[str, Any]],
        gt_mask_data: np.ndarray = None,
        save_path: str = "mask_matching_results.png"
    ) -> None:
        """
        可视化匹配结果，同时展示真值mask和匹配后的mask

        Arguments:
          image (np.ndarray): 原始图像
          matched_masks (List[Dict]): 匹配后的mask列表
          gt_mask_data (np.ndarray): 真值mask数据
          save_path (str): 保存路径
        """

        # 创建三列布局：原始图像、真值mask、匹配后的mask
        fig, axes = plt.subplots(1, 3, figsize=(22, 7))

        # 第一列：原始图像
        axes[0].imshow(image)
        axes[0].set_title("Original Image", fontsize=14)
        axes[0].axis('off')

        # 为每个mask分配一致的颜色
        cmap = plt.get_cmap('tab20', len(matched_masks))

        # 第二列：真值mask（纯mask显示，不叠加图像）
        if gt_mask_data is not None:
            # 创建纯mask可视化
            h, w = gt_mask_data.shape
            gt_mask_vis = np.zeros((h, w, 3), dtype=np.float32)

            # 提取真值ID列表
            gt_ids = [mask_info['ground_truth_id'] for mask_info in matched_masks]

            for i, gt_id in enumerate(gt_ids):
                gt_mask = (gt_mask_data == gt_id).astype(np.uint8)

                if gt_mask.sum() > 0:
                    # 使用与匹配mask相同的颜色
                    color = np.array(cmap(i)[:3])

                    # 直接在mask可视化上绘制颜色
                    for c in range(3):
                        gt_mask_vis[gt_mask > 0, c] = color[c]

            axes[1].imshow(gt_mask_vis)

            # 添加标签
            for i, gt_id in enumerate(gt_ids):
                gt_mask = (gt_mask_data == gt_id).astype(np.uint8)
                if gt_mask.sum() > 0:
                    y_coords, x_coords = np.where(gt_mask > 0)
                    center_y, center_x = np.mean(y_coords), np.mean(x_coords)

                    axes[1].text(center_x, center_y, f"ID:{gt_id}",
                               color='white', fontsize=8, ha='center', va='center',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))

            axes[1].set_title(f"Ground Truth Masks ({len(gt_ids)} objects)", fontsize=14)
        else:
            axes[1].text(0.5, 0.5, "Ground Truth\nNot Available",
                        ha='center', va='center', transform=axes[1].transAxes,
                        fontsize=16, color='gray')
            axes[1].set_title("Ground Truth Masks", fontsize=14)

        axes[1].axis('off')

        # 第三列：匹配后的mask（纯mask显示，不叠加图像）
        if len(matched_masks) > 0:
            # 获取mask尺寸
            first_mask = matched_masks[0]['segmentation']
            h, w = first_mask.shape
            matched_mask_vis = np.zeros((h, w, 3), dtype=np.float32)

            for i, mask_info in enumerate(matched_masks):
                mask = mask_info['segmentation']
                gt_id = mask_info['ground_truth_id']
                matching_iou = mask_info['matching_iou']
                is_fallback = mask_info.get('is_fallback', False)
                is_low_confidence = mask_info.get('is_low_confidence', False)
                is_merged = mask_info.get('is_merged', False)
                num_merged = mask_info.get('num_merged_masks', 1)

                # 创建彩色mask（使用与真值mask相同的颜色）
                color = np.array(cmap(i)[:3])

                # 直接在mask可视化上绘制颜色
                for c in range(3):
                    matched_mask_vis[mask > 0, c] = color[c]

            axes[2].imshow(matched_mask_vis)

            # 添加标签
            for i, mask_info in enumerate(matched_masks):
                mask = mask_info['segmentation']
                gt_id = mask_info['ground_truth_id']
                matching_iou = mask_info['matching_iou']
                is_fallback = mask_info.get('is_fallback', False)
                is_low_confidence = mask_info.get('is_low_confidence', False)
                is_merged = mask_info.get('is_merged', False)
                num_merged = mask_info.get('num_merged_masks', 1)

                if mask.sum() > 0:
                    # 找到mask的中心点
                    y_coords, x_coords = np.where(mask > 0)
                    center_y, center_x = np.mean(y_coords), np.mean(x_coords)

                    label = f"ID:{gt_id}"
                    if is_fallback:
                        label += " (GT)"
                    elif is_merged:
                        label += f" (IoU:{matching_iou:.2f}+{num_merged})"
                    elif is_low_confidence:
                        label += f" (IoU:{matching_iou:.2f}*)"
                    else:
                        label += f" (IoU:{matching_iou:.2f})"

                    axes[2].text(center_x, center_y, label,
                               color='white', fontsize=8, ha='center', va='center',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))

        axes[2].set_title(f"Matched Masks ({len(matched_masks)} objects)", fontsize=14)
        axes[2].axis('off')

        # 添加图例说明
        legend_text = "Legend: (GT) = Ground Truth Fallback, (*) = Low Confidence Match, (+N) = Merged from N masks"
        fig.text(0.5, 0.02, legend_text, ha='center', fontsize=10, style='italic')

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)  # 为图例留出空间
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to: {save_path}")

    def calculate_invalid_area_ratio(
        self,
        matched_masks: List[Dict[str, Any]],
        gt_mask_data: np.ndarray
    ) -> float:
        """
        计算invalid区域的比例

        Arguments:
          matched_masks (List[Dict]): 匹配后的mask列表
          gt_mask_data (np.ndarray): 真值mask数据

        Returns:
          invalid_ratio (float): invalid区域占总图像的比例
        """

        h, w = gt_mask_data.shape
        total_pixels = h * w

        # 创建一个mask来标记所有被覆盖的区域
        covered_mask = np.zeros((h, w), dtype=bool)

        for mask_info in matched_masks:
            mask = mask_info['segmentation'].astype(bool)
            covered_mask |= mask

        # 计算未被覆盖的区域
        uncovered_pixels = np.sum(~covered_mask)
        invalid_ratio = uncovered_pixels / total_pixels

        return invalid_ratio

    def save_matched_masks(
        self,
        matched_masks: List[Dict[str, Any]],
        output_path: str
    ) -> None:
        """
        保存匹配后的mask到文件

        Arguments:
          matched_masks (List[Dict]): 匹配后的mask列表
          output_path (str): 输出文件路径
        """

        # 创建输出数据结构
        output_data = {
            'masks': [],
            'ground_truth_ids': [],
            'matching_ious': [],
            'is_fallbacks': []
        }

        for mask_info in matched_masks:
            output_data['masks'].append(mask_info['segmentation'])
            output_data['ground_truth_ids'].append(mask_info['ground_truth_id'])
            output_data['matching_ious'].append(mask_info['matching_iou'])
            output_data['is_fallbacks'].append(mask_info.get('is_fallback', False))

        # 保存为numpy格式
        if output_path.endswith('.npz'):
            np.savez_compressed(output_path, **output_data)
        elif output_path.endswith('.pth'):
            torch.save(output_data, output_path)
        else:
            # 默认保存为npz格式
            output_path = output_path + '.npz'
            np.savez_compressed(output_path, **output_data)

        print(f"Matched masks saved to: {output_path}")


def main():
    """
    示例用法
    """

    # 配置参数
    model_cfg = "sam2_hiera_l.yaml"
    checkpoint = "model_zoo/sam/sam2_hiera_large.pt"

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
        print("错误：找不到SAM2模型文件。请确保以下路径之一存在模型文件：")
        for path in possible_checkpoints:
            print(f"  - {path}")
        print("\n你可以从以下链接下载模型：")
        print("https://github.com/facebookresearch/segment-anything-2#model-checkpoints")
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
        stability_score_thresh=0.5,
        # stability_score_thresh=0.85,
        box_nms_thresh=0.7,
        iou_threshold=0.3,  # 匹配阈值
    )

    # 读取图像
    image = Image.open(img_path).convert('RGB')
    image = np.array(image)

    # 处理图像并匹配mask
    matched_masks = matcher.process_image_with_ground_truth(image, mask_path)

    # 加载真值mask数据用于计算invalid区域
    gt_mask_data, valid_ids = matcher.load_ground_truth_masks(mask_path)

    # 计算invalid区域比例
    invalid_ratio = matcher.calculate_invalid_area_ratio(matched_masks, gt_mask_data)

    # 打印结果统计
    print(f"\n=== 匹配结果统计 ===")
    print(f"总共匹配到 {len(matched_masks)} 个对象")

    sam_matched = sum(1 for m in matched_masks if not m.get('is_fallback', False))
    fallback_count = len(matched_masks) - sam_matched

    print(f"SAM2成功匹配: {sam_matched} 个")
    print(f"使用真值填充: {fallback_count} 个")
    print(f"Invalid区域比例: {invalid_ratio:.3f} ({invalid_ratio*100:.1f}%)")

    # 计算平均IoU（仅对SAM2匹配的mask）
    sam_ious = [m['matching_iou'] for m in matched_masks if not m.get('is_fallback', False)]
    if sam_ious:
        avg_iou = np.mean(sam_ious)
        print(f"SAM2匹配的平均IoU: {avg_iou:.3f}")

    print("\n详细匹配结果:")
    for mask_info in matched_masks:
        gt_id = mask_info['ground_truth_id']
        matching_iou = mask_info['matching_iou']
        is_fallback = mask_info.get('is_fallback', False)
        area = mask_info['area']
        coverage_ratio = mask_info.get('coverage_ratio', 0.0)
        is_low_confidence = mask_info.get('is_low_confidence', False)

        if is_fallback:
            print(f"  ID {gt_id}: 使用真值填充 (面积: {area})")
        else:
            confidence_str = " [低置信度]" if is_low_confidence else ""
            print(f"  ID {gt_id}: SAM2匹配 (IoU: {matching_iou:.3f}, 覆盖率: {coverage_ratio:.3f}, 面积: {area}){confidence_str}")

    # 可视化结果
    matcher.visualize_results(image, matched_masks, gt_mask_data, "mask_matching_results.png")

    # 保存结果
    matcher.save_matched_masks(matched_masks, "matched_masks_output.npz")

    print("\n=== 处理完成 ===")


if __name__ == "__main__":
    main()
