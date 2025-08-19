import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.ops.boxes import batched_nms, box_area
from typing import Any, Dict, List, Optional, Tuple

from sam2.utils.amg import (
    area_from_rle,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    calculate_stability_score,
    coco_encode_rle,
    mask_to_rle_pytorch,
    MaskData,
    remove_small_regions,
    rle_to_mask,
)


class SAM2MaskRefinerWithNMS:
    """
    基于SAM2的mask细化器，包含完整的NMS处理流程
    仿照SAM2AutomaticMaskGenerator的处理方式，但用于细化预先存在的mask标注
    """

    def __init__(
        self,
        model_cfg: str,
        checkpoint: str,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        mask_threshold: float = 0.0,
        box_nms_thresh: float = 0.7,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
        use_m2m: bool = False,
        **kwargs,
    ) -> None:
        """
        初始化SAM2 Mask Refiner

        Arguments:
          model_cfg (str): SAM2模型配置文件路径
          checkpoint (str): SAM2模型权重文件路径
          pred_iou_thresh (float): IoU阈值，用于过滤低质量mask
          stability_score_thresh (float): 稳定性分数阈值
          stability_score_offset (float): 计算稳定性分数时的偏移量
          mask_threshold (float): mask二值化阈值
          box_nms_thresh (float): NMS的IoU阈值
          min_mask_region_area (int): 最小mask区域面积
          output_mode (str): 输出格式 ('binary_mask', 'uncompressed_rle', 'coco_rle')
          use_m2m (bool): 是否使用mask-to-mask细化
        """

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."

        if output_mode == "coco_rle":
            try:
                from pycocotools import mask as mask_utils  # type: ignore  # noqa: F401
            except ImportError as e:
                print("Please install pycocotools")
                raise e

        # 初始化SAM2预测器
        sam_model = build_sam2(model_cfg, checkpoint)
        self.predictor = SAM2ImagePredictor(
            sam_model,
            max_hole_area=min_mask_region_area,
            max_sprinkle_area=min_mask_region_area,
        )

        # 设置参数
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.mask_threshold = mask_threshold
        self.box_nms_thresh = box_nms_thresh
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode
        self.use_m2m = use_m2m

    def refine_masks(self, image: np.ndarray, mask_data: np.ndarray) -> List[Dict[str, Any]]:
        """
        细化给定的mask标注

        Arguments:
          image (np.ndarray): 输入图像，HWC uint8格式
          mask_data (np.ndarray): mask标注数据，每个像素值代表不同的对象ID

        Returns:
          list(dict(str, any)): 细化后的mask列表，每个记录包含以下键值：
            segmentation: mask数据
            bbox: 边界框 (XYWH格式)
            area: mask面积
            predicted_iou: 预测的IoU分数
            stability_score: 稳定性分数
        """

        # 检查图像和mask尺寸是否匹配
        img_h, img_w = image.shape[:2]
        mask_h, mask_w = mask_data.shape

        if (img_h, img_w) != (mask_h, mask_w):
            print(f"Warning: Image size ({img_h}, {img_w}) != Mask size ({mask_h}, {mask_w})")
            print("Resizing image to match mask size...")
            # 将图像resize到mask的尺寸
            from PIL import Image as PILImage
            image_pil = PILImage.fromarray(image)
            image_resized = image_pil.resize((mask_w, mask_h), PILImage.LANCZOS)
            image = np.array(image_resized)
            print(f"Image resized to: {image.shape}")

        # 设置图像到predictor
        self.predictor.set_image(image)

        # 提取有效的对象ID
        unique_ids = np.unique(mask_data)
        valid_ids = unique_ids[unique_ids != 0]  # 跳过背景ID (0)

        if len(valid_ids) == 0:
            return []

        # 生成初始mask数据
        mask_data_obj = self._generate_initial_masks(mask_data, valid_ids, image.shape[:2])

        # 应用NMS和过滤
        mask_data_obj = self._apply_filtering_and_nms(mask_data_obj)

        # 后处理小区域
        if self.min_mask_region_area > 0:
            mask_data_obj = self.postprocess_small_regions(
                mask_data_obj, self.min_mask_region_area, self.box_nms_thresh
            )

        # 编码mask
        if self.output_mode == "coco_rle":
            mask_data_obj["segmentations"] = [
                coco_encode_rle(rle) for rle in mask_data_obj["rles"]
            ]
        elif self.output_mode == "binary_mask":
            mask_data_obj["segmentations"] = [rle_to_mask(rle) for rle in mask_data_obj["rles"]]
        else:
            mask_data_obj["segmentations"] = mask_data_obj["rles"]

        # 生成最终结果
        curr_anns = []
        for idx in range(len(mask_data_obj["segmentations"])):
            ann = {
                "segmentation": mask_data_obj["segmentations"][idx],
                "area": area_from_rle(mask_data_obj["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data_obj["boxes"][idx]).tolist(),
                "predicted_iou": mask_data_obj["iou_preds"][idx].item(),
                "stability_score": mask_data_obj["stability_score"][idx].item(),
            }
            curr_anns.append(ann)

        return curr_anns

    def _generate_initial_masks(
        self,
        mask_data: np.ndarray,
        valid_ids: np.ndarray,
        orig_size: Tuple[int, int]
    ) -> MaskData:
        """生成初始的mask数据"""

        orig_h, orig_w = orig_size
        low_res_size = self.predictor.model.image_size // 4  # e.g., 1024//4=256

        # 为每个有效ID生成mask
        masks = np.stack([(mask_data == obj_id) for obj_id in valid_ids], axis=0)  # [N, H, W]

        # 转换为低分辨率logits
        mask_tensor = torch.as_tensor(masks, dtype=torch.float32).unsqueeze(0)  # [1, N, H, W]
        mask_lowres = F.interpolate(
            mask_tensor,
            size=(low_res_size, low_res_size),
            mode='bilinear',
            align_corners=False
        )
        mask_lowres_np = mask_lowres.squeeze(0).cpu().numpy()  # [N, H, W]

        # 收集所有细化后的mask
        refined_masks = []
        iou_preds = []

        for i in range(len(valid_ids)):
            current_mask = masks[i].astype(np.uint8)  # [H, W]

            if self.use_m2m:
                # 使用mask-to-mask细化
                # 将mask转换为低分辨率logits作为mask_input
                low_res_size = self.predictor.model.image_size // 4  # e.g., 1024//4=256
                mask_tensor = torch.as_tensor(current_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                mask_lowres = F.interpolate(
                    mask_tensor,
                    size=(low_res_size, low_res_size),
                    mode='bilinear',
                    align_corners=False
                )
                # 转换为logits，使用更温和的缩放，并确保正确的形状
                mask_logits = (mask_lowres.squeeze(0) - 0.5) * 6.0  # [1, H, W] 降低缩放强度

                # 同时提供边界框作为额外的prompt来约束预测
                y_indices, x_indices = np.where(current_mask > 0)
                if len(y_indices) == 0:  # 空mask，跳过
                    continue

                x_min, x_max = x_indices.min(), x_indices.max()
                y_min, y_max = y_indices.min(), y_indices.max()

                padding = 10
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(orig_w - 1, x_max + padding)
                y_max = min(orig_h - 1, y_max + padding)

                bbox = np.array([x_min, y_min, x_max, y_max])

                pred_masks, pred_iou, _ = self.predictor.predict(
                    box=bbox,  # 提供边界框约束
                    mask_input=mask_logits,  # 提供mask输入
                    multimask_output=False,
                    return_logits=True,
                )
            else:
                # 使用边界框作为prompt进行预测
                # 计算当前mask的边界框
                y_indices, x_indices = np.where(current_mask > 0)
                if len(y_indices) == 0:  # 空mask，跳过
                    continue

                x_min, x_max = x_indices.min(), x_indices.max()
                y_min, y_max = y_indices.min(), y_indices.max()

                padding = 5
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(orig_w - 1, x_max + padding)
                y_max = min(orig_h - 1, y_max + padding)

                bbox = np.array([x_min, y_min, x_max, y_max])

                pred_masks, pred_iou, _ = self.predictor.predict(
                    box=bbox,
                    multimask_output=False,
                    return_logits=True,
                )

            refined_masks.append(pred_masks[0])  # 取第一个mask
            iou_preds.append(pred_iou[0])

        if len(refined_masks) == 0:
            # 如果没有有效的mask，返回空的MaskData
            return MaskData(
                masks=torch.empty((0, orig_h, orig_w), dtype=torch.bool),
                iou_preds=torch.empty((0,), dtype=torch.float32),
            )

        # 堆叠所有结果
        refined_masks = np.stack(refined_masks, axis=0)  # [N, H, W]
        iou_preds = np.array(iou_preds)  # [N]

        # 创建MaskData对象
        data = MaskData(
            masks=torch.from_numpy(refined_masks),
            iou_preds=torch.from_numpy(iou_preds),
        )

        return data

    def _apply_filtering_and_nms(self, data: MaskData) -> MaskData:
        """应用过滤和NMS"""

        print(f"Initial masks: {len(data['masks'])}")

        # 按预测IoU过滤
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)
            print(f"After IoU filtering (thresh={self.pred_iou_thresh}): {len(data['masks'])}")

        # 计算稳定性分数并过滤
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.mask_threshold, self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)
            print(f"After stability filtering (thresh={self.stability_score_thresh}): {len(data['masks'])}")

        # 二值化mask并计算边界框
        data["masks"] = data["masks"] > self.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # 应用NMS去除重复
        if len(data["boxes"]) > 1:
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                data["iou_preds"],
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.box_nms_thresh,
            )
            data.filter(keep_by_nms)
            print(f"After NMS (thresh={self.box_nms_thresh}): {len(data['masks'])}")

        # 转换为RLE格式
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]  # 释放内存

        return data

    @staticmethod
    def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        移除小的断开区域和孔洞，然后重新运行NMS去除新的重复
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        # 过滤小的断开区域和孔洞
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # 给未改变的mask更高的分数，这样NMS会优先保留它们
            scores.append(float(unchanged))

        # 重新计算边界框并移除新的重复
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # 只重新计算改变了的mask的RLE
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]

        mask_data.filter(keep_by_nms)
        return mask_data


def visualize_masks(masks_list: List[Dict[str, Any]], save_path: str = "refined_masks_nms.png"):
    """可视化细化后的mask"""
    if not masks_list:
        print("No masks to visualize")
        return

    # 获取第一个mask的尺寸
    first_mask = masks_list[0]["segmentation"]
    if isinstance(first_mask, dict):  # RLE格式
        h, w = first_mask["size"]
        mask_vis = np.zeros((h, w, 3), dtype=np.float32)
    else:  # binary mask格式
        h, w = first_mask.shape
        mask_vis = np.zeros((h, w, 3), dtype=np.float32)

    # 为每个mask分配不同颜色
    N = len(masks_list)
    cmap = plt.get_cmap('tab20', N)

    for i, mask_info in enumerate(masks_list):
        color = np.array(cmap(i)[:3])  # RGB

        # 获取mask
        if isinstance(mask_info["segmentation"], dict):  # RLE格式
            mask = rle_to_mask(mask_info["segmentation"])
        else:  # binary mask格式
            mask = mask_info["segmentation"]

        mask = mask.astype(np.float32)

        # 叠加颜色
        for c in range(3):
            mask_vis[..., c] += mask * color[c]

    # 防止溢出
    mask_vis = np.clip(mask_vis, 0, 1)

    # 保存图像
    mask_vis_uint8 = (mask_vis * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_vis_uint8)
    mask_img.save(save_path)
    print(f"Refined masks saved to {save_path}")


def visualize_original_masks(mask_data: np.ndarray, save_path: str = "original_masks.png"):
    """可视化原始mask数据"""
    unique_ids = np.unique(mask_data)
    valid_ids = unique_ids[unique_ids != 0]  # 跳过背景ID (0)

    if len(valid_ids) == 0:
        print("No valid masks to visualize")
        return

    h, w = mask_data.shape
    mask_vis = np.zeros((h, w, 3), dtype=np.float32)

    # 为每个mask分配不同颜色
    N = len(valid_ids)
    cmap = plt.get_cmap('tab20', N)

    for i, obj_id in enumerate(valid_ids):
        color = np.array(cmap(i)[:3])  # RGB
        mask = (mask_data == obj_id).astype(np.float32)

        # 叠加颜色
        for c in range(3):
            mask_vis[..., c] += mask * color[c]

    # 防止溢出
    mask_vis = np.clip(mask_vis, 0, 1)

    # 保存图像
    mask_vis_uint8 = (mask_vis * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_vis_uint8)
    mask_img.save(save_path)
    print(f"Original masks saved to {save_path}")


def debug_mask_refinement():
    """调试mask细化过程的函数"""
    print("=== 开始调试mask细化过程 ===")

    # 示例用法
    checkpoint = "model_zoo/sam/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    # 创建refiner - 使用非常宽松的阈值
    refiner = SAM2MaskRefinerWithNMS(
        model_cfg=model_cfg,
        checkpoint=checkpoint,
        pred_iou_thresh=0.1,  # 非常低的IoU阈值
        stability_score_thresh=0.5,  # 非常低的稳定性阈值
        box_nms_thresh=0.3,  # 非常低的NMS阈值
        min_mask_region_area=10,  # 非常小的最小区域面积
        output_mode="binary_mask",
        use_m2m=True,  # 启用mask-to-mask细化
    )

    # 加载图像和mask数据
    img_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/mm_datasets/lihao_3d_data/processed_scannetpp_fix/0cf2e9402d/images/frame_000130.jpg"
    mask_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/mm_datasets/lihao_3d_data/processed_scannetpp_fix/0cf2e9402d/obj_ids/frame_000130.jpg.pth"

    # 读取数据
    image = np.array(Image.open(img_path).convert('RGB'))
    mask_data = torch.load(mask_path, weights_only=False)

    print(f"Image shape: {image.shape}")
    print(f"Mask data shape: {mask_data.shape}")
    print(f"Mask data type: {mask_data.dtype}")
    print(f"Unique mask IDs: {np.unique(mask_data)}")

    # 可视化原始mask
    print("\n1. 可视化原始mask...")
    visualize_original_masks(mask_data, "debug_original_masks.png")

    # 细化mask
    print("\n2. 开始细化mask...")
    refined_masks = refiner.refine_masks(image, mask_data)

    print(f"\n3. 结果对比:")
    print(f"   原始mask数量: {len(np.unique(mask_data)) - 1}")  # -1 for background
    print(f"   细化后mask数量: {len(refined_masks)}")

    # 可视化结果
    if refined_masks:
        print("\n4. 可视化细化后的mask...")
        visualize_masks(refined_masks, "debug_refined_masks.png")

        # 详细统计
        iou_scores = [mask["predicted_iou"] for mask in refined_masks]
        stability_scores = [mask["stability_score"] for mask in refined_masks]
        areas = [mask["area"] for mask in refined_masks]

        print(f"\n5. 详细统计:")
        print(f"   IoU分数: 平均={np.mean(iou_scores):.3f}, 最小={np.min(iou_scores):.3f}, 最大={np.max(iou_scores):.3f}")
        print(f"   稳定性分数: 平均={np.mean(stability_scores):.3f}, 最小={np.min(stability_scores):.3f}, 最大={np.max(stability_scores):.3f}")
        print(f"   面积: 平均={np.mean(areas):.1f}, 最小={np.min(areas):.1f}, 最大={np.max(areas):.1f}")
    else:
        print("\n4. 警告: 没有mask通过过滤!")
        print("   可能的原因:")
        print("   - SAM2模型预测质量低")
        print("   - 过滤阈值仍然过于严格")
        print("   - 输入mask质量问题")

    print("\n=== 调试完成 ===")


def analyze_mask_differences():
    """详细分析原始mask和细化后mask的差异"""
    print("=== 开始详细分析mask差异 ===")

    # 示例用法
    checkpoint = "model_zoo/sam/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    # 创建refiner - 使用非常宽松的阈值，禁用NMS
    refiner = SAM2MaskRefinerWithNMS(
        model_cfg=model_cfg,
        checkpoint=checkpoint,
        pred_iou_thresh=0.0,  # 禁用IoU过滤
        stability_score_thresh=0.0,  # 禁用稳定性过滤
        box_nms_thresh=1.0,  # 禁用NMS
        min_mask_region_area=0,  # 禁用最小区域过滤
        output_mode="binary_mask",
        use_m2m=True,  # 启用mask-to-mask细化
    )

    # 加载图像和mask数据
    img_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/mm_datasets/lihao_3d_data/processed_scannetpp_fix/0cf2e9402d/images/frame_000130.jpg"
    mask_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/mm_datasets/lihao_3d_data/processed_scannetpp_fix/0cf2e9402d/obj_ids/frame_000130.jpg.pth"

    # 读取数据
    image = np.array(Image.open(img_path).convert('RGB'))
    mask_data = torch.load(mask_path, weights_only=False)

    # 获取原始mask信息
    unique_ids = np.unique(mask_data)
    valid_ids = unique_ids[unique_ids != 0]

    print(f"原始mask数量: {len(valid_ids)}")
    print(f"原始mask IDs: {valid_ids}")

    # 细化mask (禁用所有过滤)
    refined_masks = refiner.refine_masks(image, mask_data)
    print(f"细化后mask数量: {len(refined_masks)}")

    # 逐个比较mask
    print("\n=== 逐个mask对比 ===")
    for i, obj_id in enumerate(valid_ids):
        original_mask = (mask_data == obj_id).astype(np.uint8)
        original_area = np.sum(original_mask)

        if i < len(refined_masks):
            refined_mask = refined_masks[i]["segmentation"]
            refined_area = refined_masks[i]["area"]
            iou_score = refined_masks[i]["predicted_iou"]
            stability_score = refined_masks[i]["stability_score"]

            # 计算IoU between original and refined
            intersection = np.sum(original_mask * refined_mask)
            union = np.sum((original_mask + refined_mask) > 0)
            actual_iou = intersection / union if union > 0 else 0

            print(f"Mask {i+1} (ID={obj_id}):")
            print(f"  原始面积: {original_area}")
            print(f"  细化面积: {refined_area}")
            print(f"  面积比率: {refined_area/original_area:.3f}")
            print(f"  预测IoU: {iou_score:.3f}")
            print(f"  实际IoU: {actual_iou:.3f}")
            print(f"  稳定性分数: {stability_score:.3f}")
        else:
            print(f"Mask {i+1} (ID={obj_id}): 被过滤掉了")
        print()

    # 保存对比图像
    print("保存对比图像...")

    # 创建并排对比图
    h, w = mask_data.shape
    comparison_vis = np.zeros((h, w*2, 3), dtype=np.float32)

    # 左侧：原始mask
    N = len(valid_ids)
    cmap = plt.get_cmap('tab20', N)

    for i, obj_id in enumerate(valid_ids):
        color = np.array(cmap(i)[:3])
        mask = (mask_data == obj_id).astype(np.float32)
        for c in range(3):
            comparison_vis[:, :w, c] += mask * color[c]

    # 右侧：细化后的mask
    for i, mask_info in enumerate(refined_masks):
        if i < N:
            color = np.array(cmap(i)[:3])
            mask = mask_info["segmentation"].astype(np.float32)
            for c in range(3):
                comparison_vis[:, w:, c] += mask * color[c]

    # 防止溢出并保存
    comparison_vis = np.clip(comparison_vis, 0, 1)
    comparison_vis_uint8 = (comparison_vis * 255).astype(np.uint8)
    comparison_img = Image.fromarray(comparison_vis_uint8)
    comparison_img.save("mask_comparison_side_by_side.png")
    print("对比图像保存到: mask_comparison_side_by_side.png")

    print("\n=== 分析完成 ===")


if __name__ == "__main__":
    # 示例用法
    checkpoint = "model_zoo/sam/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    # 创建refiner - 使用更宽松的阈值以保留更多mask
    refiner = SAM2MaskRefinerWithNMS(
        model_cfg=model_cfg,
        checkpoint=checkpoint,
        pred_iou_thresh=0.3,  # 降低IoU阈值
        stability_score_thresh=0.7,  # 降低稳定性阈值
        box_nms_thresh=0.5,  # 降低NMS阈值
        min_mask_region_area=25,  # 降低最小区域面积
        output_mode="binary_mask",
        use_m2m=True,  # 启用mask-to-mask细化
    )

    # 加载图像和mask数据
    img_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/mm_datasets/lihao_3d_data/processed_scannetpp_fix/0cf2e9402d/images/frame_000000.jpg"
    mask_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/mm_datasets/lihao_3d_data/processed_scannetpp_fix/0cf2e9402d/obj_ids/frame_000000.jpg.pth"

    # 读取数据
    image = np.array(Image.open(img_path).convert('RGB'))
    mask_data = torch.load(mask_path, weights_only=False)

    # 可视化原始mask
    print("Visualizing original masks...")
    visualize_original_masks(mask_data, "original_masks.png")

    # 细化mask
    print("Refining masks...")
    refined_masks = refiner.refine_masks(image, mask_data)

    print(f"Original masks: {len(np.unique(mask_data)) - 1}")  # -1 for background
    print(f"Refined masks after NMS: {len(refined_masks)}")

    # 可视化结果
    visualize_masks(refined_masks, "refined_masks_with_nms.png")

    # 打印详细的统计信息
    if refined_masks:
        iou_scores = [mask["predicted_iou"] for mask in refined_masks]
        stability_scores = [mask["stability_score"] for mask in refined_masks]
        areas = [mask["area"] for mask in refined_masks]

        print(f"\n=== 细化结果统计 ===")
        print(f"Average IoU: {np.mean(iou_scores):.3f} (min: {np.min(iou_scores):.3f}, max: {np.max(iou_scores):.3f})")
        print(f"Average Stability Score: {np.mean(stability_scores):.3f} (min: {np.min(stability_scores):.3f}, max: {np.max(stability_scores):.3f})")
        print(f"Average Area: {np.mean(areas):.1f} (min: {np.min(areas):.1f}, max: {np.max(areas):.1f})")

        # 打印每个mask的详细信息
        print(f"\n=== 每个mask的详细信息 ===")
        for i, mask_info in enumerate(refined_masks):
            print(f"Mask {i+1}: IoU={mask_info['predicted_iou']:.3f}, "
                  f"Stability={mask_info['stability_score']:.3f}, "
                  f"Area={mask_info['area']:.1f}, "
                  f"BBox={mask_info['bbox']}")
    else:
        print("No masks survived the filtering process!")
        print("Try lowering the thresholds:")
        print("- pred_iou_thresh (current: 0.3)")
        print("- stability_score_thresh (current: 0.7)")
        print("- min_mask_region_area (current: 50)")


def create_conservative_mask_refiner():
    """创建一个保守的mask细化器，用于测试"""
    print("=== 测试保守的mask细化方法 ===")

    # 示例用法
    checkpoint = "model_zoo/sam/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    # 创建refiner - 禁用所有过滤
    refiner = SAM2MaskRefinerWithNMS(
        model_cfg=model_cfg,
        checkpoint=checkpoint,
        pred_iou_thresh=0.0,  # 禁用IoU过滤
        stability_score_thresh=0.0,  # 禁用稳定性过滤
        box_nms_thresh=1.0,  # 禁用NMS
        min_mask_region_area=0,  # 禁用最小区域过滤
        output_mode="binary_mask",
        use_m2m=False,  # 禁用mask-to-mask，只使用边界框
    )

    # 加载图像和mask数据
    img_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/mm_datasets/lihao_3d_data/processed_scannetpp_fix/0cf2e9402d/images/frame_000170.jpg"
    mask_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/mm_datasets/lihao_3d_data/processed_scannetpp_fix/0cf2e9402d/obj_ids/frame_000170.jpg.pth"

    # 读取数据
    image = np.array(Image.open(img_path).convert('RGB'))
    mask_data = torch.load(mask_path, weights_only=False)

    # 获取原始mask信息
    unique_ids = np.unique(mask_data)
    valid_ids = unique_ids[unique_ids != 0]

    print(f"原始mask数量: {len(valid_ids)}")

    # 细化mask (只使用边界框)
    refined_masks = refiner.refine_masks(image, mask_data)
    print(f"细化后mask数量: {len(refined_masks)}")

    # 逐个比较mask (只显示前5个)
    print("\n=== 前5个mask对比 ===")
    for i in range(min(5, len(valid_ids))):
        obj_id = valid_ids[i]
        original_mask = (mask_data == obj_id).astype(np.uint8)
        original_area = np.sum(original_mask)

        if i < len(refined_masks):
            refined_mask = refined_masks[i]["segmentation"]
            refined_area = refined_masks[i]["area"]
            iou_score = refined_masks[i]["predicted_iou"]
            stability_score = refined_masks[i]["stability_score"]

            # 计算IoU between original and refined
            intersection = np.sum(original_mask * refined_mask)
            union = np.sum((original_mask + refined_mask) > 0)
            actual_iou = intersection / union if union > 0 else 0

            print(f"Mask {i+1} (ID={obj_id}):")
            print(f"  原始面积: {original_area}")
            print(f"  细化面积: {refined_area}")
            print(f"  面积比率: {refined_area/original_area:.3f}")
            print(f"  预测IoU: {iou_score:.3f}")
            print(f"  实际IoU: {actual_iou:.3f}")
            print(f"  稳定性分数: {stability_score:.3f}")
        print()

    # 保存对比图像
    print("保存保守细化对比图像...")

    # 创建并排对比图 (显示所有mask)
    h, w = mask_data.shape
    comparison_vis = np.zeros((h, w*2, 3), dtype=np.float32)

    # 左侧：原始mask (所有)
    N = len(valid_ids)
    cmap = plt.get_cmap('tab20', N)  # 使用tab20以支持更多颜色

    for i in range(N):
        obj_id = valid_ids[i]
        color = np.array(cmap(i % 20)[:3])  # 使用模运算确保颜色索引不超出范围
        mask = (mask_data == obj_id).astype(np.float32)
        for c in range(3):
            comparison_vis[:, :w, c] += mask * color[c]

    # 右侧：细化后的mask (所有)
    for i in range(len(refined_masks)):
        if i < N:  # 确保有对应的颜色
            color = np.array(cmap(i % 20)[:3])
            mask = refined_masks[i]["segmentation"].astype(np.float32)
            for c in range(3):
                comparison_vis[:, w:, c] += mask * color[c]

    # 防止溢出并保存
    comparison_vis = np.clip(comparison_vis, 0, 1)
    comparison_vis_uint8 = (comparison_vis * 255).astype(np.uint8)
    comparison_img = Image.fromarray(comparison_vis_uint8)
    comparison_img.save("conservative_mask_comparison_all.png")
    print(f"保守细化对比图像保存到: conservative_mask_comparison_all.png (共{N}个mask)")

    print("\n=== 保守细化测试完成 ===")
