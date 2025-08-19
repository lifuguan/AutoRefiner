#!/usr/bin/env python3
"""
批量处理脚本v2：改进版本，更好的错误处理和日志
"""

import os
import sys
import glob
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
import argparse
import time
import logging
import contextlib
from mask_refiner_matching import SAM2MaskMatcher


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('batch_processing.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def get_available_gpus():
    """获取可用的GPU数量"""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    else:
        return 0


def get_scene_list(data_root):
    """获取所有场景列表"""
    scene_dirs = []
    for scene_path in glob.glob(os.path.join(data_root, "*")):
        if os.path.isdir(scene_path):
            scene_name = os.path.basename(scene_path)
            # 检查是否有images和obj_ids目录
            images_dir = os.path.join(scene_path, "images")
            obj_ids_dir = os.path.join(scene_path, "obj_ids")
            if os.path.exists(images_dir) and os.path.exists(obj_ids_dir):
                scene_dirs.append(scene_path)

    return sorted(scene_dirs)


def get_image_pairs(scene_path):
    """获取场景中的图像-mask对"""
    images_dir = os.path.join(scene_path, "images")
    obj_ids_dir = os.path.join(scene_path, "obj_ids")

    pairs = []
    for img_file in glob.glob(os.path.join(images_dir, "frame_*.jpg")):
        img_name = os.path.basename(img_file)
        mask_file = os.path.join(obj_ids_dir, img_name + ".pth")

        if os.path.exists(mask_file):
            pairs.append((img_file, mask_file))

    return sorted(pairs)


def process_scene_sequential(scene_path, model_cfg, checkpoint, gpu_id):
    """顺序处理单个场景（避免多进程问题）"""

    scene_name = os.path.basename(scene_path)
    logger = logging.getLogger(__name__)

    # 设置GPU - 在子进程中重新设置CUDA环境
    if gpu_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        # 重新导入torch以确保CUDA正确初始化
        import torch
        # 确保CUDA重新初始化
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_device(0)  # 使用第一个可见的GPU
            logger.info(f"GPU {gpu_id} 初始化成功，设备数量: {torch.cuda.device_count()}")
        else:
            logger.warning(f"GPU {gpu_id} 不可用，将使用CPU")

    # 创建输出目录
    output_dir = os.path.join(scene_path, "refined_ins_ids")
    os.makedirs(output_dir, exist_ok=True)

    # 获取图像对
    image_pairs = get_image_pairs(scene_path)

    if not image_pairs:
        logger.warning(f"场景 {scene_name} 没有找到有效的图像-mask对")
        return {
            'scene': scene_name,
            'status': 'no_images',
            'total_images': 0,
            'successful': 0,
            'failed': 0
        }

    logger.info(f"开始处理场景 {scene_name}，共 {len(image_pairs)} 张图像")

    # 初始化匹配器
    try:
        # 抑制SAM2MaskMatcher的输出
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                matcher = SAM2MaskMatcher(
                    model_cfg=model_cfg,
                    checkpoint=checkpoint,
                    points_per_side=32,
                    pred_iou_thresh=0.7,
                    stability_score_thresh=0.5,
                    box_nms_thresh=0.7,
                    iou_threshold=0.3,
                )
        logger.info(f"场景 {scene_name} SAM2MaskMatcher初始化成功")
    except Exception as e:
        logger.error(f"场景 {scene_name} SAM2MaskMatcher初始化失败: {e}")
        return {
            'scene': scene_name,
            'status': 'init_failed',
            'error': str(e),
            'total_images': len(image_pairs),
            'successful': 0,
            'failed': len(image_pairs)
        }

    # 处理每张图像
    results = []
    successful_count = 0
    failed_count = 0

    with tqdm(image_pairs, desc=f"处理 {scene_name}", leave=False) as pbar:
        for img_path, mask_path in pbar:
            img_name = os.path.basename(img_path)
            output_path = os.path.join(output_dir, img_name + ".pth")

            # 如果输出文件已存在，跳过
            if os.path.exists(output_path):
                successful_count += 1
                pbar.set_postfix({"成功": successful_count, "失败": failed_count})
                continue

            try:
                # 读取图像
                image = Image.open(img_path).convert('RGB')
                image = np.array(image)

                # 处理图像并匹配mask（抑制输出）
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull):
                        matched_masks = matcher.process_image_with_ground_truth(image, mask_path)

                # 加载真值mask数据
                gt_mask_data, valid_ids = matcher.load_ground_truth_masks(mask_path)

                # 创建输出mask
                output_mask = np.zeros_like(gt_mask_data, dtype=np.uint16)

                for mask_info in matched_masks:
                    gt_id = mask_info['ground_truth_id']
                    mask = mask_info['segmentation'].astype(bool)
                    output_mask[mask] = gt_id

                # 保存结果（转换为int32以兼容PyTorch保存）
                torch.save(torch.from_numpy(output_mask.astype(np.int32)), output_path)

                # 计算统计信息
                invalid_ratio = matcher.calculate_invalid_area_ratio(matched_masks, gt_mask_data)
                sam_matched = sum(1 for m in matched_masks if not m.get('is_fallback', False))
                merged_count = sum(1 for m in matched_masks if m.get('is_merged', False))

                result = {
                    'success': True,
                    'img_path': img_path,
                    'total_objects': len(matched_masks),
                    'sam_matched': sam_matched,
                    'merged_count': merged_count,
                    'invalid_ratio': invalid_ratio,
                    'error': None
                }

                successful_count += 1

            except Exception as e:
                logger.error(f"处理图像 {img_path} 失败: {e}")
                result = {
                    'success': False,
                    'img_path': img_path,
                    'error': str(e)
                }
                failed_count += 1

            results.append(result)
            pbar.set_postfix({"成功": successful_count, "失败": failed_count})

    # 计算场景统计
    successful_results = [r for r in results if r['success']]
    if successful_results:
        avg_invalid_ratio = np.mean([r['invalid_ratio'] for r in successful_results])
        total_sam_matched = sum(r['sam_matched'] for r in successful_results)
        total_merged = sum(r['merged_count'] for r in successful_results)
        total_objects = sum(r['total_objects'] for r in successful_results)
    else:
        avg_invalid_ratio = 0
        total_sam_matched = 0
        total_merged = 0
        total_objects = 0

    scene_result = {
        'scene': scene_name,
        'status': 'completed',
        'total_images': len(image_pairs),
        'successful': successful_count,
        'failed': failed_count,
        'avg_invalid_ratio': avg_invalid_ratio,
        'total_sam_matched': total_sam_matched,
        'total_merged': total_merged,
        'total_objects': total_objects
    }

    logger.info(f"场景 {scene_name} 处理完成: {successful_count}/{len(image_pairs)} 成功")

    return scene_result


def process_scenes_on_gpu(scene_list, model_cfg, checkpoint, gpu_id):
    """在指定GPU上处理多个场景"""

    # 在子进程中重新设置环境
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # 重新导入必要的模块
    import torch
    import numpy as np
    from PIL import Image

    # 在子进程中重新设置日志
    logger = setup_logging()
    logger.info(f"GPU {gpu_id} 开始处理 {len(scene_list)} 个场景")

    # 验证CUDA设置
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"GPU {gpu_id} CUDA可用，设备数量: {torch.cuda.device_count()}")
    else:
        logger.warning(f"GPU {gpu_id} CUDA不可用，将使用CPU")

    results = []
    for scene_path in scene_list:
        result = process_scene_sequential(scene_path, model_cfg, checkpoint, gpu_id)
        results.append(result)

        # 记录每个场景的完成情况
        if result['status'] == 'completed':
            logger.info(f"GPU {gpu_id} 完成场景 {result['scene']}: {result['successful']}/{result['total_images']} 成功")
        else:
            logger.warning(f"GPU {gpu_id} 场景 {result['scene']} 处理失败: {result['status']}")

    logger.info(f"GPU {gpu_id} 完成所有场景处理")
    return results


def main():
    # 设置多进程启动方法为spawn以避免CUDA问题
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="批量处理SAM2 mask匹配 v2")
    parser.add_argument("--data_root", type=str,
                       default="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/mm_datasets/lihao_3d_data/processed_scannetpp_fix/",
                       help="数据根目录")
    parser.add_argument("--model_cfg", type=str, default="sam2_hiera_l.yaml", help="SAM2模型配置文件")
    parser.add_argument("--checkpoint", type=str, default="model_zoo/sam/sam2_hiera_large.pt", help="SAM2模型权重文件")
    parser.add_argument("--max_scenes", type=int, default=None, help="最大处理场景数（用于测试）")
    parser.add_argument("--sequential", action="store_true", help="顺序处理（避免多进程问题）")

    args = parser.parse_args()

    # 设置日志
    logger = setup_logging()

    logger.info("=== 批量SAM2 Mask匹配处理 v2 ===")
    logger.info(f"数据根目录: {args.data_root}")
    logger.info(f"模型配置: {args.model_cfg}")
    logger.info(f"模型权重: {args.checkpoint}")

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
        logger.error("找不到SAM2模型文件")
        return

    logger.info(f"使用模型文件: {checkpoint}")

    # 检查GPU
    num_gpus = get_available_gpus()
    if num_gpus == 0:
        logger.warning("没有检测到可用的GPU，将使用CPU处理（速度较慢）")
        num_gpus = 1
    else:
        logger.info(f"检测到 {num_gpus} 个GPU")

    # 获取场景列表
    scene_list = get_scene_list(args.data_root)
    if not scene_list:
        logger.error(f"在 {args.data_root} 中没有找到有效的场景")
        return

    if args.max_scenes:
        scene_list = scene_list[:args.max_scenes]

    logger.info(f"找到 {len(scene_list)} 个场景")

    # 处理所有场景
    all_results = []

    if not args.sequential:
        # 多GPU并行处理（强制使用多进程）
        logger.info(f"使用 {num_gpus} 个GPU并行处理")

        # 将场景分组，每个GPU处理一组
        scene_groups = [[] for _ in range(num_gpus)]
        for i, scene_path in enumerate(scene_list):
            scene_groups[i % num_gpus].append(scene_path)

        logger.info(f"场景分组: {[len(group) for group in scene_groups]}")

        # 创建进程池
        with mp.Pool(processes=num_gpus) as pool:
            # 为每个GPU创建一个任务
            tasks = []
            for gpu_id in range(num_gpus):
                if scene_groups[gpu_id]:  # 如果该GPU有场景要处理
                    task = pool.apply_async(
                        process_scenes_on_gpu,
                        (scene_groups[gpu_id], args.model_cfg, checkpoint, gpu_id)
                    )
                    tasks.append(task)

            # 等待所有任务完成并收集结果
            with tqdm(total=len(scene_list), desc="处理场景") as pbar:
                completed_scenes = 0
                while completed_scenes < len(scene_list):
                    for task in tasks:
                        if task.ready():
                            try:
                                gpu_results = task.get()
                                all_results.extend(gpu_results)
                                completed_scenes += len(gpu_results)
                                pbar.update(len(gpu_results))
                                tasks.remove(task)
                                break
                            except Exception as e:
                                logger.error(f"GPU任务失败: {e}")
                                tasks.remove(task)
                                break
                    time.sleep(0.1)  # 短暂等待
    else:
        # 单GPU或CPU顺序处理
        logger.info("使用单GPU/CPU顺序处理")
        with tqdm(scene_list, desc="处理场景") as pbar:
            for i, scene_path in enumerate(pbar):
                gpu_id = 0 if num_gpus > 0 else -1

                result = process_scene_sequential(scene_path, args.model_cfg, checkpoint, gpu_id)
                all_results.append(result)

                # 更新进度条描述
                if result['status'] == 'completed':
                    pbar.set_postfix({
                        "当前场景": result['scene'],
                        "成功率": f"{result['successful']}/{result['total_images']}"
                    })

    # 输出最终统计
    logger.info("\n=== 最终统计 ===")
    total_scenes = len(all_results)
    completed_scenes = sum(1 for r in all_results if r['status'] == 'completed')
    total_images = sum(r['total_images'] for r in all_results)
    total_successful = sum(r['successful'] for r in all_results)

    logger.info(f"总场景数: {total_scenes}")
    logger.info(f"完成场景数: {completed_scenes}")
    logger.info(f"总图像数: {total_images}")
    logger.info(f"成功处理: {total_successful}")
    logger.info(f"成功率: {total_successful/total_images*100:.1f}%")

    # 详细结果
    for result in all_results:
        if result['status'] == 'completed':
            logger.info(f"✓ {result['scene']}: {result['successful']}/{result['total_images']} 成功, "
                       f"Invalid: {result['avg_invalid_ratio']:.1%}, "
                       f"SAM匹配: {result['total_sam_matched']}, "
                       f"合并: {result['total_merged']}")
        else:
            logger.warning(f"⚠ {result['scene']}: {result['status']}")

    logger.info("\n=== 批量处理完成 ===")
    logger.info("输出文件保存在各场景的 obj_ids_refine/ 目录下")


if __name__ == "__main__":
    main()
