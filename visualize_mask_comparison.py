import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from typing import List, Tuple, Dict, Any


def load_mask_data(mask_path: str) -> Tuple[np.ndarray, List[int]]:
    """
    加载mask数据并提取有效ID

    Arguments:
        mask_path (str): mask文件路径

    Returns:
        mask_data (np.ndarray): mask标注数据
        valid_ids (List[int]): 有效的对象ID列表
    """
    if mask_path.endswith('.pth'):
        # 加载PyTorch tensor格式的mask
        mask_data = torch.load(mask_path, weights_only=False)
        if isinstance(mask_data, torch.Tensor):
            mask_data = mask_data.cpu().numpy()
    else:
        raise ValueError(f"Unsupported mask format: {mask_path}")

    # 提取有效的对象ID
    unique_ids = np.unique(mask_data)
    valid_ids = unique_ids[unique_ids != 0].tolist()  # 跳过背景ID (0)

    return mask_data, valid_ids


def create_colored_mask_visualization(mask_data: np.ndarray, valid_ids: List[int],
                                    color_map: Dict[int, np.ndarray] = None) -> np.ndarray:
    """
    创建彩色mask可视化

    Arguments:
        mask_data (np.ndarray): mask数据
        valid_ids (List[int]): 有效ID列表
        color_map (Dict[int, np.ndarray]): ID到颜色的映射，如果为None则自动生成

    Returns:
        colored_mask (np.ndarray): 彩色mask图像
    """
    h, w = mask_data.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.float32)

    # 如果没有提供颜色映射，则生成一个
    if color_map is None:
        cmap = plt.get_cmap('tab20', len(valid_ids))
        color_map = {}
        for i, obj_id in enumerate(valid_ids):
            color_map[obj_id] = np.array(cmap(i)[:3])

    # 为每个ID分配颜色
    for obj_id in valid_ids:
        if obj_id in color_map:
            mask = (mask_data == obj_id)
            color = color_map[obj_id]
            for c in range(3):
                colored_mask[mask, c] = color[c]

    return colored_mask


def visualize_mask_comparison(mask_path1: str, mask_path2: str,
                            save_path: str = "mask_comparison.png"):
    """
    可视化两个mask文件的比较，检查ID一致性

    Arguments:
        mask_path1 (str): 第一个mask文件路径
        mask_path2 (str): 第二个mask文件路径
        save_path (str): 保存路径
    """

    print(f"加载第一个mask文件: {mask_path1}")
    mask_data1, valid_ids1 = load_mask_data(mask_path1)
    print(f"  尺寸: {mask_data1.shape}")
    print(f"  有效ID数量: {len(valid_ids1)}")
    print(f"  有效ID: {sorted(valid_ids1)}")

    print(f"\n加载第二个mask文件: {mask_path2}")
    mask_data2, valid_ids2 = load_mask_data(mask_path2)
    print(f"  尺寸: {mask_data2.shape}")
    print(f"  有效ID数量: {len(valid_ids2)}")
    print(f"  有效ID: {sorted(valid_ids2)}")

    # 检查ID一致性
    common_ids = set(valid_ids1) & set(valid_ids2)
    unique_ids1 = set(valid_ids1) - set(valid_ids2)
    unique_ids2 = set(valid_ids2) - set(valid_ids1)

    print(f"\n=== ID一致性分析 ===")
    print(f"共同ID数量: {len(common_ids)}")
    print(f"共同ID: {sorted(list(common_ids))}")
    print(f"仅在第一个mask中的ID: {sorted(list(unique_ids1))}")
    print(f"仅在第二个mask中的ID: {sorted(list(unique_ids2))}")

    # 检查invalid区域（ID为0的区域）
    invalid_pixels1 = np.sum(mask_data1 == 0)
    invalid_pixels2 = np.sum(mask_data2 == 0)
    total_pixels1 = mask_data1.size
    total_pixels2 = mask_data2.size

    print(f"\n=== Invalid区域分析 ===")
    print(f"第一个mask的invalid像素数: {invalid_pixels1} / {total_pixels1} ({invalid_pixels1/total_pixels1*100:.2f}%)")
    print(f"第二个mask的invalid像素数: {invalid_pixels2} / {total_pixels2} ({invalid_pixels2/total_pixels2*100:.2f}%)")

    # 为所有ID创建统一的颜色映射
    all_ids = sorted(list(set(valid_ids1) | set(valid_ids2)))
    cmap = plt.get_cmap('tab20', len(all_ids))
    color_map = {}
    for i, obj_id in enumerate(all_ids):
        color_map[obj_id] = np.array(cmap(i)[:3])

    # 创建可视化
    colored_mask1 = create_colored_mask_visualization(mask_data1, valid_ids1, color_map)
    colored_mask2 = create_colored_mask_visualization(mask_data2, valid_ids2, color_map)

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 第一行：原始mask可视化
    axes[0, 0].imshow(colored_mask1)
    axes[0, 0].set_title(f"Mask 1: frame_000000.jpg.pth\n({len(valid_ids1)} objects)", fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(colored_mask2)
    axes[0, 1].set_title(f"Mask 2: frame_000050.jpg.pth\n({len(valid_ids2)} objects)", fontsize=12)
    axes[0, 1].axis('off')

    # 第二行：共同ID和差异分析
    # 创建只显示共同ID的mask
    common_mask1 = np.zeros_like(mask_data1)
    common_mask2 = np.zeros_like(mask_data2)

    for obj_id in common_ids:
        common_mask1[mask_data1 == obj_id] = obj_id
        common_mask2[mask_data2 == obj_id] = obj_id

    common_colored1 = create_colored_mask_visualization(common_mask1, list(common_ids), color_map)
    common_colored2 = create_colored_mask_visualization(common_mask2, list(common_ids), color_map)

    axes[1, 0].imshow(common_colored1)
    axes[1, 0].set_title(f"Common IDs in Mask 1\n({len(common_ids)} objects)", fontsize=12)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(common_colored2)
    axes[1, 1].set_title(f"Common IDs in Mask 2\n({len(common_ids)} objects)", fontsize=12)
    axes[1, 1].axis('off')

    # 添加标签到每个mask
    for ax, mask_data, valid_ids, title_suffix in [
        (axes[0, 0], mask_data1, valid_ids1, "1"),
        (axes[0, 1], mask_data2, valid_ids2, "2"),
        (axes[1, 0], common_mask1, list(common_ids), "1 (common)"),
        (axes[1, 1], common_mask2, list(common_ids), "2 (common)")
    ]:
        # 为每个ID添加标签
        for obj_id in (list(common_ids) if "common" in title_suffix else valid_ids):
            if obj_id == 0:
                continue
            mask = (mask_data == obj_id)
            if mask.sum() > 0:
                y_coords, x_coords = np.where(mask)
                center_y, center_x = np.mean(y_coords), np.mean(x_coords)

                # 根据是否为共同ID选择标签颜色
                label_color = 'white' if obj_id in common_ids else 'yellow'
                label_text = f"ID:{obj_id}"

                ax.text(center_x, center_y, label_text,
                       color=label_color, fontsize=8, ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))

    # 添加图例和统计信息
    legend_text = f"""
    统计信息:
    • 共同ID: {len(common_ids)} 个 (用白色标签显示)
    • 仅在Mask1中: {len(unique_ids1)} 个 (用黄色标签显示)
    • 仅在Mask2中: {len(unique_ids2)} 个 (用黄色标签显示)
    • Mask1 Invalid区域: {invalid_pixels1/total_pixels1*100:.1f}%
    • Mask2 Invalid区域: {invalid_pixels2/total_pixels2*100:.1f}%
    """

    fig.text(0.02, 0.02, legend_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # 为图例留出空间
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n可视化结果已保存到: {save_path}")

    return {
        'common_ids': list(common_ids),
        'unique_ids1': list(unique_ids1),
        'unique_ids2': list(unique_ids2),
        'invalid_ratio1': invalid_pixels1/total_pixels1,
        'invalid_ratio2': invalid_pixels2/total_pixels2
    }


def main():
    """
    主函数
    """
    # 指定的两个mask文件路径
    mask_path1 = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/mm_datasets/lihao_3d_data/processed_scannetpp_fix/036bce3393/obj_ids_refine/frame_000000.jpg.pth"
    mask_path2 = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/mm_datasets/lihao_3d_data/processed_scannetpp_fix/036bce3393/obj_ids_refine/frame_000050.jpg.pth"

    # 检查文件是否存在
    if not os.path.exists(mask_path1):
        print(f"错误: 文件不存在 {mask_path1}")
        return

    if not os.path.exists(mask_path2):
        print(f"错误: 文件不存在 {mask_path2}")
        return

    # 进行可视化比较
    results = visualize_mask_comparison(mask_path1, mask_path2, "mask_comparison_036bce3393.png")

    print(f"\n=== 最终总结 ===")
    print(f"两个mask文件的ID一致性分析完成")
    print(f"共同ID数量: {len(results['common_ids'])}")
    print(f"是否存在invalid区域 (ID=0):")
    print(f"  - frame_000000.jpg.pth: {'是' if results['invalid_ratio1'] > 0 else '否'} ({results['invalid_ratio1']*100:.2f}%)")
    print(f"  - frame_000050.jpg.pth: {'是' if results['invalid_ratio2'] > 0 else '否'} ({results['invalid_ratio2']*100:.2f}%)")


if __name__ == "__main__":
    main()
