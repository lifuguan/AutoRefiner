#!/usr/bin/env python3
"""
测试批量处理脚本
"""

import os
import sys
import subprocess

def test_batch_processing():
    """测试批量处理功能"""

    print("=== 测试批量SAM2 Mask匹配处理 ===")

    # 测试参数
    data_root = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/mm_datasets/lihao_3d_data/processed_scannetpp_fix/"
    max_scenes = 2  # 只测试前2个场景

    # 构建命令
    cmd = [
        "python", "batch_mask_refining.py",
        "--data_root", data_root,
        "--max_scenes", str(max_scenes)
    ]

    print(f"运行命令: {' '.join(cmd)}")
    print(f"测试场景数: {max_scenes}")
    print()

    try:
        # 运行批量处理
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1小时超时

        print("=== 标准输出 ===")
        print(result.stdout)

        if result.stderr:
            print("=== 标准错误 ===")
            print(result.stderr)

        if result.returncode == 0:
            print("✅ 测试成功完成！")
        else:
            print(f"❌ 测试失败，返回码: {result.returncode}")

    except subprocess.TimeoutExpired:
        print("❌ 测试超时")
    except Exception as e:
        print(f"❌ 测试出错: {e}")


if __name__ == "__main__":
    test_batch_processing()
