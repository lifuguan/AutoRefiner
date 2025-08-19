# SAM2 批量Mask匹配处理

这个批量处理系统可以自动处理大量场景的mask匹配任务，支持多GPU并行处理。

## 🚀 主要特性

- **多GPU并行处理**：自动检测可用GPU数量，并行处理多个场景
- **智能mask合并**：解决墙壁等大型区域被割裂的问题
- **进度监控**：使用tqdm实时显示处理进度
- **断点续传**：自动跳过已处理的文件
- **统计报告**：详细的处理统计信息

## 📁 文件结构

```
AutoRefiner/
├── batch_mask_refining.py      # v1: 原始多进程版本
├── batch_mask_refining_v2.py   # v2: 改进版，支持多GPU并行
├── batch_mask_refining_v3.py   # v3: 简化多GPU并行版本（推荐）
├── run_batch_refining.sh       # 启动脚本
├── test_batch_refining.py      # 测试脚本
├── debug_single_image.py       # 单张图像调试脚本
├── mask_refiner_matching.py    # 核心mask匹配器
└── README_batch_processing.md  # 使用说明
```

## 🔧 环境要求

- Python环境：vggt
- GPU：支持CUDA的GPU（可选，CPU也可运行但较慢）
- 依赖包：torch, numpy, PIL, tqdm, opencv-python

## 📊 数据格式

### 输入数据结构
```
data_root/
├── scene1/
│   ├── images/
│   │   ├── frame_000000.jpg
│   │   ├── frame_000001.jpg
│   │   └── ...
│   └── obj_ids/
│       ├── frame_000000.jpg.pth
│       ├── frame_000001.jpg.pth
│       └── ...
├── scene2/
│   ├── images/
│   └── obj_ids/
└── ...
```

### 输出数据结构
```
data_root/
├── scene1/
│   ├── images/
│   ├── obj_ids/
│   └── obj_ids_refine/          # 新生成的目录
│       ├── frame_000000.jpg.pth # uint16格式的refined mask
│       ├── frame_000001.jpg.pth
│       └── ...
└── ...
```

## 🚀 使用方法

### 版本选择

- **v1 (batch_mask_refining.py)**: 原始版本，基本多进程
- **v2 (batch_mask_refining_v2.py)**: 改进版本，支持多GPU并行，有详细日志
- **v3 (batch_mask_refining_v3.py)**: 简化版本，真正的多GPU并行（推荐）

### 1. 基本使用

```bash
# 激活环境
export PATH="/home/sankuai/conda/envs/vggt/bin:$PATH"

# 推荐使用v3版本（多GPU并行）
python batch_mask_refining_v3.py \
    --data_root "/path/to/your/data/" \
    --model_cfg "sam2_hiera_l.yaml" \
    --checkpoint "model_zoo/sam/sam2_hiera_large.pt"

# 或使用v2版本（更详细的日志）
python batch_mask_refining_v2.py \
    --data_root "/path/to/your/data/" \
    --model_cfg "sam2_hiera_l.yaml" \
    --checkpoint "model_zoo/sam/sam2_hiera_large.pt"
```

### 2. 使用启动脚本

```bash
# 给脚本添加执行权限
chmod +x run_batch_refining.sh

# 运行
./run_batch_refining.sh
```

### 3. 测试模式

```bash
# 只处理前2个场景进行测试
python batch_mask_refining.py --max_scenes 2
```

## ⚙️ 参数说明

- `--data_root`: 数据根目录路径
- `--model_cfg`: SAM2模型配置文件
- `--checkpoint`: SAM2模型权重文件路径
- `--max_scenes`: 最大处理场景数（用于测试）

## 📈 处理流程

1. **场景扫描**：自动扫描数据目录，找到所有有效场景
2. **GPU分配**：根据可用GPU数量分配处理任务
3. **并行处理**：每个GPU处理不同的场景
4. **Mask匹配**：对每张图像进行SAM2 mask生成和匹配
5. **智能合并**：合并割裂的mask片段
6. **结果保存**：以uint16格式保存refined mask

## 📊 输出统计

处理完成后会显示详细统计信息：
- 总场景数和处理成功数
- 每个场景的处理结果
- SAM2匹配成功率
- Invalid区域比例
- 合并mask数量

## 🔍 监控和调试

### 进度监控
- 使用tqdm显示整体进度
- 实时显示当前处理的场景和图像
- 显示每个场景的处理统计

### 错误处理
- 自动跳过损坏的文件
- 记录处理失败的图像
- 提供详细的错误信息

## 🎯 性能优化

### GPU利用率
- 自动检测GPU数量
- 智能分配GPU资源
- 避免GPU内存溢出

### 处理速度
- 多进程并行处理
- 断点续传功能
- 优化的mask合并算法

## 📝 示例输出

```
=== 批量SAM2 Mask匹配处理 ===
数据根目录: /path/to/data/
模型配置: sam2_hiera_l.yaml
模型权重: model_zoo/sam/sam2_hiera_large.pt
使用模型文件: model_zoo/sam/sam2_hiera_large.pt
检测到 2 个GPU
找到 150 个场景

处理场景: 100%|██████████| 150/150 [2:30:45<00:00, 60.30s/scene]

✓ scene001: 50/50 成功, Invalid: 12.3%, SAM匹配: 780, 合并: 45
✓ scene002: 48/50 成功, Invalid: 15.1%, SAM匹配: 720, 合并: 38
...

=== 批量处理完成 ===
所有场景处理完成！
输出文件保存在各场景的 obj_ids_refine/ 目录下
```

## 🛠️ 故障排除

### 常见问题

1. **GPU内存不足**
   - 减少`points_per_side`参数
   - 降低`stability_score_thresh`

2. **处理速度慢**
   - 检查GPU是否正常工作
   - 确认数据读取路径是否在高速存储上

3. **文件权限问题**
   - 确保对输出目录有写权限
   - 检查输入文件的读权限

### 调试模式

```bash
# 启用详细日志
python batch_mask_refining.py --max_scenes 1 --verbose

# 测试单个场景
python test_batch_refining.py
```

## 📞 技术支持

如有问题，请检查：
1. 环境配置是否正确
2. 模型文件是否存在
3. 数据格式是否符合要求
4. GPU驱动是否正常

---

**注意**：首次运行时会下载SAM2模型，请确保网络连接正常。
