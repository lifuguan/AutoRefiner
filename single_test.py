import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

checkpoint = "/data/model_zoo/sam/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

img_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/mm_datasets/lihao_3d_data/processed_scannetpp_fix/0cf2e9402d/images/frame_000130.jpg"
mask_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/3A/multimodal/mm_datasets/lihao_3d_data/processed_scannetpp_fix/0cf2e9402d/obj_ids/frame_000130.jpg.pth"
# 读取图片
image = Image.open(img_path).convert('RGB')

# 读取mask标注
mask_data = torch.load(mask_path, weights_only=False)
unique_ids = np.unique(mask_data)
# unique_ids[0] is invalid, so skip it
valid_ids = unique_ids[1:]
# 生成N,H,W的mask，N为有效id的个数
masks = np.stack([(mask_data == obj_id) for obj_id in valid_ids], axis=0)  # [N, H, W]

# 获取image的尺寸
img_w, img_h = image.size
# mask: bool tensor, shape与mask_data一致 -> 转为与image相同预处理空间的低分辨率logits（H=W=image_size//4）
mask_tensor = torch.as_tensor(masks, dtype=torch.float32).unsqueeze(0)  # [1,1,H,W]
low_res_size = predictor.model.image_size // 4  # e.g., 1024//4=256
mask_lowres = F.interpolate(mask_tensor, size=(low_res_size, low_res_size), mode='bilinear', align_corners=False)
# mask_lowres: [1, N, H, W] -> [N, H, W]
mask_lowres_np = mask_lowres.squeeze(0).cpu().numpy()  # [N, H, W]

# 叠加所有mask，每个mask用不同颜色
N, H, W = mask_lowres_np.shape
# 归一化到[0,1]，可选
mask_vis = np.zeros((H, W, 3), dtype=np.float32)
cmap = plt.get_cmap('tab20', N)
for i in range(N):
    color = np.array(cmap(i)[:3])  # RGB
    mask = mask_lowres_np[i]
    # 可以用sigmoid激活
    mask = 1 / (1 + np.exp(-mask))
    mask = (mask > 0.5).astype(np.float32)  # 二值化
    for c in range(3):
        mask_vis[..., c] += mask * color[c]
# 防止叠加溢出
mask_vis = np.clip(mask_vis, 0, 1)

# 将mask_vis从float32[0,1]转为uint8[0,255]
mask_vis_uint8 = (mask_vis * 255).astype(np.uint8)
mask_img = Image.fromarray(mask_vis_uint8)
mask_img.save('mask.png')

# resized_mask_logits = (mask_lowres[0] - 0.5) * 20.0  # [1, low_res, low_res] logits

# 设置图片到SAM2
predictor.set_image(image)

# 重新梳理：对每个低分辨率mask，送入predictor进行细化，收集所有refined mask，最后可视化
num_masks = mask_lowres_np.shape[0]
refine_masks = []

for i in range(num_masks):
    # 取出第i个mask的低分辨率logits，转为torch tensor
    single_logits = (mask_lowres_np[i] - 0.5) * 20.0  # [H, W]
    single_logits_tensor = torch.from_numpy(single_logits).unsqueeze(0)  # [1, H, W]
    # 送入predictor进行refine
    pred_masks, _, _ = predictor.predict(mask_input=single_logits_tensor)
    # pred_masks: [C, H, W]，通常C=1
    # 只取第一个mask
    refine_masks.append(pred_masks[0])

# 堆叠所有refined mask，得到[N, H, W]
refine_masks = np.stack(refine_masks, axis=0)
N, H, W = refine_masks.shape

# 可视化所有refined mask，每个mask用不同颜色
mask_vis = np.zeros((H, W, 3), dtype=np.float32)
cmap = plt.get_cmap('tab20', N)
for i in range(N):
    color = np.array(cmap(i)[:3])  # RGB
    mask = refine_masks[i]
    # 可选：sigmoid激活
    mask = 1 / (1 + np.exp(-mask))
    mask = (mask > 0.5).astype(np.float32)  # 二值化
    for c in range(3):
        mask_vis[..., c] += mask * color[c]
mask_vis = np.clip(mask_vis, 0, 1)

# 使用PIL直接保存
mask_vis_uint8 = (mask_vis * 255).astype(np.uint8)
mask_img = Image.fromarray(mask_vis_uint8)
mask_img.save('mask_refine.png')