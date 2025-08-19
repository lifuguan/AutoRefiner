from fastapi import FastAPI
from fastapi import BackgroundTasks
from ray import serve
import threading
import asyncio
import pycocotools.mask as mask_util
from loguru import logger
from pydantic import BaseModel
from stepcast import ServeEngine
from stepcast.models import ModelConfig, CustomEngineConfig
import megfile
import json
import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from tqdm import tqdm
import tempfile

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2, build_sam2_video_predictor

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

os.environ["OSS_ENDPOINT"] = "http://oss.i.basemind.com"

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0,0,1,0.4), thickness=1) 

    ax.imshow(img)

def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.
    Accelerated version.

    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """
    # Ensure tensors are on the correct device (e.g., GPU)
    device = masks.device
    
    # 1. Early filtering by score threshold
    # Sort in descending order of scores
    scores_sorted, original_indices = scores.sort(0, descending=True)
    
    # Apply score threshold *before* NMS
    keep_by_score = (scores_sorted > score_thr)
    scores_filtered = scores_sorted[keep_by_score]
    indices_filtered = original_indices[keep_by_score]
    
    if indices_filtered.numel() == 0:
        # If no masks pass the score threshold, return top 3 original masks if specified in kwargs,
        # otherwise return empty. This logic is a bit unusual, might need clarification.
        # Original logic: if keep_conf.sum() == 0: index = scores.topk(3).indices; keep_conf[index, 0] = True
        # For simplicity, if no masks pass score_thr, return empty. Adjust if specific fallback is needed.
        if 'return_top_k_if_empty' in kwargs and kwargs['return_top_k_if_empty'] > 0:
            top_k = kwargs['return_top_k_if_empty']
            return original_indices[:min(top_k, original_indices.numel())]
        else:
            return torch.empty(0, dtype=torch.long, device=device)

    num_masks = indices_filtered.shape[0]
    masks_ord = masks[indices_filtered] # Already sorted by score and filtered
    masks_area = torch.sum(masks_ord, dim=(-1, -2), dtype=torch.float)

    # Use a chunk size that balances memory and parallelism.
    # The original mask_chunk_size was 20. For GPU, you might be able to increase this,
    # but it depends on your GPU memory.
    mask_chunk_size = 25 # Increased for potential GPU parallelism
    
    # If num_masks is small, avoid chunking overhead
    if num_masks < mask_chunk_size * 2: # Arbitrary heuristic
        mask_chunk_size = num_masks

    mask_chunks = masks_ord.split(mask_chunk_size, dim=0)
    area_chunks = masks_area.split(mask_chunk_size, dim=0)

    # Initialize full matrices for IoU and inner IoU
    # It's crucial to pre-allocate on the GPU
    iou_matrix = torch.zeros((num_masks, num_masks), device=device, dtype=torch.float)
    inner_iou_matrix = torch.zeros((num_masks, num_masks), device=device, dtype=torch.float)

    # Fill the matrices in a blocked fashion
    current_row_idx = 0
    for i, (i_areas, i_chunk) in enumerate(zip(area_chunks, mask_chunks)):
        current_col_idx = 0
        for j, (j_areas, j_chunk) in enumerate(zip(area_chunks, mask_chunks)):
            # Perform batch intersection and union calculations
            # unsqueeze(1) and unsqueeze(0) for broadcasting
            intersection = torch.logical_and(i_chunk.unsqueeze(1), j_chunk.unsqueeze(0)).sum(dim=(-1, -2))
            union = torch.logical_or(i_chunk.unsqueeze(1), j_chunk.unsqueeze(0)).sum(dim=(-1, -2))
            
            # Avoid division by zero for empty masks (union=0). Set IoU to 0.
            # This handles cases where both masks are empty or one is empty and the other isn't.
            local_iou_mat = torch.where(union > 0, intersection / union, torch.zeros_like(intersection, dtype=torch.float))
            
            # Inner IoU calculations
            row_inter_mat = torch.where(i_areas[:, None] > 0, intersection / i_areas[:, None], torch.zeros_like(intersection, dtype=torch.float))
            col_inter_mat = torch.where(j_areas[None, :] > 0, intersection / j_areas[None, :], torch.zeros_like(intersection, dtype=torch.float))

            inter_condition = torch.logical_and(row_inter_mat < 0.5, col_inter_mat >= 0.85)
            
            local_inner_iou_mat = torch.zeros_like(local_iou_mat) # Use same shape
            local_inner_iou_mat[inter_condition] = 1 - row_inter_mat[inter_condition] * col_inter_mat[inter_condition]

            # Assign results to the pre-allocated matrices
            iou_matrix[current_row_idx : current_row_idx + i_chunk.shape[0], 
                       current_col_idx : current_col_idx + j_chunk.shape[0]] = local_iou_mat
            inner_iou_matrix[current_row_idx : current_row_idx + i_chunk.shape[0], 
                             current_col_idx : current_col_idx + j_chunk.shape[0]] = local_inner_iou_mat
            
            current_col_idx += j_chunk.shape[0]
        current_row_idx += i_chunk.shape[0]

    # Apply upper triangle for IoU and max over columns
    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)

    # Apply upper and lower triangles for inner IoU and max over columns
    # Note: If inner_iou_max_u and inner_iou_max_l are applied to the same original matrix,
    # the diagonal=1 for upper and diagonal=0 for lower would be more common.
    # Your original code used diagonal=1 for triu and diagonal=1 for tril, which means
    # the main diagonal is excluded from triu and included in tril.
    # Let's assume this is intended.
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=0) # Changed from 1 to 0 to be more standard, 
                                                               # if you want to include diagonal in lower part.
                                                               # Revert to diagonal=1 if original behavior is desired.
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    # Calculate final 'keep' mask
    keep = iou_max <= iou_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr

    # Apply all conditions
    keep = keep * keep_inner_u * keep_inner_l

    # Handle the cases where all masks might be suppressed by one of the inner conditions.
    # This part of your original logic is quite specific and might be a fallback
    # to ensure at least some masks are kept.
    # If a condition suppresses all masks, it forces the top 3 (by original score) to be kept for that condition.
    # This can override the NMS logic for certain cases, so be mindful of its implications.
    # For a standard NMS, you'd usually just return what's left after filtering.
    
    # Example for `keep_inner_u`:
    if keep_inner_u.sum() == 0 and scores_filtered.numel() > 0: # Check if there are any filtered masks to consider
        # Take the indices of the top 3 *from the filtered scores*, relative to `indices_filtered`
        top_k_indices = scores_filtered.topk(min(3, scores_filtered.numel())).indices
        keep_inner_u[top_k_indices] = True
        
    if keep_inner_l.sum() == 0 and scores_filtered.numel() > 0:
        top_k_indices = scores_filtered.topk(min(3, scores_filtered.numel())).indices
        keep_inner_l[top_k_indices] = True
        
    # Re-apply final combined filter after potential top-k insertions
    # No need to re-apply `keep_conf` as it was applied at the very beginning by filtering `scores_filtered`
    final_keep_mask = (iou_max <= iou_thr) * keep_inner_u * keep_inner_l
    
    selected_idx = indices_filtered[final_keep_mask]

    # 显式释放大 tensor
    for var in ['masks_ord', 'masks_area', 'iou_matrix', 'inner_iou_matrix']:
        if var in locals():
            del locals()[var]
    torch.cuda.empty_cache()

    return selected_idx

def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep

def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    masks_new = ()
    for masks_lvl in (args):
        if isinstance(masks_lvl, tuple):
            masks_lvl = masks_lvl[0]  # If it's a tuple, take the first element
        if len(masks_lvl) == 0:
            masks_new += (masks_lvl,)
            continue
            
        # Check if masks_lvl is a list of dictionaries
        if isinstance(masks_lvl[0], dict):
            seg_pred = torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0)).cuda()
            iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0)).cuda()
            stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0)).cuda()
        else:
            # If it's a direct list of masks, use them directly
            seg_pred = torch.from_numpy(np.stack(masks_lvl, axis=0)).cuda()
            # Create default values for cases without iou and stability
            iou_pred = torch.ones(len(masks_lvl)).cuda()
            stability = torch.ones(len(masks_lvl)).cuda()

        # 计算 mask 的综合分数，分数由稳定性分数和预测的 IOU 相乘得到
        scores = stability * iou_pred
        print(f"准备NMS，掩码数量: {len(seg_pred)}, 分数范围: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        print(f"NMS后保留掩码数量: {len(keep_mask_nms)}")
        masks_lvl = filter(keep_mask_nms, masks_lvl)
        masks_new += (masks_lvl,)
    return masks_new

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab20")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def save_mask(mask,frame_idx,save_dir):
    image_array = (mask * 255).astype(np.uint8)
    # Create image object
    image = Image.fromarray(image_array[0])

    # Save image
    image.save(os.path.join(save_dir,f'{frame_idx:03}.png'))

def save_masks(mask_list,frame_idx,save_dir):
    os.makedirs(save_dir,exist_ok=True)
    if len(mask_list[0].shape) == 3:
        # Calculate dimensions for concatenated image
        total_width = mask_list[0].shape[2] * len(mask_list)
        max_height = mask_list[0].shape[1]
        # Create large image
        final_image = Image.new('RGB', (total_width, max_height))
        for i, img in enumerate(mask_list):
            img = Image.fromarray((img[0] * 255).astype(np.uint8)).convert("RGB")
            final_image.paste(img, (i * img.width, 0))
        final_image.save(os.path.join(save_dir,f"mask_{frame_idx:03}.png"))
    else:
        # Calculate dimensions for concatenated image
        total_width = mask_list[0].shape[1] * len(mask_list)
        max_height = mask_list[0].shape[0]
        # Create large image
        final_image = Image.new('RGB', (total_width, max_height))
        for i, img in enumerate(mask_list):
            img = Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")
            final_image.paste(img, (i * img.width, 0))
        final_image.save(os.path.join(save_dir,f"mask_{frame_idx:03}.png"))

def save_masks_npy(mask_list,frame_idx,save_dir):
    np.save(os.path.join(save_dir,f"mask_{frame_idx:03}.npy"),np.array(mask_list))
    
def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)  

def make_enlarge_bbox(origin_bbox, max_width,max_height,ratio):
    width = origin_bbox[2]
    height = origin_bbox[3]
    new_box = [max(origin_bbox[0]-width*(ratio-1)/2,0),max(origin_bbox[1]-height*(ratio-1)/2,0)]
    new_box.append(min(width*ratio,max_width-new_box[0]))
    new_box.append(min(height*ratio,max_height-new_box[1]))
    return new_box

def sample_points(masks, enlarge_bbox,positive_num=1,negtive_num=40):
    ex, ey, ewidth, eheight = enlarge_bbox
    positive_count = positive_num
    negtive_count = negtive_num
    output_points = []
    while True:
        x = int(np.random.uniform(ex, ex + ewidth))
        y = int(np.random.uniform(ey, ey + eheight))
        if masks[y][x]==True and positive_count>0:
            output_points.append((x,y,1))
            positive_count-=1
        elif masks[y][x]==False and negtive_count>0:
            output_points.append((x,y,0))
            negtive_count-=1
        if positive_count == 0 and negtive_count == 0:
            break

    return output_points

def sample_points_from_mask(mask):
    # Get indices of all True values
    true_indices = np.argwhere(mask)

    # Check if there are any True values
    if true_indices.size == 0:
        raise ValueError("The mask does not contain any True values.")

    # Randomly select a point from True value indices
    random_index = np.random.choice(len(true_indices))
    sample_point = true_indices[random_index]

    return tuple(sample_point)

def search_new_obj(masks_from_prev, mask_list,other_masks_list=None,mask_ratio_thresh=0.00,ratio=0.5, area_threash = 5000):
    new_mask_list = []

    # Calculate mask_none, representing areas not included in any previous masks
    mask_none = ~masks_from_prev[0].copy()[0]
    for prev_mask in masks_from_prev[1:]:
        mask_none &= ~prev_mask[0]

    for mask in mask_list:
        seg = mask['segmentation']
        if (mask_none & seg).sum()/seg.sum() > ratio and seg.sum() > area_threash:
            new_mask_list.append(mask)
    
    for mask in new_mask_list:
        mask_none &= ~mask['segmentation']
    print(len(new_mask_list))
    print("now ratio:",mask_none.sum() / (mask_none.shape[0] * mask_none.shape[1]) )
    print("expected ratios:",mask_ratio_thresh)
    if other_masks_list is not None:
        for mask in other_masks_list:
            if mask_none.sum() / (mask_none.shape[0] * mask_none.shape[1]) > mask_ratio_thresh: # Still a lot of gaps, greater than current thresh
                seg = mask['segmentation']
                if (mask_none & seg).sum()/seg.sum() > ratio and seg.sum() > area_threash:
                    new_mask_list.append(mask)
                    mask_none &= ~seg
            else:
                break
    print(len(new_mask_list))

    return new_mask_list

def get_bbox_from_mask(mask):
    # Get row and column indices of non-zero elements
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    # Find min and max indices of non-zero rows and columns
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    # Calculate width and height
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    
    return xmin, ymin, width, height

def cal_no_mask_area_ratio(out_mask_list):
    h = out_mask_list[0].shape[1]
    w = out_mask_list[0].shape[2]
    mask_none = ~out_mask_list[0].copy()
    for prev_mask in out_mask_list[1:]:
        mask_none &= ~prev_mask
    return(mask_none.sum() / (h * w))


class Prompts:
    def __init__(self,bs:int):
        self.batch_size = bs
        self.prompts = {}
        self.obj_list = []
        self.key_frame_list = []
        self.key_frame_obj_begin_list = []

    def add(self,obj_id,frame_id,mask):
        if obj_id not in self.obj_list:
            new_obj = True
            self.prompts[obj_id] = []
            self.obj_list.append(obj_id)
        else:
            new_obj = False
        self.prompts[obj_id].append((frame_id,mask))
        if frame_id not in self.key_frame_list and new_obj:
            self.key_frame_list.append(frame_id)
            self.key_frame_obj_begin_list.append(obj_id)
            print("key_frame_obj_begin_list:",self.key_frame_obj_begin_list)
    
    def get_obj_num(self):
        return len(self.obj_list)
    
    def __len__(self):
        if self.obj_list % self.batch_size == 0:
            return len(self.obj_list) // self.batch_size
        else:
            return len(self.obj_list) // self.batch_size +1
    
    def __iter__(self):
        # self.batch_index = 0
        self.start_idx = 0
        self.iter_frameindex = 0
        return self

    def __next__(self):
        if self.start_idx < len(self.obj_list):
            if self.iter_frameindex == len(self.key_frame_list)-1:
                end_idx = min(self.start_idx+self.batch_size, len(self.obj_list))
            else:
                if self.start_idx+self.batch_size < self.key_frame_obj_begin_list[self.iter_frameindex+1]:
                    end_idx = self.start_idx+self.batch_size
                else:
                    end_idx =  self.key_frame_obj_begin_list[self.iter_frameindex+1]
                    self.iter_frameindex+=1
                # end_idx = min(self.start_idx+self.batch_size, self.key_frame_obj_begin_list[self.iter_frameindex+1])
            batch_keys = self.obj_list[self.start_idx:end_idx]
            batch_prompts = {key: self.prompts[key] for key in batch_keys}
            self.start_idx = end_idx
            return batch_prompts
        # if self.batch_index * self.batch_size < len(self.obj_list):
        #     start_idx = self.batch_index * self.batch_size
        #     end_idx = min(start_idx + self.batch_size, len(self.obj_list))
        #     batch_keys = self.obj_list[start_idx:end_idx]
        #     batch_prompts = {key: self.prompts[key] for key in batch_keys}
        #     self.batch_index += 1
        #     return batch_prompts
        else:
            raise StopIteration
        
def get_video_segments(prompts_loader,predictor,inference_state,final_output=False):

    video_segments = {}
    print(f"processing prompts batch, total batches...")
    for _, batch_prompts in enumerate(prompts_loader):
        predictor.reset_state(inference_state)
        for id, prompt_list in batch_prompts.items():
            for prompt in prompt_list:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=prompt[0],
                        obj_id=id,
                        mask=prompt[1]
                    )
        # start_frame_idx = 0 if final_output else None
        # import ipdb; ipdb.set_trace()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                if out_frame_idx not in video_segments:
                    video_segments[out_frame_idx] = { }
                for i, out_obj_id in enumerate(out_obj_ids):
                    video_segments[out_frame_idx][out_obj_id]= (out_mask_logits[i] > 0.0).cpu().numpy()
            
            if final_output:
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,reverse=True):
                    for i, out_obj_id in enumerate(out_obj_ids):
                        video_segments[out_frame_idx][out_obj_id]= (out_mask_logits[i] > 0.0).cpu().numpy()
    return video_segments

app = FastAPI()

class MaskRequest(BaseModel):
    image_dir: str
    output_dir: str
    level: str = "large"
    batch_size: int = 200
    detect_stride: int = 20
    use_other_level: int = 1
    postnms: int = 1
    pred_iou_thresh: float = 0.7
    box_nms_thresh: float = 0.7
    stability_score_thresh: float = 0.85

class MaskResponse(BaseModel):
    status: str = "success"
    message: str = "OK"
    output_path: str = "nowhere"

@serve.deployment(
    ray_actor_options={
        "num_cpus": 16, 
        "num_gpus": 1,
    }
)
@serve.ingress(app)
class AutoMaskService(ServeEngine):
    def __init__(self, name: str, model_config: ModelConfig, engine_config: CustomEngineConfig):
        super().__init__()
        
        # Load SAM2 model
        model_path = model_config.storage.path
        custom_args = engine_config.custom_args
        self.sam2_checkpoint = custom_args.get("sam2_checkpoint", "/data/model_zoo/sam/sam2_hiera_large.pt") # "/data/model_zoo/sam/sam2_hiera_large.pt"
        self.model_cfg = "sam2_hiera_l.yaml"
        self.predictor = build_sam2_video_predictor("sam2_hiera_l.yaml", self.sam2_checkpoint)
        
        # Load SAM1 model
        self.sam_ckpt_path = custom_args.get("sam_ckpt_path", "/data/model_zoo/sam/sam_vit_l_0b3195.pth")
        self.sam = sam_model_registry["vit_l"](checkpoint=self.sam_ckpt_path).to('cuda')
        
        # Initialize mask generator
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=0.7,
            box_nms_thresh=0.7,
            stability_score_thresh=0.85,
            crop_n_layers=1,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=100,
        )
        
        self._logger = self.get_logger()
        self._lock = threading.Lock()

    def process_video(self, request: MaskRequest) -> dict:
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA current device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

        # try:
        if True:
            print("image dir: ", request.image_dir)
            # Get frame names
            frame_names = [
                p for p in megfile.s3_listdir(request.image_dir)
                if os.path.splitext(p)[-1].lower() in ['.jpg', '.jpeg', '.png']
            ]
            frame_names.sort(key=lambda p: int(p.split('_')[1].split('.')[0]))
            
            # Get video dimensions
            first_image_path = os.path.join(request.image_dir, frame_names[0])
            if isinstance(first_image_path, str) and first_image_path.startswith("s3://"):
                with megfile.smart_open(first_image_path, 'rb') as f:
                    img_pil = Image.open(f)
                    img_pil = img_pil.convert("RGB") 
                first_image = np.array(img_pil)
            else:
                first_image = cv2.imread(first_image_path)
            if first_image is None:
                raise ValueError(f"Cannot read image: {first_image_path}")
            height, width = first_image.shape[:2]
            
            # Initialize processing
            now_frame = 0
            inference_state = self.predictor.init_state(video_path=request.image_dir)
            masks_from_prev = []
            sum_id = 0 # 记录物体总数

            prompts_loader = Prompts(bs=request.batch_size)  # 保存所有的点击用于可视化
            
            while True:
                print(f"正在处理帧: {now_frame} / 总帧数: {len(frame_names)} (帧名: {frame_names[now_frame]})")
                
                sum_id = prompts_loader.get_obj_num()
                image_path = os.path.join(request.image_dir, frame_names[now_frame])
            
                if isinstance(image_path, str) and image_path.startswith("s3://"):
                    with megfile.smart_open(image_path, 'rb') as f:
                        img_pil = Image.open(f)
                        img_pil = img_pil.convert("RGB") 
                    image = np.array(img_pil)
                else:
                    image = cv2.imread(image_path)
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # 如果输入太大则调整大小
                orig_h, orig_w = image.shape[:2]
                if orig_h > 1080:
                    print("将原始图像调整为1080P...")
                    scale = 1080 / orig_h
                    h = int(orig_h * scale)
                    w = int(orig_w * scale)
                    image = cv2.resize(image, (w, h))

                # 只生成大掩码
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    masks_l = self.mask_generator.generate(image)
                if request.postnms:
                    masks_l = masks_update(masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)[0]

                # 使用大级别掩码
                masks = masks_l
                other_masks = None
                
                if not request.use_other_level:
                    other_masks = None

                if now_frame == 0: # 第一帧
                    ann_obj_id_list = range(len(masks))
                    for ann_obj_id in ann_obj_id_list:
                        seg = masks[ann_obj_id]['segmentation']
                        prompts_loader.add(ann_obj_id, 0, seg)
                else:  
                    new_mask_list = search_new_obj(masks_from_prev, masks, other_masks, mask_ratio_thresh)
                    print(f"新物体数量: {len(new_mask_list)}，当前帧: {now_frame}，上一帧物体数: {len(masks_from_prev)}，当前检测掩码数: {len(masks)}，掩码阈值: {mask_ratio_thresh:.4f}")

                    for id, mask in enumerate(masks_from_prev):
                        if mask.sum() == 0:
                            continue
                        prompts_loader.add(id, now_frame, mask[0])

                    # 再将新物体的掩码加入prompts_loader
                    for i in range(len(new_mask_list)):
                        new_mask = new_mask_list[i]['segmentation']
                        prompts_loader.add(sum_id+i, now_frame, new_mask)

                # 打印当前物体总数及当前帧索引
                print(f"[Frame {now_frame}] 当前物体总数: {prompts_loader.get_obj_num()} (sum_id: {sum_id})")

                # 如果是第一帧或有新物体，重新生成所有帧的分割掩码
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if now_frame==0 or len(new_mask_list)!=0:
                        video_segments = get_video_segments(prompts_loader, self.predictor, inference_state)
                
                # 设置检测步长
                vis_frame_stride = request.detect_stride
                max_area_no_mask = (0, -1)
                # 遍历后续帧，查找未被掩码覆盖区域最大的帧
                print(f"正在查找未被掩码覆盖区域最大的帧，当前帧: {now_frame}, 步长: {vis_frame_stride}")
                for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
                    if out_frame_idx < now_frame:
                        continue
                    
                    # 收集当前帧所有物体的掩码
                    out_mask_list = []
                    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                        out_mask_list.append(out_mask)
                    
                    # 计算未被掩码覆盖的区域比例
                    no_mask_ratio = cal_no_mask_area_ratio(out_mask_list)
                    if now_frame == out_frame_idx:
                        mask_ratio_thresh = no_mask_ratio

                    # 如果未被掩码覆盖的区域比例大于阈值，记录该帧
                    if no_mask_ratio > mask_ratio_thresh + 0.01 and out_frame_idx > now_frame:
                        masks_from_prev = out_mask_list
                        max_area_no_mask = (no_mask_ratio, out_frame_idx)
                        print(max_area_no_mask)
                        break
                # 如果没有找到新的未掩码区域较大的帧，则跳出循环
                if max_area_no_mask[1] == -1:
                    break
                print("max_area_no_mask:", max_area_no_mask)
                # 跳转到未掩码区域最大的帧，继续处理
                now_frame = max_area_no_mask[1]

            ###### 最终输出 ######
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                video_segments = get_video_segments(prompts_loader, self.predictor, inference_state, final_output=True)
                
            # Save results
            json_output_path = os.path.join(request.output_dir, f"{os.path.basename(request.image_dir)}auto_masks.json")
            video_output_path = os.path.join(request.output_dir, f"{os.path.basename(request.image_dir)}auto_masks.avi")
            self._save_results(video_segments, height, width, len(frame_names), json_output_path, video_output_path)
            
            return {
                "status": "success",
                "message": "Video processing completed successfully",
                "output_path": json_output_path
            }
            
        # except Exception as e:
        #     self._logger.error(f"Error processing video: {str(e)}")
        #     return {
        #         "status": "error",
        #         "message": str(e)
        #     }

    def _save_results(self, video_segments, height, width, duration, output_path, video_output_path):
        masklet = []
        video_size = [height, width]
        
        for frame_id in sorted(video_segments.keys()):
            frame_objects = video_segments[frame_id]
            frame_rles = []
            
            for obj_id, mask_array in frame_objects.items():
                mask = mask_array.squeeze(0).astype(np.uint8)
                rle = mask_util.encode(np.asfortranarray(mask))
                rle_dict = {
                    "size": video_size,
                    "counts": rle["counts"].decode("utf-8")
                }
                frame_rles.append(rle_dict)
            
            masklet.append(frame_rles)
        
        save_data = {
            "video_id": os.path.basename(os.path.dirname(output_path)),
            "video_duration": duration,
            "video_frame_count": len(video_segments),
            "video_height": height,
            "video_width": width,
            "video_resolution": height * width,
            "video_environment": "indoor",
            "video_split": "train",
            "masklet": masklet,
            "masklet_id": list(range(len(masklet))),
            "masklet_type": ["manual"] * len(masklet),
            "masklet_num": len(masklet)
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if output_path.startswith("s3://"):
            with megfile.smart_open(output_path, 'w', encoding='utf-8') as f_write:
                json_data = json.dumps(save_data, indent=2)
                f_write.write(json_data)
        else:
            with open(output_path, "w") as f:
                json.dump(save_data, f, indent=2)
        print(f"JSON标注已保存至: {output_path}")

        # 这样子是可以保存视频的，下面是详细解释和稍作注释的版本：
        fourcc = cv2.VideoWriter_fourcc(*'FFV1')
        
        def write_video(writer):
            # 生成每个mask对象的随机颜色
            num_masks = max(len(video_segments[idx]) for idx in video_segments)
            colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_masks)]
        
            # 遍历每一帧，生成彩色掩码帧并写入视频
            for frame_idx in range(len(video_segments)):
                mask_combined = np.zeros((height, width, 3), dtype=np.uint8)
                for obj_id, mask in video_segments[frame_idx].items():
                    mask_area = mask[0] > 0
                    for c in range(3):
                        mask_combined[:, :, c][mask_area] = colors[obj_id][c]
                writer.write(mask_combined)

        if video_output_path.startswith("s3://"):
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_video_file = os.path.join(temp_dir, os.path.basename(video_output_path))
                mask_video = cv2.VideoWriter(temp_video_file, fourcc, 30, (width, height), isColor=True)
                write_video(mask_video)
                mask_video.release()
                megfile.smart_copy(temp_video_file, video_output_path)
        else:
            # 使用FFV1编码器创建VideoWriter对象，输出路径为video_output_path
            mask_video = cv2.VideoWriter(video_output_path, fourcc, 30, (width, height), isColor=True)
            write_video(mask_video)
            # 释放VideoWriter资源，完成视频写入
            mask_video.release()
        
        print(f"掩码视频已保存至: {video_output_path}")

    @app.post('/sam2/video_seg')
    async def api(self, request: MaskRequest, background_tasks: BackgroundTasks):
        # 先判断锁
        if not self._lock.acquire(blocking=False):
            return MaskResponse(
                status="rejected",
                message="Service is busy, another request is being processed. Please try again later."
            )
        # 能获得锁，说明没有任务在跑
        try:
            # 这里把释放锁的动作放到后台任务最后
            def wrapped_process():
                try:
                    self.process_video(request)
                finally:
                    self._lock.release()
            background_tasks.add_task(wrapped_process)
            return MaskResponse(status="success", message="Task started", output_path="nowhere")
        except Exception as e:
            self._lock.release()
            raise e

    @app.get("/healthz")
    def health(self):
        return {"status": "ok"}

    @staticmethod
    def get_logger():
        from loguru import logger
        return logger
