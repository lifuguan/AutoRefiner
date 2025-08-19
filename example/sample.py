# import megfile 
# import os
# os.environ["OSS_ENDPOINT"] = "http://oss.i.basemind.com"
# image_save_path = "s3://oss-lihao/cut3r_processed/du.log"
# with megfile.smart_open(image_save_path, 'r') as f:
#     content = f.read()
# # print(content)

# # 读取一个oss文件夹下的所有文件名
# oss_dir = "s3://oss-lihao/cut3r_processed/"
# # file_list = megfile.scandir(oss_dir)
# # print(file_list)
# res = megfile.s3_listdir(oss_dir)
# print(res)

# File name: translator.py

import megfile
import ray
import os
from PIL import Image
import numpy as np
import torch
from ray import serve
from stepcast import ServeEngine
from fastapi import FastAPI
from stepcast.models import ModelConfig, CustomEngineConfig
from pydantic import BaseModel
import os

os.environ["OSS_ENDPOINT"] = "http://oss.i.basemind.com"

app = FastAPI()

class ListFilesRequest(BaseModel):
    path: str
    image_size: int = 224 # 添加一个参数用于指定图片大小

class ListFilesResponse(BaseModel):
    results: list[str]
    processed_files: list[str] = [] # 新增一个字段用于记录处理过的图片路径

@serve.deployment(ray_actor_options={"num_cpus": 1, "num_gpus": 0.1})
@serve.ingress(app)
class Translator(ServeEngine):
    def __init__(self, name: str, model_config: ModelConfig, engine_config: CustomEngineConfig):
        super().__init__()

    def _load_img_as_tensor(self, img_path, image_size):
        # 使用 megfile.smart_open 来读取OSS上的图片
        with megfile.smart_open(img_path, 'rb') as f:
            img_pil = Image.open(f)
            img_pil = img_pil.convert("RGB") 
            
        img_np = np.array(img_pil.resize((image_size, image_size)))
        if img_np.dtype == np.uint8:  # np.uint8 is expected for JPEG images
            img_np = img_np / 255.0
        else:
            raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")
        img = torch.from_numpy(img_np).permute(2, 0, 1)
        video_width, video_height = img_pil.size  # the original video size
        return img, video_height, video_width
            
    def list_files(self, path: str, image_size: int) -> tuple[list[str], list[str]]:
        processed_files = [] # 用于存储处理成功的图片路径
        
        # Use megfile to list all files in the given path
        file_list = megfile.s3_listdir(path)
        
        for file_name in file_list:
            full_img_path = os.path.join(path, file_name) # 拼接完整的图片路径
            
            # 尝试加载图片
            img_tensor, h, w = self._load_img_as_tensor(full_img_path, image_size)
            print(f"Successfully loaded image: {full_img_path}, tensor shape: {img_tensor.shape}")

            # # 定义保存路径，这里我们将处理后的图片保存在 cut3r_processed_tensors 目录下
            # # 注意：将 tensor 直接保存为图片通常需要先转换回PIL Image
            # # 如果你只是想保存 tensor，可以考虑用 torch.save 或 numpy.save
            # # 这里我们以保存为新的JPEG图片为例，需要将tensor转换回图片格式
            
            # # 示例：将 tensor 转换回 PIL Image 并保存为新的JPEG图片
            # # 假设你希望保存为原始尺寸或者处理后的尺寸
            # # 这里我们以处理后的尺寸为例，并将 tensor 转换回 uint8 0-255 范围
            # img_np_processed = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            # processed_pil_img = Image.fromarray(img_np_processed)
            
            # # 构造新的保存路径
            # # 示例： s3://oss-lihao/cut3r_processed_output/original_filename.jpg
            # base_name, ext = os.path.splitext(file_name)
            # # 可以添加后缀或改变子目录来区分
            # save_dir = os.path.join(os.path.dirname(path), "cut3r_processed_output")
            # # 确保目标目录存在，megfile.smart_open 会自动创建 S3 路径
            # new_image_path = os.path.join(save_dir, f"{base_name}_processed{ext}")

            # # 使用 megfile.smart_open 保存处理后的图片
            # with megfile.smart_open(new_image_path, 'wb') as f_out:
            #     processed_pil_img.save(f_out, format='JPEG') # 根据需要选择格式
            # print(f"Successfully saved processed image to: {new_image_path}")
            # processed_files.append(new_image_path)
        
        return file_list, processed_files

 
    
    @app.post('/v1/list_files') 
    def api(self, request: ListFilesRequest) -> ListFilesResponse:
        path: str = request.path
        image_size: int = request.image_size
        files, processed_files = self.list_files(path, image_size)
        response = ListFilesResponse(results=files, processed_files=processed_files)
        return response
    @app.get("/healthz")
    def health(self):
        return {"status": "ok"}

