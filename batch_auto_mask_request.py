import requests
import json
from typing import Optional
from pydantic import BaseModel
import megfile
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

os.environ["OSS_ENDPOINT"] = "http://oss.i.basemind.com"


class MaskRequest(BaseModel):
    image_dir: str
    output_dir: str
    level: str = "large"
    batch_size: int = 20
    detect_stride: int = 10
    use_other_level: int = 1
    postnms: int = 1
    pred_iou_thresh: float = 0.7
    box_nms_thresh: float = 0.7
    stability_score_thresh: float = 0.85

class MaskResponse(BaseModel):
    status: str
    message: str
    output_path: Optional[str] = None

def send_mask_request(request: MaskRequest) -> MaskResponse:
    # url = "http://localhost:8000/sam2/video_seg"
    # url = 'http://10.53.1.205:9200/v1/proxy/SAM2_4090_48G/sam2/video_seg'
    # url = 'http://10.53.1.205:9200/v1/proxy/SAM2-4090-24G/sam2/video_seg'
    url = 'http://10.53.1.205:9200/v1/proxy/SAM2_H20/sam2/video_seg'
    # url = 'http://10.53.1.205:9200/v1/proxy/SAM2-H100/sam2/video_seg'
    # url = 'http://10.53.1.205:9200/v1/proxy/SAM2-H800/sam2/video_seg'
    # url = 'http://10.53.1.205:9200/v1/proxy/SAM2-A800/sam2/video_seg'
    # url = 'http://10.53.1.205:9200/v1/proxy/SAM2-L40/sam2/video_seg'
    headers = {
        "Content-Type": "application/json",
    }

    response = requests.post(
        url,
        headers=headers,
        data=request.json(),
        # timeout=60000  # 5 minutes timeout for video processing
    )

    # Check HTTP status code
    response.raise_for_status()

    # Parse JSON response
    try:
        json_response = response.json()
        return MaskResponse(**json_response)
    except json.JSONDecodeError:
        return MaskResponse(
            status="error",
            message="Failed to parse JSON response",
            output_path=None
        )

def process_video_folder(folder_name, base_s3_path, base_output_path):
    """
    Processes a single video folder.
    Constructs image and output paths, sends request, and returns result.
    """
    max_retries = 20000
    retry_wait = 30  # seconds
    attempt = 0
    print("Processing: ", folder_name)
    while attempt < max_retries:
        try:
            image_dir = os.path.join(base_s3_path, folder_name, "images")
            output_dir = os.path.join(base_output_path, folder_name)
            
            # Check if the output already exists, if so, skip processing
            if megfile.s3_exists(os.path.join(output_dir, "auto_masks.json")):
                 print(f"Skipping {folder_name}, output already exists.")
                 return "skipped"

            request = MaskRequest(
                image_dir=image_dir,
                output_dir=output_dir,
            )
            response = send_mask_request(request)
            
            if response.status == "rejected":
                attempt += 1
                if attempt < max_retries:
                    # print(f"Service busy for {folder_name}, retrying ({attempt}/{max_retries}) in {retry_wait}s...")
                    time.sleep(retry_wait)
                    continue
                else:
                    print(f"Skipping {folder_name} due to repeated rejections after {max_retries} attempts.")
                    return "skipped_rejected"

            print("Processing status:", response.status)
            print("Message:", response.message)
            return response.message
                
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code in [404, 502, 500, 503, 504]:
                attempt += 1
                if attempt < max_retries:
                    print(f"Server error for {folder_name}, retrying ({attempt}/{max_retries}) in {retry_wait}s...")
                    time.sleep(retry_wait)
                    continue
                else:
                    print(f"Skipping {folder_name} due to repeated server errors after {max_retries} attempts.")
                    return "skipped_server_error"
            else:
                print(f"HTTPError for {folder_name}: {e}")
                raise
        except Exception as e:
            print(f"An error occurred while processing {folder_name}: {e}")
            return "failed_exception"


if __name__ == "__main__":
    base_s3_path = "s3://oss-lihao/cut3r_processed/processed_scannetpp_fix/"
    base_output_path = "s3://oss-lihao/sam2_results/processed_scannetpp_fix/"
    
    # List all subdirectories in the base S3 path
    all_entries = megfile.s3_listdir(base_s3_path)
    video_folders = [entry for entry in all_entries if megfile.s3_isdir(os.path.join(base_s3_path, entry))]
    video_folders = list(reversed(video_folders))
    print(f"Found {len(video_folders)} video folders to process.")
    
    # Use ThreadPoolExecutor for parallel processing
    # 使用多线程处理视频文件夹的代码已注释，改为for循环顺序处理
    with ThreadPoolExecutor(max_workers=16) as executor:
        # Create a future for each folder to be processed
        futures = {executor.submit(process_video_folder, folder, base_s3_path, base_output_path): folder for folder in video_folders}
        
        # Use tqdm to track progress
        for future in tqdm(as_completed(futures), total=len(video_folders), desc="Processing videos"):
            folder_name = futures[future]
            try:
                result = future.result()
                # You can add more detailed logging here based on the result
                # if "saved" in result:
                    #  print(f"Folder {folder_name} processed with status: {result}")
            except Exception as exc:
                print(f'{folder_name} generated an exception: {exc}')

    # 改为for循环顺序处理
    # for folder_name in tqdm(video_folders, desc="Processing videos"):
    #     try:
    #         result = process_video_folder(folder_name, base_s3_path, base_output_path)
    #         # 可以根据结果添加更详细的日志
    #         if result != "success":
    #             print(f"Folder {folder_name} processed with status: {result}")
    #     except Exception as exc:
    #         print(f'{folder_name} generated an exception: {exc}')