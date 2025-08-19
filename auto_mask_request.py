import requests
import json
from typing import Optional
from pydantic import BaseModel

class MaskRequest(BaseModel):
    image_dir: str
    output_dir: str
    level: str = "large"
    batch_size: int = 20
    detect_stride: int = 30
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
    url = "http://localhost:8000/sam2/video_seg"
    # url = 'http://10.53.1.205:9200/v1/proxy/SAM2-4090-48G/sam2/video_seg'
    headers = {
        "Content-Type": "application/json",
    }

    response = requests.post(
        url,
        headers=headers,
        data=request.json(),
        timeout=60000  # 5 minutes timeout for video processing
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

if __name__ == "__main__":
    # Example usage
    while True:
        try:
            request = MaskRequest(
                # image_dir="/mnt/juicefs/datasets/cut3r_processed/processed_dl3dv_ours_parts/processed_dl3dv_ours/1K/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f/dense/rgb/",
                # output_dir="sam2_results/1K/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f/",
                image_dir="s3://oss-lihao/processed_dl3dv_ours_parts/processed_dl3dv_ours/4K/0345dca90e23c50447ba82d81f870281521689112fbefc96d28603611a109389/dense/rgb/",
                output_dir="s3://oss-lihao/sam2_results_v1/4K/0345dca90e23c50447ba82d81f870281521689112fbefc96d28603611a109389/"
            )
            response = send_mask_request(request)
            print("Processing status:", response.status)
            print("Message:", response.message)
            if response.output_path:
                print("Output saved at:", response.output_path)
            break
        # except:
        #     continue
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                continue
            else:
                print(e.response)
                raise