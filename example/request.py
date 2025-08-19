import requests
import json

from PIL import Image  
from io import BytesIO
import base64

def send_list_files_request(path: str):
    url = "http://localhost:8000/v1/list_files"
    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "path": path
    }

    response = requests.post(
        url,
        headers=headers,
        data=json.dumps(payload),
        timeout=30  # 设置超时时间（秒）
    )

    # 检查HTTP状态码
    response.raise_for_status()

    # 尝试解析JSON响应
    try:
        json_response = response.json()
        return json_response['results']
    except json.JSONDecodeError:
        return {
            "status": "error",
            "message": "无法解析JSON响应",
            "raw_response": response.text
        }

if __name__ == "__main__":
    # 使用示例
    path = "s3://oss-lihao/cut3r_processed/processed_dl3dv_ours_parts/processed_dl3dv_ours/1K/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f/dense/rgb/"
    file_list = send_list_files_request(path)
    print("Files in directory:", file_list)