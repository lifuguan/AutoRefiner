import megfile
import os
import json

# --- 配置你的 OSS 路径 ---
# 请将 'your-bucket' 替换为你的实际 OSS Bucket 名称
os.environ["OSS_ENDPOINT"] = "http://oss.i.basemind.com"
OSS_INPUT_PATH = "s3://oss-lihao/hello.txt"
OSS_OUTPUT_PATH = "s3://oss-lihao/hello_modified.json"

# --- 待保存的 Python 数据 (字典) ---
data_to_save = {
    "name": "Megfile Example",
    "version": "1.0",
    "settings": {
        "theme": "dark",
        "notifications": True,
        "language": "zh-CN"
    },
    "items": [
        {"id": 1, "value": "apple"},
        {"id": 2, "value": "banana"},
        {"id": 3, "value": "orange"}
    ],
    "timestamp": "2025-06-24T10:00:00Z"
}
def save_json_to_oss(data: dict, output_path: str):
    """
    将 Python 字典保存为 JSON 文件到 OSS。
    """
    try:
        print(f"准备将数据保存到 {output_path}...")

        # 1. 将 Python 字典转换为 JSON 字符串
        # indent=4 使输出的 JSON 格式化，易于阅读；ensure_ascii=False 允许保存非 ASCII 字符（如中文）
        json_string = json.dumps(data, indent=4, ensure_ascii=False)

        # 2. 使用 megfile.smart_open 以写入模式 'w' 保存 JSON 字符串
        # 确保使用 UTF-8 编码，以避免中文乱码
        with megfile.smart_open(output_path, 'w', encoding='utf-8') as f_write:
            f_write.write(json_string)
        print("JSON 文件保存成功！")

        # 可选：验证保存后的文件内容
        print(f"\n验证文件 {output_path} 的内容...")
        with megfile.smart_open(output_path, 'r', encoding='utf-8') as f_read_verify:
            read_back_json_string = f_read_verify.read()
            read_back_data = json.loads(read_back_json_string)
        print("验证成功！保存后的数据：")
        print("---")
        print(json.dumps(read_back_data, indent=4, ensure_ascii=False))
        print("---")
        
        # 比较原始数据和读取回来的数据
        if data == read_back_data:
            print("原始数据与读取回来的数据一致。")
        else:
            print("警告：原始数据与读取回来的数据不一致。")

    except Exception as e:
        print(f"操作失败: {e}")
        print("请检查你的 OSS_ENDPOINT, ACCESS_KEY_ID, ACCESS_KEY_SECRET 是否配置正确，以及 Bucket 和路径的权限。")

if __name__ == "__main__":
    save_json_to_oss(data_to_save, OSS_OUTPUT_PATH)