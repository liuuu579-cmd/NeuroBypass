import os
import re
from datasets import load_dataset
from tqdm import tqdm

# --- 代理设置 (保持您的配置) ---
os.environ['http_proxy'] = "http://192.168.143.85:7890"
os.environ['https_proxy'] = "http://192.168.143.85:7890"

# --- 配置 ---

# 使用由 AllenAI 维护的、基于 Parquet 的现代版 C4 数据集。
# 这是目前最标准、最推荐的版本。
DATASET_PATH = "allenai/c4"
DATASET_NAME = "en"  # C4的英文子集

TARGET_SIZE_MB = 1024
OUTPUT_FILE = f'c4_subset_{TARGET_SIZE_MB}mb.txt'


def normalize_whitespace(text: str) -> str:
    """一个非常简单的清洗函数，仅用于统一空白符。C4本身已经非常干净。"""
    return re.sub(r'\s+', ' ', text).strip()

def main():
    if os.path.exists(OUTPUT_FILE):
        print(f"'{OUTPUT_FILE}' already exists. Skipping.")
        return

    print(f"Starting to download and process the modern Parquet version of C4: '{DATASET_PATH}'")

    target_bytes = int(TARGET_SIZE_MB * 1024**2)
    total_written_bytes = 0

    try:
        # 加载基于 Parquet 的数据集，无需 trust_remote_code
        dataset = load_dataset(
            path=DATASET_PATH,
            name=DATASET_NAME,
            split="train",
            streaming=True
        )
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
            pbar = tqdm(total=target_bytes, unit='B', unit_scale=True, desc=f"Downloading {TARGET_SIZE_MB}MB from C4")
            for item in dataset:
                # 获取文本并进行最基础的空白符规范化
                original_text = item.get('text', '')
                cleaned_text = normalize_whitespace(original_text)

                if not cleaned_text:
                    continue
                
                # 写入文件，每条记录占一行
                f_out.write(cleaned_text + '\n')
                
                bytes_to_add = len(cleaned_text.encode('utf-8'))
                total_written_bytes += bytes_to_add
                pbar.update(bytes_to_add)

                if total_written_bytes >= target_bytes:
                    break
            
            pbar.close()

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("This may be a network issue or a problem with the dataset stream.")
        if os.path.exists(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)
        return

    print(f"\nFinished creating a clean dataset subset from C4.")
    total_size_mb = total_written_bytes / 1024**2
    print(f"Total size: {total_size_mb:.2f} MB")
    print(f"Output file: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()