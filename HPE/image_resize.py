import os
import cv2
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# ================= 설정 =================
source_dir = '/media/otter/otterHD/AXData/TotalAX/train/images'
target_dir = '/media/otter/otterHD/AXData/TotalAX/train/images_640' # 새로 저장할 폴더
img_size = 640
# ========================================

os.makedirs(target_dir, exist_ok=True)
img_files = glob.glob(os.path.join(source_dir, "*.jpg")) + glob.glob(os.path.join(source_dir, "*.png"))

def process_img(img_path):
    try:
        # 파일명 유지
        file_name = os.path.basename(img_path)
        save_path = os.path.join(target_dir, file_name)
        
        # 이미 존재하면 스킵 (중단 후 이어하기 가능)
        if os.path.exists(save_path):
            return

        img = cv2.imread(img_path)
        if img is None: return

        # 리사이즈
        img_resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        
        # 저장 (압축률 조정하여 용량 최소화)
        cv2.imwrite(save_path, img_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    except Exception as e:
        print(f"Error {img_path}: {e}")

print(f"Start resizing {len(img_files)} images...")

# 멀티스레드로 빠르게 처리
with ThreadPoolExecutor(max_workers=16) as executor:
    list(tqdm(executor.map(process_img, img_files), total=len(img_files)))

print("Done! 이제 학습 코드에서 경로를 'images_640'으로 바꾸세요.")