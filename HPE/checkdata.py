# check_dataset.py
import os 
import cv2
import numpy as np
bimg = np.zeros((224, 224, 3))
cv2.imshow('DINOv3 Pose Result', bimg)
import torch
from train import YoloPoseDataset
def check_data():
    # 경로 수정 필요
    base_folder = '/media/otter/otterHD/AXData/TotalAX/'
    img_dir = os.path.join(base_folder, "images")
    label_dir = os.path.join(base_folder, 'labels')
    
    # 데이터셋 로드
    dataset = YoloPoseDataset(img_dir, label_dir, img_size=640, nkpts=4)
    
    # 첫 번째 데이터 가져오기
    for datas in dataset:
        img_tensor, labels = datas
        
        # 1. 이미지 복원 (Tensor -> Numpy Image)
        img = img_tensor.permute(1, 2, 0).numpy() * 255.0
        img = img.astype(np.uint8).copy()
        
        # 2. 라벨 그리기
        h, w = img.shape[:2]
        
        for label in labels:
            # label: [batch_idx, cls, cx, cy, w, h, kpt1_x, kpt1_y, kpt1_vis, ...]
            
            # Box 복원 (Normalized xywh -> Absolute xyxy)
            cx, cy, bw, bh = label[2:6]
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2) # 빨간 박스
            
            # Keypoint 복원
            kpts = label[6:]
            num_kpts = len(kpts) // 3
            for cnt, i in enumerate(range(num_kpts)):
                kx = kpts[3*i] * w
                ky = kpts[3*i+1] * h
                vis = kpts[3*i+2]
                
                if vis > 0:
                    cv2.circle(img, (int(kx), int(ky)), 5, (0, 255, 0), -1) # 초록 점
                    cv2.putText(img, f'{cnt}', (int(kx), int(ky)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 결과 출력
        cv2.imshow('DINOv3 Pose Result', img)
        # 'q'를 누르면 종료
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
if __name__ == '__main__':
    check_data()