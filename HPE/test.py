import cv2
import numpy as np
bimg = np.zeros((224, 224, 3), dtype=np.uint8)
cv2.imshow('DINOv3 Pose Result', bimg)
cv2.waitKey(1)
import torch
import torchvision
from models.pose import DINOv3Pose
import os 
import natsort

# ==========================================
# 1. Simple Pre/Post Processing Functions
# ==========================================

def preprocess_simple(img0, img_size=640, device='cuda'):
    """
    학습과 똑같이: 비율 무시하고 강제 리사이즈 (Stretch)
    """
    img = cv2.resize(img0, (img_size, img_size))
    img = img.transpose((2, 0, 1)) 
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).to(device).float()
    img_tensor /= 255.0
    
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor.unsqueeze(0)
        
    return img_tensor

def scale_coords_simple(coords, img0_shape, img1_shape=(640, 640)):
    """
    패딩 계산 없이 단순 비율만 곱해서 복원
    """
    h0, w0 = img0_shape[:2]
    h1, w1 = img1_shape[:2]
    
    coords = coords.clone() if isinstance(coords, torch.Tensor) else coords.copy()
    
    # X 좌표
    coords[..., 0] *= (w0 / w1) 
    if coords.shape[-1] > 2:
        coords[..., 2] *= (w0 / w1)

    # Y 좌표
    coords[..., 1] *= (h0 / h1)
    if coords.shape[-1] > 3:
        coords[..., 3] *= (h0 / h1)
        
    return coords

def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def point2box(points):
    min_vals, _ = torch.min(points, dim=1)
    max_vals, _ = torch.max(points, dim=1)
    return torch.cat([min_vals, max_vals], 1)
    
import torch
import torchvision
import numpy as np

def non_max_suppression_pose(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300, nc=7, nkpts=4):
    """
    NMS for Pose Estimation (No Box in Input)
    prediction: (Batch, nc + nk, Anchors) or (Batch, Anchors, nc + nk)
    Channel layout: [cls(nc), kpts(nk=nkpts*3)]
    
    Args:
        nc: number of classes (default 7)
        nkpts: number of keypoints (default 4)
    """
    
    # 1. Shape Transformation & Check
    # (Batch, Channels, Anchors) -> (Batch, Anchors, Channels)
    # 보통 Anchors 개수(예: 8400)가 Channels(예: 19)보다 훨씬 큽니다.
    if prediction.shape[1] < prediction.shape[2]:
        prediction = prediction.transpose(1, 2)
        
    bs = prediction.shape[0]     # Batch size
    num_channels = prediction.shape[2] # nc + nk
    output = []

    # 2. Keypoint Dimension Check
    # nc를 제외한 나머지 채널이 Keypoint 정보
    nk = num_channels - nc 
    kpt_dim = nk // nkpts # 2(xy) or 3(xyc)
    
    print(f"[DEBUG] Input Shape: {prediction.shape}")
    print(f"[DEBUG] Parsed: nc={nc}, nk={nk} (nkpts={nkpts}, dim={kpt_dim})")

    # 3. Pre-compute Scores for efficiency
    # Box나 Objectness Score가 없으므로, Max Class Probability를 점수로 사용
    # (B, A, nc) -> (B, A)
    max_cls_scores, _ = prediction[..., :nc].max(dim=2)
    
    # 1차 필터링용 마스크 (속도 향상)
    xc = max_cls_scores > conf_thres 

    for xi, x in enumerate(prediction):
        # 결과 담을 그릇
        result = {
            'boxes': torch.zeros((0, 4), device=prediction.device),
            'scores': torch.zeros((0,), device=prediction.device),
            'labels': torch.zeros((0,), device=prediction.device),
            'keypoints': torch.zeros((0, nkpts, 3), device=prediction.device)
        }

        # 4. Apply Confidence Filtering
        # 해당 배치의 앵커들 중 score가 높은 것만 추출
        x = x[xc[xi]] # (Num_Candidates, nc + nk)
        
        if not x.shape[0]: 
            output.append(result)
            continue

        # 5. Parse Data
        # Layout: [cls(nc) | kpts(nk)]
        cls_probs = x[:, :nc]      # (N, nc)
        kpt_flat = x[:, nc:]       # (N, nk)
        
        # 6. Score & Label
        # 각 Detection의 가장 높은 클래스와 점수 추출
        conf_scores, labels = cls_probs.max(1) # (N, )
        
        # 2차 Threshold 필터링 (Class specific confidence)
        # 이미 위에서 필터링 했지만, argmax 이후 정확한 값으로 다시 체크
        mask = conf_scores > conf_thres
        
        # 필터링 적용
        scores = conf_scores[mask]
        labels = labels[mask]
        kpt_flat = kpt_flat[mask]
        
        if not scores.shape[0]:
            output.append(result)
            continue
            
        # 7. Reshape Keypoints
        # (N, nk) -> (N, nkpts, kpt_dim)
        kpts_data = kpt_flat.view(-1, nkpts, kpt_dim)
        
        # 만약 kpt_dim이 2라면(x,y), conf(1.0)를 추가해서 3으로 맞춤
        if kpt_dim == 2:
            ones = torch.ones((kpts_data.shape[0], nkpts, 1), device=x.device)
            kpts_final = torch.cat([kpts_data, ones], dim=2)
        else:
            kpts_final = kpts_data
            
        # 8. Point Feature -> Bounding Box (Min/Max)
        # xy 좌표만 추출 (N, nkpts, 2)
        kpts_xy = kpts_final[..., :2]
        
        min_coord, _ = torch.min(kpts_xy, dim=1) # (min_x, min_y)
        max_coord, _ = torch.max(kpts_xy, dim=1) # (max_x, max_y)
        
        # Box 생성 [x1, y1, x2, y2]
        # (옵션) 0보다 작은 좌표 클램핑: min_coord = torch.clamp(min_coord, min=0)
        calc_boxes = torch.cat([min_coord, max_coord], dim=1)
        
        # 9. NMS (Non-Maximum Suppression)
        # Class-agnostic NMS를 위해 오프셋 적용
        c = labels.float() * 7680 
        boxes_for_nms = calc_boxes + c.unsqueeze(1)
        
        i = torchvision.ops.nms(boxes_for_nms, scores, iou_thres)
        
        if i.shape[0] > max_det: 
            i = i[:max_det]
        
        # 최종 결과 저장
        result['boxes'] = calc_boxes[i]
        result['scores'] = scores[i]
        result['labels'] = labels[i].long()
        result['keypoints'] = kpts_final[i]
        
        output.append(result)

    return output
# ==========================================
# 2. Visualization Function
# ==========================================
def draw_detections(im0, res, nkpts=4, conf_threshold=0.5):
    """
    Draw bounding boxes, connected keypoints (Green), and center line (Red)
    """
    im0_copy = im0.copy()
    h0, w0 = im0_copy.shape[:2]
    
    scores = res['scores']
    kpts = res['keypoints']  # (N, nkpts, 3)
    labels = res['labels']
    
    # 클래스별 색상 팔레트
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
    ]
    
    # 연결할 점의 순서 정의 (Green Line)
    # 요청하신 0-1, 1-2, 2-3, 3-0 (박스 형태)로 연결합니다.
    # 만약 '3-1' 연결을 원하시면 [3, 1]로 수정하세요.
    connections = [[0, 1], [1, 2], [2, 3], [3, 0]]
    
    for box_idx, (score, label, kpt) in enumerate(zip(scores, labels, kpts)):
        if kpt.shape[0] < 4:
            continue

        # -----------------------------------------------------------
        # 1. 초록색 선 그리기 (Box Outline)
        # -----------------------------------------------------------
        for start_idx, end_idx in connections:
            pt1 = kpt[start_idx].cpu().numpy()[:2].astype(int)
            pt2 = kpt[end_idx].cpu().numpy()[:2].astype(int)
            
            # 이미지 범위 체크 (옵션)
            # if (0 <= pt1[0] < w0) and (0 <= pt2[0] < w0):
            cv2.line(im0_copy, tuple(pt1), tuple(pt2), (0, 255, 0), 2) # Green, Thickness 2

        # -----------------------------------------------------------
        # 2. 빨간색 선 그리기 (Left Midpoint <-> Right Midpoint)
        # -----------------------------------------------------------
        # 점들을 좌표값만 추출 (N, 2)
        pts_xy = kpt[:, :2].cpu().numpy().astype(float)
        
        # X축 기준으로 정렬하여 왼쪽 2개, 오른쪽 2개 구분
        # np.argsort를 사용하여 x좌표가 작은 순서대로 인덱스 추출
        sorted_indices = np.argsort(pts_xy[:, 0])
        
        left_indices = sorted_indices[:2]  # x가 작은 2개
        right_indices = sorted_indices[2:] # x가 큰 2개
        
        # 각 그룹의 중점 계산
        left_pts = pts_xy[left_indices]
        right_pts = pts_xy[right_indices]
        
        mid_left = np.mean(left_pts, axis=0).astype(int)
        mid_right = np.mean(right_pts, axis=0).astype(int)
        
        # 빨간색 선 그리기
        cv2.line(im0_copy, tuple(mid_left), tuple(mid_right), (0, 0, 255), 1) # Red
        
        # 중점 표시 (옵션: 파란색 작은 점)
        cv2.circle(im0_copy, tuple(mid_left), 3, (255, 0, 0), -1)
        cv2.circle(im0_copy, tuple(mid_right), 3, (255, 0, 0), -1)

        # -----------------------------------------------------------
        # 3. Keypoints 및 Label 그리기 (기존 로직)
        # -----------------------------------------------------------
        color = colors[int(label.item()) % len(colors)]
        
        for kpt_idx, kpt_data in enumerate(kpt):
            kx, ky, conf = kpt_data.cpu().numpy()
            kx, ky = int(kx), int(ky)
            
            if 0 <= kx < w0 and 0 <= ky < h0:
                cv2.circle(im0_copy, (kx, ky), 3, (0, 255, 0), -1)
                cv2.putText(im0_copy, str(kpt_idx), (kx+5, ky-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        text_label = f"Class {int(label.item())} {score.item():.2f}"
        # 라벨 위치를 박스의 첫 번째 점 근처로 설정
        label_x, label_y = kpt[0, :2].cpu().numpy().astype(int)
        cv2.putText(im0_copy, text_label, (label_x, label_y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return im0_copy


# ==========================================
# 3. Main Inference
# ==========================================
import time 
def run_inference(image_folder, model_path, device='cuda'):
    print(f"Loading model from {model_path}...")
    nkpts = 4
    
    # Load model
    model = DINOv3Pose(backbone='dinov3_convnext_small', nkpts=(nkpts, 3), ncls=10, device=device)
    
    ckpt = torch.load(model_path, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    
    # Stride setup
    if hasattr(model, 'head'):
        model.head.register_buffer("stride", torch.tensor([8., 16., 32.], device=device))
        model.head.anchors = torch.empty(0, device=device)
        model.head.strides = torch.empty(0, device=device)

    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully. Processing images from {image_folder}...")
    print(f"Number of keypoints: {nkpts}\n")
    
    image_files = natsort.natsorted([f for f in os.listdir(image_folder) 
                                     if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))])
    
    if not image_files:
        print(f"No images found in {image_folder}")
        return
    
    print(f"Found {len(image_files)} images to process\n")

    for img_idx, filename in enumerate(image_files):
        image_path = os.path.join(image_folder, filename)
        
        # Load image
        img0 = cv2.imread(image_path)
        if img0 is None:
            print(f"⚠ Failed to load: {filename}")
            continue

        print(f"\n[{img_idx+1}/{len(image_files)}] Processing: {filename}")
        
        # Preprocess
        img_tensor = preprocess_simple(img0, 640, device)

        # Inference
        with torch.no_grad():
            start = time.time()
            pred = model(img_tensor)
            if isinstance(pred, tuple):
                pred = pred[0]

        print(f"[DEBUG] Raw prediction shape: {pred[0].shape if isinstance(pred, list) else pred.shape}")

        # NMS (pass nkpts parameter!)
        pred_nms = non_max_suppression_pose(
            pred, 
            conf_thres=0.5, 
            iou_thres=0.5, 
            nc=10, 
            nkpts=nkpts   # number of keypoints
        )
        print("inference time :", time.time() - start)

        # Visualization
        for batch_idx, res in enumerate(pred_nms):
            # Scale coordinates to original image size
            if res['keypoints'].shape[0] > 0:
                
                h0, w0 = img0.shape[:2]
                sx, sy = w0 / 640, h0 / 640
                
                # Scale keypoints
                res['keypoints'][..., 0] *= sx
                res['keypoints'][..., 1] *= sy
            
            # Draw detections (pass nkpts!)
            im0_result = draw_detections(img0, res, nkpts=nkpts, conf_threshold=0.5)
            
            # Update display
            cv2.imshow('DINOv3 Pose Result', im0_result)
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                print("Exiting...")
                cv2.destroyAllWindows()
                return
            elif key == ord('s'):
                # Save result
                output_filename = f"result_{filename}"
                cv2.imwrite(output_filename, im0_result)
                print(f"  ✓ Saved: {output_filename}")

    cv2.destroyAllWindows()
    print("\n✓ Inference completed!")

if __name__ == '__main__':
    # IMG_PATH = '/media/otter/otterHD/pallet_data/data_yolo/clear_data/train/sampling_500/images'
    IMG_PATH = '/media/otter/otterHD/AXData/raw_data/images'

    MODEL_PATH = './weights/best.pt'
    
    if not os.path.exists(IMG_PATH):
        print(f"❌ Image folder not found: {IMG_PATH}")
    elif not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        run_inference(IMG_PATH, MODEL_PATH, device=device)