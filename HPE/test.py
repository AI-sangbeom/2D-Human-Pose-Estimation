import cv2
import numpy as np
bimg = np.zeros((224, 224, 3))
cv2.imshow('DINOv3 Pose Result', bimg)
import torch
import torchvision
from models.pose import DINOv3Pose


# 모델 클래스 import (사용자 환경에 맞게 경로 수정)
# 예: from models.pose import DINOv3Pose 
# 여기서는 가상의 클래스로 대체합니다.

# ==========================================
# 1. Utils (Pre/Post Processing & Scaling)
# ==========================================

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """이미지 비율을 유지하며 리사이즈하고 패딩을 추가합니다."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    [핵심] 박스 좌표를 모델 입력 크기(img1_shape)에서 원본 이미지 크기(img0_shape)로 변환
    패딩을 제거하고 비율에 맞춰 스케일링합니다.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding 제거
    coords[:, [1, 3]] -= pad[1]  # y padding 제거
    coords[:, :4] /= gain        # scale 복원
    
    clip_coords(coords, img0_shape) # 원본 이미지 밖으로 나가는 좌표 자르기
    return coords

def clip_coords(boxes, shape):
    """Clip bounding xyxy bounding boxes to image shape (height, width)"""
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

def scale_kpts(kpts, img1_shape, img0_shape, ratio_pad=None):
    """
    [핵심] 키포인트 좌표를 모델 입력 크기에서 원본 이미지 크기로 변환
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    kpts[..., 0::3] = (kpts[..., 0::3] - pad[0]) / gain # x 좌표 변환
    kpts[..., 1::3] = (kpts[..., 1::3] - pad[1]) / gain # y 좌표 변환
    
    # 클리핑 (선택 사항)
    kpts[..., 0::3] = kpts[..., 0::3].clamp(0, img0_shape[1])
    kpts[..., 1::3] = kpts[..., 1::3].clamp(0, img0_shape[0])
    
    return kpts

def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]"""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def non_max_suppression_pose(prediction, conf_thres=0.1, iou_thres=0.45, max_det=300, nc=1, nk=12):
    """
    NMS 수행 후 Dictionary 형태로 반환
    """
    # 1. Shape Transformation: (B, C, A) -> (B, A, C)
    if prediction.shape[1] == 4 + nc + nk:
        prediction = prediction.transpose(1, 2)
        
    bs = prediction.shape[0]
    output = [] 
    
    # 2. Candidate Filtering (1차 필터링)
    # 여기서 이미 conf_thres보다 낮은 것들은 False가 됨
    xc = prediction[..., 4:4+nc].amax(-1) > conf_thres

    for xi, x in enumerate(prediction):
        # 결과 딕셔너리 초기화
        num_kpts = nk // 3 
        result = {
            'boxes': torch.zeros((0, 4), device=prediction.device),
            'scores': torch.zeros((0,), device=prediction.device),
            'labels': torch.zeros((0,), device=prediction.device),
            'keypoints': torch.zeros((0, num_kpts, 3), device=prediction.device)
        }

        # 3. Apply 1st Filter
        x = x[xc[xi]]
        
        # 필터링 후 남은 게 없으면 빈 결과 추가 후 다음으로
        if not x.shape[0]: 
            output.append(result)
            continue

        # 4. Split & Decode
        box = x[:, :4]
        cls = x[:, 4:4+nc]
        kpt = x[:, 4+nc:]

        box = xywh2xyxy(box)

        conf, j = cls.max(1, keepdim=True)
        x = torch.cat((box, conf, j.float(), kpt), 1)
        c = x[:, 5:6] * 7680
        boxes, scores = x[:, :4] + c, x[:, 4]
        
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        
        # 7. Limit Detections
        if i.shape[0] > max_det: 
            i = i[:max_det]
        
        det = x[i]
        
        # 8. Fill Dictionary
        result['boxes'] = det[:, :4]
        result['scores'] = det[:, 4]
        result['labels'] = det[:, 5].long()
        result['keypoints'] = det[:, 6:].reshape(-1, num_kpts, 3)
        
        output.append(result)

    return output

def plot_skeleton(img, kpts, steps=3):
    """키포인트 시각화"""
    num_kpts = len(kpts) // steps
    for i in range(num_kpts):
        x, y, conf = kpts[3*i], kpts[3*i+1], kpts[3*i+2]
        if conf < 0.5: continue 
        
        cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
        # cv2.putText(img, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def run_inference(image_path, model_path, device='cuda'):
    # 1. 모델 로드
    print(f"Loading model from {model_path}...")
    nkpts = 4
    # 실제 DINOv3Pose 클래스로 교체해야 합니다.
    model = DINOv3Pose(backbone='dinov3_convnext_base', nkpts=(nkpts, 3), ncls=7, device=device)
    
    ckpt = torch.load(model_path, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    
    model.to(device)
    model.eval()

    # 2. 이미지 로드
    img0 = cv2.imread(image_path) # BGR, 원본 이미지
    if img0 is None:
        print("Image Not Found")
        return

    # 3. 전처리 (Letterbox)
    # 640x640으로 리사이즈하며 비율 유지, 빈 공간은 패딩
    img, ratio, pad = letterbox(img0, new_shape=640, stride=32, auto=False)
    
    # HWC -> CHW, BGR -> RGB
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).to(device).float()
    img_tensor /= 255.0
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor[None] 

    # 4. 추론
    print("Running inference...")
    with torch.no_grad():
        pred = model(img_tensor)
        if isinstance(pred, tuple): pred = pred[0]

    # 5. NMS
    pred_nms = non_max_suppression_pose(pred, conf_thres=0.1, iou_thres=0.45, nc=10, nk=nkpts*3)


    # 6. 좌표 복원 및 시각화 [여기가 핵심입니다!]
    for i, det in enumerate(pred_nms):
        im0_copy = img0.copy() # 원본 이미지 복사 (그리기 용)
        # 이제 인덱싱(det[:, :4]) 대신 키값으로 접근 가능
        boxes = det['boxes']
        scores = det['scores']
        labels = det['labels']
        keypoints = det['keypoints'] # 이미 (N, 4, 3)으로 reshape 되어 있음
        if len(boxes) > 0:
            # scale_coords 등에 넣을 때도 편리함
            boxes = scale_coords(img_tensor.shape[2:], boxes, img0.shape).round()
            # keypoints는 flatten해서 넣거나 scale_kpts 함수를 (N, K, 3) 지원하게 수정해서 사용
            # 기존 scale_kpts를 쓴다면:
            flat_kpts = keypoints.reshape(len(boxes), -1)
            flat_kpts = scale_kpts(flat_kpts, img_tensor.shape[2:], img0.shape).round()
            keypoints = flat_kpts.reshape(len(boxes), -1, 3)

            # 복원된 좌표로 그리기
            for box, score, label, kpt in zip(boxes, scores, labels, keypoints):
                # 박스 그리기
                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                cv2.rectangle(im0_copy, p1, p2, (0, 0, 255), 2)
                cv2.putText(im0_copy, str(label), (int(box[0]), int(box[1])-10), 0, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)
                
                for kpts in kpt:
                    # 키포인트 그리기
                    plot_skeleton(im0_copy, kpts.cpu().numpy(), steps=3)

        # 결과 출력
        cv2.imshow('DINOv3 Pose Result', im0_copy)
        # 'q'를 누르면 종료
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 경로를 실제 파일 위치로 수정해주세요.
    # IMG_PATH = './frame_2.jpg' 
    IMG_PATH = './examples/00017.png' 
    MODEL_PATH = './weights/best.pt' 
    
    run_inference(IMG_PATH, MODEL_PATH)