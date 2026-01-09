import cv2 
import os 
import glob
import numpy as np 
import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from models.pose import DINOv3Pose
from core.loss import VarifocalLoss, FocalLoss

class YoloPoseDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640, nkpts=None, transform=None):
        self.img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
        self.label_dir = label_dir
        self.img_size = img_size
        self.transform = transform
        self.nkpts = nkpts

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        label_path = os.path.join(self.label_dir, os.path.basename(img_path).rsplit('.', 1)[0] + ".txt")
        
        # 1. Image Load
        img = cv2.imread(img_path)
        h0, w0 = img.shape[:2]
        
        # 2. Letterbox Resize (Center Padding) - 추론과 동일한 방식 적용
        r = self.img_size / max(h0, w0)
        if r != 1: 
            interp = cv2.INTER_LINEAR if (self.img_size > r) else cv2.INTER_AREA
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            
        h, w = img.shape[:2]
        # Center Padding 계산
        dw, dh = self.img_size - w, self.img_size - h
        dw /= 2; dh /= 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # To Tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # 3. Label Load & Transform
        targets = np.zeros((0, 6 + self.nkpts * 3)) # 기본 빈 배열
        
        if os.path.exists(label_path):
            # 파일을 한 번에 2D 배열로 로드 (데이터가 없으면 빈 배열 반환)
            labels = np.loadtxt(label_path, ndmin=2) 
            
            if labels.shape[0] > 0:
                # 1) Keypoints 처리 (xy -> xyv 자동 확장)
                kpts = labels[:, 5:]
                if kpts.shape[1] // self.nkpts == 2: # vis 정보가 없는 경우 (N, K*2)
                    # (N, K, 2) -> (N, K, 1)인 1.0 추가 -> (N, K*3)로 flatten
                    kpts = kpts.reshape(-1, self.nkpts, 2)
                    kpts = np.concatenate([kpts, np.ones((len(labels), self.nkpts, 1))], axis=2).reshape(-1, self.nkpts * 3)
                
                # 2) Box 좌표 변환 (Scale -> Shift -> Normalize)
                # Box: [x, y, w, h]
                box_scale = np.array([w0, h0, w0, h0]) * r
                box_shift = np.array([left, top, 0, 0])
                boxes = (labels[:, 1:5] * box_scale + box_shift) / self.img_size
                
                # 3) Keypoints 좌표 변환
                # x, y 좌표만 변환하고 visibility(인덱스 2::3)는 유지
                kpts[:, 0::3] = (kpts[:, 0::3] * w0 * r + left) / self.img_size # x
                kpts[:, 1::3] = (kpts[:, 1::3] * h0 * r + top)  / self.img_size # y
                
                # 4) 결과 합치기 [batch_idx(0), cls, box, kpts]
                zeros = np.zeros((len(labels), 1))
                cls = labels[:, 0:1]
                targets = np.concatenate([zeros, cls, boxes, kpts], axis=1)

        labels_out = torch.from_numpy(targets).float()
        return img_tensor, labels_out

    @staticmethod
    def collate_fn(batch):
        imgs, labels = zip(*batch)
        imgs = torch.stack(imgs, 0)
        
        new_labels = []
        for i, label in enumerate(labels):
            if label.shape[0] > 0:
                l = label.clone()
                l[:, 0] = i
                new_labels.append(l)
        
        labels = torch.cat(new_labels, 0) if new_labels else torch.zeros((0, 6 + 4*3))
        return imgs, labels
    
def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right"""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def non_max_suppression_pose(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        max_det=300,
        nc=0,  # number of classes (optional)
        nk=17 * 3  # number of keypoints channels
):
    """
    Non-Maximum Suppression (NMS) on inference results for Pose Estimation.
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls, kpts]
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    
    # prediction shape: (Batch, 4 + nc + nk, Anchors) -> Transpose to (Batch, Anchors, Channels)
    # 작성하신 코드는 (B, C, A)로 나오므로 transpose가 필요합니다.
    if prediction.shape[1] == 4 + nc + nk: 
        prediction = prediction.transpose(1, 2) 

    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[2] - 4 - nk)  # number of classes
    nm = prediction.shape[2] - nc - 4  # number of masks/keypoints
    mi = 4 + nc  # mask start index
    xc = prediction[..., 4:mi].amax(-1) > conf_thres

    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence filtering

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf (만약 분리되어 있다면. YOLOv8은 통합되어 있으므로 이 줄은 상황에 따라 다름)
        # 작성하신 Head는 cls.sigmoid()가 바로 나오므로, 
        # box(4) + cls_scores(nc) + kpts(nk) 구조입니다.
        
        box = x[:, :4] # cx, cy, w, h
        cls = x[:, 4:4+nc] # class scores
        kpt = x[:, 4+nc:] # keypoints

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(box)

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (cls > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), kpt[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), kpt), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:
            continue
        elif n > max_det:  # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_det]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else 7680)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        
        output[xi] = x[i]

    return output

def preprocess(image_input, device='cuda'):
    """이미지 로드, 리사이징, 패딩, 정규화"""
    # 1. 입력 처리 (경로 or Tensor)
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Image not found: {image_input}")
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        raise TypeError("Input must be a file path(str) or numpy array")

    h0, w0 = img.shape[:2]
    
    # 2. Resize (비율 유지)
    target_size = 640
    scale = min(target_size / h0, target_size / w0)
    h, w = int(h0 * scale), int(w0 * scale)
    
    img_resized = cv2.resize(img, (w, h))
    
    # 3. Padding (32의 배수로 맞춤)
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    
    # Right, Bottom 방향으로만 패딩 추가
    img_padded = cv2.copyMakeBorder(
        img_resized, 0, pad_h, 0, pad_w, 
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    
    # 4. To Tensor
    img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # 복원용 메타 데이터
    meta = {
        'orig_shape': (h0, w0),
        'scale': scale,
        'pad': (0, pad_h, 0, pad_w) # top, bottom, left, right
    }
    
    return img_tensor, img, meta

def run_inference(model, image_path, device='cuda'):
    # 1. 이미지 준비
    img_raw = cv2.imread(image_path)
    img = cv2.resize(img_raw, (640, 640)) # 모델 입력 사이즈로 리사이즈
    img_in = img[:, :, ::-1].transpose(2, 0, 1) # BGR -> RGB, HWC -> CHW
    img_in = np.ascontiguousarray(img_in)
    img_in = torch.from_numpy(img_in).to(device).float()
    img_in /= 255.0 # 0~1 정규화
    if img_in.ndimension() == 3:
        img_in = img_in.unsqueeze(0)

    # 2. 추론 (Inference)
    model.eval()
    with torch.no_grad():
        # returns: (batch, 4 + nc + nk, anchors) tuple
        preds = model(img_in) 
        
        # Tuple의 첫 번째 요소가 예측값입니다 (Detection + Keypoints)
        if isinstance(preds, tuple):
            preds = preds[0]

    # 3. NMS 후처리
    # nc=1 (사람 클래스만 있다고 가정), nk=17*3 (x,y,conf)
    det = non_max_suppression_pose(preds, conf_thres=0.5, iou_thres=0.45, nc=10, nk=4*3)

    # 4. 결과 시각화
    for i, d in enumerate(det): # batch 내 각 이미지에 대해
        if len(d) == 0:
            print("No detection")
            continue
            
        # d shape: (num_obj, 6 + nk) -> [x1, y1, x2, y2, conf, cls, kpt1_x, kpt1_y, kpt1_conf, ...]
        
        # 원본 이미지 크기에 맞게 스케일 조정 (640x640 -> 원본크기)
        gain = min(img.shape[0] / img_raw.shape[0], img.shape[1] / img_raw.shape[1])
        pad_w = (img.shape[1] - img_raw.shape[1] * gain) / 2
        pad_h = (img.shape[0] - img_raw.shape[0] * gain) / 2
        
        d[:, :4] = scale_coords(img.shape[1:], d[:, :4], img_raw.shape).round()
        
        # 키포인트 스케일 조정 (Box 뒤 6번째 인덱스부터 키포인트)
        kpts = d[:, 6:]
        kpts = scale_kpts(kpts, img.shape[1:], img_raw.shape)

        for j in range(d.shape[0]):
            box = d[j, :4].cpu().numpy().astype(int)
            score = d[j, 4].cpu().numpy()
            kpt = kpts[j].cpu().numpy()
            
            # 박스 그리기
            cv2.rectangle(img_raw, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(img_raw, f'{score:.2f}', (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 키포인트 그리기 (17개 기준)
            plot_skeleton_kpts(img_raw, kpt, 3) # step=3 (x, y, conf)

    cv2.imshow('Result', img_raw)
    cv2.waitKey(0)

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """Rescale coords (xyxy) from img1_shape to img0_shape"""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, shape):
    """Clip bounding xyxy bounding boxes to image shape (height, width)."""
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

def scale_kpts(kpts, img1_shape, img0_shape, ratio_pad=None):
    """Rescale keypoints (x,y,conf) from img1_shape to img0_shape"""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    kpts[:, 0::3] = (kpts[:, 0::3] - pad[0]) / gain # x
    kpts[:, 1::3] = (kpts[:, 1::3] - pad[1]) / gain # y
    # conf는 변경 없음
    return kpts

def plot_skeleton_kpts(im, kpts, steps=3):
    # COCO Keypoint Index & Skeleton definition
    # skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    # 0-based index 변환 필요시 수정
    
    num_kpts = len(kpts) // steps
    for kid in range(num_kpts):
        x, y, conf = kpts[3*kid], kpts[3*kid+1], kpts[3*kid+2]
        if conf < 0.5: continue
        cv2.circle(im, (int(x), int(y)), 5, (255, 0, 0), -1)

class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses (OKS-based)."""
    def __init__(self, sigmas: torch.Tensor) -> None:
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  
        
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()

class PoseLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        nkps = model.head.kpt_shape[0]
        self.device = next(model.parameters()).device
        self.bbox_loss = BboxLoss(model.head.reg_max)
        sigmas = torch.ones(nkps, dtype=torch.float16)/nkps
        self.kpt_loss = KeypointLoss(sigmas).to(self.device)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.no = model.head.no # outputs per anchor
        self.nc = model.head.nc # num classes

    def forward(self, preds, batch):
        """
        preds: (feats, kpts) from Head.forward()
        batch: dict with 'img', 'bboxes', 'cls', 'kpts'
        """
        # NOTE: Real implementation requires 'TaskAlignedAssigner' to match anchors to targets.
        # This is a simplified placeholder to show the flow.
        
        # 1. Unpack Predictions
        feats, pred_kpts = preds # feats list, kpts list
        
        # In this simplified code, we assume you have an 'assigner' 
        # For now, we will compute a dummy loss to ensure the loop runs.
        # **IMPORTANT**: You must integrate 'TaskAlignedAssigner' from ultralytics or mmdet 
        # for correct convergence.
        
        loss_box = torch.tensor(0.0, device=self.device, requires_grad=True)
        loss_cls = torch.tensor(0.0, device=self.device, requires_grad=True)
        loss_kpt = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Total Loss
        return loss_box * 7.5 + loss_cls * 0.5 + loss_kpt * 12.0, \
               torch.stack([loss_box, loss_cls, loss_kpt]).detach()

class DFLoss(nn.Module):
    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)
    
class BboxLoss(nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        # [수정] DFLoss 모듈 인스턴스화
        self.dfl_loss = DFLoss(reg_max) 

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """
        Input tensors are already masked (N_pos, ...).
        fg_mask argument is kept for compatibility but ignored if input is already masked.
        """
        # 1. IoU Loss
        # target_scores: (N_pos, num_classes)
        # weight calculation: sum classes to get objectness weight
        weight = target_scores.sum(-1).unsqueeze(-1)
        
        # [수정] [fg_mask] 인덱싱 제거 (이미 필터링된 데이터임)
        iou = self.bbox_iou(pred_bboxes, target_bboxes, CIoU=True)
        
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # 2. DFL Loss
        target_ltrb = self.bbox2dist(anchor_points, target_bboxes, self.reg_max)
        
        # [수정] [fg_mask] 인덱싱 제거
        loss_dfl = self.dfl_loss(
            pred_dist.reshape(-1, self.reg_max), 
            target_ltrb.reshape(-1)
        ) * weight.reshape(-1)
        
        loss_dfl = loss_dfl.sum() / target_scores_sum
        
        # iou 반환 (Shape: (N_pos,))
        return loss_iou, loss_dfl, iou.detach().clamp(0)

    @staticmethod
    def bbox_iou(box1, box2, CIoU=True, eps=1e-7):
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
        w1, h1 = box1[:, 2], box1[:, 3]
        w2, h2 = box2[:, 2], box2[:, 3]
        union = w1 * h1 + w2 * h2 - inter + eps
        iou = inter / union

        if CIoU:
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((b1_x1 + b1_x2 - b2_x1 - b2_x2) ** 2 + (b1_y1 + b1_y2 - b2_y1 - b2_y2) ** 2) / 4
            v = (4 / (3.141592 ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
            with torch.no_grad():
                alpha = v / (v - iou + (1 + eps))
            return iou - (rho2 / c2 + v * alpha)
        return iou

    @staticmethod
    def bbox2dist(anchor_points, bbox, reg_max):
        x1y1 = bbox[..., :2] - bbox[..., 2:] / 2
        x2y2 = bbox[..., :2] + bbox[..., 2:] / 2
        dist = torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp(0, reg_max - 0.01)
        return dist

# ==========================================
# 2. Main ComputeLoss Class
# ==========================================
class ComputeLoss:
    def __init__(self, model, use_focal=True, use_vari_focal=True):
        self.device = next(model.parameters()).device
        self.num_kpts = model.head.nk // model.head.kpt_shape[1] 
        self.num_classes = model.head.nc
        self.reg_max = model.head.reg_max 

        # Weights
        self.box_weight = 7.5
        self.dfl_weight = 1.5 
        self.cls_weight = 0.5
        self.kpt_xy_weight = 12.0
        self.kpt_vis_weight = 2.0
        self.vari_focal_loss = False
        # Loss Functions
        if use_focal:
            if use_vari_focal:
                self.vari_focal_loss = True
                self.cls_loss_fn = VarifocalLoss(gamma=2.0, alpha=0.75)
            else:    
                self.cls_loss_fn = FocalLoss(gamma=2.0, reduction='sum', num_classes=self.num_classes)
        else:
            self.cls_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
            
        self.kpt_vis_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
        
        # [수정] Keypoint Loss (OKS) 설정
        # 17개(사람)인 경우 COCO Sigmas 사용, 그 외(4개 등)는 균등 Sigma 사용
        if self.num_kpts == 17:
            kpt_sigmas = torch.tensor([
                .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 
                1.07, 1.07, .87, .87, .89, .89
            ], device=self.device) / 10.0
        else:
            # 모든 키포인트에 대해 균등한 sigma (0.05 ~ 0.1 정도가 적당)
            kpt_sigmas = torch.ones(self.num_kpts, device=self.device) * 0.1
            
        self.kpt_loss_fn = KeypointLoss(kpt_sigmas)
        
        # Bbox Loss
        self.bbox_loss = BboxLoss(self.reg_max).to(self.device)
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=self.device)

    def decode_bboxes(self, pred_dist, anchor_points):
        prob = pred_dist.softmax(2)
        dist = prob.matmul(self.proj)
        lt, rb = dist.chunk(2, -1)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], -1)

    def __call__(self, preds, targets):
        loss_box = torch.zeros(1, device=self.device)
        loss_dfl = torch.zeros(1, device=self.device)
        loss_cls = torch.zeros(1, device=self.device)
        loss_kpt_xy = torch.zeros(1, device=self.device)
        loss_kpt_vis = torch.zeros(1, device=self.device)
        
        num_pos_total = 0

        for pred in preds:
            B, C, H, W = pred.shape
            pred = pred.permute(0, 2, 3, 1) # (B, H, W, C)
            
            target_mask = torch.zeros(B, H, W, device=self.device, dtype=torch.bool)
            target_box = torch.zeros(B, H, W, 4, device=self.device)
            target_kpts = torch.zeros(B, H, W, self.num_kpts * 3, device=self.device)
            target_cls = torch.zeros(B, H, W, self.num_classes, device=self.device)
            
            yv, xv = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij')
            anchor_points = torch.stack((xv, yv), -1).float() + 0.5
            
            if targets.shape[0] > 0:
                gt_boxes = targets[:, 2:6] * torch.tensor([W, H, W, H], device=self.device)
                gt_kpts = targets[:, 6:].clone()
                gt_kpts[:, 0::3] *= W
                gt_kpts[:, 1::3] *= H
                
                b_idx = targets[:, 0].long()
                c_idx = targets[:, 1].long()
                gx = gt_boxes[:, 0].long().clamp(0, W-1)
                gy = gt_boxes[:, 1].long().clamp(0, H-1)
                
                target_mask[b_idx, gy, gx] = True
                target_cls[b_idx, gy, gx, c_idx] = 1.0
                target_box[b_idx, gy, gx] = gt_boxes 
                
                t_kpts = gt_kpts.clone()
                t_kpts[:, 0::3] = gt_kpts[:, 0::3] - (gx.unsqueeze(1) + 0.5)
                t_kpts[:, 1::3] = gt_kpts[:, 1::3] - (gy.unsqueeze(1) + 0.5)
                target_kpts[b_idx, gy, gx] = t_kpts

            # --- Loss Calculation ---
            idx_box_end = 4 * self.reg_max
            idx_cls_end = idx_box_end + self.num_classes
            
            # Positive Sample Processing
            num_pos = target_mask.sum()
            num_pos_total += num_pos
            
            if num_pos > 0:
                pred_pos = pred[target_mask]
                t_box_pos = target_box[target_mask]
                t_kpts_pos = target_kpts[target_mask]
                anchors_pos = anchor_points.unsqueeze(0).repeat(B, 1, 1, 1)[target_mask]
                
                # Box Loss (DFL + IoU)
                pred_dist = pred_pos[:, :idx_box_end].reshape(-1, 4, self.reg_max)
                decoded_bboxes = self.decode_bboxes(pred_dist, anchors_pos)
                
                l_iou, l_dfl, pred_iou = self.bbox_loss(
                    pred_dist, 
                    decoded_bboxes, 
                    anchors_pos, 
                    t_box_pos, 
                    target_scores=target_cls[target_mask], 
                    target_scores_sum=num_pos, 
                    fg_mask=None
                )
                
                loss_box += l_iou
                loss_dfl += l_dfl
                
                # [수정] Keypoint Loss (OKS)
                pred_iou = pred_iou.clamp(0, 1)
                t_cls_pos = target_cls[target_mask]
                t_cls_pos = t_cls_pos * pred_iou.unsqueeze(-1) 
                target_cls[target_mask] = t_cls_pos
                pred_kpts = pred_pos[:, idx_cls_end:].reshape(-1, self.num_kpts, 3)
                t_kpts_grid = t_kpts_pos.reshape(-1, self.num_kpts, 3)
                
                # Visibility Mask (0: 안보임, 1: 보임)
                kpt_mask = (t_kpts_grid[..., 2] > 0).float()
                
                # OKS Loss에는 Area 정보가 필요 (Width * Height)
                # t_box_pos는 (x, y, w, h) 형태이므로 w * h 계산
                area = t_box_pos[:, 2] * t_box_pos[:, 3]
                
                loss_kpt_xy += self.kpt_loss_fn(
                    pred_kpts[..., :2],   # 예측 좌표
                    t_kpts_grid[..., :2], # 정답 좌표
                    kpt_mask,             # Visibility Mask
                    area.unsqueeze(-1)    # (N, 1) 형태로 전달
                )
                
                # Visibility Loss
                loss_kpt_vis += self.kpt_vis_loss_fn(
                    pred_kpts[..., 2], 
                    kpt_mask
                )
            # Classification Loss
            pred_cls = pred[..., idx_box_end:idx_cls_end]
            if self.vari_focal_loss:
                target_labels = (target_cls > 0).float() # IoU가 들어있으므로 0보다 크면 1로 변환
                
                loss_cls += self.cls_loss_fn(
                    pred_cls.reshape(-1, self.num_classes), 
                    target_cls.reshape(-1, self.num_classes),
                    target_labels.reshape(-1, self.num_classes)
                )
            else:
                loss_cls += self.cls_loss_fn(pred_cls, target_cls)

        num_pos_total = max(num_pos_total, 1)
        
        loss_box = (loss_box / num_pos_total) * self.box_weight
        loss_dfl = (loss_dfl / num_pos_total) * self.dfl_weight
        loss_cls = (loss_cls / num_pos_total) * self.cls_weight
        loss_kpt_xy = (loss_kpt_xy / num_pos_total) * self.kpt_xy_weight
        loss_kpt_vis = (loss_kpt_vis / num_pos_total) * self.kpt_vis_weight
        
        total_loss = loss_box + loss_dfl + loss_cls + loss_kpt_xy + loss_kpt_vis
        
        return total_loss, (
            loss_box.item(), 
            loss_dfl.item(), 
            loss_cls.item(),
            loss_kpt_xy.item(), 
            loss_kpt_vis.item()
        )
    
hyp = {
    'lr': 1e-3,
    'epochs': 100,
    'batch_size': 24,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'weight_decay': 0.05,
    'warmup_epochs': 3
}


def train(ckps=None):
    device = hyp['device']
    
    # 1. Model Initialization
    print(f"Initializing DINOv3Pose on {device}...")
    # [중요] nkpts=4로 데이터셋 설정했으니 모델도 4로 맞춰야 함 (이전 코드에선 17이었음)
    model = DINOv3Pose(backbone='dinov3_convnext_base', nkpts=(4, 3), ncls=7, device=device)
    if ckps:
        param = torch.load(ckps)
        model.load_state_dict(param)
    model.train() 

    # 2. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=hyp['lr'], weight_decay=hyp['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyp['epochs'], eta_min=1e-5)

    # 3. Dataset & Dataloader
    print("Loading Data...")
    train_img_dir = "/media/otter/otterHD/pallet_data/data_yolo/clear_data/train/images"
    train_label_dir = "/media/otter/otterHD/pallet_data/data_yolo/clear_data/train/labels"
    
    batch_size = hyp['batch_size']
    img_size = 640
    
    # nkpts=4 확인 필수!
    dataset = YoloPoseDataset(train_img_dir, train_label_dir, img_size=img_size, nkpts=4)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            collate_fn=YoloPoseDataset.collate_fn, num_workers=4)

    # 4. Loss Function
    print("Setting up Loss...")
    criterion = ComputeLoss(model, use_focal=True)
    
    # 5. AMP Scaler
    scaler = torch.amp.GradScaler(enabled=True)
    top_loss = 100
    # --- Training Loop ---
    for epoch in range(hyp['epochs']):
        model.train() # Epoch 시작마다 확실히 모드 설정
        pbar = tqdm(enumerate(dataloader), 
                    total=len(dataloader),
                    desc = f'Epoch {epoch+1}',
                    dynamic_ncols= True,
                    smoothing=0.5)
        epoch_loss = 0
        box_loss_sum = 0
        dfl_loss_sum = 0
        cls_loss_sum = 0
        kpt_xy_loss_sum = 0
        
        for i, (imgs, targets) in pbar:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            # [수정됨] Gradient 초기화는 루프 시작 시 1번만
            optimizer.zero_grad()
            
            # Automatic Mixed Precision (AMP)
            with torch.amp.autocast(device_type=hyp['device'], enabled=True):
                preds = model(imgs) # Forward
                loss, loss_items = criterion(preds, targets)
            
            # [수정됨] Backward Pass (AMP 적용)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            # Logging
            epoch_loss += loss.item()
            box_loss_sum += loss_items[0]
            dfl_loss_sum += loss_items[1]
            cls_loss_sum += loss_items[2]
            kpt_xy_loss_sum += (loss_items[3] + loss_items[4])
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "box": f"{loss_items[0]:.4f}",
                "dfl": f"{loss_items[1]:.4f}",
                "cls": f"{loss_items[2]:.4f}",
                "kpts": f"{(loss_items[3] + loss_items[4]):.4f}",
            })
            
        scheduler.step()
        
        avg_loss = epoch_loss / len(dataloader)
        avg_box = box_loss_sum / len(dataloader)
        avg_obj = dfl_loss_sum / len(dataloader)
        avg_cls = cls_loss_sum / len(dataloader)
        avg_kpt_xy = kpt_xy_loss_sum / len(dataloader)
        
        print(f" Toal : {avg_loss:.4f} | box: {avg_box:.4f} | dfl: {avg_obj:.4f} | cls : {avg_cls:.4f} | kpts: {avg_kpt_xy:.4f}")
        print(f" ")
        
        # Save Checkpoint
        if (epoch + 1) % 10 == 0 or (epoch + 1) == hyp['epochs']:
            os.makedirs("weights", exist_ok=True)
            torch.save(model.state_dict(), f"weights/pose_dino_epoch_{epoch+1}.pt")
            print(f"Saved checkpoint to weights/pose_dino_epoch_{epoch+1}.pt")
        if top_loss > avg_loss:
            top_loss = avg_loss
            torch.save(model.state_dict(), f"weights/best.pt")

    print("Training Completed.")


def test():
    pass 
if __name__ == '__main__':
    # hyp 변수가 선언되어 있어야 합니다.
    # hyp = {'device': 'cuda', 'lr': 1e-3, 'weight_decay': 5e-4, 'epochs': 100, 'batch_size': 8}
    train()

