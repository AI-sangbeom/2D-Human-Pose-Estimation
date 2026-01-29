import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Helper Classes
# ==========================================

class VarifocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # pred_score: Logits
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        
        with torch.autocast(device_type=pred_score.device.type, enabled=False):
            loss = (
                F.binary_cross_entropy_with_logits(
                    pred_score.float(), 
                    gt_score.float(), 
                    reduction="none"
                ) * weight
            ).sum()
        return loss

class KeypointLoss(nn.Module):
    def __init__(self, sigmas: torch.Tensor) -> None:
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
        area = area.unsqueeze(-1)  # (N, 1)
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()

class ImprovedKeypointLoss(nn.Module):
    """
    개선된 Keypoint Loss
    - OKS 기반 유지
    - Wing Loss 스타일 gradient 개선
    - 더 안정적인 정규화
    """
    def __init__(self, sigmas: torch.Tensor, omega: float = 10.0) -> None:
        super().__init__()
        self.sigmas = sigmas
        self.omega = omega
        
    def forward(
        self, 
        pred_kpts: torch.Tensor,  # (N, nkpts, 2)
        gt_kpts: torch.Tensor,     # (N, nkpts, 2)
        kpt_mask: torch.Tensor,    # (N, nkpts)
        area: torch.Tensor         # (N,)
    ) -> torch.Tensor:
        
        area = area.unsqueeze(-1).clamp(min=1.0)  # 최소값 보장
        
        # Euclidean distance squared
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + \
            (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        
        # OKS normalization
        variance = (2 * self.sigmas).pow(2) * area * 2
        e = d / (variance + 1e-6)  # 더 큰 epsilon
        
        # Wing Loss 스타일: 작은 오차에 더 집중
        threshold = 2.0
        loss = torch.where(
            e < threshold,
            self.omega * torch.log(1 + e / self.omega + 0.0 * e),
            e - threshold + self.omega * torch.log(1 + threshold / self.omega + 0.0 * e)
        )
        
        # Visibility masking
        loss = loss * kpt_mask
        
        # 정규화: visible keypoint 개수로
        num_visible = kpt_mask.sum().clamp(min=1.0)
        return loss.sum() / num_visible


class RobustKeypointLoss(nn.Module):
    """
    더욱 강건한 버전 - Adaptive Wing Loss + OKS
    """
    def __init__(
        self, 
        sigmas: torch.Tensor,
        alpha: float = 2.1,
        omega: float = 14,
        epsilon: float = 1,
        theta: float = 0.5
    ) -> None:
        super().__init__()
        self.sigmas = sigmas
        self.alpha = alpha
        self.omega = omega
        self.epsilon = epsilon
        self.theta = theta
        
    def forward(
        self,
        pred_kpts: torch.Tensor,
        gt_kpts: torch.Tensor,
        kpt_mask: torch.Tensor,
        area: torch.Tensor
    ) -> torch.Tensor:
        
        area = area.unsqueeze(-1).clamp(min=1.0)
        
        # L2 distance
        diff = torch.sqrt(
            (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + 
            (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2) + 1e-8
        )
        
        # OKS normalization
        sigma_area = self.sigmas * torch.sqrt(area)
        diff_normalized = diff / (sigma_area + 1e-6)
        
        # Adaptive Wing Loss
        C = self.omega * (1 - torch.log(torch.tensor(1 + self.omega / self.epsilon)))
        
        loss = torch.where(
            diff_normalized < self.theta,
            self.omega * torch.log(1 + torch.pow(diff_normalized / self.epsilon, self.alpha - diff_normalized)),
            diff_normalized - C
        )
        
        loss = loss * kpt_mask
        num_visible = kpt_mask.sum().clamp(min=1.0)
        
        return loss.sum() / num_visible
    
class MultiScaleKeypointLoss(nn.Module):
    """
    Multi-scale 일관성을 고려한 Loss
    """
    def __init__(self, sigmas: torch.Tensor, scales: list = [0.5, 1.0, 2.0]):
        super().__init__()
        self.sigmas = sigmas
        self.scales = scales
        self.base_loss = ImprovedKeypointLoss(sigmas)
        
    def forward(
        self,
        pred_kpts: torch.Tensor,
        gt_kpts: torch.Tensor,
        kpt_mask: torch.Tensor,
        area: torch.Tensor
    ) -> torch.Tensor:
        
        total_loss = 0
        
        # Multi-scale evaluation
        for scale in self.scales:
            scaled_pred = pred_kpts * scale
            scaled_gt = gt_kpts * scale
            scaled_area = area * (scale ** 2)
            
            loss = self.base_loss(scaled_pred, scaled_gt, kpt_mask, scaled_area)
            total_loss += loss
            
        return total_loss / len(self.scales)
    
class HybridKeypointLoss(nn.Module):
    """
    여러 Loss의 장점을 결합
    - OKS Loss (metric consistency)
    - L1 Loss (stable gradient)
    - Smoothness regularization
    """
    def __init__(
        self, 
        sigmas: torch.Tensor,
        lambda_oks: float = 1.0,
        lambda_l1: float = 0.5,
        lambda_smooth: float = 0.1
    ):
        super().__init__()
        self.sigmas = sigmas
        self.lambda_oks = lambda_oks
        self.lambda_l1 = lambda_l1
        self.lambda_smooth = lambda_smooth
        
        self.oks_loss = ImprovedKeypointLoss(sigmas)
        
    def forward(
        self,
        pred_kpts: torch.Tensor,
        gt_kpts: torch.Tensor,
        kpt_mask: torch.Tensor,
        area: torch.Tensor
    ) -> tuple:
        
        # 1. OKS-based loss
        loss_oks = self.oks_loss(pred_kpts, gt_kpts, kpt_mask, area)
        
        # 2. L1 loss (더 안정적인 gradient)
        diff = torch.abs(pred_kpts - gt_kpts).sum(dim=-1)
        loss_l1 = (diff * kpt_mask).sum() / kpt_mask.sum().clamp(min=1.0)
        
        # 3. Smoothness regularization (인접 keypoint 간)
        if pred_kpts.shape[1] > 1:
            kpt_diff = pred_kpts[:, 1:] - pred_kpts[:, :-1]
            loss_smooth = kpt_diff.pow(2).sum() / pred_kpts.shape[0]
        else:
            loss_smooth = torch.tensor(0.0, device=pred_kpts.device)
        
        # Combined loss
        total_loss = (
            self.lambda_oks * loss_oks +
            self.lambda_l1 * loss_l1 +
            self.lambda_smooth * loss_smooth
        )
        
        return total_loss
    
# ==========================================
# 2. Main ComputeLoss Class
# ==========================================

class ComputeLoss:
    def __init__(self, model, use_focal=True, use_vari_focal=True):
        self.device = next(model.parameters()).device
        self.nc = model.head.nc
        self.nk = model.head.nk
        # 모델 구조에 따라 kpt_shape[1]이 2(xy)인지 3(xyc)인지 확인 필요. 여기선 3으로 가정
        self.num_kpts = self.nk // 3 
        
        # 1. Loss Functions
        if use_focal:
            if use_vari_focal:
                self.cls_loss_fn = VarifocalLoss(gamma=2.0, alpha=0.75)
            else:
                # FocalLoss 구현체가 따로 있다고 가정
                self.cls_loss_fn = nn.BCEWithLogitsLoss(reduction='mean') 
        else:
            self.cls_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        
        # Keypoint Sigmas (COCO 17 points 기준)
        if self.num_kpts == 17:
            kpt_sigmas = torch.tensor([
                .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 
                1.07, 1.07, .87, .87, .89, .89
            ], device=self.device) / 10.0
        else:
            # Custom keypoints인 경우 기본값
            kpt_sigmas = torch.ones(self.num_kpts, device=self.device) * 0.05
        
        self.kpt_loss_fn = HybridKeypointLoss(kpt_sigmas)
        self.kpt_vis_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
        
        # Weights
        self.cls_weight = 5.0
        self.kpt_weight = 10.0 

    # def point2box_xywh(self, points):
    #     """
    #     Keypoints에서 Min/Max를 찾아 Bounding Box (cx, cy, w, h) 형태로 반환
    #     """
    #     min_vals, _ = torch.min(points, dim=1) # (N, 2)
    #     max_vals, _ = torch.max(points, dim=1) # (N, 2)
        
    #     # xyxy -> xywh 변환
    #     w = max_vals[:, 0] - min_vals[:, 0]
    #     h = max_vals[:, 1] - min_vals[:, 1]
    #     cx = min_vals[:, 0] + w * 0.5
    #     cy = min_vals[:, 1] + h * 0.5
        
    #     return torch.stack([cx, cy, w, h], dim=-1)
    def point2box_xywh(self, points, visibility):
        """
        Visibility를 고려한 안정적인 box 추정
        points: (N, nkpts, 2)
        visibility: (N, nkpts)
        """
        valid_mask = visibility > 0
        
        boxes = []
        for i in range(points.shape[0]):
            visible_points = points[i][valid_mask[i]]
            
            if visible_points.shape[0] < 2:
                visible_points = points[i]
            
            # Outlier rejection: percentile 기반
            if visible_points.shape[0] > 4:
                x_sorted, _ = torch.sort(visible_points[:, 0])
                y_sorted, _ = torch.sort(visible_points[:, 1])
                
                trim = max(1, int(visible_points.shape[0] * 0.1))
                x_min = x_sorted[trim]
                x_max = x_sorted[-trim-1]
                y_min = y_sorted[trim]
                y_max = y_sorted[-trim-1]
            else:
                x_min = visible_points[:, 0].min()
                x_max = visible_points[:, 0].max()
                y_min = visible_points[:, 1].min()
                y_max = visible_points[:, 1].max()
            
            w = (x_max - x_min).clamp(min=1.0)
            h = (y_max - y_min).clamp(min=1.0)
            cx = (x_min + x_max) * 0.5
            cy = (y_min + y_max) * 0.5
            
            boxes.append(torch.stack([cx, cy, w, h]))
        
        return torch.stack(boxes)
    
    @staticmethod
    def bbox_iou(box1, box2, CIoU=True, eps=1e-7):
        """
        box1, box2: (N, 4) -> (cx, cy, w, h) format
        """
        # xywh -> xyxy
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

        # Intersection
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
            v = (4 / (3.141592 ** 2)) * torch.pow(torch.atan(w2 / (h2+eps)) - torch.atan(w1 / (h1+eps)), 2)
            with torch.no_grad():
                alpha = v / (v - iou + (1 + eps))
            ciou = iou - (rho2 / c2 + v * alpha)
            return torch.nan_to_num(ciou, nan=0.0)
        return torch.nan_to_num(iou, nan=0.0)
    
    def __call__(self, preds, targets):
        loss_cls = torch.zeros(1, device=self.device)
        loss_kpt = torch.zeros(1, device=self.device)
        loss_kpt_vis = torch.zeros(1, device=self.device)
        
        num_pos_total = 0
        
        for i, pred in enumerate(preds):
            B, C, H, W = pred.shape
            
            # (B, H, W, C)
            pred = pred.permute(0, 2, 3, 1)
            
            # 1. 배경을 포함한 전체 Target 초기화 (0으로 채움)
            target_cls = torch.zeros(B, H, W, self.nc, device=self.device)
            target_mask = torch.zeros(B, H, W, device=self.device, dtype=torch.bool)
            
            # 2. 정답(Positive) 데이터 준비
            if targets.shape[0] > 0:
                gt_box = targets[:, 2:6].clone()
                gt_kpts = targets[:, 6:].clone()
                
                # Scaling
                gt_box[:, [0, 2]] *= W
                gt_box[:, [1, 3]] *= H
                gt_kpts[:, 0::3] *= W
                gt_kpts[:, 1::3] *= H
                
                b_idx = targets[:, 0].long()
                c_idx = targets[:, 1].long()

                gx = gt_box[:, 0].long().clamp(0, W-1)
                gy = gt_box[:, 1].long().clamp(0, H-1)

                # Positive Mask 마킹
                target_mask[b_idx, gy, gx] = True
                
                num_pos = target_mask.sum()
                num_pos_total += num_pos

                if num_pos > 0:
                    # -----------------------------------------------------------
                    # A. Positive Sample에 대해서만 IoU 및 Keypoint Loss 계산
                    # -----------------------------------------------------------
                    
                    # Offset 계산
                    gt_box_offset = gt_box.clone()
                    gt_box_offset[:, 0] -= gx.float()
                    gt_box_offset[:, 1] -= gy.float()
                    gt_kpts[:, 0::3] -= gx.unsqueeze(1).float()
                    gt_kpts[:, 1::3] -= gy.unsqueeze(1).float()
                    
                    pred_pos = pred[b_idx, gy, gx]
                    idx_cls_end = self.nc
                    
                    pred_kpts_raw = pred_pos[:, idx_cls_end:]
                    pred_kpts = pred_kpts_raw.reshape(-1, self.num_kpts, 3)
                    pred_kpts_xy = pred_kpts[..., :2]
                    pred_kpts_conf = pred_kpts[..., 2]

                    gt_kpts_pos = gt_kpts.reshape(-1, self.num_kpts, 3)
                    gt_kpts_xy = gt_kpts_pos[..., :2]
                    kpt_mask = (gt_kpts_pos[..., 2] > 0).float()
                    area = gt_box[:, 2] * gt_box[:, 3]

                    # IoU 계산
                    pred_box_from_kpts = self.point2box_xywh(pred_kpts_xy, kpt_mask)
                    pred_iou = self.bbox_iou(
                        pred_box_from_kpts, 
                        gt_box_offset, 
                        CIoU=True
                    )
                    
                    
                    loss_kpt += self.kpt_loss_fn(pred_kpts_xy, gt_kpts_xy, kpt_mask, area)
                    loss_kpt_vis += self.kpt_vis_loss_fn(pred_kpts_conf, kpt_mask)

                    pred_iou_score = pred_iou.detach().clamp(0, 1)
                    target_cls[b_idx, gy, gx, c_idx] = pred_iou_score

            pred_cls_all = pred[..., :self.nc].reshape(-1, self.nc)
            target_cls_all = target_cls.reshape(-1, self.nc)
            
            target_label_all = (target_cls_all > 0).float()

            # 전체 픽셀에 대해 Loss 수행
            loss_cls += self.cls_loss_fn(pred_cls_all, target_cls_all, target_label_all)

        # Normalize Loss
        num_pos_total = max(num_pos_total, 1)
        
        # Class Loss는 전체 픽셀 합이므로 보통 num_pos로 나누거나 H*W로 나눔.
        # YOLO 계열은 보통 num_pos로 나눕니다.
        loss_cls = (loss_cls / num_pos_total) * 1.0
        loss_kpt = (loss_kpt / num_pos_total) * 10.0
        loss_kpt_vis = (loss_kpt_vis / num_pos_total) * 5.0
        
        total_loss = loss_cls + loss_kpt + loss_kpt_vis
        
        return total_loss, (loss_cls.item(), loss_kpt.item(), loss_kpt_vis.item())