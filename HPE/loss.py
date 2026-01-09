import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp as amp  # amp 임포트 확인

# ==========================================
# 1. Helper Classes
# ==========================================

class VarifocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        pred_score: (N, C) Logits
        gt_score: (N, C) IoU Score (0~1 실수)
        label: (N, C) Binary Label (0 or 1)
        """
        # VFL 식에 따라 가중치 계산
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        
        # Loss 계산 (BCE with Logits)
        # AMP 비활성화 구간 (안전성 확보)
        with torch.autocast(device_type=pred_score.device.type, enabled=False):
            loss = (
                F.binary_cross_entropy_with_logits(
                    pred_score.float(), 
                    gt_score.float(), 
                    reduction="none"
                ) * weight
            ).sum() # 논문 구현체에 따라 mean(1).sum() 대신 전체 sum 사용 (Reduction은 외부에서 처리)
            
        return loss

class DFLoss(nn.Module):
    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        loss = (
            F.cross_entropy(pred_dist, tl.reshape(-1), reduction="none").reshape(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.reshape(-1), reduction="none").reshape(tl.shape) * wr
        )
        return loss.mean(-1, keepdim=True)

class KeypointLoss(nn.Module):
    def __init__(self, sigmas: torch.Tensor) -> None:
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()

class BboxLoss(nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        self.dfl_loss = DFLoss(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        
        # IoU 계산 및 Loss
        iou = self.bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL Loss
        target_ltrb = self.bbox2dist(anchor_points, target_bboxes, self.reg_max)
        loss_dfl = self.dfl_loss(
            pred_dist[fg_mask].reshape(-1, self.reg_max), 
            target_ltrb[fg_mask].reshape(-1)
        ) * weight.reshape(-1)
        
        return loss_iou, loss_dfl.sum() / target_scores_sum, iou # IoU 반환 추가

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
# 2. ComputeLoss (VFL 적용됨)
# ==========================================

class ComputeLoss:
    def __init__(self, model, use_varifocal=True):
        self.device = next(model.parameters()).device
        self.num_kpts = model.head.nk // model.head.kpt_shape[1] 
        self.num_classes = model.head.nc
        self.reg_max = model.head.reg_max 

        if hasattr(model.head, 'stride'):
            self.strides = model.head.stride
        else:
            self.strides = [8, 16, 32]

        # Weights
        self.box_weight = 7.5
        self.dfl_weight = 1.5 
        self.cls_weight = 0.5
        self.kpt_xy_weight = 12.0
        self.kpt_vis_weight = 2.0
        
        # [수정] Varifocal Loss 사용 설정
        if use_varifocal:
            self.cls_loss_fn = VarifocalLoss(gamma=2.0, alpha=0.75)
        else:
            self.cls_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
            
        self.kpt_vis_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
        
        # Keypoint Sigmas
        if self.num_kpts == 17: 
            kpt_sigmas = torch.tensor([
                .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 
                1.07, 1.07, .87, .87, .89, .89
            ], device=self.device) / 10.0
        else:
            kpt_sigmas = torch.ones(self.num_kpts, device=self.device) * 0.05
            
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

        for scale_idx, pred in enumerate(preds):
            B, C, H, W = pred.shape
            pred = pred.permute(0, 2, 3, 1) # (B, H, W, C)
            
            # 1. Target 준비
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
                target_cls[b_idx, gy, gx, c_idx] = 1.0 # 일단 1.0으로 초기화 (이후 IoU로 교체)
                target_box[b_idx, gy, gx] = gt_boxes 
                
                t_kpts = gt_kpts.clone()
                t_kpts[:, 0::3] = gt_kpts[:, 0::3] - (gx.unsqueeze(1) + 0.5)
                t_kpts[:, 1::3] = gt_kpts[:, 1::3] - (gy.unsqueeze(1) + 0.5)
                target_kpts[b_idx, gy, gx] = t_kpts

            # --- Loss Calculation ---
            idx_box_end = 4 * self.reg_max
            idx_cls_end = idx_box_end + self.num_classes
            
            # [순서 변경] Positive Sample 처리를 먼저 수행하여 IoU를 구함
            num_pos = target_mask.sum()
            num_pos_total += num_pos
            
            if num_pos > 0:
                pred_pos = pred[target_mask]
                t_box_pos = target_box[target_mask]
                t_kpts_pos = target_kpts[target_mask]
                anchors_pos = anchor_points.unsqueeze(0).repeat(B, 1, 1, 1)[target_mask]
                
                # 2-1. Box Loss (IoU 계산을 위해 필요)
                pred_dist = pred_pos[:, :idx_box_end].reshape(-1, 4, self.reg_max)
                decoded_bboxes = self.decode_bboxes(pred_dist, anchors_pos)
                
                # BboxLoss에서 IoU를 리턴받음
                l_iou, l_dfl, pred_iou = self.bbox_loss(
                    pred_dist, 
                    decoded_bboxes, 
                    anchors_pos, 
                    t_box_pos, 
                    target_scores=target_cls[target_mask], # weighting
                    target_scores_sum=num_pos, 
                    fg_mask=None
                )
                
                loss_box += l_iou
                loss_dfl += l_dfl
                
                # [핵심] VFL을 위해 Target Class Score를 IoU로 업데이트
                # detach()를 해서 score target에는 그래디언트가 흐르지 않게 함
                pred_iou = pred_iou.detach().clamp(0, 1)
                
                # 현재 Positive Sample인 위치의 target_cls 값을 IoU로 교체
                # (B, H, W, C) 중 (N_pos, C) 부분을 가져와서 업데이트
                # target_cls[target_mask]는 (N_pos, C) 형태
                
                # 방법: target_cls가 1.0인 부분(정답 클래스)만 IoU 값으로 바꿈
                # target_cls[target_mask] * pred_iou.unsqueeze(-1) 형태로 곱해주면 됨
                # 단, pred_iou는 (N_pos,) 이고 target_cls는 (N_pos, C) (One-hot)
                t_cls_pos = target_cls[target_mask]
                t_cls_pos = t_cls_pos * pred_iou.unsqueeze(-1) # 정답 클래스 자리에 IoU가 들어감
                target_cls[target_mask] = t_cls_pos

                # 2-2. Keypoint Loss
                pred_kpts = pred_pos[:, idx_cls_end:].reshape(-1, self.num_kpts, 3)
                t_kpts_grid = t_kpts_pos.reshape(-1, self.num_kpts, 3)
                kpt_mask = (t_kpts_grid[..., 2] > 0).float()
                area = t_box_pos[:, 2] * t_box_pos[:, 3]
                
                loss_kpt_xy += self.kpt_loss_fn(
                    pred_kpts[..., :2], t_kpts_grid[..., :2], kpt_mask, area.unsqueeze(-1)
                )
                loss_kpt_vis += self.kpt_vis_loss_fn(pred_kpts[..., 2], kpt_mask)

            # 3. Classification Loss (VFL)
            # 이제 target_cls에는 Positive인 경우 IoU가, Negative인 경우 0이 들어있음
            pred_cls = pred[..., idx_box_end:idx_cls_end]
            
            # VFL Forward: (pred_score, gt_score, label)
            # pred_score: (N_total, C)
            # gt_score: (N_total, C) -> IoU (0~1)
            # label: (N_total, C) -> Binary (0 or 1)
            
            target_labels = (target_cls > 0).float() # IoU가 들어있으므로 0보다 크면 1로 변환
            
            loss_cls += self.cls_loss_fn(
                pred_cls.reshape(-1, self.num_classes), 
                target_cls.reshape(-1, self.num_classes),
                target_labels.reshape(-1, self.num_classes)
            )

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