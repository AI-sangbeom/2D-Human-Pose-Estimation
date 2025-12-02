import torch
from typing import List, Dict, Optional, Sequence


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    boxes1: (N, 4), boxes2: (M, 4)
    each box: [x1, y1, x2, y2]
    return: (N, M) IoU matrix
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.size(0), boxes2.size(0)))

    # (N, 1, 4), (1, M, 4)
    b1 = boxes1.unsqueeze(1)
    b2 = boxes2.unsqueeze(0)

    # intersection
    inter_x1 = torch.max(b1[..., 0], b2[..., 0])
    inter_y1 = torch.max(b1[..., 1], b2[..., 1])
    inter_x2 = torch.min(b1[..., 2], b2[..., 2])
    inter_y2 = torch.min(b1[..., 3], b2[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h  # (N, M)

    # areas
    area1 = (boxes1[..., 2] - boxes1[..., 0]).clamp(min=0) * \
            (boxes1[..., 3] - boxes1[..., 1]).clamp(min=0)  # (N,)
    area2 = (boxes2[..., 2] - boxes2[..., 0]).clamp(min=0) * \
            (boxes2[..., 3] - boxes2[..., 1]).clamp(min=0)  # (M,)

    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter
    iou = inter / union.clamp(min=1e-10)
    return iou


class DetectionMAP:
    """
    IoU 기반 object detection mAP 계산용 클래스.

    - multi-class 지원
    - COCO 스타일 IoU threshold [0.50:0.95] 평균 mAP
    - VOC 스타일 AP@0.5도 함께 뽑을 수 있음

    사용법 요약:
        metric = DetectionMAP(num_classes=80)
        metric.update(...)
        results = metric.compute()
    """

    def __init__(
        self,
        num_classes: int,
        iou_thresholds: Optional[Sequence[float]] = None,
    ):
        self.num_classes = num_classes
        if iou_thresholds is None:
            # COCO 스타일: 0.50 ~ 0.95 step 0.05
            self.iou_thresholds = torch.arange(0.5, 0.96, 0.05)
        else:
            self.iou_thresholds = torch.tensor(iou_thresholds, dtype=torch.float32)

        # 누적 버퍼
        # class별 GT, detection을 모아둘 리스트
        self.gt_boxes_per_image: List[List[torch.Tensor]] = []
        self.gt_labels_per_image: List[List[torch.Tensor]] = []

        self.pred_boxes_per_image: List[List[torch.Tensor]] = []
        self.pred_scores_per_image: List[List[torch.Tensor]] = []
        self.pred_labels_per_image: List[List[torch.Tensor]] = []

    def reset(self):
        self.gt_boxes_per_image = []
        self.gt_labels_per_image = []
        self.pred_boxes_per_image = []
        self.pred_scores_per_image = []
        self.pred_labels_per_image = []

    @torch.no_grad()
    def update(
        self,
        gt_boxes: List[torch.Tensor],       # List[Tensor(ngt_i, 4)]
        gt_labels: List[torch.Tensor],      # List[Tensor(ngt_i)]
        pred_boxes: List[torch.Tensor],     # List[Tensor(npred_i, 4)]
        pred_scores: List[torch.Tensor],    # List[Tensor(npred_i)]
        pred_labels: List[torch.Tensor],    # List[Tensor(npred_i)]
    ):
        """
        한 배치에 대한 GT / Pred 정보를 누적.

        각 리스트의 길이 = batch_size, i번째 요소는 i번째 이미지의 annotations.
        """
        B = len(gt_boxes)
        assert len(gt_labels) == B
        assert len(pred_boxes) == B
        assert len(pred_scores) == B
        assert len(pred_labels) == B

        self.gt_boxes_per_image.extend(gt_boxes)
        self.gt_labels_per_image.extend(gt_labels)
        self.pred_boxes_per_image.extend(pred_boxes)
        self.pred_scores_per_image.extend(pred_scores)
        self.pred_labels_per_image.extend(pred_labels)

    @torch.no_grad()
    def _compute_ap_for_class(
        self,
        cls_id: int,
        iou_thr: float,
    ) -> float:
        """
        특정 클래스 & 특정 IoU threshold에 대한 AP 계산.
        """
        # GT 수 카운트 + GT 마스크
        gt_boxes_all = []
        gt_img_ids = []  # GT가 속한 이미지 인덱스
        gt_detected = [] # per-GT matched flag

        # detection 모음
        det_boxes = []
        det_scores = []
        det_img_ids = []

        num_images = len(self.gt_boxes_per_image)

        for img_id in range(num_images):
            gt_boxes_img = self.gt_boxes_per_image[img_id]
            gt_labels_img = self.gt_labels_per_image[img_id]

            # 해당 클래스 GT
            gt_mask = (gt_labels_img == cls_id)
            gt_boxes_c = gt_boxes_img[gt_mask]
            gt_boxes_all.append(gt_boxes_c)
            gt_img_ids.extend([img_id] * gt_boxes_c.size(0))
            gt_detected.extend([False] * gt_boxes_c.size(0))

            # 해당 클래스 detection
            pred_boxes_img = self.pred_boxes_per_image[img_id]
            pred_scores_img = self.pred_scores_per_image[img_id]
            pred_labels_img = self.pred_labels_per_image[img_id]

            det_mask = (pred_labels_img == cls_id)
            det_boxes_c = pred_boxes_img[det_mask]
            det_scores_c = pred_scores_img[det_mask]

            det_boxes.append(det_boxes_c)
            det_scores.append(det_scores_c)
            det_img_ids.extend([img_id] * det_boxes_c.size(0))

        if len(gt_boxes_all) == 0:
            num_gt = 0
        else:
            num_gt = sum([g.size(0) for g in gt_boxes_all])

        # GT가 아예 없으면 AP 정의하기 어렵다 → 0으로
        if num_gt == 0:
            return 0.0

        if len(det_boxes) == 0:
            return 0.0

        gt_boxes_all = torch.cat(gt_boxes_all, dim=0) if len(gt_boxes_all) > 0 else torch.zeros((0, 4))
        det_boxes_all = torch.cat(det_boxes, dim=0) if len(det_boxes) > 0 else torch.zeros((0, 4))
        det_scores_all = torch.cat(det_scores, dim=0) if len(det_scores) > 0 else torch.zeros((0,))

        det_img_ids = torch.tensor(det_img_ids, dtype=torch.long)
        gt_img_ids = torch.tensor(gt_img_ids, dtype=torch.long)
        gt_detected = [False] * len(gt_img_ids)

        # score 기준으로 detection 정렬
        order = torch.argsort(det_scores_all, descending=True)
        det_boxes_all = det_boxes_all[order]
        det_scores_all = det_scores_all[order]
        det_img_ids = det_img_ids[order]

        # TP/FP 판정
        tp = torch.zeros(det_boxes_all.size(0))
        fp = torch.zeros(det_boxes_all.size(0))

        for d_idx in range(det_boxes_all.size(0)):
            img_id = det_img_ids[d_idx].item()
            box_d = det_boxes_all[d_idx].unsqueeze(0)  # (1, 4)

            # 해당 이미지의 GT 중 cls_id인 것만 추출
            gt_indices = (gt_img_ids == img_id).nonzero().squeeze(-1)
            if gt_indices.numel() == 0:
                fp[d_idx] = 1.0
                continue

            gt_boxes_img_c = gt_boxes_all[gt_indices]
            ious = box_iou(box_d, gt_boxes_img_c).squeeze(0)  # (num_gt_in_img,)

            max_iou, max_idx = ious.max(dim=0)
            if max_iou >= iou_thr:
                if not gt_detected[gt_indices[max_idx].item()]:
                    tp[d_idx] = 1.0
                    gt_detected[gt_indices[max_idx].item()] = True
                else:
                    fp[d_idx] = 1.0
            else:
                fp[d_idx] = 1.0

        # 누적 TP/FP → PR curve → AP
        tp_cum = torch.cumsum(tp, dim=0)
        fp_cum = torch.cumsum(fp, dim=0)

        recall = tp_cum / float(num_gt)
        precision = tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-10)


        # COCO/VOC 2010+ 방식 (integral of PR curve)
        mrec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
        mpre = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])

        # precision envelope
        for i in range(mpre.numel() - 2, -1, -1):
            mpre[i] = torch.maximum(mpre[i], mpre[i + 1])

        idx = torch.nonzero(mrec[1:] != mrec[:-1]).squeeze(-1)
        ap = 0.0
        for i in idx:
            ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
        return float(ap)

    @torch.no_grad()
    def compute(self) -> Dict[str, float]:
        """
        전체 데이터셋에 대한 mAP 계산.

        Returns
        -------
        result: dict
            {
                "mAP":   COCO-style mean AP over IoU thresholds & classes,
                "mAP_50": AP at IoU=0.50,
                "mAP_75": AP at IoU=0.75 (있으면),
                "ap_per_class": 평균 AP (IoU thresholds 평균) per class
            }
        """
        iou_thrs = self.iou_thresholds

        ap_per_class_per_thr = torch.zeros((len(iou_thrs), self.num_classes), dtype=torch.float32)

        for t_idx, thr in enumerate(iou_thrs):
            for c in range(self.num_classes):
                ap_c_t = self._compute_ap_for_class(c, float(thr))
                ap_per_class_per_thr[t_idx, c] = ap_c_t

        # IoU threshold와 class 둘 다 평균 → mAP
        mAP = ap_per_class_per_thr.mean().item()

        # mAP@0.5, mAP@0.75도 추출
        mAP_50 = float('nan')
        mAP_75 = float('nan')
        if (iou_thrs == 0.5).any():
            t50_idx = (iou_thrs == 0.5).nonzero().item()
            mAP_50 = ap_per_class_per_thr[t50_idx].mean().item()
        if (iou_thrs == 0.75).any():
            t75_idx = (iou_thrs == 0.75).nonzero().item()
            mAP_75 = ap_per_class_per_thr[t75_idx].mean().item()

        # class별 AP (IoU threshold 평균)
        ap_per_class = ap_per_class_per_thr.mean(dim=0)  # (num_classes,)

        return {
            "mAP": mAP,
            "mAP_50": mAP_50,
            "mAP_75": mAP_75,
            "ap_per_class": ap_per_class,
        }
