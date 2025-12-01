import torch
from typing import Optional, Sequence

class OKS:
    """
    COCO Object Keypoint Similarity (OKS) 계산용 클래스.

    Parameters
    ----------
    sigmas : torch.Tensor, shape (K,)
        각 keypoint type별 sigma.
    in_vis_thres : float, default=0.0
        visibility threshold (v > in_vis_thres 인 joint만 사용).
    eps : float, default=1e-10
        수치 안정용 epsilon.
    """

    def __init__(
        self,
        sigmas: torch.Tensor,
        in_vis_thres: float = 0.0,
        eps: float = 1e-10,
    ):
        if not isinstance(sigmas, torch.Tensor):
            sigmas = torch.tensor(sigmas, dtype=torch.float32)
        self.sigmas = sigmas.float()
        self.in_vis_thres = in_vis_thres
        self.eps = eps

    @torch.no_grad()
    def compute(
        self,
        gt_keypoints: torch.Tensor,   # (N, K, 3) [x, y, v]
        pred_keypoints: torch.Tensor, # (N, K, 2)
        areas: torch.Tensor,          # (N,)
    ) -> torch.Tensor:
        """
        각 샘플(사람 인스턴스)에 대한 OKS를 계산.

        Returns
        -------
        oks : torch.Tensor, shape (N,)
        """
        if not isinstance(gt_keypoints, torch.Tensor):
            gt_keypoints = torch.tensor(gt_keypoints)
        if not isinstance(pred_keypoints, torch.Tensor):
            pred_keypoints = torch.tensor(pred_keypoints)
        if not isinstance(areas, torch.Tensor):
            areas = torch.tensor(areas)

        N, K, D = gt_keypoints.shape
        assert D == 3, "gt_keypoints는 (N, K, 3) [x, y, v] 형식이어야 합니다."
        assert pred_keypoints.shape == (N, K, 2)
        assert self.sigmas.shape[0] == K, "sigmas 길이는 K와 같아야 합니다."
        assert areas.shape[0] == N

        device = gt_keypoints.device
        pred_keypoints = pred_keypoints.to(device)
        areas = areas.to(device)
        sigmas = self.sigmas.to(device)

        gt_xy = gt_keypoints[..., :2].float()    # (N, K, 2)
        gt_v  = gt_keypoints[..., 2].float()     # (N, K)
        pred_xy = pred_keypoints.float()

        # visibility mask: v > in_vis_thres 인 joint만 사용
        vis_mask = (gt_v > self.in_vis_thres).float()  # (N, K)
        dist2 = ((pred_xy - gt_xy) ** 2).sum(dim=-1)      # (N, K)
        denom = 2.0 * areas.view(N, 1) * ((sigmas * 2.0) ** 2).view(1, K) + self.eps  # (N, K)

        return (torch.exp(-dist2 / denom) * vis_mask).sum(dim=1)/(vis_mask.sum(dim=1) + self.eps)


class OKSAP:
    """
    OKS 기반 COCO style Average Precision (AP) 계산용 클래스.

    - 멀티 클래스 지원 (num_classes)
    - 여러 OKS threshold (기본: 0.50~0.95 step 0.05)
    - 클래스별 AP + AP50, AP75, 전체 mAP 계산

    Parameters
    ----------
    sigmas : torch.Tensor
        OKS 계산용 sigma (K,)
    num_classes : int
        클래스 개수 (label은 0 ~ num_classes-1 가정)
    in_vis_thres : float, default=0.0
        OKS 계산 시 사용할 visibility threshold
    eps : float, default=1e-10
        수치 안정용 epsilon
    oks_thresholds : Sequence[float] or None
        AP를 계산할 OKS threshold 리스트.
        None이면 [0.50, 0.55, ..., 0.95] 사용.
    """

    def __init__(
        self,
        sigmas: torch.Tensor,
        num_classes: int,
        in_vis_thres: float = 0.0,
        eps: float = 1e-10,
        oks_thresholds: Optional[Sequence[float]] = None,
    ):
        self.oks = OKS(sigmas, in_vis_thres, eps)
        self.num_classes = num_classes

        if oks_thresholds is None:
            self.oks_thresholds = torch.arange(0.5, 0.96, 0.05, dtype=torch.float32)
        else:
            self.oks_thresholds = torch.tensor(oks_thresholds, dtype=torch.float32)

    @torch.no_grad()
    def _compute_ap_single_threshold(
        self,
        oks: torch.Tensor,      # (N_c,)
        scores: torch.Tensor,   # (N_c,)
        num_gt: int,
        thr: float,
    ) -> float:
        """
        하나의 OKS threshold에 대해 AP 계산.
        (한 클래스에 대한 detection list와 GT 개수라고 가정)
        """
        if num_gt == 0:
            return 0.0

        if not isinstance(oks, torch.Tensor):
            oks = torch.tensor(oks, dtype=torch.float32)
        if not isinstance(scores, torch.Tensor):
            scores = torch.tensor(scores, dtype=torch.float32)

        device = oks.device
        scores = scores.to(device)
        oks = oks.to(device)

        N = oks.shape[0]
        if N == 0:
            # GT는 있는데 예측이 하나도 없는 경우
            return 0.0

        # score 기준 내림차순 정렬
        order = torch.argsort(scores, descending=True)
        oks_sorted = oks[order]

        tp = (oks_sorted >= thr).float()
        fp = (oks_sorted <  thr).float()

        tp_cum = torch.cumsum(tp, dim=0)
        fp_cum = torch.cumsum(fp, dim=0)

        denom = torch.clamp(tp_cum + fp_cum, min=1e-10)
        recall = tp_cum / max(float(num_gt), 1e-10)
        precision = tp_cum / denom

        # precision envelope 만들고 area 계산
        mrec = torch.cat([
            torch.tensor([0.0], device=device),
            recall,
            torch.tensor([1.0], device=device),
        ])
        mpre = torch.cat([
            torch.tensor([0.0], device=device),
            precision,
            torch.tensor([0.0], device=device),
        ])

        # 뒤에서부터 누적 max
        for i in range(mpre.numel() - 2, -1, -1):
            mpre[i] = torch.maximum(mpre[i], mpre[i + 1])

        # recall이 증가하는 구간만 적분
        idx = torch.nonzero(mrec[1:] != mrec[:-1]).squeeze(-1)
        ap = 0.0
        for i in idx:
            ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]

        return float(ap)

    @torch.no_grad()
    def compute(
        self,
        gt_keypoints: torch.Tensor,   # (N, K, 3) [x, y, v]
        pred_keypoints: torch.Tensor, # (N, K, 2)
        areas: torch.Tensor,          # (N,)
        scores: torch.Tensor,         # (N,)
        labels: torch.Tensor,         # (N,) 0~C-1
    ):
        """
        여러 OKS threshold에 대해, 각 클래스별 AP 및 전체 mAP 계산.

        Returns
        -------
        result : dict
            {
                "mAP": float,                   # 모든 클래스 + 모든 thr 평균
                "mAP_50": float or nan,         # OKS=0.50에서 클래스 평균
                "mAP_75": float or nan,         # OKS=0.75에서 클래스 평균
                "ap_per_class": Tensor(C,),     # OKS thresholds 평균 per class
                "ap_per_class_per_thr": Tensor(T, C),  # (num_thr, num_classes)
                "ap_per_thr": Tensor(T,),       # 클래스 평균 per threshold
            }
        """
        # OKS 계산
        oks = self.oks.compute(gt_keypoints, pred_keypoints, areas)  # (N,)

        if not isinstance(oks, torch.Tensor):
            oks = torch.tensor(oks, dtype=torch.float32)
        if not isinstance(scores, torch.Tensor):
            scores = torch.tensor(scores, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)

        oks = oks.float()
        scores = scores.float()
        labels = labels.long()

        N = oks.shape[0]
        assert scores.shape[0] == N
        assert labels.shape[0] == N

        device = oks.device
        thrs = self.oks_thresholds.to(device)
        C = self.num_classes
        T = thrs.numel()

        # (T, C) : threshold x class
        ap_per_class_per_thr = torch.zeros((T, C), dtype=torch.float32, device=device)

        for t_idx, thr in enumerate(thrs):
            for c in range(C):
                cls_mask = (labels == c)
                oks_c = oks[cls_mask]
                scores_c = scores[cls_mask]

                # 간단한 가정: detection 개수 = GT 개수
                num_gt_c = int(oks_c.shape[0])
                if num_gt_c == 0:
                    ap_c_t = 0.0
                else:
                    ap_c_t = self._compute_ap_single_threshold(
                        oks_c, scores_c, num_gt_c, float(thr)
                    )
                ap_per_class_per_thr[t_idx, c] = ap_c_t

        # 클래스 + threshold 모두 평균 → 전체 mAP (COCO style)
        mAP = ap_per_class_per_thr.mean().item()

        # mAP@OKS=0.50, 0.75
        mAP_50, mAP_75 = float("nan"), float("nan")
        if (thrs == 0.5).any():
            idx50 = (thrs == 0.5).nonzero().item()
            mAP_50 = ap_per_class_per_thr[idx50].mean().item()
        if (thrs == 0.75).any():
            idx75 = (thrs == 0.75).nonzero().item()
            mAP_75 = ap_per_class_per_thr[idx75].mean().item()

        # 클래스별 AP (threshold 평균)
        ap_per_class = ap_per_class_per_thr.mean(dim=0)  # (C,)

        return {
            "mAP": mAP,
            "mAP_50": mAP_50,
            "mAP_75": mAP_75,
            "ap_per_class": ap_per_class.cpu(),                 # (C,)
        }



import torch

if __name__ == '__main__':
    # torch.manual_seed(42)

    C = 3          # 클래스 3개
    N, K = 10, 4   # 인스턴스 10, keypoint 4개
    sigmas = torch.ones(K) / K

    # -------------------------
    # (0~1 normalzied) GT keypoints
    # -------------------------
    gt = torch.rand(N, K, 3)
    gt[..., 2] = torch.randint(0, 3, (N, K))  # visibility 0/1/2

    # prediction = GT + small noise
    pred = gt[..., :2] + torch.randn(N, K, 2) * 0.12
    pred = pred.clamp(0, 1)

    # -------------------------
    # area (0~1 normalized bbox area)
    # -------------------------
    x_min = gt[..., 0].min(dim=1).values
    x_max = gt[..., 0].max(dim=1).values
    y_min = gt[..., 1].min(dim=1).values
    y_max = gt[..., 1].max(dim=1).values

    widths  = (x_max - x_min).clamp(min=1e-6)
    heights = (y_max - y_min).clamp(min=1e-6)
    areas = widths * heights

    # -------------------------
    # scores
    # -------------------------
    scores = torch.rand(N)

    # -------------------------
    # labels: 0,1,2 중 하나
    # -------------------------
    labels = torch.randint(0, C, (N,), dtype=torch.long)

    print("샘플별 클래스 라벨:", labels.tolist())

    # -------------------------
    # Compute OKS AP
    # -------------------------
    oksap = OKSAP(sigmas=sigmas, num_classes=C)
    result = oksap.compute(
        gt_keypoints=gt,
        pred_keypoints=pred,
        areas=areas,
        scores=scores,
        labels=labels,
    )

    print("mAP@50-95:", result["mAP"])
    print("mAP@50:", result["mAP_50"])
    print("mAP@75:", result["mAP_75"])
    print("AP per class:", result["ap_per_class"])          # 길이 3

