import torch


class MPJPE:
    """
    Mean Per Joint Position Error (MPJPE)
    GT, Pred shape: (N, K, 3)

    기본 MPJPE (Protocol #1)
    """

    def __init__(self, reduction: str = "mean"):
        """
        reduction: 'mean' | 'sum' | 'none'
        """
        self.reduction = reduction

    @torch.no_grad()
    def compute(
        self,
        gt: torch.Tensor,       # (N, K, 3)
        pred: torch.Tensor,     # (N, K, 3)
        visible: torch.Tensor = None  # optional (N, K) bool
    ) -> torch.Tensor:
        """
        Returns:
            mpjpe (float or tensor)
        """
        if not isinstance(gt, torch.Tensor):
            gt = torch.tensor(gt, dtype=torch.float32)
        if not isinstance(pred, torch.Tensor):
            pred = torch.tensor(pred, dtype=torch.float32)

        gt = gt.float()
        pred = pred.float()

        assert gt.shape == pred.shape
        N, K, D = gt.shape
        assert D == 3, "MPJPE는 (N, K, 3) 좌표가 필요합니다."

        if visible is None:
            visible = torch.ones((N, K), dtype=torch.bool, device=gt.device)
        else:
            visible = visible.bool().to(gt.device)

        # joint 단위 거리
        diff = gt - pred                             # (N, K, 3)
        dist = torch.linalg.norm(diff, dim=-1)       # (N, K)

        # visible joint만 사용
        dist = dist * visible.float()

        if self.reduction == "mean":
            denom = visible.sum().clamp(min=1)
            return dist.sum() / denom

        elif self.reduction == "sum":
            return dist.sum()

        elif self.reduction == "none":
            return dist  # (N, K)

        else:
            raise ValueError("reduction must be one of ['mean','sum','none']")
