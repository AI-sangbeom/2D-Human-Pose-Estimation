import torch
from typing import Optional


class PCK:
    """
    Percentage of Correct Keypoints (PCK) metric.
    """

    def __init__(self, alpha: float = 0.5, eps: float = 1e-6):
        self.alpha = alpha
        self.eps = eps

    @torch.no_grad()
    def _compute_normalizer_from_gt(
        self,
        gt_keypoints: torch.Tensor,              # (N, K, 2)
        visible: Optional[torch.Tensor] = None,  # (N, K)
    ) -> torch.Tensor:
        """
        각 이미지별 bounding box 크기를 normalizer로 사용.
        torch.nanmin 없이 동작하도록 마스킹 처리한다.
        """
        N, K, _ = gt_keypoints.shape
        gt = gt_keypoints.float()

        if visible is None:
            visible = torch.ones((N, K), dtype=torch.bool, device=gt.device)
        else:
            visible = visible.bool()

        mask = visible.unsqueeze(-1)  # (N, K, 1)

        # min 계산용: 보이지 않는 joint는 +inf
        x_min = torch.where(mask[..., 0], gt[..., 0], torch.full_like(gt[..., 0], float('inf'))).min(dim=1).values
        y_min = torch.where(mask[..., 0], gt[..., 1], torch.full_like(gt[..., 1], float('inf'))).min(dim=1).values

        # max 계산용: 보이지 않는 joint는 -inf
        x_max = torch.where(mask[..., 0], gt[..., 0], torch.full_like(gt[..., 0], float('-inf'))).max(dim=1).values
        y_max = torch.where(mask[..., 0], gt[..., 1], torch.full_like(gt[..., 1], float('-inf'))).max(dim=1).values

        width = x_max - x_min
        height = y_max - y_min

        L = torch.max(width, height)  # (N,)
        L = torch.clamp(L, min=self.eps)

        return L

    @torch.no_grad()
    def compute(
        self,
        gt_keypoints: torch.Tensor,                # (N, K, 2)
        pred_keypoints: torch.Tensor,              # (N, K, 2)
        visible: Optional[torch.Tensor] = None,    # (N, K)
        normalizer: Optional[torch.Tensor] = None, # None, scalar, (N,)
    ) -> float:

        if not isinstance(gt_keypoints, torch.Tensor):
            gt_keypoints = torch.tensor(gt_keypoints)
        if not isinstance(pred_keypoints, torch.Tensor):
            pred_keypoints = torch.tensor(pred_keypoints)

        N, K, _ = gt_keypoints.shape
        gt = gt_keypoints.float()
        pred = pred_keypoints.float()

        # default visible mask
        if visible is None:
            visible = torch.ones((N, K), dtype=torch.bool, device=gt.device)
        else:
            visible = visible.bool()

        # normalizer 계산
        if normalizer is None:
            L = self._compute_normalizer_from_gt(gt, visible)  # (N,)
        else:
            if not isinstance(normalizer, torch.Tensor):
                normalizer = torch.tensor(normalizer, dtype=torch.float32, device=gt.device)
            normalizer = normalizer.float()

            if normalizer.dim() == 0:
                L = normalizer.repeat(N)
            else:
                assert normalizer.shape == (N,)
                L = normalizer

            L = torch.clamp(L, min=self.eps)

        # 거리 계산
        diff = gt - pred
        dist = torch.linalg.norm(diff, ord=2, dim=-1)  # (N, K)

        # threshold
        thr = self.alpha * L.view(N, 1)  # (N, 1)

        correct = (dist <= thr) & visible
        total_correct = correct.sum().item()
        total_keypoints = visible.sum().item()

        if total_keypoints == 0:
            return 0.0

        return 100.0 * total_correct / total_keypoints


if __name__=='__main__':

    # 예시 데이터 (N=2 images, K=5 joints)
    gt = torch.tensor([
        [[0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.]],
        [[0., 0.], [1., 1.], [2., 1.], [3., 1.], [4., 1.]],
    ])

    pred = gt + 0.3 * torch.randn_like(gt)

    # 전체 joint가 다 보인다고 가정
    visible = torch.ones(gt.shape[:2], dtype=torch.bool)

    metric_pck = PCK(alpha=0.5)

    # 1) GT bbox 기준 자동 normalizer 사용
    pck_auto = metric_pck.compute(gt, pred, visible=visible)
    print(f"PCK (auto bbox norm): {pck_auto:.2f}%")

    # 2) 특정 scalar normalizer 사용 (예: head size 등을 대신했다고 가정)
    pck_scalar = metric_pck.compute(gt, pred, visible=visible, normalizer=0.5)
    print(f"PCK (scalar L=1.0): {pck_scalar:.2f}%")

    # 3) 이미지별 normalizer (예: head size per image)
    normalizer_per_image = torch.tensor([1.0, 1.5])  # (N,)
    pck_per_img = metric_pck.compute(gt, pred, visible=visible, normalizer=normalizer_per_image)
    print(f"PCK (per-image L): {pck_per_img:.2f}%")
