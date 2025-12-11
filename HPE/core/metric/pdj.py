import torch
from typing import Optional, Tuple


class PDJ:
    """
    Percentage of Detected Joints (PDJ) metric (PyTorch).

    PDJ는 각 joint의 GT-예측 거리 d가
        d <= alpha * L
    이면 detected 라고 보고, 전체 joint 중 비율(%)을 계산.

    여기서는 L을 'torso size'로 잡는다.

    Parameters
    ----------
    alpha : float, default=0.5
        d <= alpha * torso_size 이면 detected (correct).
    torso_joint_indices : Optional[Tuple[int, int]]
        GT keypoints에서 torso size를 측정할 두 joint 인덱스 (예: (left_shoulder, right_hip)).
        torso_size 인자를 직접 넘길 거라면 None이어도 됨.
    eps : float, default=1e-6
        torso_size가 0에 가까울 때를 피하기 위한 epsilon.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        torso_joint_indices: Optional[Tuple[int, int]] = None,
        eps: float = 1e-6,
    ):
        self.alpha = alpha
        self.torso_joint_indices = torso_joint_indices
        self.eps = eps

    @torch.no_grad()
    def _compute_torso_size_from_joints(
        self,
        gt_keypoints: torch.Tensor,              # (N, K, 2)
        visible: Optional[torch.Tensor] = None,  # (N, K)
    ) -> torch.Tensor:
        """
        torso_joint_indices=(j1, j2) 에 해당하는 두 관절 사이의 거리를
        이미지별 torso size로 사용.
        """
        if self.torso_joint_indices is None:
            raise ValueError(
                "torso_joint_indices가 None입니다. "
                "torso_size를 직접 넘기거나 torso_joint_indices를 설정하세요."
            )

        j1, j2 = self.torso_joint_indices
        N, K, D = gt_keypoints.shape
        assert D == 2

        gt = gt_keypoints.float()

        # 두 torso joint의 좌표: (N, 2)
        p1 = gt[:, j1, :]
        p2 = gt[:, j2, :]

        # 기본 torso size: 두 점 사이의 L2 distance
        torso_size = torch.linalg.norm(p1 - p2, ord=2, dim=-1)  # (N,)

        if visible is not None:
            visible = visible.bool()
            # 두 torso joint 모두 visible인 경우만 신뢰
            torso_vis = visible[:, j1] & visible[:, j2]  # (N,)
            torso_size = torch.where(
                torso_vis,
                torso_size,
                torch.full_like(torso_size, self.eps),
            )

        torso_size = torch.clamp(torso_size, min=self.eps)
        return torso_size  # (N,)

    @torch.no_grad()
    def compute(
        self,
        gt_keypoints: torch.Tensor,                 # (N, K, 2)
        pred_keypoints: torch.Tensor,               # (N, K, 2)
        visible: Optional[torch.Tensor] = None,     # (N, K)
        torso_size: Optional[torch.Tensor] = None,  # (N,) optional
    ) -> float:
        """
        전체 데이터셋에 대한 PDJ(%) 계산.

        Parameters
        ----------
        gt_keypoints : torch.Tensor
            GT keypoints, shape (N, K, 2).
        pred_keypoints : torch.Tensor
            예측 keypoints, shape (N, K, 2).
        visible : Optional[torch.Tensor]
            joint 유효 여부, shape (N, K), bool 또는 0/1.
        torso_size : Optional[torch.Tensor]
            이미지별 torso size, shape (N,).
            None이면 torso_joint_indices로부터 자동 계산.

        Returns
        -------
        pdj : float
            전체 PDJ (%).
        """
        if not isinstance(gt_keypoints, torch.Tensor):
            gt_keypoints = torch.tensor(gt_keypoints)
        if not isinstance(pred_keypoints, torch.Tensor):
            pred_keypoints = torch.tensor(pred_keypoints)

        assert gt_keypoints.shape == pred_keypoints.shape, \
            "gt_keypoints와 pred_keypoints의 shape가 같아야 합니다."

        N, K, D = gt_keypoints.shape
        assert D == 2, "keypoints는 (x, y) 2차원 좌표여야 합니다."

        gt = gt_keypoints.float()
        pred = pred_keypoints.float()

        # visible mask
        if visible is None:
            visible = torch.ones((N, K), dtype=torch.bool, device=gt.device)
        else:
            if not isinstance(visible, torch.Tensor):
                visible = torch.tensor(visible, device=gt.device)
            visible = visible.bool()

        # torso size 설정
        if torso_size is None:
            torso_size = self._compute_torso_size_from_joints(gt, visible)  # (N,)
        else:
            if not isinstance(torso_size, torch.Tensor):
                torso_size = torch.tensor(torso_size, dtype=torch.float32, device=gt.device)
            torso_size = torso_size.float().to(gt.device)
            assert torso_size.shape == (N,), "torso_size는 (N,) 텐서여야 합니다."
            torso_size = torch.clamp(torso_size, min=self.eps)

        # joint별 거리 (N, K)
        diff = gt - pred
        dist = torch.linalg.norm(diff, ord=2, dim=-1)  # (N, K)

        # threshold = alpha * torso_size_i
        thr = self.alpha * torso_size.view(N, 1)  # (N, 1) -> (N, K)

        detected = (dist <= thr) & visible
        total_detected = detected.sum().item()
        total_joints = visible.sum().item()

        if total_joints == 0:
            return 0.0

        pdj = 100.0 * total_detected / total_joints
        return pdj


if __name__=='__main__':

    N, K = 8, 17
    gt = torch.randn(N, K, 2)
    pred = gt + torch.randn_like(gt) * 0.3  # 약간 noisy 한 예측

    visible = torch.ones(N, K, dtype=torch.bool)

    # torso_joint_indices = (left_shoulder=5, right_hip=12) 라고 가정
    metric_pdj = PDJ(alpha=0.5, torso_joint_indices=(5, 12))

    pdj_val = metric_pdj.compute(gt, pred, visible=visible)
    print(f"PDJ: {pdj_val:.2f}%")
