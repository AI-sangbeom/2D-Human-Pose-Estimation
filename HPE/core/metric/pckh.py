import torch
from typing import Optional, Tuple


class PCKh:
    """
    PCKh (Head-normalized Percentage of Correct Keypoints)

    PCK와 차이:
      - normalizer L_i 로 'head size'를 사용
      - head size는
        (1) head_joint_indices=(j1, j2) 로부터 자동 계산하거나
        (2) head_size (N,) 텐서로 직접 입력 가능

    Parameters
    ----------
    alpha : float, default=0.5
        d <= alpha * head_size 이면 correct.
    head_joint_indices : Optional[Tuple[int, int]]
        GT keypoints에서 머리 크기를 추정할 두 관절 인덱스 (예: (head_top, upper_neck)).
        head_size 인자를 직접 넘길 거라면 None으로 둬도 됨.
    eps : float, default=1e-6
        head_size가 0에 가까울 때를 피하기 위한 epsilon.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        head_joint_indices: Optional[Tuple[int, int]] = None,
        eps: float = 1e-6,
    ):
        self.alpha = alpha
        self.head_joint_indices = head_joint_indices
        self.eps = eps

    @torch.no_grad()
    def _compute_head_size_from_joints(
        self,
        gt_keypoints: torch.Tensor,              # (N, K, 2)
        visible: Optional[torch.Tensor] = None,  # (N, K)
    ) -> torch.Tensor:
        """
        head_joint_indices=(j1, j2) 에 해당하는 두 관절 사이의 거리를
        이미지별 head size로 사용.
        """
        if self.head_joint_indices is None:
            raise ValueError(
                "head_joint_indices가 None 입니다. "
                "head_size를 직접 넘기거나 head_joint_indices를 설정하세요."
            )

        j1, j2 = self.head_joint_indices
        N, K, D = gt_keypoints.shape
        assert D == 2

        gt = gt_keypoints.float()

        # head joint 좌표 (N, 2)
        p1 = gt[:, j1, :]
        p2 = gt[:, j2, :]

        # 기본 head size: 두 점 사이의 L2 distance
        head_size = torch.linalg.norm(p1 - p2, ord=2, dim=-1)  # (N,)

        if visible is not None:
            visible = visible.bool()
            # 두 head joint 모두 visible인 경우만 사용
            head_vis = visible[:, j1] & visible[:, j2]  # (N,)
            # visible=False 인 샘플은 eps로 대체
            head_size = torch.where(
                head_vis,
                head_size,
                torch.full_like(head_size, self.eps),
            )

        head_size = torch.clamp(head_size, min=self.eps)
        return head_size  # (N,)

    @torch.no_grad()
    def compute(
        self,
        gt_keypoints: torch.Tensor,                # (N, K, 2)
        pred_keypoints: torch.Tensor,              # (N, K, 2)
        visible: Optional[torch.Tensor] = None,    # (N, K)
        head_size: Optional[torch.Tensor] = None,  # (N,) optional
    ) -> float:
        """
        전체 데이터셋에 대한 PCKh(%) 계산.

        Parameters
        ----------
        gt_keypoints : torch.Tensor
            GT keypoints, shape (N, K, 2).
        pred_keypoints : torch.Tensor
            예측 keypoints, shape (N, K, 2).
        visible : Optional[torch.Tensor]
            joint 유효 여부, shape (N, K), bool 또는 0/1.
        head_size : Optional[torch.Tensor]
            이미지별 head size, shape (N,).
            None이면 head_joint_indices로부터 자동 계산.

        Returns
        -------
        pckh : float
            전체 PCKh (%).
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

        # head size 계산/정리
        if head_size is None:
            # head_joint_indices 기반 자동 추정
            head_size = self._compute_head_size_from_joints(gt, visible)  # (N,)
        else:
            if not isinstance(head_size, torch.Tensor):
                head_size = torch.tensor(head_size, dtype=torch.float32, device=gt.device)
            head_size = head_size.float().to(gt.device)
            assert head_size.shape == (N,), "head_size는 (N,) 텐서여야 합니다."
            head_size = torch.clamp(head_size, min=self.eps)

        # keypoint별 거리 (N, K)
        diff = gt - pred
        dist = torch.linalg.norm(diff, ord=2, dim=-1)  # (N, K)

        # pckhs = []

        # alphas = (0.1, 0.2, 0.5)
        # for alpha in alphas:
        #     thr = alpha * head_size.view(N, 1)
        #     correct = (dist <= thr) & visible
        #     total_correct = correct.sum().item()
        #     total_keypoints = visible.sum().item()
            
        #     if total_keypoints == 0:
        #         val = 0
        #     else:
        #         val = 100.0 * total_correct / total_keypoints
        #     pckhs.append(val)
        # return pckhs

        # threshold = alpha * head_size_i
        thr = self.alpha * head_size.view(N, 1)  # (N, 1) -> (N, K) broadcast

        correct = (dist <= thr) & visible
        total_correct = correct.sum().item()
        total_keypoints = visible.sum().item()

        if total_keypoints == 0:
            return 0.0

        pckh = 100.0 * total_correct / total_keypoints
        return pckh

if __name__=='__main__':

    # 예시 데이터
    N, K = 4, 16
    head_size = torch.rand(N) 
    gt = torch.randn(N, K, 2)
    pred = gt + torch.randn_like(gt) * 0.1  # 약간의 noise

    visible = torch.ones(N, K, dtype=torch.bool)

    # head_top=9, upper_neck=8 라고 가정
    metric_pckh = PCKh(alpha=0.2, head_joint_indices=(9, 8))

    pckh_val = metric_pckh.compute(gt, pred, visible=visible, head_size=head_size)
    print(f"PCKh (head joints 9-8 기준): {pckh_val:.2f}%")
