import torch
from typing import List, Tuple, Optional

class PCP:
    """
    Percentage of Correct Parts (PCP) 

    Parameters
    ----------
    skeleton : List[Tuple[int, int]]
        각 part를 구성하는 (joint_idx1, joint_idx2) 쌍의 리스트.
    alpha : float, default=0.5
        한 part를 Correct로 인정하기 위한 threshold 비율.
        d1 <= alpha * L and d2 <= alpha * L 이면 correct.
    ignore_short_parts : bool, default=True
        GT part 길이가 0에 가깝거나 매우 짧을 때 (수치 불안정),
        해당 part를 평가에서 제외할지 여부.
    eps : float, default=1e-6
        길이가 0인 경우를 피하기 위한 epsilon.
    """

    def __init__(
        self,
        skeleton: List[Tuple[int, int]],
        alpha: float = 0.5,
        ignore_short_parts: bool = True,
        eps: float = 1e-6,
    ):
        self.skeleton = skeleton
        self.alpha = alpha
        self.ignore_short_parts = ignore_short_parts
        self.eps = eps

    @torch.no_grad()
    def _compute_pcp_per_image(
        self,
        gt: torch.Tensor,            # (K, 2)
        pred: torch.Tensor,          # (K, 2)
        visible: Optional[torch.Tensor] = None,  # (K,)
    ) -> Tuple[int, int]:
        """
        한 이미지에 대한 correct_part_count, total_part_count를 리턴.
        """
        # safety: float으로 통일
        gt = gt.float()
        pred = pred.float()

        correct = 0
        total = 0

        for (j1, j2) in self.skeleton:
            # 가시성 체크
            if visible is not None:
                # visible은 bool 또는 0/1 tensor를 가정
                if not (bool(visible[j1]) and bool(visible[j2])):
                    continue

            gt1, gt2 = gt[j1], gt[j2]
            p1, p2 = pred[j1], pred[j2]

            # GT part 길이 L
            L = torch.linalg.norm(gt1 - gt2, ord=2).item()

            # 너무 짧은 part는 무시
            if L < self.eps and self.ignore_short_parts:
                continue

            # endpoint 오차
            d1 = torch.linalg.norm(gt1 - p1, ord=2).item()
            d2 = torch.linalg.norm(gt2 - p2, ord=2).item()

            if d1 <= self.alpha * L and d2 <= self.alpha * L:
                correct += 1
            total += 1

        return correct, total

    @torch.no_grad()
    def compute(
        self,
        gt_keypoints: torch.Tensor,              # (N, K, 2)
        pred_keypoints: torch.Tensor,            # (N, K, 2)
        visible: Optional[torch.Tensor] = None,  # (N, K) optional
    ) -> float:
        """
        전체 데이터셋에 대한 PCP(%) 계산.

        Parameters
        ----------
        gt_keypoints : torch.Tensor
            shape (N, K, 2) 의 GT keypoints.
        pred_keypoints : torch.Tensor
            shape (N, K, 2) 의 예측 keypoints.
        visible : Optional[torch.Tensor]
            shape (N, K) 의 bool/0-1 mask. joint 가 유효할 때 True/1.

        Returns
        -------
        pcp : float
            전체 Percentage of Correct Parts (0~100).
        """
        if not isinstance(gt_keypoints, torch.Tensor):
            gt_keypoints = torch.tensor(gt_keypoints)
        if not isinstance(pred_keypoints, torch.Tensor):
            pred_keypoints = torch.tensor(pred_keypoints)

        assert gt_keypoints.shape == pred_keypoints.shape, \
            "gt_keypoints와 pred_keypoints의 shape가 같아야 합니다."

        N, K, D = gt_keypoints.shape
        assert D == 2, "keypoints는 (x, y) 2차원 좌표여야 합니다."

        if visible is not None and not isinstance(visible, torch.Tensor):
            visible = torch.tensor(visible)

        if visible is not None:
            assert visible.shape == (N, K), \
                "visible은 (N, K) shape의 bool/0-1 배열이어야 합니다."
            visible = visible.bool()

        total_correct = 0
        total_parts = 0

        for i in range(N):
            v_i = None if visible is None else visible[i]
            c, t = self._compute_pcp_per_image(
                gt_keypoints[i],
                pred_keypoints[i],
                v_i,
            )
            total_correct += c
            total_parts += t

        if total_parts == 0:
            return 0.0

        pcp = 100.0 * float(total_correct) / float(total_parts)
        return pcp
    
if __name__=='__main__':
    # 예시 데이터 (N=2 images, K=4 joints)
    gt = torch.tensor([
        [[0., 0.], [1., 0.], [2., 0.], [3., 0.]],  # image 0
        [[0., 0.], [1., 1.], [2., 1.], [3., 1.]],  # image 1
    ])

    # 임의의 noise를 더한 예측
    pred = gt + 0.3 * torch.randn_like(gt)

    # skeleton: 3개의 part (0-1), (1-2), (2-3)
    skeleton = [(0, 1), (1, 2), (2, 3)]

    metric = PCP(skeleton=skeleton, alpha=0.5)

    pcp_value = metric.compute(gt, pred)
    print(f"PCP: {pcp_value:.2f}%")
