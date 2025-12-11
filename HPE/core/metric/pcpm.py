import torch
from typing import List, Tuple, Optional


class PCPm:
    """
    PCPm (Percentage of Correct Parts, modified) 계산용 PyTorch 버전 클래스.

    PCP와의 차이:
        - PCP: 각 part마다 GT 길이 L_i 를 threshold로 사용
        - PCPm: 전체 test set에 대해 계산한 평균 GT part 길이 L_mean 을
                고정 threshold로 사용 (d1, d2 <= alpha * L_mean)

    Parameters
    ----------
    skeleton : List[Tuple[int, int]]
        각 part를 구성하는 (joint_idx1, joint_idx2) 쌍의 리스트.
    alpha : float, default=0.5
        한 part를 Correct로 인정하기 위한 threshold 비율.
        d1 <= alpha * L_mean and d2 <= alpha * L_mean 이면 correct.
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
    def _compute_mean_part_length(
        self,
        gt_keypoints: torch.Tensor,              # (N, K, 2)
        visible: Optional[torch.Tensor] = None,  # (N, K)
    ) -> float:
        """
        전체 데이터에 대해 GT part 길이의 평균 L_mean을 계산.
        PCPm threshold = alpha * L_mean 에서 사용.
        """
        N, K, D = gt_keypoints.shape
        assert D == 2

        if visible is not None:
            assert visible.shape == (N, K)
            visible = visible.bool()

        sum_len = 0.0
        cnt_len = 0

        for i in range(N):
            gt = gt_keypoints[i].float()
            v_i = None if visible is None else visible[i]

            for (j1, j2) in self.skeleton:
                if v_i is not None:
                    if not (bool(v_i[j1]) and bool(v_i[j2])):
                        continue

                gt1, gt2 = gt[j1], gt[j2]
                L = torch.linalg.norm(gt1 - gt2, ord=2).item()

                if L < self.eps and self.ignore_short_parts:
                    continue

                sum_len += L
                cnt_len += 1

        if cnt_len == 0:
            return 0.0

        return float(sum_len / cnt_len)

    @torch.no_grad()
    def _compute_pcp_per_image(
        self,
        gt: torch.Tensor,             # (K, 2)
        pred: torch.Tensor,           # (K, 2)
        L_mean: float,
        visible: Optional[torch.Tensor] = None,  # (K,)
    ) -> Tuple[int, int]:
        """
        한 이미지에 대한 correct_part_count, total_part_count를 리턴.
        threshold는 L_mean을 사용.
        """
        gt = gt.float()
        pred = pred.float()

        correct = 0
        total = 0

        # mean 길이가 0이면 의미 있게 평가 불가
        if L_mean <= self.eps:
            return 0, 0

        thr = self.alpha * L_mean

        for (j1, j2) in self.skeleton:
            if visible is not None:
                if not (bool(visible[j1]) and bool(visible[j2])):
                    continue

            gt1, gt2 = gt[j1], gt[j2]
            p1, p2 = pred[j1], pred[j2]

            # endpoint 오차
            d1 = torch.linalg.norm(gt1 - p1, ord=2).item()
            d2 = torch.linalg.norm(gt2 - p2, ord=2).item()

            if d1 <= thr and d2 <= thr:
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
        전체 데이터셋에 대한 PCPm(%) 계산.

        Returns
        -------
        pcp_m : float
            전체 PCPm (0~100).
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

        # 1) 전체 데이터에 대해 평균 GT part 길이 L_mean 계산
        L_mean = self._compute_mean_part_length(gt_keypoints, visible)
        if L_mean <= self.eps:
            return 0.0

        # 2) L_mean을 threshold로 사용하여 PCPm 계산
        total_correct = 0
        total_parts = 0

        for i in range(N):
            v_i = None if visible is None else visible[i]
            c, t = self._compute_pcp_per_image(
                gt_keypoints[i],
                pred_keypoints[i],
                L_mean=L_mean,
                visible=v_i,
            )
            total_correct += c
            total_parts += t

        if total_parts == 0:
            return 0.0

        pcp_m = 100.0 * float(total_correct) / float(total_parts)
        return pcp_m

if __name__=='__main__':

    # 예시 데이터 (N=2 images, K=4 joints)
    gt = torch.tensor([
        [[0., 0.], [1., 0.], [2., 0.], [3., 0.]],  # image 0
        [[0., 0.], [1., 1.], [2., 1.], [3., 1.]],  # image 1
    ])

    pred = gt + 0.3 * torch.randn_like(gt)

    skeleton = [(0, 1), (1, 2), (2, 3)]

    metric_pcpm = PCPm(skeleton=skeleton, alpha=0.5)

    pcpm = metric_pcpm.compute(gt, pred)

    print(f"PCPm: {pcpm:.2f}%")
