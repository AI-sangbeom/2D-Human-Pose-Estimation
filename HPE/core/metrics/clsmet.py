import torch
import torch.nn.functional as F
from typing import Optional, List


class ClassifyMet:
    def __init__(self, num_classes: int, default_topk: int = 5):
        self.num_classes = num_classes
        self.default_topk = default_topk

        # 누적 버퍼
        self._pred_list: List[torch.Tensor] = []
        self._target_list: List[torch.Tensor] = []

    # ---------- 기본 단일 배치용 metric 함수들 ----------

    @torch.no_grad()
    def accuracy(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: (N, C) logits or probabilities
        target: (N,)
        """
        pred_label = pred.argmax(dim=1)
        return (pred_label == target).float().mean()

    @torch.no_grad()
    def topk_accuracy(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        k: int = 5,
    ) -> torch.Tensor:
        """
        pred: (N, C)
        target: (N,)
        """
        topk = pred.topk(k, dim=1).indices          # (N, k)
        correct = topk.eq(target.unsqueeze(1))      # (N, k)
        return correct.any(dim=1).float().mean()

    @torch.no_grad()
    def precision_recall_f1(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ):
        """
        클래스별 precision / recall / f1 계산

        Returns
        -------
        precision: (num_classes,)
        recall:    (num_classes,)
        f1:        (num_classes,)
        """
        pred_label = pred.argmax(dim=1)

        precision = []
        recall = []
        f1 = []

        for c in range(self.num_classes):
            tp = ((pred_label == c) & (target == c)).sum().float()
            fp = ((pred_label == c) & (target != c)).sum().float()
            fn = ((pred_label != c) & (target == c)).sum().float()

            p = tp / (tp + fp + 1e-10)
            r = tp / (tp + fn + 1e-10)
            f = 2 * p * r / (p + r + 1e-10)

            precision.append(p)
            recall.append(r)
            f1.append(f)

        return torch.stack(precision), torch.stack(recall), torch.stack(f1)

    @torch.no_grad()
    def compute_once(self, pred: torch.Tensor, target: torch.Tensor, k: int = 5):
        """
        한 번에 pred/target 텐서를 넣어서 바로 metric 계산.
        """
        acc = self.accuracy(pred, target)
        topk_acc = self.topk_accuracy(pred, target, k)
        prec, reca, f1 = self.precision_recall_f1(pred, target)

        return {
            "acc": acc,
            "topk": topk_acc,
            "prec": prec,
            "reca": reca,
            "f1-score": f1,
        }

    # ---------- 배치 누적용 API (update / reset / compute_accumulated) ----------

    def reset(self):
        """
        누적된 prediction / target 버퍼 초기화.
        """
        self._pred_list = []
        self._target_list = []

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        한 배치에 대한 prediction / target을 누적.

        pred:   (B, C)
        target: (B,)
        """
        if not isinstance(pred, torch.Tensor):
            pred = torch.tensor(pred)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target)

        B = pred.shape[0]
        assert target.shape[0] == B, "pred와 target의 batch 크기가 다릅니다."

        # GPU 메모리 아끼려면 CPU에 올려서 저장
        self._pred_list.append(pred.detach().cpu())
        self._target_list.append(target.detach().cpu())

    @torch.no_grad()
    def compute(self, k: Optional[int] = None):
        """
        지금까지 update()로 누적된 전체 데이터에 대해 metric 계산.

        k: top-k accuracy에 사용할 k (None이면 default_topk 사용)
        """
        if len(self._pred_list) == 0:
            # 아무것도 누적되지 않은 경우
            return {
                "acc": torch.tensor(0.0),
                "topk": torch.tensor(0.0),
                "prec": torch.zeros(self.num_classes),
                "reca": torch.zeros(self.num_classes),
                "f1-score": torch.zeros(self.num_classes),
            }

        pred = torch.cat(self._pred_list, dim=0)     # (N_total, C)
        target = torch.cat(self._target_list, dim=0) # (N_total,)

        if k is None:
            k = self.default_topk

        return self.compute_once(pred, target, k=k)
