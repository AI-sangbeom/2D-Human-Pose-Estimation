import torch
import torch.nn.functional as F

class ClassifyMet:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def accuracy(self, pred, target):
        pred_label = pred.argmax(dim=1)
        return (pred_label == target).float().mean()

    def topk_accuracy(self, pred, target, k=5):
        topk = pred.topk(k, dim=1).indices
        correct = topk.eq(target.unsqueeze(1))
        return correct.any(dim=1).float().mean()

    def precision_recall_f1(self, pred, target):
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
    
    def compute(self, pred, target, k):
        acc = self.accuracy(pred, target)
        topk_acc = self.topk_accuracy(pred, target, k)
        prec, reca, f1 = self.precision_recall_f1(pred, target)

        return {
            "acc" : acc,
            "topk" : topk_acc,
            "prec" : prec,
            "reca" : reca,
            "f1-score":f1
        }