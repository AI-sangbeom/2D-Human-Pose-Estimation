import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        """
        Standard Cross Entropy Loss for multi-class classification.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        """
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass to compute the Cross Entropy Loss.
        :param inputs: Predictions (logits) from the model. Shape: (batch_size, num_classes)
        :param targets: Ground truth labels. Shape: (batch_size,)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        if self.reduction == 'mean':
            return ce_loss.mean()
        elif self.reduction == 'sum':
            return ce_loss.sum()
        return ce_loss