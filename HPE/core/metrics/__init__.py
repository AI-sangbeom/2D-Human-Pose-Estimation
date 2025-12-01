import torch 
from .pcp import PCP
from .pcpm import PCPm
from .pck import PCK
from .pckh import PCKh
from .pdj import PDJ
from .mpjpe import MPJPE
from .oksap import OKSAP as AP
from .detap import DetectionMAP as mAP
from .clsmet import ClassifyMet as cMet

__all__ = (
    "PCKh",
    "AP",
    "mAP",
    "cMet"
)

OKS_SIGMAS = torch.tensor([
    0.26, 0.25, 0.25, 0.35, 0.35,
    0.79, 0.79, 0.72, 0.72, 0.62,
    0.62, 1.07, 1.07, 0.87, 0.87,
    0.89, 0.89
], dtype=torch.float32)

def mask_iou(mask1: torch.Tensor, mask2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Calculate masks IoU.

    Args:
        mask1 (torch.Tensor): A tensor of shape (N, n) where N is the number of ground truth objects and n is the
            product of image width and height.
        mask2 (torch.Tensor): A tensor of shape (M, n) where M is the number of predicted objects and n is the product
            of image width and height.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing masks IoU.
    """
    intersection = torch.matmul(mask1, mask2.T).clamp_(0)
    union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]) - intersection  # (area1 + area2) - intersection
    return intersection / (union + eps)

