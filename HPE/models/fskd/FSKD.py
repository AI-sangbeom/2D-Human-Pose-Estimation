import torch
import torch.nn as nn
from typing import List, Dict, Optional

from models.backbones.dinov3vit import Dinov3ViT, vit_sizes

class FSKD(nn.Module):
    def __init__(
            self,
            nkpts: int,
            backbone: str = 'base',
            pretrained: bool = True,
        ):
        super(FSKD, self).__init__()
        backbone_cfg = vit_sizes[backbone]
        embed_dim = backbone_cfg['embed_dim']

        self.backbone = Dinov3ViT        (
            patch_size=backbone_cfg["patch_size"],
            embed_dim=embed_dim,
            depth=backbone_cfg["depth"],
            num_heads=backbone_cfg["num_heads"],
            ffn_ratio=backbone_cfg["ffn_ratio"],
            pretrained=pretrained,
        )

        self.neck = nn.Identity()
        self.head = nn.Linear(embed_dim, nkpts * 2)

    def forward_features(
            self,
            x: torch.Tensor,
            masks: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        _, all_xes = self.backbone.forward_features_list([x], [masks])
        
        last_block_features = all_xes[-1][0]
        cls_token = last_block_features[:, 0]

        pose_feature = self.neck(cls_token)
        result = self.head(pose_feature)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.forward_features(x)
        return result
