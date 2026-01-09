import torch.nn as nn 
from models.nn.modules import FeatureAdaptor, SPPF, PAN, PoseHead

BACKBONE = {
    'convnext',
    'vit'
}

class DINOv3Pose(nn.Module):
    def __init__(self, backbone='dinov3_convnext_base', 
                 nkpts=(4, 3), 
                 ncls=10, 
                 backbone_ckps=None,
                 finetuning=True,
                 device='cuda'):
        super().__init__()
        tar_info = backbone.split('_')
        model_name = tar_info[1]
        model_size = tar_info[2]
        self.tc = [192, 384, 768]
        
        if 'convnext' in model_name:
            from models.backbones.dinov3convnext import Dinov3ConvNext, convnext_sizes, convnext_ckps
            self.backbone = Dinov3ConvNext(
                depths=convnext_sizes[model_size]["depths"],
                dims=convnext_sizes[model_size]["dims"],    
                weights=backbone_ckps if backbone_ckps else convnext_ckps[model_size],
                )
            channels = convnext_sizes[model_size]["dims"][1:]
            self.adaptor = FeatureAdaptor(channels, self.tc)
            
        elif 'vit' in model_name:
            from models.backbones.dinov3vit import Dinov3ViT, vit_sizes, vit_ckps
            self.backbone = Dinov3ViT(
                patch_size=vit_sizes[model_size]["patch_size"],
                embed_dim=vit_sizes[model_size]["embed_dim"],
                depth=vit_sizes[model_size]["depth"],
                num_heads=vit_sizes[model_size]["num_heads"],
                ffn_ratio=vit_sizes[model_size]["ffn_ratio"],
                weights=backbone_ckps if backbone_ckps else vit_ckps[model_size],
                )
            channels = convnext_sizes[model_size]["embed_dim"]
            self.adaptor = FeatureAdaptor(channels, self.tc)
        else:
            raise TypeError(f'not supported backbone type : {model_name}')
        
        if finetuning:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.sppf = SPPF(self.tc[-1], self.tc[-1])
        self.pan = PAN(self.tc)
        self.head = PoseHead(ncls, nkpts, self.tc)
        self.to(device)

    def forward(self, x):
        feat = self.forward_features(x)
        return self.head(feat)
        
    def forward_features(self, x):
        feature_list = self.backbone.forward_features_list([x], [None])[1:]
        feature_list = self.adaptor(feature_list)
        feature_list[-1] = self.sppf(feature_list[-1])
        features = self.pan(feature_list)
        return features
        


        