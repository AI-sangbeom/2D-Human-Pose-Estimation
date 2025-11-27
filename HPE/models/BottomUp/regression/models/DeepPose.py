import timm
import torch.nn as nn

class DeepPose(nn.Module):
    def __init__(self, num_joints=8, backbone='resnet50', pretrained=True):
        super(DeepPose, self).__init__()
        # Load a pre-trained ResNet model from timm
        self.njoints = num_joints
        self.backbone = timm.create_model(backbone, pretrained=pretrained)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, self.njoints*2)
        
    def forward(self, x):
        return self.backbone(x)