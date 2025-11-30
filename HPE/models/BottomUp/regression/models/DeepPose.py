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
        return self.backbone(x).reshape(-1, 2, self.njoints)
    

if __name__=='__main__':
    import torch 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DeepPose(8, 'resnet50', pretrained=True)
    model = model.to(device)

    random_noise = torch.randn((32, 3, 224, 224)).to(device)
    result = model(random_noise)

    print(f'INPUT SIZE : {random_noise.shape}')
    print(f'OUTPUT SIZE : {result.shape}')