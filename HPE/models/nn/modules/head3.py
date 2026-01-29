import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.nn.modules import ConvBlock, DFL
from models.utils import make_anchors

class ClassifyHead(nn.Module):
    def __init__(self, ncls, in_ch: int, c_:int=1280, k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        super().__init__()
        self.conv = ConvBlock(in_ch, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, ncls)  # to x(b,c2)

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor | tuple:
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)  # get final output
        return (y, x)


class DetectHead(nn.Module):

    dynamic = False 
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)
    shape = None

    def __init__(self, ncls=80, in_ch = ()):
        super().__init__()
        self.nc = ncls
        self.reg_max = 4 # DFL 사용 (Loss 계산 시 필수)
        
        # 채널 수 계산
        c2, c3 = max((16, in_ch[0] // 4, self.reg_max * 4)), max(in_ch[0], min(self.nc, 100))
        self.no = self.nc + self.reg_max * 4
        self.nl = len(in_ch)
        
        # [수정] Stride를 Buffer로 등록하여 Device 자동 이동 지원
        self.register_buffer("stride", torch.tensor([8, 16, 32], dtype=torch.float32))

        # Box Branch (CV2)
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                ConvBlock(x, c2, 3), 
                ConvBlock(c2, c2, 3), 
                nn.Conv2d(c2, 4 * self.reg_max + 1, 1)
            ) for x in in_ch
        )

        
        # Cls Branch (CV3)
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(
                    ConvBlock(c1=x, c2=x, k=3, g=math.gcd(x, x)),
                    ConvBlock(x, c3, 1)
                ),
                nn.Sequential(
                    ConvBlock(c3, c3, 3),
                    ConvBlock(c3, c3, 1),
                ),
                nn.Conv2d(c3, ncls, 1)
            ) for x in in_ch

        )
        
        # DFL 모듈
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        self._init_biases()

    def _init_biases(self):
        """Initialize biases to predict background with high confidence."""
        # 배경일 확률을 높게(0.99), 객체일 확률을 낮게(0.01) 설정
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        
        # 1. Classification Branch (cv3) 초기화 (기존 코드)
        for cv3_block in self.cv3:
            final_conv = cv3_block[-1] 
            if hasattr(final_conv, 'bias') and final_conv.bias is not None:
                nn.init.constant_(final_conv.bias, bias_value)


    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Concatenate and return predicted bounding boxes and class probabilities."""

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x

        y = self._inference(x)
        return (y, x)

    def _inference(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps.

        Args:
            x (list[torch.Tensor]): List of feature maps from different detection layers.

        Returns:
            (torch.Tensor): Concatenated tensor of decoded bounding boxes and class probabilities.
        """
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape        
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        return torch.cat((dbox, cls.sigmoid()), 1)

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor:
        """Decode bounding boxes from predictions."""
        return self.dist2bbox(
            bboxes,
            anchors,
            xywh=xywh,
            dim=1,
        )
    
    def dist2bbox(self, distance, anchor_points, xywh=True, dim=-1):
        """Transform distance(ltrb) to box(xywh or xyxy)."""
        lt, rb = distance.chunk(2, dim)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        if xywh:
            c_xy = (x1y1 + x2y2) / 2
            wh = x2y2 - x1y1
            return torch.cat([c_xy, wh], dim)  # xywh bbox
        return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

class PoseHead(DetectHead):
    def __init__(self, ncls=80, kpt_shape: tuple = (17, 3), in_ch: tuple=()):
        super().__init__(ncls, in_ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total

        c4 = max(in_ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(
            nn.Sequential(
                ConvBlock(x, c4, 3), 
                ConvBlock(c4, c4, 3), 
                nn.Conv2d(c4, self.nk, 1)
            ) for x in in_ch
        )

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Perform forward pass."""
        
        for i in range(self.nl):
            # det + obj + cls 
            det_feat = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1) 
            kpt_feat = self.cv4[i](x[i])
            x[i] = torch.cat([det_feat, kpt_feat], 1)
        if self.training:
            return x 
        
        return self._inference_pose(x)

    def _inference_pose(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Pose 전용 Inference 로직 (DFL 호환 수정)"""
        shape = x[0].shape  # BCHW
        box_ch = 4 * self.reg_max
        total_ch = box_ch + self.nc + 1 + self.nk
        
        x_cat = torch.cat([xi.view(shape[0], total_ch, -1) for xi in x], 2)
        
        # Anchor 생성
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, obj, cls, kpt = x_cat.split((box_ch, 1, self.nc, self.nk), 1)
        dbox = (self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0))) * self.strides
        
        # Keypoint Decoding
        dkpt = self.kpts_decode(kpt)
        
        return torch.cat((dbox, obj.sigmoid(), cls.sigmoid(), dkpt), 1)
    
    def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor:
        """Decode keypoints from predictions."""
        ndim = self.kpt_shape[1]
        y = kpts.clone()  - 0.5
        
        # x, y 좌표 복원 (Grid 좌표 -> 실제 이미지 좌표)
        y[:, 0::ndim] = (y[:, 0::ndim] + self.anchors[0]) * self.strides
        y[:, 1::ndim] = (y[:, 1::ndim] + self.anchors[1]) * self.strides
        if ndim == 3:
            y[:, 2::ndim] = y[:, 2::ndim].sigmoid() # Visibility는 Sigmoid
        return y