import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.nn.modules import ConvBlock, DFL

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

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
        self.reg_max = 16 # DFL 사용 (Loss 계산 시 필수)
        
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
                nn.Conv2d(c2, 4 * self.reg_max, 1)
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

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)



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
            det_feat = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1) 
            kpt_feat = self.cv4[i](x[i])
            x[i] = torch.cat([det_feat, kpt_feat], 1)
        # det_feat.shape = 74 
        # kpt_feat.shape = 12
        if self.training:
            return x 
        
        return self._inference_pose(x)

    def _inference_pose(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Pose 전용 Inference 로직 (DFL 호환 수정)"""
        shape = x[0].shape  # BCHW
        box_ch = 4 * self.reg_max
        total_ch = box_ch + self.nc + self.nk
        
        x_cat = torch.cat([xi.view(shape[0], total_ch, -1) for xi in x], 2)
        
        # Anchor 생성
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls, kpt = x_cat.split((box_ch, self.nc, self.nk), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        
        # Keypoint Decoding
        dkpt = self.kpts_decode(kpt)
        
        return torch.cat((dbox, cls.sigmoid(), dkpt), 1)
    
    def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor:
        """Decode keypoints from predictions."""
        ndim = self.kpt_shape[1]
        y = kpts.clone()
        if ndim == 3:
            y[:, 2::ndim] = y[:, 2::ndim].sigmoid() # Visibility는 Sigmoid
        
        # x, y 좌표 복원 (Grid 좌표 -> 실제 이미지 좌표)
        y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
        y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
        return y

    def xywh2xyxy(self, x):
        """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]"""
        y = x.clone()
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y

    def pose_postprocessor(
            self,
            prediction: torch.Tensor,
            conf_thres: float = 0.25,
            iou_thres: float = 0.5,
            classes: list = None,
            agnostic: bool = False,
            multi_label: bool = False,
            max_det: int = 300,
            nc: int = 10,  # number of classes
            nk: int = 12  # number of keypoints * 3 (ex: 17*3=51)
    ):
        """
        Post-process Pose Estimation model predictions.
        
        Args:
            prediction: (Batch, 4 + nc + nk, Anchors)
            conf_thres: Confidence Threshold
            iou_thres: IoU Threshold for NMS
            
        Returns:
            List of torch.Tensor: [(N, 6 + nk), (M, 6 + nk), ...] 
            각 Tensor는 [x1, y1, x2, y2, conf, cls_idx, kpt1_x, kpt1_y, kpt1_conf, ...]
        """
        
        # 1. Shape Check & Transpose
        # 모델 출력 (B, C, A) -> NMS 처리를 위해 (B, A, C)로 변환
        if prediction.shape[1] == 4 + nc + nk:
            prediction = prediction.transpose(1, 2)
            
        bs = prediction.shape[0]  # batch size
        # 채널 수 자동 계산 (만약 nc, nk 인자가 틀렸을 경우를 대비)
        xc = prediction[..., 4:4+nc].amax(1) > conf_thres

        output = [torch.zeros((0, 6 + nk), device=prediction.device)] * bs
        
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            x = x[xc[xi]]  # confidence filtering

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Split Box, Cls, Kpt
            box = x[:, :4]
            cls = x[:, 4:4+nc]
            kpt = x[:, 4+nc:]

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(box)

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                # 다중 레이블인 경우 (한 박스가 여러 클래스일 수 있음)
                i, j = (cls > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), kpt[i]), 1)
            else:  
                # 가장 높은 확률의 클래스 하나만 선택
                conf, j = cls.max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), kpt), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:
                continue
            elif n > max_det:  # sort by confidence
                x = x[x[:, 4].argsort(descending=True)[:max_det]]

            # Batched NMS
            # 클래스별로 독립적인 NMS를 수행하기 위해 오프셋을 더함
            c = x[:, 5:6] * (0 if agnostic else 7680)  
            boxes, scores = x[:, :4] + c, x[:, 4]  
            
            # NMS 수행
            i = torchvision.ops.nms(boxes, scores, iou_thres)
            
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            
            output[xi] = x[i]

        return output
    

# bimg = np.zeros((224, 224, 3))
# cv2.imshow('DINOv3 Pose Result', bimg)