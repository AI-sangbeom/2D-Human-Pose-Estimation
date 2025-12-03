import torch 
from metrics import * 

class ClassifyMetric:
    def __init__(self, cfg):
        ncls = cfg.data.ncls
        topk = cfg.data.topk
        self.metric = CMet(ncls, topk)

class DetectionMetric:
    def __init__(self, cfg):
        ncls = cfg.data.ncls
        iou_thrs = cfg.model.iou_threshold
        super(DetectionMetric, self).__init__(ncls, iou_thrs)


class PoseMetric:
    def __init__(self, cfg):
        ncls = cfg.data.ncls
        sigmas = OKS_SIGMAS if cfg.data.name == 'coco' else torch.ones(ncls)/ncls
        super(PoseMetric, self).__init__(ncls, sigmas)
