import torch 
from metrics import * 

class Metric:
    def __init__(self):
        self.p = []  # (nc, )
        self.r = []  # (nc, )
        self.f1 = []  # (nc, )
        self.all_ap = []  # (nc, 10)
        self.ap_class_index = []  # (nc, )
        self.nc = 0

    def ap(self):
        return 
    
    def ap50(self):
        return 
    
    def ap75(self):
        return 
    
    def ap95(self):
        return 

    def mp(self):
        return 
    
    def mr(self):
        return 
    
    def map50(self):
        return 
    
    def map75(self):
        return 
    
    def map95(self):
        return 
    
    def mean_results(self):
        return 
    
    def class_results(self):
        return 

    def update(self, results:tuple):
        """Update the evaluation metrics with a new set of results.

        Args:
            results (tuple): A tuple containing evaluation metrics:
                - p (list): Precision for each class.
                - r (list): Recall for each class.
                - f1 (list): F1 score for each class.
                - all_ap (list): AP scores for all classes and all IoU thresholds.
                - ap_class_index (list): Index of class for each AP score.
                - p_curve (list): Precision curve for each class.
                - r_curve (list): Recall curve for each class.
                - f1_curve (list): F1 curve for each class.
                - px (list): X values for the curves.
                - prec_values (list): Precision values for each class.
        """
        (
            self.p,
            self.r,
            self.f1,
            self.all_ap,
            self.ap_class_index,
        ) = results

class ClassifyMetric(CMet):
    def __init__(self, cfg):
        ncls = cfg.data.ncls
        topk = cfg.data.topk
        super(ClassifyMetric, self).__init__(ncls, topk)


class DetectionMetric(DMet):
    def __init__(self, cfg):
        ncls = cfg.data.ncls
        iou_thrs = cfg.model.iou_threshold
        super(DetectionMetric, self).__init__(ncls, iou_thrs)


class PoseMetric(PMet):
    def __init__(self, cfg):
        ncls = cfg.data.ncls
        sigmas = OKS_SIGMAS if cfg.data.name == 'coco' else torch.ones(ncls)/ncls
        super(PoseMetric, self).__init__(ncls, sigmas)
