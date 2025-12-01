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


class DetectionMetric(Metric):
    pass 

class PoseMetric(Metric):
    pass 