import torch
import torch.nn as nn
import torch.nn.functional as F

class TaskAlignedAssigner(nn.Module):
    """
    Task-Aligned Assigner for YOLOv8/v11
    
    This assigner combines both classification and localization information
    to select positive samples based on a task-aligned metric:
    
    t = s^alpha * u^beta
    
    where:
    - s: classification score
    - u: IoU between prediction and ground truth
    - alpha: weight for classification (default 1.0)
    - beta: weight for localization (default 6.0)
    """
    
    def __init__(self, topk=10, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt):
        """
        Args:
            pd_scores: (B, N, C) - Predicted classification scores
            pd_bboxes: (B, N, 4) - Predicted bboxes in xyxy format
            anchor_points: (N, 2) - Anchor points coordinates
            gt_labels: (B, M, 1) - Ground truth labels
            gt_bboxes: (B, M, 4) - Ground truth bboxes in xyxy format
            mask_gt: (B, M, 1) - Valid ground truth mask
            
        Returns:
            target_bboxes: (B, N, 4) - Assigned target bboxes
            target_scores: (B, N, C) - Assigned target scores
            fg_mask: (B, N) - Foreground mask
            target_gt_idx: (B, N) - Index of assigned GT for each anchor
        """
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)
        
        # Handle empty GT case
        if self.n_max_boxes == 0:
            device = pd_scores.device
            return (
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros(self.bs, pd_scores.shape[1], dtype=torch.bool, device=device),
                torch.zeros(self.bs, pd_scores.shape[1], dtype=torch.long, device=device)
            )

        # 1. Get positive candidate mask
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anchor_points, mask_gt
        )

        # 2. Select highest overlaps when one anchor matches multiple GTs
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes
        )
        
        # 3. Get target labels and bboxes
        # Flatten indices for gathering
        bs_idx = torch.arange(self.bs, device=pd_scores.device)[:, None]
        target_gt_idx_offset = target_gt_idx + bs_idx * self.n_max_boxes
        
        target_labels = gt_labels.view(-1)[target_gt_idx_offset]  # (B, N, 1)
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx_offset]  # (B, N, 4)
        target_labels = target_labels.squeeze(-1).clamp(0, self.num_classes - 1)  # (B, N)
        
        # 4. Normalize alignment metric
        # For each GT, find the max alignment metric across all anchors
        # Then normalize by this max value
        max_metric_per_gt = align_metric.amax(dim=1, keepdim=True).clamp(min=self.eps)  # (B, 1, M)
        norm_align_metric = (align_metric / max_metric_per_gt).amax(dim=2)  # (B, N)
        norm_align_metric = norm_align_metric * fg_mask  # Only for foreground

        # 5. Assign target scores (one-hot encoded with normalized alignment metric)
        target_scores = torch.zeros(
            (self.bs, pd_scores.shape[1], self.num_classes),
            dtype=pd_scores.dtype,
            device=pd_scores.device
        )
        
        if fg_mask.any():
            # Create one-hot encoding for positive samples
            fg_scores_mask = fg_mask.unsqueeze(-1).expand(-1, -1, self.num_classes)
            
            # Get the scores for assigned class
            assigned_scores = torch.zeros_like(target_scores).long()
            assigned_scores.scatter_(
                2, 
                target_labels.unsqueeze(-1).long(), 
                norm_align_metric.unsqueeze(-1).long()
            )
            
            # Only keep scores for foreground anchors
            target_scores = torch.where(fg_scores_mask, assigned_scores, target_scores)
        
        return target_bboxes, target_scores, fg_mask, target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anchor_points, mask_gt):
        """
        Get positive anchor mask based on:
        1. Anchor center inside GT box
        2. Top-k alignment metric
        """
        # 1. Check if anchor centers are inside GT boxes
        mask_in_gts = self.select_candidates_in_gts(anchor_points, gt_bboxes)  # (B, N, M)
        
        # 2. Calculate alignment metric and overlaps
        align_metric, overlaps = self.get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt
        )
        
        # 3. Select top-k candidates based on alignment metric
        mask_topk = self.select_topk_candidates(
            align_metric, mask_gt=mask_gt
        )  # (B, N, M)
        
        # 4. Combine masks: must be inside GT AND in top-k AND valid GT
        mask_gt_expanded = mask_gt.squeeze(-1).unsqueeze(1)  # (B, 1, M)
        mask_pos = mask_in_gts * mask_topk * mask_gt_expanded
        
        return mask_pos, align_metric, overlaps

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, eps=1e-9):
        """
        Check if anchor centers are inside GT boxes.
        
        Args:
            xy_centers: (N, 2) - Anchor center coordinates
            gt_bboxes: (B, M, 4) - GT boxes in xyxy format
            
        Returns:
            (B, N, M) - Boolean mask indicating if anchor is inside GT
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        
        # Split into x1y1 (left-top) and x2y2 (right-bottom)
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # (B*M, 1, 2), (B*M, 1, 2)
        
        # Calculate distances: left-top and right-bottom
        bbox_deltas = torch.cat(
            (xy_centers[None] - lt, rb - xy_centers[None]), 
            dim=2
        ).view(bs, n_boxes, n_anchors, -1)  # (B, M, N, 4)
        
        # If all deltas are positive, anchor is inside GT
        # Return (B, N, M) by transposing
        return bbox_deltas.amin(3).gt_(eps).transpose(1, 2)  # (B, N, M)

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """
        Calculate alignment metric and IoU overlaps.
        
        Alignment metric = score^alpha * iou^beta
        """
        bs, n_anchors = pd_scores.shape[:2]
        n_gt = gt_bboxes.shape[1]
        
        if n_gt == 0:
            device = pd_scores.device
            return (
                torch.zeros((bs, n_anchors, n_gt), device=device),
                torch.zeros((bs, n_anchors, n_gt), device=device)
            )

        # 1. Calculate IoU between predictions and GTs
        # Expand dimensions for broadcasting
        pd_boxes_expanded = pd_bboxes.unsqueeze(2).expand(-1, -1, n_gt, -1)  # (B, N, M, 4)
        gt_boxes_expanded = gt_bboxes.unsqueeze(1).expand(-1, n_anchors, -1, -1)  # (B, N, M, 4)
        
        overlaps = self.bbox_iou(pd_boxes_expanded, gt_boxes_expanded)  # (B, N, M)
        overlaps = overlaps.squeeze(-1) if overlaps.dim() == 4 else overlaps
        overlaps = overlaps.clamp(min=0)
        
        # 2. Get classification scores for GT classes
        gt_labels_long = gt_labels.long().squeeze(-1)  # (B, M)
        gt_labels_expanded = gt_labels_long.unsqueeze(1).expand(-1, n_anchors, -1)  # (B, N, M)
        
        # Gather scores for GT classes
        bbox_scores = pd_scores.gather(
            dim=2,
            index=gt_labels_expanded
        )  # (B, N, M)
        
        # 3. Calculate alignment metric
        align_metric = bbox_scores.sigmoid().pow(self.alpha) * overlaps.pow(self.beta)
        
        # 4. Mask invalid GTs
        mask_gt_expanded = mask_gt.squeeze(-1).unsqueeze(1)  # (B, 1, M)
        align_metric = align_metric * mask_gt_expanded
        
        return align_metric, overlaps

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None, mask_gt=None):
        """
        Select top-k candidates based on alignment metrics.
        
        For each GT, select the top-k anchors with highest alignment metric.
        """
        bs, n_anchors, n_gt = metrics.shape
        
        # Get top-k values and indices per GT
        # topk along anchor dimension (dim=1)
        topk_metrics, topk_idxs = torch.topk(
            metrics, 
            min(self.topk, n_anchors), 
            dim=1, 
            largest=largest
        )  # (B, K, M)
        
        # Create mask for top-k positions
        # Check if top-k metric is valid (> 0)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(dim=1, keepdim=True)[0] > self.eps)
        
        # Create binary mask
        mask_topk = torch.zeros_like(metrics, dtype=torch.bool)  # (B, N, M)
        
        # Scatter 1s at top-k positions
        mask_topk.scatter_(
            1,  # dim
            topk_idxs,  # indices
            torch.ones_like(topk_idxs, dtype=torch.bool)
        )
        
        # Apply topk_mask
        mask_topk = mask_topk * topk_mask
        
        return mask_topk.float()

    def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes):
        """
        Handle case where one anchor is assigned to multiple GTs.
        Select the GT with highest IoU.
        """
        # Count how many GTs each anchor is assigned to
        fg_mask = mask_pos.sum(dim=2)  # (B, N)
        
        if fg_mask.max() > 1:
            # Some anchors are assigned to multiple GTs
            # Create mask for these anchors
            mask_multi_gts = (fg_mask.unsqueeze(2) > 1).expand(-1, -1, n_max_boxes)  # (B, N, M)
            
            # Find GT with highest IoU for each anchor
            max_overlaps_idx = overlaps.argmax(dim=2)  # (B, N)
            
            # Create one-hot encoding for highest IoU GT
            is_max_overlaps = F.one_hot(
                max_overlaps_idx, 
                n_max_boxes
            ).to(overlaps.dtype)  # (B, N, M)
            
            # Keep only highest IoU assignment for multi-assigned anchors
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
            fg_mask = mask_pos.sum(dim=2)
        
        # Get GT index for each anchor
        target_gt_idx = mask_pos.argmax(dim=2)  # (B, N)
        
        return target_gt_idx, fg_mask > 0, mask_pos

    @staticmethod
    def bbox_iou(box1, box2, eps=1e-7):
        """
        Calculate IoU between box1 and box2.
        
        Args:
            box1: (..., 4) in xyxy format
            box2: (..., 4) in xyxy format
            
        Returns:
            iou: (...) IoU values
        """
        # Get coordinates
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, dim=-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, dim=-1)
        
        # Calculate intersection area
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        
        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        
        # Calculate union area
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area - inter_area + eps
        
        # Calculate IoU
        iou = inter_area / union_area
        
        return iou