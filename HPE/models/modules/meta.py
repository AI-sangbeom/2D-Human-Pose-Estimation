import torch
import torch.nn as nn
from typing import Tuple, List
from models.layers import GatedAttention, GaussianPooling

class MetaSpace(nn.Module):
    """
    MetaSpace: 학습 중 keypoint feature를 저장하고 업데이트하는 메타 공간
    
    각 feature level과 keypoint마다 평균 feature를 유지하며,
    Gated Attention을 통해 meta feature와 current feature를 융합
    """
    def __init__(self, 
                 original_size: Tuple[int, int],
                 feature_dims: List[int], 
                 num_kpts: int,
                 num_heads: int = 8,
                 momentum: float = 0.9):
        """
        Args:
            original_size: 원본 이미지 크기 (H, W)
            feature_dims: 각 feature level의 dimension 리스트
            num_kpts: keypoint 개수
            num_heads: attention head 수
            momentum: EMA 업데이트 momentum (0.9면 90% old, 10% new)
        """
        super().__init__()
        self.original_size = original_size
        self.num_kpts = num_kpts
        self.num_levels = len(feature_dims)
        self.momentum = momentum
        
        # Gaussian pooling for local feature extraction
        self.pool = GaussianPooling(kernel_size=5, sigma=2.0)
        
        # Meta spaces: 각 level과 keypoint마다 learnable feature 저장

        # xavier 초기화
        self.meta_spaces = nn.ParameterList()
        for feat_dim in feature_dims:
            param = nn.Parameter(torch.empty(num_kpts, feat_dim))
            nn.init.xavier_uniform_(param)
            self.meta_spaces.append(param)
        
        # Gated Multi-Head Attention for each level
        self.gmha = nn.ModuleList([
            GatedAttention(
                dim=feat_dim,
                num_heads=num_heads,
                gate_type="headwise"
            )
            for feat_dim in feature_dims
        ])
        
        # Projection layers (optional): feature fusion
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim * 2, feat_dim),
                nn.LayerNorm(feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, feat_dim)
            )
            for feat_dim in feature_dims
        ])
        
        # Running statistics for accumulation (training only)
        # Register as buffers (not parameters, won't be trained)
        for level_idx in range(self.num_levels):
            self.register_buffer(
                f'feature_sum_{level_idx}',
                torch.zeros(num_kpts, feature_dims[level_idx])
            )
            self.register_buffer(
                f'feature_count_{level_idx}',
                torch.zeros(num_kpts)
            )
    
    def cal_resized_keypoints(self, 
                             keypoints: torch.Tensor, 
                             target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Resize keypoints to match feature map size
        
        Args:
            keypoints: (B, N, 2) or (N, 2) in [x, y] format
            target_size: (H, W) of feature map
            
        Returns:
            resized_keypoints: same shape as input
        """
        orig_h, orig_w = self.original_size
        target_h, target_w = target_size
        
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        resized_kpts = keypoints.clone()
        resized_kpts[..., 0] *= scale_x
        resized_kpts[..., 1] *= scale_y
        
        return resized_kpts
    
    def extract_keypoint_features(self,
                                  feature_map: torch.Tensor,
                                  keypoints: torch.Tensor) -> torch.Tensor:
        """
        Extract features at keypoint locations using Gaussian pooling
        
        Args:
            feature_map: (B, C, H, W) feature map
            keypoints: (B, N, 2) keypoint coordinates [x, y]
            
        Returns:
            kpt_features: (B, N, C) features at each keypoint
        """
        B, C, H, W = feature_map.shape
        _, N, _ = keypoints.shape
        
        # Resize keypoints to feature map scale
        resized_kpts = self.cal_resized_keypoints(keypoints, (H, W))
        
        batch_features = []
        for b in range(B):
            # Extract features for this batch item
            kpt_feats = self.pool(feature_map[b], resized_kpts[b])  # (N, C)
            batch_features.append(kpt_feats)
        
        return torch.stack(batch_features)  # (B, N, C)
    
    def accumulate_features(self,
                           level_idx: int,
                           kpt_features: torch.Tensor,
                           valid_mask: torch.Tensor = None):
        """
        Accumulate keypoint features for later update (training only)
        
        Args:
            level_idx: feature level index
            kpt_features: (B, N, C) extracted features
            valid_mask: (B, N) boolean mask for valid keypoints
        """
        if not self.training:
            return
        
        B = kpt_features.shape[0]
        
        feature_sum = getattr(self, f'feature_sum_{level_idx}')
        feature_count = getattr(self, f'feature_count_{level_idx}')
        
        # Average over batch and accumulate
        if valid_mask is not None:
            # Only accumulate valid keypoints
            valid_mask = valid_mask.float()  # (B, N)
            masked_features = kpt_features * valid_mask.unsqueeze(-1)  # (B, N, C)
            feature_sum += masked_features.sum(dim=0)  # (N, C)
            feature_count += valid_mask.sum(dim=0)  # (N,)
        else:
            feature_sum += kpt_features.sum(dim=0)  # (N, C)
            feature_count += B
    
    def update_meta_spaces(self):
        """
        Update meta spaces with accumulated features (call at end of epoch/batch)
        Uses momentum-based EMA update
        """
        if not self.training:
            return
        
        for level_idx in range(self.num_levels):
            feature_sum = getattr(self, f'feature_sum_{level_idx}')
            feature_count = getattr(self, f'feature_count_{level_idx}')
            
            # Compute mean for keypoints with non-zero count
            valid_kpts = feature_count > 0
            
            if valid_kpts.any():
                # Mean features for valid keypoints
                mean_features = torch.zeros_like(feature_sum)
                mean_features[valid_kpts] = (
                    feature_sum[valid_kpts] / feature_count[valid_kpts].unsqueeze(-1)
                )
                
                # EMA update: new = momentum * old + (1 - momentum) * new
                self.meta_spaces[level_idx].data[valid_kpts] = (
                    self.momentum * self.meta_spaces[level_idx].data[valid_kpts] +
                    (1 - self.momentum) * mean_features[valid_kpts]
                )
            
            # Reset accumulators
            feature_sum.zero_()
            feature_count.zero_()
    
    def fuse_with_meta_features(self,
                               level_idx: int,
                               kpt_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse current keypoint features with meta space features using attention
        
        Args:
            level_idx: feature level index
            kpt_features: (B, N, C) current features
            
        Returns:
            fused_features: (B, N, C) fused features
        """
        B, N, C = kpt_features.shape
        
        # Get meta features for this level
        meta_features = self.meta_spaces[level_idx].unsqueeze(0).expand(B, -1, -1)  # (B, N, C)
        
        # Concatenate current and meta features
        combined = torch.cat([kpt_features, meta_features], dim=-1)  # (B, N, 2C)
        
        # Project and fuse
        projected = self.projections[level_idx](combined)  # (B, N, C)
        
        # Apply gated attention for refinement
        # Stack features for attention: [current, meta]
        stacked = torch.stack([kpt_features, meta_features], dim=1)  # (B, 2, N, C)
        stacked_reshaped = stacked.transpose(1, 2).reshape(B * N, 2, C)  # (B*N, 2, C)
        
        attended = self.gmha[level_idx](stacked_reshaped)  # (B*N, 2, C)
        attended = attended.reshape(B, N, 2, C)  # (B, N, 2, C)
        
        # Weighted combination
        fused = attended.mean(dim=2) + projected  # (B, N, C)
        
        return fused
    
    def forward(self, 
                feature_maps: List[torch.Tensor],
                keypoints: torch.Tensor,
                valid_mask: torch.Tensor = None) -> List[torch.Tensor]:
        """
        Forward pass: extract, fuse, and accumulate keypoint features
        
        Args:
            feature_maps: List of (B, C, H, W) feature maps from different levels
            keypoints: (B, N, 2) keypoint coordinates [x, y] in original image space
            valid_mask: (B, N) boolean mask for valid keypoints
            
        Returns:
            fused_features: List of (B, N, C) fused features for each level
        """
        assert len(feature_maps) == self.num_levels, \
            f"Expected {self.num_levels} feature maps, got {len(feature_maps)}"
        
        fused_features = []
        
        for level_idx, feature_map in enumerate(feature_maps):
            # 1. Extract keypoint features
            kpt_feats = self.extract_keypoint_features(feature_map, keypoints)
            
            # 2. Accumulate for meta space update (training only)
            self.accumulate_features(level_idx, kpt_feats, valid_mask)
            
            # 3. Fuse with meta features
            fused = self.fuse_with_meta_features(level_idx, kpt_feats)
            
            fused_features.append(fused)
        
        return fused_features


# ===== Usage Example =====
if __name__ == "__main__":
    print("=" * 60)
    print("MetaSpace with Keypoint Features - Example")
    print("=" * 60)
    
    # Hyperparameters
    batch_size = 4
    num_kpts = 17  # e.g., human pose keypoints
    original_size = (640, 640)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Multi-scale feature maps (e.g., from backbone)
    feature_maps = [
        torch.randn(batch_size, 256, 64, 64).to(device),   # Level 0: 1/4 scale
        torch.randn(batch_size, 512, 32, 32).to(device),   # Level 1: 1/8 scale
        torch.randn(batch_size, 1024, 16, 16).to(device),  # Level 2: 1/16 scale
    ]
    
    # Keypoints in original image coordinates
    keypoints = torch.rand(batch_size, num_kpts, 2).to(device) * 640  # Random [0, 640]
    
    # Valid mask (e.g., some keypoints are occluded)
    valid_mask = torch.rand(batch_size, num_kpts).to(device) > 0.2
    
    # Initialize MetaSpace
    meta_space = MetaSpace(
        original_size=original_size,
        feature_dims=[256, 512, 1024],
        num_kpts=num_kpts,
        num_heads=8,
        momentum=0.9
    ).to(device)
    
    print("\n1. Initial forward pass (training mode):")
    meta_space.train()
    fused_features = meta_space(feature_maps, keypoints, valid_mask)
    
    for i, feats in enumerate(fused_features):
        print(f"   Level {i}: {feats.shape}")
    
    print("\n2. Update meta spaces:")
    meta_space.update_meta_spaces()
    print("   Meta spaces updated with accumulated features")
    
    print("\n3. Check meta space statistics:")
    for i, meta in enumerate(meta_space.meta_spaces):
        print(f"   Level {i} meta space: {meta.shape}")
        print(f"      Mean: {meta.mean().item():.4f}, Std: {meta.std().item():.4f}")
    
    print("\n4. Inference mode:")
    meta_space.eval()
    with torch.no_grad():
        fused_features_eval = meta_space(feature_maps, keypoints)
    print("   Features fused without accumulation")
    
    print("\n" + "=" * 60)
    print("주요 기능:")
    print("- Multi-scale keypoint feature extraction")
    print("- Gaussian pooling for robust local features")
    print("- EMA-based meta feature learning")
    print("- Gated attention fusion")
    print("- Valid mask support for occluded keypoints")
    print("=" * 60)