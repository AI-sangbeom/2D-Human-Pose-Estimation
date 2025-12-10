import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class Attention(nn.Module):
    """Multi-head Self Attention"""
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP as used in Vision Transformer"""
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                             attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DINOv2MAEBackbone(nn.Module):
    """
    DINOv2-inspired Masked Autoencoder Backbone
    
    Features:
    - Patch-based vision transformer
    - Multi-scale feature extraction
    - Optional masking for self-supervised learning
    - Compatible with MetaSpace
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        use_register_tokens=True,  # DINOv2 feature
        num_register_tokens=4,
        feature_levels=[3, 6, 9, 12],  # Extract features at these layers
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.feature_levels = feature_levels
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        
        # Class token (like BERT's [CLS])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Register tokens (DINOv2 innovation for better feature quality)
        self.use_register_tokens = use_register_tokens
        if use_register_tokens:
            self.num_register_tokens = num_register_tokens
            self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
        else:
            self.num_register_tokens = 0
        
        # Positional embeddings
        num_tokens = 1 + self.num_register_tokens + self.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Feature projection heads for different scales
        self.feature_projections = nn.ModuleDict()
        feature_dims = [256, 512, 768, 1024]  # Output dimensions for multi-scale
        for idx, level in enumerate(feature_levels):
            self.feature_projections[f'level_{level}'] = nn.Sequential(
                nn.Linear(embed_dim, feature_dims[idx]),
                nn.LayerNorm(feature_dims[idx]),
                nn.GELU(),
                nn.Linear(feature_dims[idx], feature_dims[idx])
            )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize patch_embed like nn.Linear
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Timm's trunc_normal_(std=.02) is effectively normal_(std=0.02)
        nn.init.normal_(self.cls_token, std=.02)
        nn.init.normal_(self.pos_embed, std=.02)
        if self.use_register_tokens:
            nn.init.normal_(self.register_tokens, std=.02)
        
        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights_module)
    
    def _init_weights_module(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def random_masking(self, x, mask_ratio=0.75):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        
        Args:
            x: [B, N, D], sequence
            mask_ratio: ratio of patches to mask
            
        Returns:
            x_masked: [B, N*(1-mask_ratio), D], masked sequence
            mask: [B, N], 0 is keep, 1 is remove
            ids_restore: [B, N], indices to restore original order
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        
        # Random noise for shuffling
        noise = torch.rand(B, N, device=x.device)
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_features(self, x, mask_ratio=0.0, return_all_features=False):
        """
        Extract features from input images
        
        Args:
            x: (B, C, H, W) input images
            mask_ratio: ratio of patches to mask (0 = no masking)
            return_all_features: return features from all specified levels
            
        Returns:
            features: dict with multi-scale features
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # Optional masking (for MAE-style pretraining)
        if mask_ratio > 0 and self.training:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            mask = None
            ids_restore = None
        
        # Add register tokens (DINOv2 feature)
        if self.use_register_tokens:
            register_tokens = self.register_tokens.expand(B, -1, -1)
            x = torch.cat([cls_tokens, register_tokens, x], dim=1)
        else:
            x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embeddings
        if mask_ratio > 0 and self.training:
            # Only add pos_embed for visible patches
            num_prefix = 1 + self.num_register_tokens
            pos_embed_prefix = self.pos_embed[:, :num_prefix, :]
            pos_embed_patches = self.pos_embed[:, num_prefix:, :]
            
            # Gather visible patch positions
            pos_embed_visible = torch.gather(
                pos_embed_patches.expand(B, -1, -1), 
                dim=1, 
                index=torch.arange(x.shape[1] - num_prefix, device=x.device).unsqueeze(0).unsqueeze(-1).expand(B, -1, self.embed_dim)
            )
            x = x + torch.cat([pos_embed_prefix.expand(B, -1, -1), pos_embed_visible], dim=1)
        else:
            x = x + self.pos_embed
        
        # Transform through blocks and collect features
        features = {}
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            
            # Extract features at specified levels
            if (idx + 1) in self.feature_levels:
                # Remove cls and register tokens for feature maps
                feat = x[:, 1 + self.num_register_tokens:, :]  # (B, N, D)
                feat = self.norm(feat)
                
                # Project to different dimensions
                feat_projected = self.feature_projections[f'level_{idx+1}'](feat)
                
                # Reshape to 2D feature map
                H = W = int(feat.shape[1] ** 0.5)
                feat_2d = feat_projected.transpose(1, 2).reshape(B, -1, H, W)
                
                features[f'level_{idx+1}'] = feat_2d
        
        if return_all_features:
            return features, mask, ids_restore
        else:
            # Return only the features (for inference)
            return features
    
    def forward(self, x, mask_ratio=0.0, return_all_features=False):
        """
        Forward pass
        
        Args:
            x: (B, C, H, W) input images
            mask_ratio: masking ratio for MAE-style training
            return_all_features: whether to return all intermediate features
            
        Returns:
            features: dict of multi-scale features
                - 'level_3': (B, 256, H/16, W/16)
                - 'level_6': (B, 512, H/16, W/16)
                - 'level_9': (B, 768, H/16, W/16)
                - 'level_12': (B, 1024, H/16, W/16)
        """
        return self.forward_features(x, mask_ratio, return_all_features)


class GaussianPooling(nn.Module):
    """Gaussian weighted pooling for local features"""
    def __init__(self, kernel_size: int = 5, sigma: float = 2.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        self.register_buffer('kernel', kernel)
    
    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def forward(self, feature_map: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        C, H, W = feature_map.shape
        N = keypoints.shape[0]
        half_k = self.kernel_size // 2
        pooled_features = []
        
        for kpt in keypoints:
            x, y = kpt[0].long(), kpt[1].long()
            x = torch.clamp(x, half_k, W - half_k - 1)
            y = torch.clamp(y, half_k, H - half_k - 1)
            
            patch = feature_map[:, y - half_k:y + half_k + 1, x - half_k:x + half_k + 1]
            
            if patch.shape[1:] == (self.kernel_size, self.kernel_size):
                weighted = patch * self.kernel
                pooled = weighted.squeeze(0).sum(dim=(1, 2))
            else:
                pooled = feature_map[:, y, x]
            
            pooled_features.append(pooled)
        
        return torch.stack(pooled_features)


class DINOv2WithMetaSpace(nn.Module):
    """
    Complete pipeline: DINOv2 MAE Backbone + MetaSpace for keypoint learning
    """
    def __init__(
        self,
        img_size=224,
        num_kpts=17,
        embed_dim=768,
        depth=12,
        num_heads=12,
        feature_levels=[3, 6, 9, 12],
        mask_ratio=0.0,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.num_kpts = num_kpts
        self.mask_ratio = mask_ratio
        
        # DINOv2 MAE Backbone
        self.backbone = DINOv2MAEBackbone(
            img_size=img_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            feature_levels=feature_levels,
        )
        
        # MetaSpace for keypoint feature learning
        from typing import List
        self.feature_dims = [256, 512, 768, 1024]
        self.pool = GaussianPooling(kernel_size=5, sigma=2.0)
        
        # Simplified MetaSpace components
        self.meta_spaces = nn.ParameterList([
            nn.Parameter(torch.randn(num_kpts, feat_dim) * 0.02)
            for feat_dim in self.feature_dims
        ])
        
        # Feature fusion layers
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim * 2, feat_dim),
                nn.LayerNorm(feat_dim),
                nn.GELU(),
                nn.Linear(feat_dim, feat_dim)
            )
            for feat_dim in self.feature_dims
        ])
    
    def extract_keypoint_features(self, feature_map, keypoints):
        """Extract features at keypoint locations"""
        B, C, H, W = feature_map.shape
        _, N, _ = keypoints.shape
        
        # Scale keypoints to feature map size
        scale_h = H / self.img_size
        scale_w = W / self.img_size
        scaled_kpts = keypoints.clone()
        scaled_kpts[..., 0] *= scale_w
        scaled_kpts[..., 1] *= scale_h
        
        batch_features = []
        for b in range(B):
            kpt_feats = self.pool(feature_map[b], scaled_kpts[b])
            batch_features.append(kpt_feats)
        
        return torch.stack(batch_features)
    
    def forward(self, x, keypoints=None):
        """
        Forward pass
        
        Args:
            x: (B, 3, H, W) input images
            keypoints: (B, N, 2) keypoint coordinates [x, y] or None
            
        Returns:
            features: dict of multi-scale features
            kpt_features: dict of keypoint features (if keypoints provided)
        """
        # Extract backbone features
        features = self.backbone(x, mask_ratio=self.mask_ratio if self.training else 0.0)
        
        if keypoints is None:
            return features, None
        
        # Extract keypoint features from each level
        kpt_features = {}
        for idx, (level_name, feat_map) in enumerate(features.items()):
            # Extract raw keypoint features
            kpt_feats = self.extract_keypoint_features(feat_map, keypoints)  # (B, N, C)
            
            # Fuse with meta space
            B, N, C = kpt_feats.shape
            meta_feats = self.meta_spaces[idx].unsqueeze(0).expand(B, -1, -1)
            
            # Concatenate and fuse
            combined = torch.cat([kpt_feats, meta_feats], dim=-1)
            fused = self.fusion_layers[idx](combined)
            
            kpt_features[level_name] = fused
        
        return features, kpt_features


# ===== Usage Example =====
if __name__ == "__main__":
    print("=" * 70)
    print("DINOv2 MAE Backbone with MetaSpace - Example")
    print("=" * 70)
    
    # Hyperparameters
    batch_size = 2
    img_size = 224
    num_kpts = 17  # Human pose keypoints
    
    # Input images and keypoints
    images = torch.randn(batch_size, 3, img_size, img_size)
    keypoints = torch.rand(batch_size, num_kpts, 2) * img_size
    
    print(f"\nInput:")
    print(f"  Images: {images.shape}")
    print(f"  Keypoints: {keypoints.shape}")
    
    # Initialize model
    model = DINOv2WithMetaSpace(
        img_size=img_size,
        num_kpts=num_kpts,
        embed_dim=768,
        depth=12,
        num_heads=12,
        feature_levels=[3, 6, 9, 12],
        mask_ratio=0.0,  # No masking for inference
    )
    
    print(f"\nModel initialized:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Forward pass
    print(f"\n1. Forward pass (inference mode):")
    model.eval()
    with torch.no_grad():
        features, kpt_features = model(images, keypoints)
    
    print(f"\n  Backbone Features:")
    for name, feat in features.items():
        print(f"    {name}: {feat.shape}")
    
    print(f"\n  Keypoint Features:")
    for name, feat in kpt_features.items():
        print(f"    {name}: {feat.shape}")
    
    # Training mode with masking
    print(f"\n2. Training mode with masking:")
    model.train()
    model.mask_ratio = 0.75
    features_train, kpt_features_train = model(images, keypoints)
    
    print(f"  Features extracted with 75% masking")
    print(f"  (useful for MAE-style self-supervised pretraining)")
    
    print("\n" + "=" * 70)
    print("Key Features:")
    print("  ✓ DINOv2-inspired architecture with register tokens")
    print("  ✓ Masked Autoencoder capability for self-supervised learning")
    print("  ✓ Multi-scale feature extraction at layers [3, 6, 9, 12]")
    print("  ✓ MetaSpace integration for keypoint feature learning")
    print("  ✓ Gaussian pooling for robust local features")
    print("  ✓ Ready for pose estimation, detection, segmentation tasks")
    print("=" * 70)