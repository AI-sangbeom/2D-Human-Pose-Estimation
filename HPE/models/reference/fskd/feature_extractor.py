import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from models.backbones.dinov3vit import Dinov3ViT, vit_sizes


class FeatureExtractor(nn.Module):
    """
    Feature extractor using DINOv3 small backbone
    Extracts multi-scale features for few-shot learning
    """
    
    def __init__(
        self,
        backbone: str = 'small',
        pretrained: bool = True,
        freeze_backbone: bool = False,
        feature_levels: int = 3
    ):
        super().__init__()
        
        # DINOv3 Small backbone configuration
        backbone_cfg = vit_sizes[backbone]
        embed_dim = backbone_cfg['embed_dim']  # 384 for small
        
        # Initialize DINOv3 backbone
        self.backbone = Dinov3ViT(
            patch_size=backbone_cfg["patch_size"],
            embed_dim=embed_dim,
            depth=backbone_cfg["depth"],
            num_heads=backbone_cfg["num_heads"],
            ffn_ratio=backbone_cfg["ffn_ratio"],
            pretrained=pretrained,
        )
        
        self.freeze_backbone = freeze_backbone
        self.feature_levels = feature_levels
        
        # Feature projection layers
        self.feature_proj = nn.ModuleDict({
            'cls_token': nn.Linear(embed_dim, embed_dim),
            'patch_tokens': nn.Linear(embed_dim, embed_dim),
        })
        
        # Feature normalization
        self.feature_norm = nn.LayerNorm(embed_dim)
        
        # Feature fusion layers for multi-scale representation
        self.fusion_layers = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(feature_levels)
        ])
        
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from input images using DINOv3 backbone
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Dict containing:
                - cls_token: Global class token [B, embed_dim]
                - patch_tokens: Patch features [B, num_patches, embed_dim]
                - multi_scale_features: List of multi-scale features
        """
        batch_size = x.shape[0]
        
        # Forward through DINOv3 backbone
        # Returns list of features from different depths
        features_list, all_xes = self.backbone.forward_features_list([x], [None])
        
        # Extract features from the last layer
        last_features = features_list[0]
        
        # Class token (global representation)
        cls_token = last_features["x_norm_clstoken"]  # [B, embed_dim]
        cls_token = self.feature_norm(cls_token)
        cls_token = self.feature_proj['cls_token'](cls_token)
        
        # Patch tokens (local representations)
        patch_tokens = last_features["x_norm_patchtokens"]  # [B, num_patches, embed_dim]
        patch_tokens = self.feature_norm(patch_tokens)
        patch_tokens = self.feature_proj['patch_tokens'](patch_tokens)
        
        # Multi-scale features from different layers
        multi_scale_features = []
        for i in range(min(self.feature_levels, len(all_xes))):
            layer_features = all_xes[i][0]  # [B, num_tokens, embed_dim]
            # Use CLS token from each layer
            layer_cls = layer_features[:, 0]  # [B, embed_dim]
            layer_cls = self.feature_norm(layer_cls)
            layer_cls = self.fusion_layers[i](layer_cls)
            multi_scale_features.append(layer_cls)
        
        return {
            'cls_token': cls_token,
            'patch_tokens': patch_tokens,
            'multi_scale_features': multi_scale_features,
            'batch_size': batch_size
        }
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through feature extractor
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Dict containing extracted features
        """
        return self.extract_features(x)


class CrossModalFeatureExtractor(FeatureExtractor):
    """
    Extended feature extractor for cross-modal scenarios
    Handles different input modalities (images, sketches, etc.)
    """
    
    def __init__(
        self,
        backbone: str = 'small',
        pretrained: bool = True,
        freeze_backbone: bool = False,
        modal_embedding_dim: int = 64
    ):
        super().__init__(backbone, pretrained, freeze_backbone)
        
        # Modal embedding for different input types
        self.modal_embedding = nn.Embedding(3, modal_embedding_dim)  # Support 3 modalities
        self.modal_proj = nn.Linear(modal_embedding_dim, 384)  # Project to backbone dim
        
        # Modal-specific feature adaptors
        self.modal_adaptors = nn.ModuleDict({
            'image': nn.Identity(),
            'sketch': nn.Linear(384, 384),
            'sparse': nn.Linear(384, 384)
        })
    
    def extract_features_with_modality(
        self, 
        x: torch.Tensor, 
        modality: str = 'image'
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features with modality awareness
        
        Args:
            x: Input data [B, C, H, W] or [B, N, 2] for sparse
            modality: Input modality ('image', 'sketch', 'sparse')
            
        Returns:
            Dict containing modality-aware features
        """
        # Extract base features
        features = self.extract_features(x)
        
        # Apply modality-specific adaptation
        adapted_features = {}
        for key, value in features.items():
            if isinstance(value, torch.Tensor):
                adapted_features[key] = self.modal_adaptors[modality](value)
            else:
                adapted_features[key] = value
        
        # Add modality information
        modal_ids = torch.tensor([self._get_modal_id(modality)] * x.shape[0], device=x.device)
        modal_emb = self.modal_embedding(modal_ids)  # [B, modal_embedding_dim]
        modal_emb = self.modal_proj(modal_emb)  # [B, embed_dim]
        
        adapted_features['modal_embedding'] = modal_emb
        adapted_features['modality'] = modality
        
        return adapted_features
    
    def _get_modal_id(self, modality: str) -> int:
        """Get modality ID for embedding lookup"""
        modality_map = {'image': 0, 'sketch': 1, 'sparse': 2}
        return modality_map.get(modality, 0)
    
    def forward(self, x: torch.Tensor, modality: str = 'image') -> Dict[str, torch.Tensor]:
        """
        Forward pass with modality awareness
        
        Args:
            x: Input data
            modality: Input modality type
            
        Returns:
            Dict containing modality-aware features
        """
        return self.extract_features_with_modality(x, modality)
