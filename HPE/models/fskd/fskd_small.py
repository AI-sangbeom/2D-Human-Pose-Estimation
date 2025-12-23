import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .feature_extractor import FeatureExtractor
from .prototypical_head import PrototypicalHead, MultiPrototypicalHead
from .attention_fusion import AttentionFusion, AdaptiveAttentionFusion


class FSKD(nn.Module):
    """
    Few-Shot Keypoint Detection model based on DINOv3 small backbone
    Implements meta-learning for rapid adaptation to new keypoint patterns
    """
    
    def __init__(
        self,
        nkpts: int = 17,
        backbone: str = 'small',
        pretrained: bool = True,
        n_way: int = 5,
        temperature: float = 1.0,
        fusion_method: str = 'cross_attention',
        use_hierarchical_prototypes: bool = False,
        freeze_backbone: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.nkpts = nkpts
        self.backbone_name = backbone
        self.n_way = n_way
        self.temperature = temperature
        self.fusion_method = fusion_method
        self.use_hierarchical_prototypes = use_hierarchical_prototypes
        
        # Get backbone configuration
        from models.backbones.dinov3vit import vit_sizes
        backbone_cfg = vit_sizes[backbone]
        self.embed_dim = backbone_cfg['embed_dim']  # 384 for small
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            backbone=backbone,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            feature_levels=3
        )
        
        # Attention fusion module
        if fusion_method == 'adaptive':
            self.attention_fusion = AdaptiveAttentionFusion(
                embed_dim=self.embed_dim,
                num_heads=8,
                dropout=dropout
            )
        else:
            self.attention_fusion = AttentionFusion(
                embed_dim=self.embed_dim,
                num_heads=8,
                dropout=dropout,
                fusion_method=fusion_method,
                num_fusion_layers=2
            )
        
        # Prototypical head
        if use_hierarchical_prototypes:
            self.prototypical_head = MultiPrototypicalHead(
                nkpts=nkpts,
                embed_dim=self.embed_dim,
                n_way=n_way,
                num_prototype_layers=3,
                temperature=temperature
            )
        else:
            self.prototypical_head = PrototypicalHead(
                nkpts=nkpts,
                embed_dim=self.embed_dim,
                n_way=n_way,
                temperature=temperature
            )
        
        # Meta-learning specific components
        self.meta_encoder = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        # Task-specific adaptation layers
        self.task_adapter = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def extract_features(
        self, 
        images: torch.Tensor,
        modality: str = 'image'
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from input images
        
        Args:
            images: Input images [B, C, H, W]
            modality: Input modality type
            
        Returns:
            Dict containing extracted features
        """
        return self.feature_extractor(images, modality)
    
    def compute_prototypes(
        self,
        support_images: torch.Tensor,
        support_keypoints: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute prototypes from support examples
        
        Args:
            support_images: Support images [N_support, C, H, W]
            support_keypoints: Support keypoints [N_support, nkpts, 2]
            
        Returns:
            Prototypes [n_way, nkpts, embed_dim]
        """
        # Extract features from support examples
        support_features = self.extract_features(support_images)
        support_cls_features = support_features['cls_token']  # [N_support, embed_dim]
        
        # Apply meta-encoding
        support_encoded = self.meta_encoder(support_cls_features)
        
        # Compute prototypes
        prototypes = self.prototypical_head.compute_prototypes(
            support_encoded.unsqueeze(0),  # Add batch dimension
            support_keypoints
        )
        
        return prototypes
    
    def meta_adapt(
        self,
        query_features: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform meta-adaptation for query examples
        
        Args:
            query_features: Query features [N_query, embed_dim]
            prototypes: Class prototypes [n_way, nkpts, embed_dim]
            
        Returns:
            Meta-adapted query features
        """
        # Apply meta-encoding to query features
        query_encoded = self.meta_encoder(query_features)
        
        # Task-specific adaptation
        query_adapted = self.task_adapter(query_encoded)
        
        # Combine with prototype information
        # Average prototype features across keypoints
        avg_prototypes = torch.mean(prototypes, dim=1)  # [n_way, embed_dim]
        
        # Compute attention weights based on similarity to prototypes
        attention_weights = F.softmax(
            torch.matmul(query_adapted, avg_prototypes.transpose(0, 1)), dim=1
        )  # [N_query, n_way]
        
        # Weighted combination with prototypes
        prototype_influence = torch.matmul(attention_weights, avg_prototypes)  # [N_query, embed_dim]
        
        # Final adapted features
        adapted_features = query_adapted + 0.1 * prototype_influence  # Small prototype influence
        
        return adapted_features
    
    def predict_keypoints(
        self,
        query_images: torch.Tensor,
        prototypes: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Predict keypoints for query examples
        
        Args:
            query_images: Query images [N_query, C, H, W]
            prototypes: Class prototypes [n_way, nkpts, embed_dim]
            return_attention: Whether to return attention maps
            
        Returns:
            Dict containing predictions and intermediate results
        """
        # Extract features from query images
        query_features = self.extract_features(query_images)
        query_cls_features = query_features['cls_token']  # [N_query, embed_dim]
        
        # Meta-adaptation
        adapted_query_features = self.meta_adapt(query_cls_features, prototypes)
        
        # Compute confidence scores
        confidence_scores = self.confidence_estimator(adapted_query_features)  # [N_query, 1]
        
        # Generate predictions using prototypical head
        predictions = self.prototypical_head.predict_keypoints(
            adapted_query_features,
            prototypes
        )
        
        # Add confidence scores to predictions
        predictions['confidence_global'] = confidence_scores
        
        # Optionally return attention maps
        if return_attention and hasattr(self.attention_fusion, 'attention_visualization'):
            # This would require support features, so we need to modify the interface
            pass
        
        return predictions
    
    def forward(
        self,
        support_images: torch.Tensor,
        support_keypoints: torch.Tensor,
        query_images: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for few-shot learning episode
        
        Args:
            support_images: Support images [N_support, C, H, W]
            support_keypoints: Support keypoints [N_support, nkpts, 2]
            query_images: Query images [N_query, C, H, W]
            return_attention: Whether to return attention maps
            
        Returns:
            Dict containing:
                - predictions: Keypoint predictions for queries
                - prototypes: Computed prototypes
                - meta_features: Intermediate meta-learning features
        """
        # Step 1: Compute prototypes from support examples
        prototypes = self.compute_prototypes(support_images, support_keypoints)
        
        # Step 2: Predict keypoints for query examples
        predictions = self.predict_keypoints(
            query_images, 
            prototypes, 
            return_attention=return_attention
        )
        
        # Step 3: Extract query features for meta-analysis
        query_features = self.extract_features(query_images)
        query_cls_features = query_features['cls_token']
        adapted_query_features = self.meta_adapt(query_cls_features, prototypes)
        
        # Compile results
        result = {
            'predictions': predictions,
            'prototypes': prototypes,
            'meta_features': {
                'query_features': query_cls_features,
                'adapted_query_features': adapted_query_features,
                'support_features': self.extract_features(support_images)['cls_token']
            }
        }
        
        return result
    
    def get_attention_maps(
        self,
        support_images: torch.Tensor,
        query_images: torch.Tensor,
        prototypes: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate attention maps for visualization
        
        Args:
            support_images: Support images [N_support, C, H, W]
            query_images: Query images [N_query, C, H, W]
            prototypes: Optional class prototypes
            
        Returns:
            Dict containing attention maps
        """
        # Extract features
        support_features = self.extract_features(support_images)
        query_features = self.extract_features(query_images)
        
        support_cls = support_features['cls_token']
        query_cls = query_features['cls_token']
        
        # Get attention maps from fusion module
        if hasattr(self.attention_fusion, 'attention_visualization'):
            attention_maps = self.attention_fusion.attention_visualization(
                support_cls, query_cls, 'cross'
            )
            return attention_maps
        else:
            return {}
    
    def adaptation_step(
        self,
        support_images: torch.Tensor,
        support_keypoints: torch.Tensor,
        adaptation_lr: float = 0.01,
        num_steps: int = 5
    ) -> nn.Module:
        """
        Perform few-shot adaptation (MAML-style)
        
        Args:
            support_images: Support images [N_support, C, H, W]
            support_keypoints: Support keypoints [N_support, nkpts, 2]
            adaptation_lr: Learning rate for adaptation
            num_steps: Number of adaptation steps
            
        Returns:
            Adapted model
        """
        # Create a copy of the model for adaptation
        adapted_model = FSKD(
            nkpts=self.nkpts,
            backbone=self.backbone_name,
            pretrained=False,  # Don't load pretrained weights
            n_way=self.n_way,
            temperature=self.temperature,
            fusion_method=self.fusion_method,
            use_hierarchical_prototypes=self.use_hierarchical_prototypes,
            freeze_backbone=True  # Freeze backbone during adaptation
        )
        
        # Copy current weights
        adapted_model.load_state_dict(self.state_dict())
        
        # Freeze backbone and feature extractor
        for param in adapted_model.feature_extractor.parameters():
            param.requires_grad = False
        
        # Adaptation loop
        optimizer = torch.optim.SGD(
            [p for p in adapted_model.parameters() if p.requires_grad],
            lr=adaptation_lr
        )
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Forward pass
            predictions = adapted_model(
                support_images, 
                support_keypoints, 
                support_images  # Use support as query for adaptation
            )
            
            # Compute adaptation loss (simplified)
            # In practice, this would depend on the specific task
            loss = torch.tensor(0.0, requires_grad=True, device=adapted_model.device)
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        return adapted_model
    


class MetaLearningFSKD(FSKD):
    """
    Extended few-shot FSKD with advanced meta-learning capabilities
    """
    
    def __init__(
        self,
        nkpts: int = 17,
        backbone: str = 'small',
        pretrained: bool = True,
        n_way: int = 5,
        meta_lr: float = 1e-3,
        update_steps: int = 5,
        **kwargs
    ):
        super().__init__(nkpts, backbone, pretrained, n_way, **kwargs)
        
        self.meta_lr = meta_lr
        self.update_steps = update_steps
        
        # Meta-optimizer (for MAML-style training)
        self.meta_optimizer = torch.optim.Adam(
            self.parameters(),
            lr=meta_lr
        )
        
        # Fast adaptation weights
        self.fast_weights = {}
        
    def maml_forward(
        self,
        support_images: torch.Tensor,
        support_keypoints: torch.Tensor,
        query_images: torch.Tensor,
        return_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        MAML-style forward pass with inner and outer loops
        
        Args:
            support_images: Support images [N_support, C, H, W]
            support_keypoints: Support keypoints [N_support, nkpts, 2]
            query_images: Query images [N_query, C, H, W]
            return_loss: Whether to compute losses
            
        Returns:
            Dict containing predictions and losses
        """
        # Inner loop: adapt to support set
        adapted_model = self.adaptation_step(
            support_images, 
            support_keypoints,
            adaptation_lr=self.meta_lr,
            num_steps=self.update_steps
        )
        
        # Outer loop: evaluate on query set
        with torch.no_grad():
            query_predictions = adapted_model.predict_keypoints(query_images, None)
        
        # Compute losses if requested
        losses = {}
        if return_loss:
            # This would require actual query keypoints
            # For now, return empty losses
            pass
        
        return {
            'query_predictions': query_predictions,
            'adapted_model': adapted_model,
            'losses': losses
        }
    
    def compute_meta_loss(
        self,
        support_predictions: Dict[str, torch.Tensor],
        query_predictions: Dict[str, torch.Tensor],
        support_keypoints: torch.Tensor,
        query_keypoints: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute meta-learning loss
        
        Args:
            support_predictions: Predictions on support set
            query_predictions: Predictions on query set
            support_keypoints: Ground truth support keypoints
            query_keypoints: Ground truth query keypoints
            
        Returns:
            Meta-learning loss
        """
        # Support set loss
        support_loss = F.mse_loss(
            support_predictions['keypoints'],
            support_keypoints
        )
        
        # Query set loss
        query_loss = F.mse_loss(
            query_predictions['keypoints'],
            query_keypoints
        )
        
        # Meta loss is typically just the query loss
        # (support loss is used for inner loop adaptation)
        meta_loss = query_loss
        
        return meta_loss
