import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class AttentionFusion(nn.Module):
    """
    Attention fusion module for combining support and query features
    Implements cross-attention and self-attention mechanisms for few-shot learning
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        fusion_method: str = 'cross_attention',
        num_fusion_layers: int = 2
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.fusion_method = fusion_method
        self.num_fusion_layers = num_fusion_layers
        
        # Multi-head attention for cross-modality fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Self-attention for feature refinement
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward networks for feature transformation
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        # Fusion layers
        self.fusion_layers = nn.ModuleList([
            self._create_fusion_layer() for _ in range(num_fusion_layers)
        ])
        
        # Feature aggregation
        self.aggregation_weights = nn.Parameter(torch.ones(num_fusion_layers) / num_fusion_layers)
        
    def _create_fusion_layer(self) -> nn.Module:
        """Create a single fusion layer"""
        return nn.ModuleDict({
            'cross_attention': self.cross_attention,
            'self_attention': self.self_attention,
            'ffn': self.ffn,
            'norm1': self.norm1,
            'norm2': self.norm2,
            'norm3': self.norm3
        })
    
    def cross_modal_attention(
        self,
        support_features: torch.Tensor,
        query_features: torch.Tensor,
        support_keypoints: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-modal attention between support and query features
        
        Args:
            support_features: Support features [N_support, embed_dim]
            query_features: Query features [N_query, embed_dim]
            support_keypoints: Optional keypoint information [N_support, nkpts, 2]
            
        Returns:
            Tuple of (attended_support, attended_query)
        """
        N_support = support_features.shape[0]
        N_query = query_features.shape[0]
        
        # Prepare attention inputs
        # Support features as query, query features as key/value
        support_as_query = support_features.unsqueeze(0)  # [1, N_support, embed_dim]
        query_as_key = query_features.unsqueeze(0)  # [1, N_query, embed_dim]
        query_as_value = query_features.unsqueeze(0)  # [1, N_query, embed_dim]

        query_as_query = query_features.unsqueeze(0)
        support_as_key = support_features.unsqueeze(0)
        support_as_value = support_features.unsqueeze(0)
        
        # Cross-attention: support attends to query
        attended_support, attention_weights_support = self.cross_attention(
            query=support_as_query,
            key=query_as_key,
            value=query_as_value
        )
        
        # Cross-attention: query attends to support
        attended_query, attention_weights_query = self.cross_attention(
            query=query_as_query,
            key=support_as_key,
            value=support_as_value
        )
        
        attended_support = attended_support.squeeze(0)  # [N_support, embed_dim]
        attended_query = attended_query.squeeze(0)  # [N_query, embed_dim]
        
        return attended_support, attended_query
    
    def self_attention_fusion(
        self,
        support_features: torch.Tensor,
        query_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply self-attention for feature refinement
        
        Args:
            support_features: Support features [N_support, embed_dim]
            query_features: Query features [N_query, embed_dim]
            
        Returns:
            Tuple of (refined_support, refined_query)
        """
        # Combine features for joint processing
        combined_features = torch.cat([support_features, query_features], dim=0)
        combined_features = combined_features.unsqueeze(0)  # [1, N_total, embed_dim]
        
        # Apply self-attention
        refined_combined, _ = self.self_attention(
            query=combined_features,
            key=combined_features,
            value=combined_features
        )
        
        refined_combined = refined_combined.squeeze(0)  # [N_total, embed_dim]
        
        # Split back
        N_support = support_features.shape[0]
        refined_support = refined_combined[:N_support]
        refined_query = refined_combined[N_support:]
        
        return refined_support, refined_query
    
    def prototype_aware_attention(
        self,
        support_features: torch.Tensor,
        query_features: torch.Tensor,
        prototypes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention mechanism aware of prototypes
        
        Args:
            support_features: Support features [N_support, embed_dim]
            query_features: Query features [N_query, embed_dim]
            prototypes: Class prototypes [n_way, nkpts, embed_dim]
            
        Returns:
            Tuple of (prototype_aware_support, prototype_aware_query)
        """
        # Flatten prototypes for attention
        flat_prototypes = prototypes.view(-1, self.embed_dim)  # [n_way * nkpts, embed_dim]
        
        # Support attends to prototypes
        support_as_query = support_features.unsqueeze(0)  # [1, N_support, embed_dim]
        prototypes_as_key = flat_prototypes.unsqueeze(0)  # [1, n_way * nkpts, embed_dim]
        
        prototype_aware_support, _ = self.cross_attention(
            query=support_as_query,
            key=prototypes_as_key,
            value=prototypes_as_key
        )
        prototype_aware_support = prototype_aware_support.squeeze(0)
        
        # Query attends to prototypes
        query_as_query = query_features.unsqueeze(0)  # [1, N_query, embed_dim]
        
        prototype_aware_query, _ = self.cross_attention(
            query=query_as_query,
            key=prototypes_as_key,
            value=prototypes_as_key
        )
        prototype_aware_query = prototype_aware_query.squeeze(0)
        
        return prototype_aware_support, prototype_aware_query
    
    def hierarchical_fusion(
        self,
        support_features: torch.Tensor,
        query_features: torch.Tensor,
        prototypes: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply hierarchical fusion at multiple levels
        
        Args:
            support_features: Support features [N_support, embed_dim]
            query_features: Query features [N_query, embed_dim]
            prototypes: Optional class prototypes
            
        Returns:
            Dict containing fused features at different levels
        """
        fused_features = {}
        
        for layer_idx, fusion_layer in enumerate(self.fusion_layers):
            # Cross-modal attention
            if self.fusion_method == 'cross_attention':
                attended_support, attended_query = self.cross_modal_attention(
                    support_features, query_features
                )
            
            # Self-attention fusion
            elif self.fusion_method == 'self_attention':
                attended_support, attended_query = self.self_attention_fusion(
                    support_features, query_features
                )
            
            # Prototype-aware fusion
            elif self.fusion_method == 'prototype_aware' and prototypes is not None:
                attended_support, attended_query = self.prototype_aware_attention(
                    support_features, query_features, prototypes
                )
            
            else:
                attended_support, attended_query = support_features, query_features
            
            # Apply residual connections and layer norm
            support_output = fusion_layer['norm1'](
                support_features + attended_support
            )
            query_output = fusion_layer['norm1'](
                query_features + attended_query
            )
            
            # Feed-forward
            support_ffn = fusion_layer['ffn'](support_output)
            query_ffn = fusion_layer['ffn'](query_output)
            
            # Final residual connections
            support_final = fusion_layer['norm2'](support_output + support_ffn)
            query_final = fusion_layer['norm2'](query_output + query_ffn)
            
            fused_features[f'layer_{layer_idx}'] = {
                'support': support_final,
                'query': query_final
            }
        
        # Weighted aggregation across layers
        final_support = torch.zeros_like(support_features)
        final_query = torch.zeros_like(query_features)
        
        for layer_idx, layer_features in fused_features.items():
            layer_weight = F.softmax(self.aggregation_weights, dim=0)[int(layer_idx.split('_')[1])]
            final_support += layer_weight * layer_features['support']
            final_query += layer_weight * layer_features['query']
        
        fused_features['final'] = {
            'support': final_support,
            'query': final_query
        }
        
        return fused_features
    
    def attention_visualization(
        self,
        support_features: torch.Tensor,
        query_features: torch.Tensor,
        attention_type: str = 'cross'
    ) -> Dict[str, torch.Tensor]:
        """
        Generate attention maps for visualization
        
        Args:
            support_features: Support features [N_support, embed_dim]
            query_features: Query features [N_query, embed_dim]
            attention_type: Type of attention ('cross', 'self', 'prototype')
            
        Returns:
            Dict containing attention maps
        """
        if attention_type == 'cross':
            # Cross-attention maps
            support_as_query = support_features.unsqueeze(0)
            query_as_key = query_features.unsqueeze(0)
            
            attn_maps, _ = self.cross_attention(
                query=support_as_query,
                key=query_as_key,
                value=query_as_key
            )
            
            return {
                'attention_weights': attn_maps,
                'attention_type': 'cross_modal'
            }
        
        elif attention_type == 'self':
            # Self-attention maps
            combined = torch.cat([support_features, query_features], dim=0).unsqueeze(0)
            
            attn_maps, _ = self.self_attention(
                query=combined,
                key=combined,
                value=combined
            )
            
            return {
                'attention_weights': attn_maps,
                'attention_type': 'self'
            }
        
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")
    
    def forward(
        self,
        support_features: torch.Tensor,
        query_features: torch.Tensor,
        prototypes: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through attention fusion module
        
        Args:
            support_features: Support features [N_support, embed_dim]
            query_features: Query features [N_query, embed_dim]
            prototypes: Optional class prototypes for prototype-aware fusion
            return_attention: Whether to return attention maps for visualization
            
        Returns:
            Dict containing fused features and optional attention maps
        """
        # Apply hierarchical fusion
        fused_features = self.hierarchical_fusion(
            support_features, query_features, prototypes
        )
        
        result = {
            'fused_support': fused_features['final']['support'],
            'fused_query': fused_features['final']['query'],
            'layer_features': fused_features
        }
        
        # Optionally return attention maps
        if return_attention:
            attention_maps = {}
            for attn_type in ['cross', 'self']:
                attention_maps[attn_type] = self.attention_visualization(
                    support_features, query_features, attn_type
                )
            result['attention_maps'] = attention_maps
        
        return result


class AdaptiveAttentionFusion(AttentionFusion):
    """
    Adaptive attention fusion that learns to weight different fusion strategies
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        fusion_strategies: List[str] = None
    ):
        if fusion_strategies is None:
            fusion_strategies = ['cross_attention', 'self_attention', 'prototype_aware']
        
        super().__init__(embed_dim, num_heads, dropout, 'cross_attention')
        
        self.fusion_strategies = fusion_strategies
        
        # Adaptive weighting for different fusion strategies
        self.strategy_weights = nn.Parameter(torch.ones(len(fusion_strategies)))
        
        # Separate modules for each strategy
        self.cross_attention_module = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.self_attention_module = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        
        # Feature projectors for different strategies
        self.feature_projectors = nn.ModuleDict({
            strategy: nn.Linear(embed_dim, embed_dim) for strategy in fusion_strategies
        })
        
    def adaptive_fusion(
        self,
        support_features: torch.Tensor,
        query_features: torch.Tensor,
        prototypes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply adaptive fusion with learned weights
        
        Args:
            support_features: Support features [N_support, embed_dim]
            query_features: Query features [N_query, embed_dim]
            prototypes: Optional class prototypes
            
        Returns:
            Adaptively fused query features
        """
        fused_features = []
        
        for strategy in self.fusion_strategies:
            if strategy == 'cross_attention':
                # Cross-modal attention
                attended_support, attended_query = self._cross_modal_fusion(
                    support_features, query_features
                )
                fused_query = attended_query
                
            elif strategy == 'self_attention':
                # Self-attention fusion
                attended_support, attended_query = self._self_attention_fusion(
                    support_features, query_features
                )
                fused_query = attended_query
                
            elif strategy == 'prototype_aware' and prototypes is not None:
                # Prototype-aware fusion
                attended_support, attended_query = self._prototype_aware_fusion(
                    support_features, query_features, prototypes
                )
                fused_query = attended_query
                
            else:
                fused_query = query_features
            
            # Apply feature projector
            fused_query = self.feature_projectors[strategy](fused_query)
            fused_features.append(fused_query)
        
        # Stack and apply adaptive weights
        stacked_features = torch.stack(fused_features, dim=0)  # [num_strategies, N_query, embed_dim]
        weights = F.softmax(self.strategy_weights, dim=0).unsqueeze(-1).unsqueeze(-1)
        
        # Weighted combination
        adaptive_fused = torch.sum(stacked_features * weights, dim=0)
        
        return adaptive_fused
    
    def _cross_modal_fusion(self, support_features, query_features):
        """Cross-modal fusion implementation"""
        support_as_query = support_features.unsqueeze(0)
        query_as_key = query_features.unsqueeze(0)
        query_as_value = query_features.unsqueeze(0)

        query_as_query = query_features.unsqeeze(0)
        support_as_key = support_features.unsqueeze(0)
        support_as_value = support_features.unsqueeze(0)
        
        attended_support, _ = self.cross_attention_module(
            query=support_as_query, key=query_as_key, value=query_as_value
        )
        
        attended_query, _ = self.cross_attention_module(
            query=query_as_query, key=support_as_key, value=support_as_value
        )
        
        return attended_support.squeeze(0), attended_query.squeeze(0)
    
    def _self_attention_fusion(self, support_features, query_features):
        """Self-attention fusion implementation"""
        combined = torch.cat([support_features, query_features], dim=0)
        combined = combined.unsqueeze(0)
        
        refined_combined, _ = self.self_attention_module(
            query=combined, key=combined, value=combined
        )
        
        refined_combined = refined_combined.squeeze(0)
        N_support = support_features.shape[0]
        
        return refined_combined[:N_support], refined_combined[N_support:]
    
    def _prototype_aware_fusion(self, support_features, query_features, prototypes):
        """Prototype-aware fusion implementation"""
        # Flatten prototypes
        flat_prototypes = prototypes.view(-1, self.embed_dim)
        
        # Query attends to prototypes
        query_as_query = query_features.unsqueeze(0)
        prototypes_as_key = flat_prototypes.unsqueeze(0)
        
        prototype_aware_query, _ = self.cross_attention_module(
            query=query_as_query, key=prototypes_as_key, value=prototypes_as_key
        )
        
        return support_features, prototype_aware_query.squeeze(0)
    
    def forward(
        self,
        support_features: torch.Tensor,
        query_features: torch.Tensor,
        prototypes: Optional[torch.Tensor] = None,
        return_weights: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with adaptive fusion
        """
        # Apply adaptive fusion
        fused_query = self.adaptive_fusion(support_features, query_features, prototypes)
        
        result = {
            'fused_support': support_features,
            'fused_query': fused_query,
            'fusion_strategy': 'adaptive'
        }
        
        if return_weights:
            result['strategy_weights'] = F.softmax(self.strategy_weights, dim=0)
        
        return result
