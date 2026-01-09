import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class PrototypicalHead(nn.Module):
    """
    Prototypical head for few-shot keypoint detection
    Computes prototypes from support examples and performs distance-based prediction
    """
    
    def __init__(
        self,
        nkpts: int,
        embed_dim: int,
        n_way: int = 5,
        temperature: float = 1.0,
        use_prototype_relu: bool = True
    ):
        super().__init__()
        
        self.nkpts = nkpts
        self.embed_dim = embed_dim
        self.n_way = n_way
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.use_prototype_relu = use_prototype_relu
        
        # Prototype computation layers
        self.prototype_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True) if use_prototype_relu else nn.Identity(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Distance computation
        self.distance_fn = nn.CosineSimilarity(dim=-1)
        
        # Keypoint offset prediction head
        self.offset_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 2)  # x, y offsets
        )
        
        # Confidence prediction
        self.confidence_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def compute_prototypes(
        self, 
        support_features: torch.Tensor, 
        support_keypoints: torch.Tensor,
        n_way: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute prototypes from support examples
        
        Args:
            support_features: Features from support examples [N_support, embed_dim]
            support_keypoints: Keypoint coordinates [N_support, nkpts, 2]
            n_way: Number of classes (default: self.n_way)
            
        Returns:
            prototypes: Computed prototypes [n_way, nkpts, embed_dim]
        """
        if n_way is None:
            n_way = self.n_way
            
        batch_size, n_support, embed_dim = support_features.shape
        
        # Reshape for processing
        support_features = support_features.view(-1, embed_dim)  # [N_support, embed_dim]
        support_keypoints = support_keypoints.view(-1, self.nkpts, 2)  # [N_support, nkpts, 2]
        
        # Apply prototype network
        processed_features = self.prototype_net(support_features)  # [N_support, embed_dim]
        
        # Reshape back to batch format
        processed_features = processed_features.view(batch_size, n_support, embed_dim)
        
        # Compute prototypes: mean pooling across support examples per class
        # Assuming equal number of support examples per class
        n_support_per_class = n_support // n_way
        prototypes = torch.zeros(n_way, self.nkpts, embed_dim, device=support_features.device)
        
        for way_idx in range(n_way):
            start_idx = way_idx * n_support_per_class
            end_idx = (way_idx + 1) * n_support_per_class
            
            class_features = processed_features[:, start_idx:end_idx, :]  # [batch_size, n_support_per_class, embed_dim]
            class_keypoints = support_keypoints[start_idx:end_idx, :, :]  # [n_support_per_class, nkpts, 2]
            
            # Average features for each keypoint across support examples
            for kpt_idx in range(self.nkpts):
                # Get all examples for this keypoint
                kpt_features = class_features[:, :, :]  # [batch_size, n_support_per_class, embed_dim]
                
                # Simple mean pooling (can be improved with attention)
                prototype = torch.mean(kpt_features, dim=1)  # [batch_size, embed_dim]
                prototypes[way_idx, kpt_idx] = prototype
        
        return prototypes  # [n_way, nkpts, embed_dim]
    
    def compute_distances(
        self, 
        query_features: torch.Tensor, 
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distances between query features and prototypes
        
        Args:
            query_features: Query features [N_query, embed_dim]
            prototypes: Prototypes [n_way, nkpts, embed_dim]
            
        Returns:
            distances: Distance scores [N_query, n_way, nkpts]
        """
        n_query = query_features.shape[0]
        distances = torch.zeros(n_query, self.n_way, self.nkpts, device=query_features.device)
        
        for way_idx in range(self.n_way):
            for kpt_idx in range(self.nkpts):
                # Compute cosine similarity
                sim = self.distance_fn(
                    query_features.unsqueeze(0),  # [1, N_query, embed_dim]
                    prototypes[way_idx, kpt_idx].unsqueeze(0)  # [1, nkpts, embed_dim]
                )  # [1, N_query]
                
                # Convert similarity to distance (higher similarity = lower distance)
                distances[:, way_idx, kpt_idx] = 1.0 - sim.squeeze(0)
        
        # Apply temperature scaling
        distances = distances / self.temperature
        
        return distances
    
    def predict_keypoints(
        self, 
        query_features: torch.Tensor,
        prototypes: torch.Tensor,
        image_coords: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict keypoints for query examples
        
        Args:
            query_features: Query features [N_query, embed_dim]
            prototypes: Prototypes [n_way, nkpts, embed_dim]
            image_coords: Image coordinates [N_query, 2] (optional)
            
        Returns:
            Dict containing:
                - keypoints: Predicted keypoint coordinates [N_query, nkpts, 2]
                - confidence: Confidence scores [N_query, nkpts]
                - distances: Distance scores [N_query, n_way, nkpts]
                - class_predictions: Predicted class for each query [N_query]
        """
        n_query = query_features.shape[0]
        
        # Compute distances
        distances = self.compute_distances(query_features, prototypes)
        
        # Predict class based on minimum average distance
        avg_distances = torch.mean(distances, dim=2)  # [N_query, n_way]
        class_predictions = torch.argmin(avg_distances, dim=1)  # [N_query]
        
        # Use predicted class for keypoint prediction
        keypoints = torch.zeros(n_query, self.nkpts, 2, device=query_features.device)
        confidence = torch.zeros(n_query, self.nkpts, device=query_features.device)
        
        for query_idx in range(n_query):
            pred_class = class_predictions[query_idx].item()
            
            for kpt_idx in range(self.nkpts):
                # Use the predicted class prototype for this keypoint
                kpt_prototype = prototypes[pred_class, kpt_idx]  # [embed_dim]
                
                # Compute attention weights
                attention_weights = F.softmax(
                    -distances[query_idx, pred_class, kpt_idx].unsqueeze(0), dim=0
                )
                
                # Predict offset from query feature to prototype
                offset = self.offset_head(query_features[query_idx])  # [2]
                confidence_score = self.confidence_head(query_features[query_idx])  # [1]
                
                # Combine offset with attention
                keypoints[query_idx, kpt_idx] = offset * attention_weights
                confidence[query_idx, kpt_idx] = confidence_score.squeeze(0)
        
        # If image coordinates provided, normalize keypoints
        if image_coords is not None:
            # Normalize keypoints to image coordinate space
            keypoints = torch.sigmoid(keypoints)  # Convert to [0, 1] range
            keypoints = keypoints * image_coords.unsqueeze(1)  # Scale to image size
        
        return {
            'keypoints': keypoints,
            'confidence': confidence,
            'distances': distances,
            'class_predictions': class_predictions
        }
    
    def forward(
        self, 
        support_features: torch.Tensor, 
        support_keypoints: torch.Tensor,
        query_features: torch.Tensor,
        image_coords: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through prototypical head
        
        Args:
            support_features: Support features [batch_size, n_support, embed_dim]
            support_keypoints: Support keypoints [n_support, nkpts, 2]
            query_features: Query features [n_query, embed_dim]
            image_coords: Image coordinates [n_query, 2] (optional)
            
        Returns:
            Dict containing predictions and intermediate results
        """
        # Compute prototypes from support examples
        prototypes = self.compute_prototypes(support_features, support_keypoints)
        
        # Predict keypoints for query examples
        predictions = self.predict_keypoints(query_features, prototypes, image_coords)
        
        # Add prototypes to output for debugging/analysis
        predictions['prototypes'] = prototypes
        
        return predictions


class MultiPrototypicalHead(PrototypicalHead):
    """
    Extended prototypical head with multiple prototype layers
    for hierarchical few-shot learning
    """
    
    def __init__(
        self,
        nkpts: int,
        embed_dim: int,
        n_way: int = 5,
        num_prototype_layers: int = 3,
        temperature: float = 1.0
    ):
        super().__init__(nkpts, embed_dim, n_way, temperature)
        
        self.num_prototype_layers = num_prototype_layers
        
        # Multiple prototype computation layers
        self.prototype_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(num_prototype_layers)
        ])
        
        # Layer-wise attention
        self.layer_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            batch_first=True
        )
    
    def compute_hierarchical_prototypes(
        self,
        support_features: torch.Tensor,
        support_keypoints: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Compute hierarchical prototypes at multiple levels
        
        Args:
            support_features: Support features [batch_size, n_support, embed_dim]
            support_keypoints: Support keypoints [n_support, nkpts, 2]
            
        Returns:
            List of prototype tensors at different levels
        """
        prototypes_list = []
        
        for layer_idx in range(self.num_prototype_layers):
            # Use different prototype layer
            original_prototype_net = self.prototype_net
            self.prototype_net = self.prototype_layers[layer_idx]
            
            # Compute prototypes
            prototypes = self.compute_prototypes(support_features, support_keypoints)
            prototypes_list.append(prototypes)
            
            # Restore original prototype net
            self.prototype_net = original_prototype_net
        
        return prototypes_list
    
    def hierarchical_predict(
        self,
        query_features: torch.Tensor,
        prototypes_list: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions using hierarchical prototypes
        
        Args:
            query_features: Query features [N_query, embed_dim]
            prototypes_list: List of prototype tensors
            
        Returns:
            Dict containing hierarchical predictions
        """
        all_predictions = []
        
        for prototypes in prototypes_list:
            pred = self.predict_keypoints(query_features, prototypes)
            all_predictions.append(pred)
        
        # Combine predictions from different levels
        # Simple averaging (can be improved with learned fusion)
        combined_keypoints = torch.stack([p['keypoints'] for p in all_predictions], dim=0)
        combined_keypoints = torch.mean(combined_keypoints, dim=0)
        
        combined_confidence = torch.stack([p['confidence'] for p in all_predictions], dim=0)
        combined_confidence = torch.mean(combined_confidence, dim=0)
        
        return {
            'keypoints': combined_keypoints,
            'confidence': combined_confidence,
            'hierarchical_predictions': all_predictions,
            'prototypes_list': prototypes_list
        }
    
    def forward(
        self,
        support_features: torch.Tensor,
        support_keypoints: torch.Tensor,
        query_features: torch.Tensor,
        image_coords: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with hierarchical prototypes
        """
        # Compute hierarchical prototypes
        prototypes_list = self.compute_hierarchical_prototypes(
            support_features, support_keypoints
        )
        
        # Make hierarchical predictions
        predictions = self.hierarchical_predict(query_features, prototypes_list)
        
        return predictions
