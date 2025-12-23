import torch
import torch.nn as nn

class GaussianPooling(nn.Module):
    """Gaussian weighted pooling for local features"""
    def __init__(self, kernel_size: int = 5, sigma: float = 2.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        
        # Create Gaussian kernel
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        self.register_buffer('kernel', kernel)
        
    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Create 2D Gaussian kernel"""
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0)  # (1, 1, k, k)
    
    def forward(self, feature_map: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        """
        Extract Gaussian-weighted features around keypoints
        
        Args:
            feature_map: (C, H, W) feature map
            keypoints: (N, 2) keypoint coordinates [x, y]
            
        Returns:
            pooled_features: (N, C) features for each keypoint
        """
        device = feature_map.device
        C, H, W = feature_map.shape
        N = keypoints.shape[0]
        half_k = self.kernel_size // 2
        
        pooled_features = []
        
        for kpt in keypoints:
            x, y = kpt[0].long(), kpt[1].long()
            
            # Boundary check and padding
            x = torch.clamp(x, half_k, W - half_k - 1)
            y = torch.clamp(y, half_k, H - half_k - 1)
            
            # Extract local patch
            patch = feature_map[:, 
                              y - half_k:y + half_k + 1,
                              x - half_k:x + half_k + 1]  # (C, k, k)
            
            # Apply Gaussian weighting
            if patch.shape[1:] == (self.kernel_size, self.kernel_size):
                weighted = patch * self.kernel  # (C, k, k)
                pooled = weighted.sum(dim=(-2, -1))  # (C,)
            else:
                # Fallback to center pixel if patch size mismatch
                pooled = feature_map[:, :, y, x]
            
            pooled_features.append(pooled)
        
        return torch.stack(pooled_features).to(device)  # (N, C)