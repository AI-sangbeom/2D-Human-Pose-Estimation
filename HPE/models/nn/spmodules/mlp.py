import torch
import torch.nn as nn
import numpy.random as random
import spconv.pytorch as spconv

class DropPath(nn.Module):
    """ Drop Path for sparse tensors.
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x: spconv.SparseConvTensor):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # For spconv, create mask based on batch indices
        batch_indices = x.indices[:, 0]
        unique_batches = torch.unique(batch_indices)
        mask_list = []
        for b in unique_batches:
            coords_b = x.indices[batch_indices == b]
            len_b = len(coords_b)
            if random.uniform(0, 1) > self.drop_prob:
                mask_list.append(torch.ones(len_b))
            else:
                mask_list.append(torch.zeros(len_b))
        mask = torch.cat(mask_list).view(-1, 1).to(x.device)
        if keep_prob > 0.0 and self.scale_by_keep:
            mask.div_(keep_prob)
        new_features = x.features * mask
        return x.replace_feature(new_features)
    

class GELU(nn.Module):
    """
    Sparse GELU activation function for SparseConvTensor
    
    GELU(x) = x * Φ(x) where Φ(x) is the cumulative distribution function 
    of the standard Gaussian distribution
    """
    def __init__(self, approximate='none'):
        """
        Args:
            approximate (str): 'none', 'tanh' approximation method
                - 'none': exact GELU
                - 'tanh': faster approximation using tanh
        """
        super().__init__()
        self.approximate = approximate
        self.gelu = nn.GELU(approximate=approximate)
    
    def forward(self, x):
        """
        Args:
            x: spconv.SparseConvTensor
        
        Returns:
            spconv.SparseConvTensor with GELU applied to features
        """
        if isinstance(x, spconv.SparseConvTensor):
            # Sparse tensor의 features에만 GELU 적용
            new_features = self.gelu(x.features)
            
            # 새로운 SparseConvTensor 생성
            return x.replace_feature(new_features)
        else:
            # Dense tensor의 경우 일반 GELU 적용
            return self.gelu(x)


class FastGELU(nn.Module):
    """
    Fast approximation of GELU for sparse tensors
    GELU(x) ≈ x * σ(1.702 * x)
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """
        Args:
            x: spconv.SparseConvTensor
        
        Returns:
            spconv.SparseConvTensor with fast GELU applied
        """
        if isinstance(x, spconv.SparseConvTensor):
            # Fast GELU approximation: x * sigmoid(1.702 * x)
            new_features = x.features * torch.sigmoid(1.702 * x.features)
            return x.replace_feature(new_features)
        else:
            return x * torch.sigmoid(1.702 * x)


class QuickGELU(nn.Module):
    """
    Quick GELU approximation for sparse tensors
    Used in CLIP and some other models
    GELU(x) ≈ x * σ(1.702 * x)
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        if isinstance(x, spconv.SparseConvTensor):
            new_features = x.features * torch.sigmoid(1.702 * x.features)
            return x.replace_feature(new_features)
        else:
            return x * torch.sigmoid(1.702 * x)