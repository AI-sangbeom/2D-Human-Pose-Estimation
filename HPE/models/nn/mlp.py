import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy.random as random
import spconv.pytorch as spconv

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
    

class SpDropPath(nn.Module):
    """ Drop Path for sparse tensors.
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(SpDropPath, self).__init__()
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
    

class SparseGELU(nn.Module):
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


class SparseFastGELU(nn.Module):
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


class SparseQuickGELU(nn.Module):
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