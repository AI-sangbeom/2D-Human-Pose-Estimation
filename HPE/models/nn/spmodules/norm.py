import torch
import torch.nn as nn
import spconv.pytorch as spconv

class GRN(nn.Module):
    def __init__(self, dim):
        super(GRN, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x: spconv.SparseConvTensor):
        features = x.features 
        # dim=0: 모든 복셀(N)에 대해 통계 집계 (Global Context)
        Gx = torch.norm(features, p=2, dim=0, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        out_features = self.gamma * (features * Nx) + self.beta + features
        return x.replace_feature(out_features)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, input: spconv.SparseConvTensor):
        output_features = self.ln(input.features)
        return input.replace_feature(output_features)
            