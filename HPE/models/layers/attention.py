import torch 
import torch.nn as nn
from typing import Optional, Tuple

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


class GatedAttention(nn.Module):
    """
    Gated Multi-head Self Attention
    Based on "Gated Attention for Large Language Models" (arXiv:2505.06708)
    
    Applies sigmoid gate after SDPA: Y' = Y ⊙ σ(XW_θ)
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., gate_type="headwise"):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.gate_type = gate_type
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Gate projection (핵심 추가)
        if gate_type == "elementwise":
            self.W_gate = nn.Linear(dim, dim, bias=qkv_bias)
        elif gate_type == "headwise":
            self.W_gate = nn.Linear(dim, num_heads, bias=qkv_bias)
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}")
    
    def feature_forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Standard SDPA
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, 1, N, N)
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        Y = (attn @ v)  # (B, num_heads, N, head_dim)
        
        # Gating mechanism: Y' = Y ⊙ σ(XW_θ)
        gate_logits = self.W_gate(x)  # (B, N, dim or num_heads)
        gate_scores = torch.sigmoid(gate_logits)
        
        if self.gate_type == "elementwise":
            # Element-wise: (B, N, dim) -> (B, num_heads, N, head_dim)
            gate_scores = gate_scores.view(B, N, self.num_heads, C // self.num_heads)
            gate_scores = gate_scores.transpose(1, 2)
        elif self.gate_type == "headwise":
            # Head-wise: (B, N, num_heads) -> (B, num_heads, N, 1)
            gate_scores = gate_scores.view(B, N, self.num_heads, 1)
            gate_scores = gate_scores.transpose(1, 2)
        
        # Apply gate
        Y_gated = Y * gate_scores
        
        # Reshape and project
        x = Y_gated.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, gate_scores

    def forward(
            self, 
            x: torch.Tensor, 
            mask: Optional[torch.Tensor] = None, 
            return_gate_scores: bool = False
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        x, gate_scores = self.feature_forward(x, mask)
        if return_gate_scores:
            return x, gate_scores
        else:
            return x
        

        
# 테스트 코드
if __name__ == "__main__":
    print("=== Gated Attention Test ===\n")
    
    # 설정
    B, N, C = 2, 10, 64
    num_heads = 8
    
    # 1. Headwise gate 테스트
    print("1. Headwise Gate Test")
    model = GatedAttention(dim=C, num_heads=num_heads, gate_type="headwise")
    x = torch.randn(B, N, C)
    
    output = model(x)
    print(f"   Output shape: {output.shape}")
    assert output.shape == (B, N, C), "Output shape mismatch!"
    
    output, gate_scores = model(x, return_gate_scores=True)
    print(f"   Gate scores shape: {gate_scores.shape}")
    assert gate_scores.shape == (B, num_heads, N, 1), "Gate scores shape mismatch!"
    print("   ✓ Passed\n")
    
    # 2. Elementwise gate 테스트
    print("2. Elementwise Gate Test")
    model_elem = GatedAttention(dim=C, num_heads=num_heads, gate_type="elementwise")
    output, gate_scores = model_elem(x, return_gate_scores=True)
    print(f"   Output shape: {output.shape}")
    print(f"   Gate scores shape: {gate_scores.shape}")
    assert gate_scores.shape == (B, num_heads, N, C // num_heads), "Gate scores shape mismatch!"
    print("   ✓ Passed\n")
    
    # 3. Mask 테스트
    print("3. Mask Test")
    mask = torch.ones(B, N, N)
    mask[:, :, 5:] = 0
    output = model(x, mask=mask)
    print(f"   Masked output shape: {output.shape}")
    print("   ✓ Passed\n")
    
    # 4. Gradient flow 테스트
    print("4. Gradient Flow Test")
    output, gate_scores = model(x, return_gate_scores=True)
    loss = output.sum() + gate_scores.sum()
    loss.backward()
    print(f"   W_gate gradient norm: {model.W_gate.weight.grad.norm().item():.4f}")
    print("   ✓ Passed\n")
    
    print("All tests passed! ✓")