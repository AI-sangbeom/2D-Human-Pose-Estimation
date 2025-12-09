import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GatedMultiHeadAttention(nn.Module):
    """
    Gated Multi-Head Attention from "Gated Attention for Large Language Models: 
    Non-linearity, Sparsity, and Attention-Sink-Free" (arXiv:2505.06708)
    
    핵심 아이디어:
    - SDPA (Scaled Dot-Product Attention) 출력 후에 sigmoid gate를 적용
    - 수식: Y' = Y ⊙ σ(XW_θ)
    - head-specific한 gating으로 attention sink 현상 완화
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        gate_type: str = "elementwise",  # "elementwise" or "headwise"
        bias: bool = True,
    ):
        """
        Args:
            d_model: 모델의 hidden dimension
            num_heads: attention head 수
            dropout: dropout 비율
            gate_type: "elementwise" (각 원소별) 또는 "headwise" (head별 단일 게이트)
            bias: linear layer에 bias 사용 여부
        """
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.gate_type = gate_type
        
        # Q, K, V projection layers
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        
        # Gate projection (핵심 추가 부분)
        # elementwise: 각 차원마다 독립적인 gate
        # headwise: 각 head마다 하나의 gate 값
        if gate_type == "elementwise":
            self.W_gate = nn.Linear(d_model, d_model, bias=bias)
        elif gate_type == "headwise":
            self.W_gate = nn.Linear(d_model, num_heads, bias=bias)
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}")
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_gate_scores: bool = False,
    ):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Attention mask (batch_size, seq_len, seq_len) or (batch_size, 1, seq_len, seq_len)
            return_gate_scores: gate 값도 반환할지 여부
            
        Returns:
            output: (batch_size, seq_len, d_model)
            gate_scores (optional): gate 값들
        """
        batch_size, seq_len, d_model = x.shape
        
        # 1. Linear projections for Q, K, V
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 2. Split into multiple heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Shape: (batch_size, num_heads, seq_len, d_k)
        
        # 3. Scaled Dot-Product Attention (SDPA)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # Add head dimension
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Attention output
        Y = torch.matmul(attn_weights, V)
        # Shape: (batch_size, num_heads, seq_len, d_k)
        
        # 4. 핵심: Gating mechanism (논문의 G1 position)
        # Compute gate scores: σ(XW_θ)
        gate_logits = self.W_gate(x)  # (batch_size, seq_len, d_model or num_heads)
        gate_scores = torch.sigmoid(gate_logits)
        
        if self.gate_type == "elementwise":
            # Element-wise gating: 각 차원마다 독립적인 gate
            gate_scores = gate_scores.view(batch_size, seq_len, self.num_heads, self.d_k)
            gate_scores = gate_scores.transpose(1, 2)
            # Shape: (batch_size, num_heads, seq_len, d_k)
            
            # Apply gate: Y' = Y ⊙ σ(XW_θ)
            Y_gated = Y * gate_scores
            
        elif self.gate_type == "headwise":
            # Head-wise gating: 각 head마다 하나의 gate 값
            gate_scores = gate_scores.view(batch_size, seq_len, self.num_heads, 1)
            gate_scores = gate_scores.transpose(1, 2)
            # Shape: (batch_size, num_heads, seq_len, 1)
            
            # Apply gate
            Y_gated = Y * gate_scores
        
        # 5. Concatenate heads
        Y_gated = Y_gated.transpose(1, 2).contiguous()
        # Shape: (batch_size, seq_len, num_heads, d_k)
        Y_gated = Y_gated.view(batch_size, seq_len, d_model)
        # Shape: (batch_size, seq_len, d_model)
        
        # 6. Final output projection
        output = self.W_o(Y_gated)
        output = self.dropout(output)
        
        if return_gate_scores:
            return output, gate_scores
        return output


# 사용 예시
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 2
    num_kpts = 17
    d_model = 768
    num_heads = 8
    
    # 입력 데이터
    x = torch.randn(batch_size, num_kpts, d_model)
    
    
    print("=" * 60)
    print("Gated Attention 구현 테스트")
    print("=" * 60)
    
    # Element-wise gating
    print("\n1. Element-wise Gating:")
    gated_attn_elem = GatedMultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
        gate_type="elementwise"
    )
    
    output_elem, gates_elem = gated_attn_elem(x, return_gate_scores=True)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output_elem.shape}")
    print(f"   Gate scores shape: {gates_elem.shape}")
    print(f"   Gate scores mean: {gates_elem.mean().item():.4f}")
    print(f"   Gate scores std: {gates_elem.std().item():.4f}")
    
    # Head-wise gating
    print("\n2. Head-wise Gating:")
    gated_attn_head = GatedMultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
        gate_type="headwise"
    )
    
    output_head, gates_head = gated_attn_head(x, return_gate_scores=True)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output_head.shape}")
    print(f"   Gate scores shape: {gates_head.shape}")
    print(f"   Gate scores mean: {gates_head.mean().item():.4f}")
    print(f"   Gate scores std: {gates_head.std().item():.4f}")
    
    # 파라미터 수 비교
    print("\n3. 파라미터 수:")
    params_elem = sum(p.numel() for p in gated_attn_elem.parameters())
    params_head = sum(p.numel() for p in gated_attn_head.parameters())
    print(f"   Element-wise gating: {params_elem:,} parameters")
    print(f"   Head-wise gating: {params_head:,} parameters")
    print(f"   추가 파라미터 (element-wise): {params_elem - params_head:,}")
    
    print("\n" + "=" * 60)
    print("주요 특징:")
    print("- Attention sink 현상 완화")
    print("- 학습 안정성 향상")
    print("- 더 큰 learning rate 사용 가능")
    print("- Long-context 성능 향상")
    print("=" * 60)