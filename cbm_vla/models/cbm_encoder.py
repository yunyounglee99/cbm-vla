"""
Module 1: CBM Encoder
======================

SigLIP vision encoder의 출력에 MLP residual을 추가하여
scene concept 방향의 정보를 보강합니다.

Architecture:
    H_output = H_siglip + MLP(H_siglip)
    MLP: Linear(siglip_dim, hidden) → ReLU → Linear(hidden, siglip_dim)

Design Decision (이전 대화에서 확정):
    - LoRA가 아닌 MLP residual: SigLIP 내부 weight를 수정하지 않고
      출력 위에 정보를 추가하는 방식. SigLIP은 완전 freeze.
    - DICAN의 low-rank residual projector(262K params)와 동일 원리.
    - Residual connection이므로 MLP가 학습되지 않아도 원본이 그대로 통과.

Params: 768 × 256 + 256 × 768 = ~393K (siglip_dim=768, hidden=256)
"""

import torch
import torch.nn as nn
from typing import Optional


class CBMEncoder(nn.Module):
    """
    CBM Encoder: MLP Residual on SigLIP Output
    
    SigLIP의 출력 feature에 scene concept 방향 정보를 보강합니다.
    Phase 1에서 image description concept으로 학습됩니다.
    
    Args:
        siglip_hidden_dim: SigLIP 출력 차원 (SigLIP-B/16 = 768)
        mlp_hidden_dim: MLP 중간 차원 (기본 256)
        dropout: Dropout 비율
    """
    
    def __init__(
        self,
        siglip_hidden_dim: int = 768,
        mlp_hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.siglip_hidden_dim = siglip_hidden_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        
        # MLP: siglip_dim → hidden → siglip_dim
        self.mlp = nn.Sequential(
            nn.Linear(siglip_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, siglip_hidden_dim),
            nn.Dropout(dropout),
        )
        
        # Layer normalization on the residual output
        self.norm = nn.LayerNorm(siglip_hidden_dim)
        
        # Learnable scaling factor for residual (initialized small)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize MLP weights"""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        siglip_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: H_output = H_siglip + scale * MLP(H_siglip)
        
        Args:
            siglip_features: [batch, num_tokens, siglip_hidden_dim]
                SigLIP vision encoder의 출력 (freeze된 상태)
                예: [B, 128, 768] (2 cameras × 64 tokens)
                
        Returns:
            enhanced_features: [batch, num_tokens, siglip_hidden_dim]
                Scene concept 방향으로 보강된 feature
        """
        # MLP residual with learnable scaling
        residual = self.mlp(siglip_features)
        output = siglip_features + self.residual_scale * residual
        output = self.norm(output)
        
        return output
    
    def get_num_params(self) -> int:
        """Return total parameter count"""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    print("Testing CBMEncoder...")
    
    encoder = CBMEncoder(siglip_hidden_dim=768, mlp_hidden_dim=256)
    print(f"Parameters: {encoder.get_num_params():,}")
    
    # Test forward
    x = torch.randn(2, 128, 768)  # [batch=2, tokens=128, dim=768]
    y = encoder(x)
    print(f"Input: {x.shape} → Output: {y.shape}")
    
    # Verify residual: if MLP outputs zero, output equals input
    with torch.no_grad():
        for m in encoder.mlp:
            if isinstance(m, nn.Linear):
                m.weight.zero_()
                m.bias.zero_()
        y_zero = encoder(x)
        diff = (y_zero - encoder.norm(x)).abs().max()
        print(f"Zero MLP residual check: max diff = {diff:.6f}")
    
    print("✓ CBMEncoder test completed!")