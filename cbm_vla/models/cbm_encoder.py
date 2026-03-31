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
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


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


# ================================================================
# Phase 1 Loss Function
# ================================================================

def compute_encoder_alignment_loss(
    enhanced_features: torch.Tensor,
    target_text_embeddings: torch.Tensor,
    temperature: float = 0.07,
    pooling: str = "mean",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Phase 1 Encoder Alignment Loss (InfoNCE Contrastive)
    
    CBM Encoder의 출력이 image description text embedding과
    정렬되도록 InfoNCE contrastive loss를 계산합니다.
    
    Batch 내에서:
    - Positive pair: (image_i의 enhanced feature, image_i의 text embedding)
    - Negative pairs: (image_i의 enhanced feature, image_j의 text embedding)
    
    왜 이 loss가 필요한가:
        CBM Encoder는 SigLIP feature 위에 MLP residual을 추가하는 모듈입니다.
        이 MLP가 아무 방향으로나 학습되면 Phase 2/3에서 concept scoring이
        올바르게 작동하지 않습니다. Image description text embedding을
        teacher signal로 사용하여, MLP가 scene concept 방향의 정보를
        보강하도록 유도합니다.
    
    왜 MSE가 아닌 contrastive loss인가:
        MSE는 절대적 위치를 맞추므로 CBM Encoder의 자유도를 과도하게 제한합니다.
        Contrastive loss는 상대적 유사도 순서만 보존하면 되므로,
        원본 SigLIP feature에 concept-relevant information을 보강할 자유도가 유지됩니다.
    
    Args:
        enhanced_features: [batch, num_tokens, siglip_hidden_dim]
            CBM Encoder 출력 (H_siglip + scale * MLP(H_siglip))
        target_text_embeddings: [batch, siglip_hidden_dim]
            SigLIP text encoder로 인코딩한 image description embedding.
            Data collection에서 생성된 scene description을 인코딩한 결과.
        temperature: Contrastive loss temperature (낮을수록 hard negatives에 집중)
        pooling: "mean" or "cls" (token pooling 방식)
            
    Returns:
        loss: scalar — InfoNCE loss
        info: dict with individual metrics for logging
    """
    # ================================================================
    # 1. Pool enhanced features: [B, T, D] → [B, D]
    # ================================================================
    if pooling == "mean":
        visual_emb = enhanced_features.mean(dim=1)  # [B, D]
    elif pooling == "cls":
        visual_emb = enhanced_features[:, 0, :]  # [B, D]
    else:
        raise ValueError(f"Unknown pooling method: {pooling}")
    
    # ================================================================
    # 2. L2 normalize for cosine similarity
    # ================================================================
    visual_emb = F.normalize(visual_emb, p=2, dim=-1)           # [B, D]
    text_emb = F.normalize(target_text_embeddings, p=2, dim=-1) # [B, D]
    
    # ================================================================
    # 3. Cosine similarity matrix: [B, B]
    # ================================================================
    sim_matrix = torch.matmul(visual_emb, text_emb.t()) / temperature
    
    batch_size = sim_matrix.shape[0]
    labels = torch.arange(batch_size, device=sim_matrix.device)
    
    # ================================================================
    # 4. Symmetric InfoNCE loss
    # ================================================================
    # Visual → Text direction
    loss_v2t = F.cross_entropy(sim_matrix, labels)
    # Text → Visual direction
    loss_t2v = F.cross_entropy(sim_matrix.t(), labels)
    # Average
    loss = (loss_v2t + loss_t2v) / 2.0
    
    # ================================================================
    # 5. Info dict for logging
    # ================================================================
    with torch.no_grad():
        v2t_acc = (sim_matrix.argmax(dim=1) == labels).float().mean().item()
        t2v_acc = (sim_matrix.t().argmax(dim=1) == labels).float().mean().item()
        pos_sim = (visual_emb * text_emb).sum(dim=-1).mean().item()
    
    info = {
        "loss_encoder_alignment": loss.item(),
        "encoder_v2t_acc": v2t_acc,
        "encoder_t2v_acc": t2v_acc,
        "encoder_pos_similarity": pos_sim,
    }
    
    return loss, info


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
    
    # Test alignment loss
    print("\nTesting compute_encoder_alignment_loss...")
    batch_size = 8
    encoder2 = CBMEncoder(siglip_hidden_dim=768, mlp_hidden_dim=256)
    
    siglip_feat = torch.randn(batch_size, 128, 768)
    text_emb = torch.randn(batch_size, 768)
    
    enhanced = encoder2(siglip_feat)
    loss, info = compute_encoder_alignment_loss(enhanced, text_emb)
    
    print(f"Alignment loss: {loss.item():.4f}")
    for k, v in info.items():
        print(f"  {k}: {v:.4f}")
    
    # Gradient check
    loss.backward()
    grad_norm = sum(
        p.grad.norm().item() for p in encoder2.parameters() if p.grad is not None
    )
    print(f"Total gradient norm: {grad_norm:.4f}")
    
    print("\n✓ CBMEncoder + alignment loss test completed!")