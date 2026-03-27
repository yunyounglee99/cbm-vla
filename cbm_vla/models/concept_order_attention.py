"""
Module 4: Concept Order Attention
===================================

Cross-attention으로 추출된 concept embedding들의 실행 순서를 학습합니다.
Causal self-attention으로 concept 간 순차적 의존성을 모델링하고,
Order head로 각 concept의 실행 순서 score를 출력합니다.

Architecture:
    Input: concept_embeddings [batch, n, 128]
    → Causal Self-Attention (concept i는 concept 0..i만 참조)
    → Order Head (Linear 128→1, pairwise ranking loss)
    Output: ordered concept embeddings + order scores

Design Decisions:
    - Ground truth order: segmentation 순서 (data collection에서 확보)
    - Causal mask: concept i가 concept 0..i만 attend → 선행 개념에만 의존
    - Phase 3에서 cross-attention과 함께 학습 (task loss 역전파)

Params: QKV(128×128×3) + Out(128×128) + FFN(128×512×2) + OrderHead(128×1)
        = ~196K
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple


class ConceptOrderAttention(nn.Module):
    """
    Causal Self-Attention for Concept Order Learning
    
    Concept들을 ground truth 순서로 배열한 뒤 causal mask를 적용하여
    순차적 의존성을 학습합니다.
    
    Args:
        concept_embed_dim: Concept embedding 차원 (128)
        num_heads: Attention head 수
        ff_dim: Feed-forward 중간 차원
        dropout: Dropout 비율
    """
    
    def __init__(
        self,
        concept_embed_dim: int = 128,
        num_heads: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.concept_embed_dim = concept_embed_dim
        self.num_heads = num_heads
        self.head_dim = concept_embed_dim // num_heads
        assert self.head_dim * num_heads == concept_embed_dim
        
        # Causal Self-Attention
        self.qkv_proj = nn.Linear(concept_embed_dim, 3 * concept_embed_dim)
        self.out_proj = nn.Linear(concept_embed_dim, concept_embed_dim)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(concept_embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, concept_embed_dim),
            nn.Dropout(dropout),
        )
        
        # Layer norms (pre-norm)
        self.norm1 = nn.LayerNorm(concept_embed_dim)
        self.norm2 = nn.LayerNorm(concept_embed_dim)
        
        # Order Head: concept_embed_dim → 1 (order score)
        self.order_head = nn.Linear(concept_embed_dim, 1)
        
        # Positional encoding for concept positions
        self.pos_encoding = nn.Parameter(
            torch.randn(1, 20, concept_embed_dim) * 0.02  # max 20 concepts
        )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
        # Causal mask buffer
        self.register_buffer("causal_mask", None, persistent=False)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.zeros_(self.qkv_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.xavier_uniform_(self.order_head.weight, gain=0.1)
        nn.init.zeros_(self.order_head.bias)
    
    def _get_causal_mask(self, n: int, device: torch.device) -> torch.Tensor:
        """Lower triangular causal mask [n, n]"""
        if self.causal_mask is None or self.causal_mask.shape[0] < n:
            self.causal_mask = torch.tril(torch.ones(n, n, device=device))
        return self.causal_mask[:n, :n]
    
    def forward(
        self,
        concept_embeddings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            concept_embeddings: [batch, n_concepts, concept_embed_dim]
                Cross-attention에서 추출된 concept embedding
                Ground truth 순서로 정렬되어 있어야 함 (학습 시)
                
        Returns:
            dict:
                "ordered_embeddings": [batch, n_concepts, concept_embed_dim]
                    순서 정보가 인코딩된 concept embedding
                "order_scores": [batch, n_concepts]
                    각 concept의 실행 순서 score (낮을수록 먼저)
        """
        batch_size, n_concepts, _ = concept_embeddings.shape
        
        # Add positional encoding
        x = concept_embeddings + self.pos_encoding[:, :n_concepts, :]
        
        # ================================================================
        # Causal Self-Attention (pre-norm)
        # ================================================================
        residual = x
        x = self.norm1(x)
        
        # QKV projection
        qkv = self.qkv_proj(x)  # [B, n, 3*D]
        qkv = qkv.view(batch_size, n_concepts, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, n, head_dim]
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # Attention scores with causal mask
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # [B, heads, n, n]
        
        causal_mask = self._get_causal_mask(n_concepts, x.device)
        scores = scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf')
        )
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, V)
        # [B, heads, n, head_dim] → [B, n, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, n_concepts, self.concept_embed_dim
        )
        
        attn_output = self.out_proj(attn_output)
        x = residual + self.dropout(attn_output)
        
        # ================================================================
        # Feed-Forward (pre-norm)
        # ================================================================
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        ordered_embeddings = x
        
        # ================================================================
        # Order Head
        # ================================================================
        order_scores = self.order_head(ordered_embeddings).squeeze(-1)
        # [B, n_concepts]
        
        return {
            "ordered_embeddings": ordered_embeddings,
            "order_scores": order_scores,
        }
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def compute_order_loss(
    order_scores: torch.Tensor,
    gt_order: torch.Tensor,
) -> torch.Tensor:
    """
    Pairwise Ranking Loss for concept order
    
    Ground truth 순서에 따라 concept i가 concept j보다 먼저 실행되어야 하면
    order_score[i] < order_score[j]가 되도록 학습합니다.
    
    Args:
        order_scores: [batch, n_concepts] — predicted order scores
        gt_order: [batch, n_concepts] — ground truth order (0, 1, 2, ...)
            값이 작을수록 먼저 실행
            
    Returns:
        loss: scalar
    """
    batch_size, n = order_scores.shape
    
    if n <= 1:
        return torch.tensor(0.0, device=order_scores.device)
    
    # Pairwise: for all i < j in gt_order, we want score[i] < score[j]
    # Use margin ranking loss
    total_loss = torch.tensor(0.0, device=order_scores.device)
    num_pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            # gt_order[i] < gt_order[j] → score[i] should be < score[j]
            # target = 1 means first input should be ranked higher (lower score)
            gt_diff = gt_order[:, j] - gt_order[:, i]  # positive if i before j
            target = torch.sign(gt_diff)  # +1 or -1
            
            pair_loss = F.margin_ranking_loss(
                order_scores[:, j],   # should be larger (later)
                order_scores[:, i],   # should be smaller (earlier)
                target,
                margin=0.5,
            )
            total_loss = total_loss + pair_loss
            num_pairs += 1
    
    if num_pairs > 0:
        total_loss = total_loss / num_pairs
    
    return total_loss


if __name__ == "__main__":
    print("Testing ConceptOrderAttention...")
    
    batch_size = 4
    n_concepts = 5
    embed_dim = 128
    
    module = ConceptOrderAttention(concept_embed_dim=embed_dim)
    print(f"Parameters: {module.get_num_params():,}")
    
    # Forward test
    concept_emb = torch.randn(batch_size, n_concepts, embed_dim)
    result = module(concept_emb)
    
    print(f"ordered_embeddings: {result['ordered_embeddings'].shape}")
    print(f"order_scores: {result['order_scores'].shape}")
    print(f"order_scores[0]: {result['order_scores'][0].tolist()}")
    
    # Order loss test
    gt_order = torch.arange(n_concepts).unsqueeze(0).expand(batch_size, -1).float()
    loss = compute_order_loss(result["order_scores"], gt_order)
    print(f"\nOrder loss: {loss.item():.4f}")
    
    # Verify causal mask: concept 0 should have same embedding regardless of later concepts
    with torch.no_grad():
        # Full sequence
        full_result = module(concept_emb)
        # Only first 3
        partial_result = module(concept_emb[:, :3])
        
        # concept 0's embedding should be identical
        diff = (full_result["ordered_embeddings"][:, 0] - 
                partial_result["ordered_embeddings"][:, 0]).abs().max()
        print(f"Causal mask check (diff for concept 0): {diff:.6f}")
        assert diff < 1e-4, "Causal mask not working!"
    
    print("\n✓ ConceptOrderAttention test completed!")