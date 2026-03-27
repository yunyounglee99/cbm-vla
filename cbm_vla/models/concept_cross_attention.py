"""
Module 3: Concept Cross-Attention
===================================

선택된 concept의 text embedding을 Query로,
backbone output (139×960)을 Key/Value로 사용하여
각 concept에 대한 풍부한 embedding을 추출합니다.

Architecture:
    Query: selected concept text embeddings [n, concept_text_dim]
    Key/Value: backbone output [139, vlm_hidden_dim]
    Output: n × k tokens → aggregation → n × concept_embed_dim

    k = num_sub_queries (하이퍼파라미터, 기본 4)
    → concept당 k개의 sub-aspect를 추출한 후 aggregation

Design Decisions:
    - Scoring module(Phase 2)과 역할 분리: scoring은 "선택", cross-attn은 "번역"
    - Scoring module이 freeze된 Phase 3에서 backbone LoRA와 함께 학습
    - Score matrix를 attention bias로 활용 가능 (concept-guided attention)

Params: ~2.7M (Q_proj + K_proj + V_proj + Out_proj + aggregation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple


class ConceptCrossAttention(nn.Module):
    """
    Concept Cross-Attention Module
    
    선택된 concept들에 대해 backbone output에서 관련 정보를 추출하여
    concept embedding을 생성합니다.
    
    Args:
        vlm_hidden_dim: Backbone output 차원 (960)
        concept_text_dim: Concept text embedding 차원
            SigLIP text encoder: 768, 또는 별도 encoder 사용 시 다를 수 있음
        concept_embed_dim: 출력 concept embedding 차원 (128)
        num_heads: Attention head 수
        num_sub_queries: Concept당 sub-query 수 (k)
            각 concept의 여러 sub-aspect를 추출
        dropout: Dropout 비율
    """
    
    def __init__(
        self,
        vlm_hidden_dim: int = 960,
        concept_text_dim: int = 768,
        concept_embed_dim: int = 128,
        num_heads: int = 8,
        num_sub_queries: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vlm_hidden_dim = vlm_hidden_dim
        self.concept_text_dim = concept_text_dim
        self.concept_embed_dim = concept_embed_dim
        self.num_heads = num_heads
        self.num_sub_queries = num_sub_queries
        
        # Internal attention dimension (must be divisible by num_heads)
        self.attn_dim = vlm_hidden_dim  # Use vlm_dim as attention space
        self.head_dim = self.attn_dim // num_heads
        assert self.head_dim * num_heads == self.attn_dim
        
        # Query projection: concept_text_dim → attn_dim
        # concept text embedding을 attention 공간으로 projection
        self.q_proj = nn.Linear(concept_text_dim, self.attn_dim)
        
        # Key/Value projections: vlm_dim → attn_dim
        self.k_proj = nn.Linear(vlm_hidden_dim, self.attn_dim)
        self.v_proj = nn.Linear(vlm_hidden_dim, self.attn_dim)
        
        # Output projection: attn_dim → concept_embed_dim
        self.out_proj = nn.Linear(self.attn_dim, concept_embed_dim)
        
        # Sub-query expansion: concept_text_dim → k × concept_text_dim
        # 각 concept에서 k개의 sub-query를 생성
        if num_sub_queries > 1:
            self.sub_query_proj = nn.Linear(
                concept_text_dim, 
                num_sub_queries * concept_text_dim,
            )
        
        # Aggregation: k sub-query outputs → 1 embedding per concept
        if num_sub_queries > 1:
            self.aggregation = nn.Sequential(
                nn.Linear(num_sub_queries * concept_embed_dim, concept_embed_dim),
                nn.GELU(),
            )
        
        # Layer normalization
        self.q_norm = nn.LayerNorm(self.attn_dim)
        self.out_norm = nn.LayerNorm(concept_embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if self.num_sub_queries > 1:
            nn.init.xavier_uniform_(self.sub_query_proj.weight, gain=0.5)
    
    def forward(
        self,
        concept_text_embeddings: torch.Tensor,
        backbone_output: torch.Tensor,
        concept_score_bias: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: concept embedding 추출
        
        Args:
            concept_text_embeddings: [batch, n_selected, concept_text_dim]
                선택된 n개 concept의 text embedding
            backbone_output: [batch, num_tokens, vlm_hidden_dim]
                예: [B, 139, 960]
            concept_score_bias: [batch, n_selected, num_tokens] (optional)
                Scoring module의 token-level concept score를 attention bias로 활용
                → concept-guided attention: 관련 높은 token에 바로 집중
            attention_mask: [batch, num_tokens] (optional)
                Padding mask
                
        Returns:
            dict:
                "concept_embeddings": [batch, n_selected, concept_embed_dim]
                    각 concept의 최종 embedding (PAM → Action Expert에 전달)
                "attention_weights": [batch, n_selected, num_tokens] (optional)
                    Attention weight (해석 가능성 분석용)
        """
        batch_size = concept_text_embeddings.shape[0]
        n_selected = concept_text_embeddings.shape[1]
        num_tokens = backbone_output.shape[1]
        
        # ================================================================
        # 1. Sub-query expansion (optional)
        # ================================================================
        if self.num_sub_queries > 1:
            # [B, n, text_dim] → [B, n, k*text_dim] → [B, n*k, text_dim]
            sub_queries = self.sub_query_proj(concept_text_embeddings)
            sub_queries = sub_queries.view(
                batch_size, n_selected * self.num_sub_queries, self.concept_text_dim
            )
        else:
            sub_queries = concept_text_embeddings
        
        n_queries = sub_queries.shape[1]  # n_selected * k (or n_selected if k=1)
        
        # ================================================================
        # 2. Q, K, V projections
        # ================================================================
        Q = self.q_norm(self.q_proj(sub_queries))  # [B, n_queries, attn_dim]
        K = self.k_proj(backbone_output)            # [B, num_tokens, attn_dim]
        V = self.v_proj(backbone_output)            # [B, num_tokens, attn_dim]
        
        # ================================================================
        # 3. Multi-head attention
        # ================================================================
        # Reshape for multi-head: [B, seq, D] → [B, heads, seq, head_dim]
        Q = Q.view(batch_size, n_queries, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores: [B, heads, n_queries, num_tokens]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Concept-guided attention bias (optional)
        if concept_score_bias is not None:
            # concept_score_bias: [B, n_selected, num_tokens]
            # Expand for sub-queries: [B, n_selected, T] → [B, n_selected*k, T]
            if self.num_sub_queries > 1:
                bias = concept_score_bias.repeat_interleave(
                    self.num_sub_queries, dim=1
                )
            else:
                bias = concept_score_bias
            # [B, n_queries, T] → [B, 1, n_queries, T] (broadcast over heads)
            attn_scores = attn_scores + bias.unsqueeze(1)
        
        # Attention mask (padding)
        if attention_mask is not None:
            # [B, T] → [B, 1, 1, T]
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax + dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention: [B, heads, n_queries, head_dim]
        attn_output = torch.matmul(attn_weights, V)
        
        # Merge heads: [B, n_queries, attn_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, n_queries, self.attn_dim
        )
        
        # Output projection: [B, n_queries, concept_embed_dim]
        concept_features = self.out_proj(attn_output)
        concept_features = self.dropout(concept_features)
        
        # ================================================================
        # 4. Aggregation (k sub-queries → 1 per concept)
        # ================================================================
        if self.num_sub_queries > 1:
            # [B, n*k, embed_dim] → [B, n, k*embed_dim]
            concept_features = concept_features.view(
                batch_size, n_selected, self.num_sub_queries * self.concept_embed_dim
            )
            # [B, n, k*embed_dim] → [B, n, embed_dim]
            concept_features = self.aggregation(concept_features)
        
        concept_embeddings = self.out_norm(concept_features)
        # [B, n_selected, concept_embed_dim]
        
        # ================================================================
        # 5. Attention weights for interpretability
        # ================================================================
        # Average over heads, reshape back to per-concept
        avg_attn = attn_weights.mean(dim=1)  # [B, n_queries, num_tokens]
        if self.num_sub_queries > 1:
            # [B, n*k, T] → [B, n, k, T] → [B, n, T] (mean over k)
            avg_attn = avg_attn.view(
                batch_size, n_selected, self.num_sub_queries, num_tokens
            ).mean(dim=2)
        
        return {
            "concept_embeddings": concept_embeddings,
            "attention_weights": avg_attn,
        }
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    print("Testing ConceptCrossAttention...")
    
    batch_size = 4
    n_selected = 5
    num_tokens = 139
    vlm_dim = 960
    text_dim = 768
    embed_dim = 128
    k = 4
    
    module = ConceptCrossAttention(
        vlm_hidden_dim=vlm_dim,
        concept_text_dim=text_dim,
        concept_embed_dim=embed_dim,
        num_sub_queries=k,
    )
    print(f"Parameters: {module.get_num_params():,}")
    
    # Forward test
    concept_text = torch.randn(batch_size, n_selected, text_dim)
    backbone = torch.randn(batch_size, num_tokens, vlm_dim)
    score_bias = torch.randn(batch_size, n_selected, num_tokens)
    
    result = module(
        concept_text_embeddings=concept_text,
        backbone_output=backbone,
        concept_score_bias=score_bias,
    )
    
    print(f"concept_embeddings: {result['concept_embeddings'].shape}")
    print(f"attention_weights:  {result['attention_weights'].shape}")
    
    # Verify attention weights sum to 1
    attn_sum = result["attention_weights"].sum(dim=-1)
    print(f"Attention weight sum (should ≈ 1.0): {attn_sum[0].tolist()}")
    
    # Gradient check
    loss = result["concept_embeddings"].sum()
    loss.backward()
    print(f"Gradient flows to q_proj: {module.q_proj.weight.grad is not None}")
    print(f"Gradient flows to k_proj: {module.k_proj.weight.grad is not None}")
    
    print("\n✓ ConceptCrossAttention test completed!")