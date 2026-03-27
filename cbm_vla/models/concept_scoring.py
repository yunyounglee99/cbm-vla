"""
Module 2: Concept Scoring Module
==================================

Backbone output (139×960)에서 concept pool의 각 concept에 대한
활성도 점수를 계산합니다. CLG-CBM 스타일의 3가지 loss로 학습합니다.

Architecture:
    1. backbone_output (139×vlm_dim) × concept_proj (vlm_dim×num_concepts)
       → 139×num_concepts score matrix
    2. Per-concept max aggregation → num_concepts-dim score vector
    3. Sigmoid → activation probabilities

Loss (CLG-CBM style):
    L_scoring = λ₁·L_similarity + λ₂·L_activation_BCE + λ₃·L_sparsity

Design Decisions (이전 대화에서 확정):
    - Attention pooling으로 압축하지 않음 → 139개 token 전체 사용
    - DICAN의 spatial similarity map → max/mean/min aggregation과 동일 원리
    - Per-concept max aggregation: "이 concept과 관련 높은 token이 하나라도 있는가?"
    - Phase 2에서 학습, Phase 3에서 freeze

Params: concept_proj (vlm_dim × num_concepts) = 960 × 50 = ~48K
        (또는 300개 concept pool이면 960 × 300 = ~288K)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class ConceptScoringModule(nn.Module):
    """
    Concept Scoring Module
    
    Backbone output의 각 token에서 concept별 관련도를 계산하고,
    aggregation을 통해 concept별 최종 활성도 score를 출력합니다.
    
    Args:
        vlm_hidden_dim: Backbone output 차원 (960)
        num_concepts: Concept pool 크기 (20~300)
        aggregation: Token-level score 집계 방법 ("max", "mean", "max_mean")
        temperature: Score softmax temperature (학습 안정성)
    """
    
    def __init__(
        self,
        vlm_hidden_dim: int = 960,
        num_concepts: int = 50,
        aggregation: str = "max",
        temperature: float = 1.0,
    ):
        super().__init__()
        
        self.vlm_hidden_dim = vlm_hidden_dim
        self.num_concepts = num_concepts
        self.aggregation = aggregation
        self.temperature = temperature
        
        # Concept projection layer: (vlm_dim → num_concepts)
        # 각 token을 num_concepts 차원으로 projection하여
        # token-concept 관련도 score matrix를 생성
        # 초기화: concept text embedding으로 초기화 가능 (set_concept_embeddings)
        self.concept_proj = nn.Linear(vlm_hidden_dim, num_concepts, bias=False)
        
        # Learnable temperature for scoring
        self.log_temperature = nn.Parameter(torch.tensor(0.0))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights"""
        nn.init.xavier_uniform_(self.concept_proj.weight, gain=0.5)
    
    def set_concept_embeddings(
        self,
        concept_text_embeddings: torch.Tensor,
    ):
        """
        Concept text embedding으로 projection weight 초기화
        
        T5/SigLIP으로 인코딩한 concept text embedding을 
        projection layer의 초기 weight로 사용합니다.
        이렇게 하면 학습 초기부터 의미적으로 올바른 concept scoring이 가능합니다.
        
        Args:
            concept_text_embeddings: [num_concepts, vlm_hidden_dim]
                각 concept의 text embedding (vlm_dim으로 projection된 상태)
        """
        assert concept_text_embeddings.shape == (self.num_concepts, self.vlm_hidden_dim), \
            f"Expected ({self.num_concepts}, {self.vlm_hidden_dim}), " \
            f"got {concept_text_embeddings.shape}"
        
        with torch.no_grad():
            # L2 normalize
            normed = F.normalize(concept_text_embeddings, dim=-1)
            self.concept_proj.weight.copy_(normed)
    
    def compute_score_matrix(
        self,
        backbone_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Token-level concept score matrix 계산
        
        Args:
            backbone_output: [batch, num_tokens, vlm_hidden_dim]
                예: [B, 139, 960]
                
        Returns:
            score_matrix: [batch, num_tokens, num_concepts]
                예: [B, 139, 50]
                각 (token, concept) 쌍의 관련도 score
        """
        # L2 normalize backbone tokens
        backbone_normed = F.normalize(backbone_output, dim=-1)
        
        # Projection: [B, T, D] × [D, C] → [B, T, C]
        # concept_proj.weight: [C, D] → transpose needed conceptually,
        # but nn.Linear handles this: output = input @ weight.T
        temperature = torch.exp(self.log_temperature) + self.temperature
        score_matrix = self.concept_proj(backbone_normed) / temperature
        
        return score_matrix
    
    def aggregate_scores(
        self,
        score_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Token-level scores를 concept별 단일 score로 집계
        
        DICAN의 spatial similarity map → max/mean/min aggregation과 동일 원리.
        139개 token에서 concept별로 max를 취하면
        "이 concept과 가장 관련 높은 token의 score"가 됩니다.
        
        Args:
            score_matrix: [batch, num_tokens, num_concepts]
            
        Returns:
            concept_scores: [batch, num_concepts]
        """
        if self.aggregation == "max":
            # Per-concept max across tokens
            concept_scores, _ = score_matrix.max(dim=1)  # [B, C]
        elif self.aggregation == "mean":
            concept_scores = score_matrix.mean(dim=1)     # [B, C]
        elif self.aggregation == "max_mean":
            # DICAN style: concat max and mean
            max_scores, _ = score_matrix.max(dim=1)       # [B, C]
            mean_scores = score_matrix.mean(dim=1)        # [B, C]
            concept_scores = (max_scores + mean_scores) / 2  # [B, C]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        return concept_scores
    
    def forward(
        self,
        backbone_output: torch.Tensor,
        return_matrix: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: concept scoring
        
        Args:
            backbone_output: [batch, num_tokens, vlm_hidden_dim]
            return_matrix: 139×C score matrix도 반환할지 여부
                (cross-attention의 attention bias로 활용 가능)
                
        Returns:
            dict:
                "concept_scores": [batch, num_concepts] — 각 concept의 활성도 score
                "concept_probs": [batch, num_concepts] — sigmoid 적용된 확률
                "active_mask": [batch, num_concepts] — threshold 초과 여부 (bool)
                "score_matrix": [batch, num_tokens, num_concepts] (optional)
        """
        # 1. Token-level score matrix
        score_matrix = self.compute_score_matrix(backbone_output)
        # [B, T, C]
        
        # 2. Aggregate to concept-level scores
        concept_scores = self.aggregate_scores(score_matrix)
        # [B, C]
        
        # 3. Activation probabilities
        concept_probs = torch.sigmoid(concept_scores)
        # [B, C]
        
        # 4. Active concept mask (used at inference for top-n selection)
        # Training에서는 사용하지 않음 (모든 concept에 대해 loss 계산)
        active_mask = concept_probs > 0.5
        
        result = {
            "concept_scores": concept_scores,
            "concept_probs": concept_probs,
            "active_mask": active_mask,
        }
        
        if return_matrix:
            result["score_matrix"] = score_matrix
        
        return result
    
    def select_top_n(
        self,
        concept_scores: torch.Tensor,
        n: int = 6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Top-n concept 선택 (inference 전용, gradient 끊김)
        
        Args:
            concept_scores: [batch, num_concepts]
            n: 선택할 concept 수
            
        Returns:
            selected_indices: [batch, n] — 선택된 concept의 인덱스
            selected_scores: [batch, n] — 선택된 concept의 score
        """
        # Top-n selection (non-differentiable — 의도적)
        selected_scores, selected_indices = torch.topk(
            concept_scores, k=min(n, self.num_concepts), dim=-1
        )
        return selected_indices, selected_scores
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ================================================================
# Loss Functions (CLG-CBM style)
# ================================================================

def compute_similarity_loss(
    concept_scores: torch.Tensor,
    reference_scores: torch.Tensor,
    visual_token_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Similarity Loss: SigLIP reference와의 정렬
    
    SigLIP이 계산한 image-concept similarity를 teacher signal로 사용하여
    scoring module이 올바른 concept scoring 방향을 학습하도록 합니다.
    
    Args:
        concept_scores: [batch, num_concepts] — scoring module 출력
        reference_scores: [batch, num_concepts] — SigLIP으로 계산한 reference
        visual_token_mask: 사용하지 않음 (aggregated score 기준)
        
    Returns:
        loss: scalar
    """
    # MSE between our scores and SigLIP reference
    loss = F.mse_loss(concept_scores, reference_scores.detach())
    return loss


def compute_activation_bce_loss(
    concept_probs: torch.Tensor,
    gt_active_concepts: torch.Tensor,
) -> torch.Tensor:
    """
    Activation BCE Loss: Ground truth 활성 concept과의 정렬
    
    Data collection에서 만든 ground truth label을 사용하여
    각 concept이 활성/비활성을 정확히 판별하도록 학습합니다.
    
    Args:
        concept_probs: [batch, num_concepts] — sigmoid 적용된 확률
        gt_active_concepts: [batch, num_concepts] — binary ground truth
            1.0 = 이 프레임에서 활성, 0.0 = 비활성
            
    Returns:
        loss: scalar
    """
    loss = F.binary_cross_entropy(
        concept_probs,
        gt_active_concepts.float(),
        reduction="mean",
    )
    return loss


def compute_sparsity_loss(
    concept_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Sparsity Loss: 전체 concept 활성도를 최소화
    
    300개 concept 중 대부분(294~295개)은 비활성이어야 합니다.
    L1 penalty로 전체 활성도를 억제합니다.
    
    이것이 similarity loss + activation BCE만으로 부족한 이유:
    GT label이 없는 concept(대다수)에 대한 직접적 supervision이 없기 때문.
    Sparsity loss가 이 빈 자리를 채웁니다.
    
    Args:
        concept_probs: [batch, num_concepts]
        
    Returns:
        loss: scalar
    """
    # L1 norm of activation probabilities
    loss = concept_probs.abs().mean()
    return loss


def compute_scoring_loss(
    concept_scores: torch.Tensor,
    concept_probs: torch.Tensor,
    reference_scores: Optional[torch.Tensor] = None,
    gt_active_concepts: Optional[torch.Tensor] = None,
    lambda_sim: float = 0.3,
    lambda_bce: float = 1.0,
    lambda_sparsity: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined Scoring Loss (CLG-CBM style)
    
    L_scoring = λ₁·L_similarity + λ₂·L_activation_BCE + λ₃·L_sparsity
    
    λ 권장값:
        activation BCE = 가장 크게 (직접적 supervision)
        similarity = 중간 (soft guidance from SigLIP)
        sparsity = 가장 작게 (regularization)
    
    Args:
        concept_scores: [batch, num_concepts]
        concept_probs: [batch, num_concepts]
        reference_scores: [batch, num_concepts] (optional, SigLIP reference)
        gt_active_concepts: [batch, num_concepts] (optional, ground truth)
        lambda_sim: Similarity loss weight
        lambda_bce: Activation BCE loss weight
        lambda_sparsity: Sparsity loss weight
        
    Returns:
        total_loss: scalar
        info: dict with individual loss components
    """
    total_loss = torch.tensor(0.0, device=concept_scores.device)
    info = {}
    
    # Similarity loss (soft teacher from SigLIP)
    if reference_scores is not None:
        l_sim = compute_similarity_loss(concept_scores, reference_scores)
        total_loss = total_loss + lambda_sim * l_sim
        info["loss_similarity"] = l_sim.item()
    
    # Activation BCE (hard ground truth)
    if gt_active_concepts is not None:
        l_bce = compute_activation_bce_loss(concept_probs, gt_active_concepts)
        total_loss = total_loss + lambda_bce * l_bce
        info["loss_activation_bce"] = l_bce.item()
    
    # Sparsity (global regularization)
    l_sparsity = compute_sparsity_loss(concept_probs)
    total_loss = total_loss + lambda_sparsity * l_sparsity
    info["loss_sparsity"] = l_sparsity.item()
    
    info["loss_scoring_total"] = total_loss.item()
    info["num_active_concepts"] = (concept_probs > 0.5).float().sum(dim=-1).mean().item()
    
    return total_loss, info


if __name__ == "__main__":
    print("Testing ConceptScoringModule...")
    
    batch_size = 4
    num_tokens = 139
    vlm_dim = 960
    num_concepts = 50
    
    module = ConceptScoringModule(
        vlm_hidden_dim=vlm_dim,
        num_concepts=num_concepts,
    )
    print(f"Parameters: {module.get_num_params():,}")
    
    # Forward test
    backbone = torch.randn(batch_size, num_tokens, vlm_dim)
    result = module(backbone, return_matrix=True)
    
    print(f"concept_scores: {result['concept_scores'].shape}")
    print(f"concept_probs:  {result['concept_probs'].shape}")
    print(f"active_mask:    {result['active_mask'].shape}")
    print(f"score_matrix:   {result['score_matrix'].shape}")
    print(f"Active concepts: {result['active_mask'].sum(dim=-1).tolist()}")
    
    # Top-n selection test
    indices, scores = module.select_top_n(result["concept_scores"], n=6)
    print(f"Top-6 indices: {indices[0].tolist()}")
    print(f"Top-6 scores:  {scores[0].tolist()}")
    
    # Loss test
    ref_scores = torch.randn(batch_size, num_concepts)
    gt_active = torch.zeros(batch_size, num_concepts)
    gt_active[:, :5] = 1.0  # First 5 concepts are active
    
    loss, info = compute_scoring_loss(
        concept_scores=result["concept_scores"],
        concept_probs=result["concept_probs"],
        reference_scores=ref_scores,
        gt_active_concepts=gt_active,
    )
    print(f"\nScoring loss: {loss.item():.4f}")
    for k, v in info.items():
        print(f"  {k}: {v:.4f}")
    
    # Gradient check
    loss.backward()
    grad_norm = module.concept_proj.weight.grad.norm().item()
    print(f"\nGradient norm on concept_proj: {grad_norm:.4f}")
    
    print("\n✓ ConceptScoringModule test completed!")