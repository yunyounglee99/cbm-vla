"""
CBM-VLA: Concept Bottleneck Model Vision-Language-Action
==========================================================

SmolVLA에 CBM 모듈을 통합한 메인 모델 클래스입니다.
기존 SmolVLA의 VisionEncoder, SmolVLMBackbone, FlowMatchingActionExpert를
그대로 재사용하면서, CBM 모듈(~4M params)을 추가합니다.

핵심 원칙: "Concept as Controller, Not Replacement"
    - Backbone output은 그대로 action expert에 전달 (정보 손실 없음)
    - Concept embedding은 action expert의 KV에 prepend되어 방향 제어만 담당

Training Phases:
    Phase 1: CBM Encoder 학습 (나머지 모두 freeze)
    Phase 2: Concept Scoring Module 학습 (backbone freeze)
    Phase 3: Backbone LoRA + CrossAttn + OrderAttn + ActionExpert 동시 학습
             (Scoring Module freeze)

Parameter Budget:
    CBM Encoder:        ~393K
    Concept Scoring:    ~288K (50 concepts) or ~48K
    Cross-Attention:    ~2.7M
    Order Attention:    ~196K
    Completion Det.:    ~9K
    Total added:        ~3.6M (~0.8% of 450M)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

# CBM-VLA 모듈
from .cbm_encoder import CBMEncoder
from .concept_scoring import (
    ConceptScoringModule,
    compute_scoring_loss,
)
from .concept_cross_attention import ConceptCrossAttention
from .concept_order_attention import (
    ConceptOrderAttention,
    compute_order_loss,
)
from .completion_detector import (
    CompletionSignal,
    RuleBasedCompletionDetector,
    MLPCompletionDetector,
)


@dataclass
class CBMVLAConfig:
    """
    CBM-VLA 설정
    
    SmolVLA Config를 확장하여 CBM 관련 설정을 추가합니다.
    """
    # === SmolVLA dimensions (모델 로드 시 자동 감지) ===
    siglip_hidden_dim: int = 768      # SigLIP-B/16 output dim
    vlm_hidden_dim: int = 960         # SmolLM2-360M hidden dim
    expert_hidden_dim: int = 720      # 0.75 × vlm_hidden_dim
    action_dim: int = 7               # 6-DOF + gripper
    state_dim: int = 7                # (or 14 if velocities included)
    
    # === CBM module dimensions ===
    cbm_mlp_hidden_dim: int = 256     # CBM Encoder MLP hidden
    num_concepts: int = 50            # Concept pool size
    concept_embed_dim: int = 128      # Final concept embedding dim
    concept_text_dim: int = 768       # Concept text embedding dim (from SigLIP text)
    num_sub_queries: int = 4          # k: sub-queries per concept in cross-attn
    num_attn_heads: int = 8           # Cross-attention heads
    order_attn_heads: int = 4         # Order self-attention heads
    order_ff_dim: int = 512           # Order FFN hidden dim
    
    # === Concept scoring ===
    scoring_aggregation: str = "max"  # "max", "mean", "max_mean"
    top_n_concepts: int = 6           # Number of concepts to select
    
    # === Completion detector ===
    use_mlp_completion: bool = True   # True=MLP, False=rule-based
    completion_threshold: float = 0.7
    
    # === Training ===
    dropout: float = 0.1
    
    # === Loss weights (Phase 2) ===
    lambda_similarity: float = 0.3
    lambda_activation_bce: float = 1.0
    lambda_sparsity: float = 0.1
    
    # === Loss weights (Phase 3) ===
    lambda_order: float = 0.1
    lambda_contrastive: float = 0.05


class CBMVLA(nn.Module):
    """
    CBM-VLA: Concept Bottleneck Model Vision-Language-Action
    
    SmolVLA의 VLM backbone과 action expert를 재사용하면서
    CBM 모듈을 추가하여 해석 가능한 행동 생성을 구현합니다.
    
    Architecture:
    ┌──────────────────────────────────────────────────────────┐
    │ Input: Multi-view Images + Instruction + Robot State     │
    └──────────────────────────────────────────────────────────┘
                             ↓
    ┌──────────────────────────────────────────────────────────┐
    │ SigLIP → [CBM Encoder (MLP residual)] → MLP Projector   │
    │                 (Phase 1)                                │
    └──────────────────────────────────────────────────────────┘
                             ↓
    ┌──────────────────────────────────────────────────────────┐
    │ VLM Backbone (SmolVLM2, frozen/LoRA)                     │
    │ → backbone_output [139 × vlm_dim]                        │
    └──────────┬─────────────────────────────┬─────────────────┘
               ↓                             ↓
    ┌──────────────────────┐     ┌──────────────────────────────┐
    │ Concept Scoring      │     │ Feature Projector            │
    │ (Phase 2, freeze P3) │     │ → vlm_features [139 × 720]  │
    │ → top-n selection    │     └──────────┬───────────────────┘
    └──────────┬───────────┘                │
               ↓                             │
    ┌──────────────────────┐                │
    │ Cross-Attention      │                │
    │ + Order Attention    │                │
    │ (Phase 3)            │                │
    │ → concept embeddings │                │
    └──────────┬───────────┘                │
               ↓                             ↓
    ┌──────────────────────────────────────────────────────────┐
    │ Action Expert (Flow Matching)                            │
    │ KV = [concept_embedding, vlm_features]                   │
    │ → action chunk [n × action_dim]                          │
    └──────────────────────────────────────────────────────────┘
    
    Args:
        config: CBMVLAConfig
        smolvla: Optional pre-loaded SmolVLA model
            If None, you must set backbone components manually
    """
    
    def __init__(
        self,
        config: Optional[CBMVLAConfig] = None,
    ):
        super().__init__()
        
        if config is None:
            config = CBMVLAConfig()
        self.config = config
        
        # ================================================================
        # SmolVLA components (set externally via load_smolvla_backbone)
        # These are placeholders — populated by init_from_smolvla()
        # ================================================================
        self.vision_encoder = None    # SmolVLA's VisionEncoder
        self.vlm_backbone = None      # SmolVLA's SmolVLMBackbone
        self.feature_projector = None # SmolVLA's FeatureProjector
        self.state_projector = None   # SmolVLA's StateProjector
        self.action_expert = None     # SmolVLA's FlowMatchingActionExpert
        
        # ================================================================
        # CBM Module 1: CBM Encoder (Phase 1)
        # ================================================================
        self.cbm_encoder = CBMEncoder(
            siglip_hidden_dim=config.siglip_hidden_dim,
            mlp_hidden_dim=config.cbm_mlp_hidden_dim,
            dropout=config.dropout,
        )
        
        # ================================================================
        # CBM Module 2: Concept Scoring (Phase 2)
        # ================================================================
        self.concept_scoring = ConceptScoringModule(
            vlm_hidden_dim=config.vlm_hidden_dim,
            num_concepts=config.num_concepts,
            aggregation=config.scoring_aggregation,
        )
        
        # ================================================================
        # CBM Module 3: Cross-Attention (Phase 3)
        # ================================================================
        self.concept_cross_attn = ConceptCrossAttention(
            vlm_hidden_dim=config.vlm_hidden_dim,
            concept_text_dim=config.concept_text_dim,
            concept_embed_dim=config.concept_embed_dim,
            num_heads=config.num_attn_heads,
            num_sub_queries=config.num_sub_queries,
            dropout=config.dropout,
        )
        
        # ================================================================
        # CBM Module 4: Order Attention (Phase 3)
        # ================================================================
        self.concept_order_attn = ConceptOrderAttention(
            concept_embed_dim=config.concept_embed_dim,
            num_heads=config.order_attn_heads,
            ff_dim=config.order_ff_dim,
            dropout=config.dropout,
        )
        
        # ================================================================
        # Concept-to-Expert projection: concept_embed_dim → expert_hidden_dim
        # Action Expert의 KV에 prepend할 때 차원을 맞춤
        # ================================================================
        self.concept_to_expert_proj = nn.Linear(
            config.concept_embed_dim,
            config.expert_hidden_dim,
        )
        
        # ================================================================
        # Completion Detector
        # ================================================================
        if config.use_mlp_completion:
            self.completion_detector = MLPCompletionDetector(
                state_dim=config.state_dim,
                concept_embed_dim=config.concept_embed_dim,
                threshold=config.completion_threshold,
            )
        else:
            self.completion_detector = RuleBasedCompletionDetector()
        
        # ================================================================
        # Concept text embeddings buffer
        # data collection에서 생성된 concept pool의 text embedding을 저장
        # ================================================================
        self.register_buffer(
            "concept_text_embeddings",
            torch.zeros(config.num_concepts, config.concept_text_dim),
        )
        
        self._print_param_summary()
    
    def _print_param_summary(self):
        """Print parameter summary for CBM modules only"""
        cbm_params = {
            "CBM Encoder": sum(p.numel() for p in self.cbm_encoder.parameters()),
            "Concept Scoring": sum(p.numel() for p in self.concept_scoring.parameters()),
            "Cross-Attention": sum(p.numel() for p in self.concept_cross_attn.parameters()),
            "Order Attention": sum(p.numel() for p in self.concept_order_attn.parameters()),
            "Concept→Expert Proj": sum(p.numel() for p in self.concept_to_expert_proj.parameters()),
        }
        if isinstance(self.completion_detector, nn.Module):
            cbm_params["Completion Detector"] = sum(
                p.numel() for p in self.completion_detector.parameters()
            )
        
        total = sum(cbm_params.values())
        print("\n" + "=" * 50)
        print("CBM-VLA Added Parameters")
        print("=" * 50)
        for name, count in cbm_params.items():
            print(f"  {name:25s}: {count:>10,}")
        print(f"  {'TOTAL':25s}: {total:>10,}")
        print("=" * 50)
    
    def init_from_smolvla(self, smolvla_model):
        """
        SmolVLA 모델에서 backbone components를 복사합니다.
        
        Args:
            smolvla_model: 로드된 SmolVLA 인스턴스
        """
        self.vision_encoder = smolvla_model.vision_encoder
        self.vlm_backbone = smolvla_model.vlm_backbone
        self.feature_projector = smolvla_model.feature_projector
        self.state_projector = smolvla_model.state_projector
        self.action_expert = smolvla_model.action_expert
        
        # VLM hidden dim 자동 감지
        detected_dim = smolvla_model.vlm_backbone.get_hidden_dim()
        if detected_dim != self.config.vlm_hidden_dim:
            print(f"[WARNING] VLM hidden dim mismatch: "
                  f"config={self.config.vlm_hidden_dim}, detected={detected_dim}")
        
        print("✓ SmolVLA components loaded into CBM-VLA")
    
    def set_concept_pool(
        self,
        concept_text_embeddings: torch.Tensor,
    ):
        """
        Concept pool의 text embedding을 설정합니다.
        
        Args:
            concept_text_embeddings: [num_concepts, concept_text_dim]
                T5/SigLIP으로 인코딩한 concept text embedding
        """
        assert concept_text_embeddings.shape == self.concept_text_embeddings.shape, \
            f"Shape mismatch: expected {self.concept_text_embeddings.shape}, " \
            f"got {concept_text_embeddings.shape}"
        
        self.concept_text_embeddings.copy_(concept_text_embeddings)
        
        # Scoring module 초기화에도 사용
        # concept_text_dim → vlm_dim으로 projection이 필요하면 여기서 처리
        # (현재 concept_proj는 vlm_dim → num_concepts이므로 직접 호환 안됨)
        print(f"✓ Concept pool set: {concept_text_embeddings.shape}")
    
    # ================================================================
    # Phase-specific forward methods
    # ================================================================
    
    def forward_phase2(
        self,
        backbone_output: torch.Tensor,
        reference_scores: Optional[torch.Tensor] = None,
        gt_active_concepts: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Phase 2 Forward: Concept Scoring Module 학습
        
        Backbone은 freeze, scoring module만 학습합니다.
        3가지 loss (similarity + activation BCE + sparsity)로 학습합니다.
        
        Args:
            backbone_output: [batch, num_tokens, vlm_hidden_dim] (detached)
            reference_scores: [batch, num_concepts] (SigLIP reference)
            gt_active_concepts: [batch, num_concepts] (binary ground truth)
            
        Returns:
            dict with loss and scoring results
        """
        # Scoring (backbone output should be detached in Phase 2)
        scoring_result = self.concept_scoring(
            backbone_output.detach(),
            return_matrix=True,
        )
        
        # Compute loss
        loss, loss_info = compute_scoring_loss(
            concept_scores=scoring_result["concept_scores"],
            concept_probs=scoring_result["concept_probs"],
            reference_scores=reference_scores,
            gt_active_concepts=gt_active_concepts,
            lambda_sim=self.config.lambda_similarity,
            lambda_bce=self.config.lambda_activation_bce,
            lambda_sparsity=self.config.lambda_sparsity,
        )
        
        return {
            "loss": loss,
            "info": loss_info,
            **scoring_result,
        }
    
    def forward_phase3(
        self,
        backbone_output: torch.Tensor,
        vlm_features: torch.Tensor,
        gt_actions: torch.Tensor,
        gt_concept_ids: torch.Tensor,
        gt_concept_order: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Phase 3 Forward: Cross-Attn + Order-Attn + Action Expert 동시 학습
        
        Scoring module은 freeze. 선택된 concept에 대해:
        1. Cross-attention으로 embedding 추출
        2. Causal self-attention으로 순서 학습
        3. Action expert로 action 생성 (flow matching loss)
        
        Gradient는 action expert → order_attn → cross_attn → backbone LoRA로 흐름.
        Scoring module에는 gradient가 흐르지 않음 (top-n selection이 gradient 끊음).
        
        Args:
            backbone_output: [batch, num_tokens, vlm_hidden_dim]
            vlm_features: [batch, num_tokens, expert_hidden_dim]
                Feature projector를 거친 backbone output
            gt_actions: [batch, chunk_size, action_dim]
            gt_concept_ids: [batch, n_active] — 활성 concept의 pool index
            gt_concept_order: [batch, n_active] — ground truth 실행 순서 (0,1,2,...)
            attention_mask: [batch, num_tokens]
            
        Returns:
            dict with losses and intermediate outputs
        """
        batch_size = backbone_output.shape[0]
        device = backbone_output.device
        
        # ================================================================
        # 1. Concept Scoring (frozen, no gradient)
        # ================================================================
        with torch.no_grad():
            scoring_result = self.concept_scoring(
                backbone_output, return_matrix=True
            )
        
        # Use scoring module's actual predictions (not GT) for robustness
        # → Phase 3에서 scoring 오류가 포함된 상태로 학습
        # → action expert가 scoring 오류에 robust해지는 효과
        selected_indices, selected_scores = self.concept_scoring.select_top_n(
            scoring_result["concept_scores"],
            n=self.config.top_n_concepts,
        )
        # selected_indices: [batch, n]
        
        n_selected = selected_indices.shape[1]
        
        # ================================================================
        # 2. Get concept text embeddings for selected concepts
        # ================================================================
        # concept_text_embeddings: [num_concepts, concept_text_dim]
        # selected_indices: [batch, n] → gather text embeddings
        expanded_indices = selected_indices.unsqueeze(-1).expand(
            -1, -1, self.config.concept_text_dim
        )
        # [batch, n, concept_text_dim]
        selected_text_embs = self.concept_text_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        ).gather(1, expanded_indices)
        
        # ================================================================
        # 3. Cross-Attention: backbone → concept embeddings
        # ================================================================
        # Score matrix에서 선택된 concept의 token-level score를 bias로 활용
        score_matrix = scoring_result["score_matrix"]  # [B, T, C]
        # selected concepts의 score만 추출: [B, T, C] → [B, n, T]
        score_indices = selected_indices.unsqueeze(1).expand(-1, score_matrix.shape[1], -1)
        selected_token_scores = score_matrix.gather(2, score_indices)  # [B, T, n]
        concept_score_bias = selected_token_scores.transpose(1, 2)     # [B, n, T]
        
        cross_attn_result = self.concept_cross_attn(
            concept_text_embeddings=selected_text_embs,
            backbone_output=backbone_output,
            concept_score_bias=concept_score_bias.detach(),  # bias는 gradient 끊음
            attention_mask=attention_mask,
        )
        concept_embeddings = cross_attn_result["concept_embeddings"]
        # [batch, n_selected, concept_embed_dim]
        
        # ================================================================
        # 4. Causal Self-Attention: order learning
        # ================================================================
        order_result = self.concept_order_attn(concept_embeddings)
        ordered_embeddings = order_result["ordered_embeddings"]
        order_scores = order_result["order_scores"]
        # [batch, n_selected, concept_embed_dim], [batch, n_selected]
        
        # ================================================================
        # 5. Select active concept for action expert
        # ================================================================
        # 학습 시: GT에서 첫 번째 concept을 active로 사용
        # (PAM은 학습에 참여하지 않으므로, 단순히 순서상 첫 번째를 선택)
        active_concept_emb = ordered_embeddings[:, 0, :]  # [batch, concept_embed_dim]
        
        # ================================================================
        # 6. Concept conditioning → Action Expert
        # ================================================================
        # concept embedding → expert dim projection
        concept_token = self.concept_to_expert_proj(active_concept_emb)
        # [batch, expert_hidden_dim]
        concept_token = concept_token.unsqueeze(1)
        # [batch, 1, expert_hidden_dim]
        
        # Prepend concept token to vlm_features
        # [batch, 1+139, expert_hidden_dim] = [batch, 140, expert_hidden_dim]
        conditioned_features = torch.cat([concept_token, vlm_features], dim=1)
        
        # Attention mask도 1칸 확장
        if attention_mask is not None:
            concept_mask = torch.ones(
                batch_size, 1, dtype=attention_mask.dtype, device=device
            )
            conditioned_mask = torch.cat([concept_mask, attention_mask], dim=1)
        else:
            conditioned_mask = None
        
        # ================================================================
        # 7. Flow Matching Loss (from SmolVLA's action expert)
        # ================================================================
        from smolvla.smolvla.action_expert import compute_flow_matching_loss
        
        flow_loss, flow_info = compute_flow_matching_loss(
            action_expert=self.action_expert,
            actions_gt=gt_actions,
            vlm_features=conditioned_features,
            attention_mask=conditioned_mask,
        )
        
        # ================================================================
        # 8. Auxiliary losses
        # ================================================================
        # Order loss
        order_loss = compute_order_loss(order_scores, gt_concept_order[:, :n_selected])
        
        # Total loss
        total_loss = (
            flow_loss
            + self.config.lambda_order * order_loss
        )
        
        # ================================================================
        # 9. Info dict
        # ================================================================
        info = {
            **flow_info,
            "loss_flow": flow_loss.item(),
            "loss_order": order_loss.item(),
            "loss_total": total_loss.item(),
            "n_selected_concepts": n_selected,
        }
        
        return {
            "loss": total_loss,
            "info": info,
            "concept_embeddings": concept_embeddings,
            "ordered_embeddings": ordered_embeddings,
            "order_scores": order_scores,
            "selected_indices": selected_indices,
        }
    
    @torch.no_grad()
    def inference(
        self,
        backbone_output: torch.Tensor,
        vlm_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 10,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference: concept selection → embedding → action generation
        
        PAM 없이 단순 inference (PAM은 inference loop에서 외부적으로 관리)
        
        Args:
            backbone_output: [batch, num_tokens, vlm_hidden_dim]
            vlm_features: [batch, num_tokens, expert_hidden_dim]
            attention_mask: [batch, num_tokens]
            num_inference_steps: Flow matching denoising steps
            
        Returns:
            dict with actions and concept information
        """
        self.eval()
        batch_size = backbone_output.shape[0]
        device = backbone_output.device
        
        # 1. Concept scoring → top-n selection
        scoring_result = self.concept_scoring(backbone_output, return_matrix=True)
        selected_indices, selected_scores = self.concept_scoring.select_top_n(
            scoring_result["concept_scores"], n=self.config.top_n_concepts
        )
        
        # 2. Get text embeddings
        n_selected = selected_indices.shape[1]
        expanded = selected_indices.unsqueeze(-1).expand(-1, -1, self.config.concept_text_dim)
        selected_text_embs = self.concept_text_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        ).gather(1, expanded)
        
        # 3. Cross-attention
        score_matrix = scoring_result["score_matrix"]
        score_idx = selected_indices.unsqueeze(1).expand(-1, score_matrix.shape[1], -1)
        bias = score_matrix.gather(2, score_idx).transpose(1, 2)
        
        cross_result = self.concept_cross_attn(
            concept_text_embeddings=selected_text_embs,
            backbone_output=backbone_output,
            concept_score_bias=bias,
            attention_mask=attention_mask,
        )
        
        # 4. Order attention
        order_result = self.concept_order_attn(cross_result["concept_embeddings"])
        
        # 5. Sort by order scores → concept sequence for PAM
        order_scores = order_result["order_scores"]
        sorted_indices = order_scores.argsort(dim=-1)  # ascending: earlier first
        
        # Reorder embeddings by predicted order
        sorted_embeddings = order_result["ordered_embeddings"].gather(
            1, sorted_indices.unsqueeze(-1).expand(-1, -1, self.config.concept_embed_dim)
        )
        
        # 6. Use first concept (earliest) for immediate action
        active_emb = sorted_embeddings[:, 0, :]  # [batch, concept_embed_dim]
        concept_token = self.concept_to_expert_proj(active_emb).unsqueeze(1)
        
        conditioned = torch.cat([concept_token, vlm_features], dim=1)
        if attention_mask is not None:
            cmask = torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=device)
            conditioned_mask = torch.cat([cmask, attention_mask], dim=1)
        else:
            conditioned_mask = None
        
        # 7. Action expert inference
        actions = self.action_expert.predict_action_chunk(
            vlm_features=conditioned,
            attention_mask=conditioned_mask,
            num_inference_steps=num_inference_steps,
        )
        
        # 8. Build concept task list for PAM
        concept_tasks = []
        for i in range(n_selected):
            sorted_idx = sorted_indices[0, i].item()
            original_idx = selected_indices[0, sorted_idx].item()
            concept_tasks.append({
                "task_id": i,
                "concept_pool_id": original_idx,
                "embedding": sorted_embeddings[0, i].cpu().numpy(),
                "order_score": order_scores[0, sorted_idx].item(),
                "activation_score": selected_scores[0, sorted_idx].item(),
            })
        
        return {
            "actions": actions,
            "concept_tasks": concept_tasks,
            "concept_scores": scoring_result["concept_scores"],
            "concept_probs": scoring_result["concept_probs"],
            "selected_indices": selected_indices,
            "sorted_embeddings": sorted_embeddings,
            "order_scores": order_scores,
        }
    
    # ================================================================
    # Phase management
    # ================================================================
    
    def configure_phase1(self):
        """Phase 1: CBM Encoder만 학습"""
        self._freeze_all()
        for p in self.cbm_encoder.parameters():
            p.requires_grad = True
        self._print_trainable_status("Phase 1")
    
    def configure_phase2(self):
        """Phase 2: Concept Scoring Module만 학습 (backbone freeze)"""
        self._freeze_all()
        for p in self.concept_scoring.parameters():
            p.requires_grad = True
        self._print_trainable_status("Phase 2")
    
    def configure_phase3(self):
        """
        Phase 3: CrossAttn + OrderAttn + ActionExpert + concept_to_expert_proj 학습
        Scoring module freeze. Backbone LoRA는 별도로 설정 필요.
        """
        self._freeze_all()
        
        # Trainable modules in Phase 3
        trainable_modules = [
            self.concept_cross_attn,
            self.concept_order_attn,
            self.concept_to_expert_proj,
            self.action_expert,
        ]
        if self.feature_projector is not None:
            trainable_modules.append(self.feature_projector)
        if self.state_projector is not None:
            trainable_modules.append(self.state_projector)
        if isinstance(self.completion_detector, nn.Module):
            trainable_modules.append(self.completion_detector)
        
        for module in trainable_modules:
            for p in module.parameters():
                p.requires_grad = True
        
        # Backbone LoRA는 train/finetuning.py에서 별도 설정
        self._print_trainable_status("Phase 3")
    
    def _freeze_all(self):
        """Freeze all parameters"""
        for p in self.parameters():
            p.requires_grad = False
    
    def _print_trainable_status(self, phase: str):
        """Print trainable parameter summary"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n[{phase}] Trainable: {trainable:,} / {total:,} "
              f"({100*trainable/max(total,1):.2f}%)")
    
    def get_num_cbm_params(self) -> int:
        """CBM 모듈만의 parameter 수"""
        cbm_modules = [
            self.cbm_encoder,
            self.concept_scoring,
            self.concept_cross_attn,
            self.concept_order_attn,
            self.concept_to_expert_proj,
        ]
        if isinstance(self.completion_detector, nn.Module):
            cbm_modules.append(self.completion_detector)
        
        return sum(
            sum(p.numel() for p in m.parameters())
            for m in cbm_modules
        )


if __name__ == "__main__":
    print("Testing CBMVLA...")
    
    config = CBMVLAConfig(
        vlm_hidden_dim=960,
        expert_hidden_dim=720,
        num_concepts=50,
        concept_embed_dim=128,
        top_n_concepts=6,
    )
    
    model = CBMVLA(config)
    print(f"\nTotal CBM params: {model.get_num_cbm_params():,}")
    
    # Simulate Phase 2
    print("\n=== Phase 2 Test ===")
    model.configure_phase2()
    
    backbone = torch.randn(2, 139, 960)
    ref_scores = torch.randn(2, 50)
    gt_active = torch.zeros(2, 50)
    gt_active[:, :5] = 1.0
    
    result = model.forward_phase2(backbone, ref_scores, gt_active)
    print(f"Phase 2 loss: {result['loss'].item():.4f}")
    result["loss"].backward()
    
    # Simulate Phase 3 (without actual SmolVLA backbone)
    print("\n=== Phase 3 Test (mock) ===")
    
    # Create mock action expert
    from smolvla.smolvla.action_expert import FlowMatchingActionExpert
    model.action_expert = FlowMatchingActionExpert(
        vlm_hidden_dim=720,  # expert_hidden_dim (already projected)
        action_dim=7,
        chunk_size=10,
        num_layers=4,
        hidden_dim_ratio=1.0,  # Already at expert dim
    )
    
    model.configure_phase3()
    
    # Set concept text embeddings
    model.set_concept_pool(torch.randn(50, 768))
    
    vlm_features = torch.randn(2, 139, 720)
    gt_actions = torch.randn(2, 10, 7)
    gt_concept_ids = torch.randint(0, 50, (2, 6))
    gt_concept_order = torch.arange(6).unsqueeze(0).expand(2, -1).float()
    
    result3 = model.forward_phase3(
        backbone_output=backbone,
        vlm_features=vlm_features,
        gt_actions=gt_actions,
        gt_concept_ids=gt_concept_ids,
        gt_concept_order=gt_concept_order,
    )
    print(f"Phase 3 loss: {result3['loss'].item():.4f}")
    for k, v in result3["info"].items():
        print(f"  {k}: {v}")
    
    # Test inference
    print("\n=== Inference Test ===")
    with torch.no_grad():
        inf_result = model.inference(
            backbone_output=backbone,
            vlm_features=vlm_features,
            num_inference_steps=5,
        )
    print(f"Actions shape: {inf_result['actions'].shape}")
    print(f"Concept tasks: {len(inf_result['concept_tasks'])}")
    for task in inf_result["concept_tasks"][:3]:
        print(f"  Task {task['task_id']}: pool_id={task['concept_pool_id']}, "
              f"order={task['order_score']:.3f}, activation={task['activation_score']:.3f}")
    
    print("\n✓ CBMVLA test completed!")