"""
Completion Detector
====================

현재 활성 concept이 완료되었는지 판단하여 PAM에 signal을 전달합니다.

Option A: Rule-based (0 params) — action norm + elapsed time 기반
Option B: Small MLP (~9K params) — robot state + concept embedding 기반

Design Decision:
    PAM은 매 timestep completion signal을 받아 task 전환을 결정합니다.
    Completion detector는 PAM 외부(AI 측)에서 동작합니다.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class CompletionSignal:
    """PAM에 전달할 완료 신호"""
    done: bool
    confidence: float  # 0.0 ~ 1.0


class RuleBasedCompletionDetector:
    """
    Option A: Rule-based Completion Detection (No AI, 0 params)
    
    두 가지 규칙:
    1. Action norm < threshold AND elapsed > min_steps → done
    2. Elapsed >= max_duration * 0.8 → done (timeout 방지)
    """
    
    def __init__(
        self,
        action_norm_threshold: float = 0.01,
        min_elapsed_steps: int = 5,
        timeout_ratio: float = 0.8,
    ):
        self.action_norm_threshold = action_norm_threshold
        self.min_elapsed_steps = min_elapsed_steps
        self.timeout_ratio = timeout_ratio
    
    def __call__(
        self,
        action_output: torch.Tensor,
        elapsed: int,
        max_duration: int,
        **kwargs,
    ) -> CompletionSignal:
        """
        Args:
            action_output: [action_dim] — 가장 최근 실행된 action
            elapsed: 현재 concept 실행 경과 timestep
            max_duration: 현재 concept의 최대 실행 시간
        """
        action_norm = action_output.norm().item()
        
        # Rule 1: 로봇이 거의 안 움직이면 완료
        if action_norm < self.action_norm_threshold and elapsed > self.min_elapsed_steps:
            return CompletionSignal(done=True, confidence=0.9)
        
        # Rule 2: 시간 80% 경과하면 완료로 간주
        if elapsed >= max_duration * self.timeout_ratio:
            return CompletionSignal(done=True, confidence=0.7)
        
        return CompletionSignal(done=False, confidence=0.0)


class MLPCompletionDetector(nn.Module):
    """
    Option B: MLP Completion Detection (~9K params)
    
    Robot state + active concept embedding을 입력으로 받아
    completion confidence를 출력합니다.
    
    Architecture:
        concat(robot_state, active_embedding) → Linear(d_in, 64) → ReLU → Linear(64, 1) → Sigmoid
        d_in = state_dim + concept_embed_dim = 14 + 128 = 142
    
    Args:
        state_dim: Robot state 차원 (14: 6 joints + 6 velocities + 1 gripper + 1 timestamp)
        concept_embed_dim: Concept embedding 차원 (128)
        hidden_dim: MLP 중간 차원 (64)
        threshold: Completion 판정 threshold (0.7)
    """
    
    def __init__(
        self,
        state_dim: int = 14,
        concept_embed_dim: int = 128,
        hidden_dim: int = 64,
        threshold: float = 0.7,
    ):
        super().__init__()
        
        self.threshold = threshold
        input_dim = state_dim + concept_embed_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        robot_state: torch.Tensor,
        active_concept_embedding: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            robot_state: [batch, state_dim] or [state_dim]
            active_concept_embedding: [batch, concept_embed_dim] or [concept_embed_dim]
            
        Returns:
            dict:
                "confidence": [batch] or scalar — completion confidence
                "done": [batch] or scalar — boolean (confidence > threshold)
        """
        # Handle unbatched input
        squeeze = False
        if robot_state.dim() == 1:
            robot_state = robot_state.unsqueeze(0)
            active_concept_embedding = active_concept_embedding.unsqueeze(0)
            squeeze = True
        
        x = torch.cat([robot_state, active_concept_embedding], dim=-1)
        confidence = self.mlp(x).squeeze(-1)  # [batch]
        done = confidence > self.threshold
        
        if squeeze:
            confidence = confidence.squeeze(0)
            done = done.squeeze(0)
        
        return {
            "confidence": confidence,
            "done": done,
        }
    
    def to_signal(
        self,
        robot_state: torch.Tensor,
        active_concept_embedding: torch.Tensor,
    ) -> CompletionSignal:
        """Convenience: output as CompletionSignal dataclass"""
        with torch.no_grad():
            result = self.forward(robot_state, active_concept_embedding)
        return CompletionSignal(
            done=result["done"].item(),
            confidence=result["confidence"].item(),
        )
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    print("Testing Completion Detectors...")
    
    # Rule-based
    print("\n=== Rule-based ===")
    rule_det = RuleBasedCompletionDetector()
    
    action_small = torch.tensor([0.001, 0.002, -0.001, 0.0, 0.0, 0.0, 0.0])
    action_large = torch.tensor([0.1, 0.2, -0.15, 0.05, 0.0, 0.0, 0.5])
    
    sig1 = rule_det(action_small, elapsed=10, max_duration=30)
    sig2 = rule_det(action_large, elapsed=10, max_duration=30)
    sig3 = rule_det(action_large, elapsed=25, max_duration=30)
    
    print(f"Small action, elapsed 10/30: done={sig1.done}, conf={sig1.confidence}")
    print(f"Large action, elapsed 10/30: done={sig2.done}, conf={sig2.confidence}")
    print(f"Large action, elapsed 25/30: done={sig3.done}, conf={sig3.confidence}")
    
    # MLP-based
    print("\n=== MLP-based ===")
    mlp_det = MLPCompletionDetector(state_dim=14, concept_embed_dim=128)
    print(f"Parameters: {mlp_det.get_num_params():,}")
    
    state = torch.randn(4, 14)
    concept_emb = torch.randn(4, 128)
    result = mlp_det(state, concept_emb)
    print(f"confidence: {result['confidence'].tolist()}")
    print(f"done: {result['done'].tolist()}")
    
    # Signal conversion
    sig = mlp_det.to_signal(torch.randn(14), torch.randn(128))
    print(f"Signal: done={sig.done}, confidence={sig.confidence:.4f}")
    
    print("\n✓ Completion Detector test completed!")