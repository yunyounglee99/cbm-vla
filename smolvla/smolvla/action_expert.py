import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class CrossAttentionLayer(nn.Module):
    """
    Cross-Attention layer where action tokens attend to VLM features
    
    Query: action tokens
    Key/Value: VLM features
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert self.head_dim * num_heads == hidden_dim, \
            "hidden_dim must be divisible by num_heads"
        
        # Query projection (from action tokens)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Key/Value projections (from VLM features)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        action_tokens: torch.Tensor,
        vlm_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            action_tokens: [batch_size, num_actions, hidden_dim]
            vlm_features: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            output: [batch_size, num_actions, hidden_dim]
        """
        batch_size, num_actions, _ = action_tokens.shape
        seq_len = vlm_features.shape[1]
        
        # Residual connection
        residual = action_tokens
        
        # Compute Q, K, V
        Q = self.q_proj(action_tokens)  # [B, num_actions, D]
        K = self.k_proj(vlm_features)   # [B, seq_len, D]
        V = self.v_proj(vlm_features)   # [B, seq_len, D]
        
        # Reshape for multi-head attention
        # [B, num_actions, D] -> [B, num_heads, num_actions, head_dim]
        Q = Q.view(batch_size, num_actions, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        # [B, num_heads, num_actions, head_dim] @ [B, num_heads, head_dim, seq_len]
        # -> [B, num_heads, num_actions, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Reshape mask: [B, seq_len] -> [B, 1, 1, seq_len]
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # [B, num_heads, num_actions, seq_len] @ [B, num_heads, seq_len, head_dim]
        # -> [B, num_heads, num_actions, head_dim]
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape back
        # [B, num_heads, num_actions, head_dim] -> [B, num_actions, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, num_actions, self.hidden_dim
        )
        
        # Output projection
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        # Residual connection and layer norm
        output = self.norm(residual + output)
        
        return output


class CausalSelfAttentionLayer(nn.Module):
    """
    Causal Self-Attention layer for action tokens
    
    Each action token can only attend to past tokens within the chunk
    This prevents future action dependencies
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert self.head_dim * num_heads == hidden_dim, \
            "hidden_dim must be divisible by num_heads"
        
        # Q, K, V projections
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Causal mask will be registered as buffer
        self.register_buffer("causal_mask", None, persistent=False)
    
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal mask for self-attention
        
        Args:
            seq_len: Sequence length
            device: Device
            
        Returns:
            mask: [seq_len, seq_len] lower triangular matrix
        """
        if self.causal_mask is None or self.causal_mask.shape[0] != seq_len:
            # Create lower triangular mask
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
            self.causal_mask = mask
        
        return self.causal_mask
    
    def forward(
        self,
        action_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            action_tokens: [batch_size, num_actions, hidden_dim]
            
        Returns:
            output: [batch_size, num_actions, hidden_dim]
        """
        batch_size, num_actions, _ = action_tokens.shape
        
        # Residual connection
        residual = action_tokens
        
        # Compute Q, K, V
        qkv = self.qkv_proj(action_tokens)  # [B, num_actions, 3*D]
        qkv = qkv.view(batch_size, num_actions, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, num_actions, head_dim]
        
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        # [B, num_heads, num_actions, head_dim] @ [B, num_heads, head_dim, num_actions]
        # -> [B, num_heads, num_actions, num_actions]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        causal_mask = self._get_causal_mask(num_actions, action_tokens.device)
        # [num_actions, num_actions] -> [1, 1, num_actions, num_actions]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, num_actions, self.hidden_dim
        )
        
        # Output projection
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        # Residual connection and layer norm
        output = self.norm(residual + output)
        
        return output


class FeedForwardLayer(nn.Module):
    """
    Feed-forward layer with GELU activation
    """
    
    def __init__(
        self,
        hidden_dim: int,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if ff_dim is None:
            ff_dim = 4 * hidden_dim
        
        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.norm(residual + x)


class InterleavedTransformerBlock(nn.Module):
    """
    Transformer block with either Cross-Attention or Self-Attention
    
    Blocks are interleaved:
    - Even blocks: Cross-Attention
    - Odd blocks: Causal Self-Attention
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        is_cross_attention: bool = True,
    ):
        super().__init__()
        
        self.is_cross_attention = is_cross_attention
        
        # Attention layer
        if is_cross_attention:
            self.attn = CrossAttentionLayer(hidden_dim, num_heads, dropout)
        else:
            self.attn = CausalSelfAttentionLayer(hidden_dim, num_heads, dropout)
        
        # Feed-forward layer
        self.ff = FeedForwardLayer(hidden_dim, dropout=dropout)
    
    def forward(
        self,
        action_tokens: torch.Tensor,
        vlm_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            action_tokens: [batch_size, num_actions, hidden_dim]
            vlm_features: [batch_size, seq_len, hidden_dim] (for cross-attention)
            attention_mask: [batch_size, seq_len] (for cross-attention)
            
        Returns:
            output: [batch_size, num_actions, hidden_dim]
        """
        # Attention
        if self.is_cross_attention:
            assert vlm_features is not None, "VLM features required for cross-attention"
            x = self.attn(action_tokens, vlm_features, attention_mask)
        else:
            x = self.attn(action_tokens)
        
        # Feed-forward
        x = self.ff(x)
        
        return x


class FlowMatchingActionExpert(nn.Module):
    """
    Flow Matching Action Expert

    Architecture:
    - Interleaved Cross-Attention and Causal Self-Attention layers
    - Hidden dim = 0.75 × VLM hidden dim (for efficiency)
    - Action chunk prediction (n=50)
    """
    
    def __init__(
        self,
        vlm_hidden_dim: int,
        action_dim: int,
        chunk_size: int = 50,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        hidden_dim_ratio: float = 0.75,
    ):
        super().__init__()
        
        self.vlm_hidden_dim = vlm_hidden_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.num_layers = num_layers
        
        # Expert hidden dim (0.75 × VLM hidden dim for efficiency)
        self.expert_hidden_dim = int(vlm_hidden_dim * hidden_dim_ratio)
        
        print(f"Action Expert: {self.expert_hidden_dim}D hidden, {num_layers} layers")
        
        # Project VLM features to expert dimension
        self.vlm_feature_proj = nn.Linear(vlm_hidden_dim, self.expert_hidden_dim)
        
        # Project actions to expert dimension
        # Action tokens: [batch, chunk_size, action_dim]
        self.action_proj = nn.Linear(action_dim, self.expert_hidden_dim)

        self.pos_encoding = nn.Parameter(
            torch.randn(1, chunk_size, self.expert_hidden_dim) * 0.02
        )
        
        # Interleaved transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            # Even: Cross-Attention, Odd: Self-Attention
            is_cross_attention = (i % 2 == 0)
            self.blocks.append(
                InterleavedTransformerBlock(
                    hidden_dim=self.expert_hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    is_cross_attention=is_cross_attention,
                )
            )
        
        # Output projection: expert_hidden_dim -> action_dim
        self.output_proj = nn.Sequential(
            nn.Linear(self.expert_hidden_dim, self.expert_hidden_dim),
            nn.GELU(),
            nn.Linear(self.expert_hidden_dim, action_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        # Initialize positional encoding
        nn.init.normal_(self.pos_encoding, std=0.02)
        
        # Initialize projections
        for module in [self.vlm_feature_proj, self.action_proj, self.output_proj]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Sequential):
                for m in module:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
    
    def forward(
        self,
        noisy_actions: torch.Tensor,
        vlm_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict vector field v_theta
        
        Args:
            noisy_actions: [batch_size, chunk_size, action_dim]
            vlm_features: [batch_size, seq_len, vlm_hidden_dim]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            vector_field: [batch_size, chunk_size, action_dim] - predicted v_theta
        """
        batch_size = noisy_actions.shape[0]
        
        # Project VLM features to expert dimension
        vlm_features_proj = self.vlm_feature_proj(vlm_features)
        # [B, seq_len, expert_hidden_dim]
        seq_len = vlm_features_proj.shape[1]
        
        # VLM feature에도 위치 정보를 주입하여 Cross-Attention의 공간 인지력 향상
        # [B, chunk_size, expert_hidden_dim]
        
        # Add positional encoding
        action_tokens = self.action_proj(noisy_actions)
        
        # Add positional encoding
        action_tokens = action_tokens + self.pos_encoding
        
        # Pass through interleaved transformer blocks
        for block in self.blocks:
            action_tokens = block(
                action_tokens=action_tokens,
                vlm_features=vlm_features_proj,
                attention_mask=attention_mask,
            )
        
        # Project to action dimension
        vector_field = self.output_proj(action_tokens)
        # [B, chunk_size, action_dim]
        
        return vector_field
    
    def sample_actions(
        self,
        vlm_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 10,
        temperature: float = 1.0,
        solver: str = "heun"
    ) -> torch.Tensor:
        """
        Sample actions using flow matching (inference)
        
        Algorithm:
        1. Start with noise: A_0 ~ N(0, I)
        2. For t = 0 to T-1:
            - Compute τ = t / T
            - Predict vector field: v = v_θ(A_t, o)
            - Update: A_{t+1} = A_t + (1/T) * v
        3. Return A_T
        
        Args:
            vlm_features: [batch_size, seq_len, vlm_hidden_dim]
            attention_mask: [batch_size, seq_len]
            num_inference_steps: Number of denoising steps (default: 10)
            temperature: Sampling temperature
            
        Returns:
            actions: [batch_size, chunk_size, action_dim]
        """
        batch_size = vlm_features.shape[0]
        device = vlm_features.device
        
        # Start with Gaussian noise
        actions = torch.randn(
            batch_size, self.chunk_size, self.action_dim,
            device=device
        ) * temperature
        
        # Denoising steps
        dt = 1.0 / num_inference_steps
        
        for step in range(num_inference_steps):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # 1. Evaluate vector field (Euler step)
                v_t = self.forward(
                    noisy_actions=actions,
                    vlm_features=vlm_features,
                    attention_mask=attention_mask,
                )
                
            if solver == "euler" or step == num_inference_steps - 1:
                actions = actions + dt * v_t
            
            elif solver == "heun":
                # 2. Heun's second-order correction
                actions_euler = actions + dt * v_t
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    v_next = self.forward(
                        noisy_actions=actions_euler,
                        vlm_features=vlm_features,
                        attention_mask=attention_mask,
                    )
                # 오일러 예측값과 다음 스텝 예측값의 평균 벡터로 업데이트
                actions = actions + dt * (v_t + v_next) / 2
        
        return actions
    
    @torch.no_grad()
    def predict_action_chunk(
        self,
        vlm_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 10,
    ) -> torch.Tensor:
        """
        Predict action chunk (inference mode)
        
        Args:
            vlm_features: [batch_size, seq_len, vlm_hidden_dim]
            attention_mask: [batch_size, seq_len]
            num_inference_steps: Number of denoising steps
            
        Returns:
            actions: [batch_size, chunk_size, action_dim]
        """
        return self.sample_actions(
            vlm_features=vlm_features,
            attention_mask=attention_mask,
            num_inference_steps=num_inference_steps,
        )


def sample_beta_distribution(
    batch_size: int,
    alpha: float = 0.5,
    beta: float = 0.5,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    Sample τ from Beta distribution
    
    Args:
        batch_size: Batch size
        alpha: Beta distribution parameter alpha
        beta: Beta distribution parameter beta
        device: Device
        
    Returns:
        tau: [batch_size, 1, 1] sampled from Beta(alpha, beta))
    """
    # Sample from Beta distribution
    tau = torch.distributions.Beta(alpha, beta).sample((batch_size,))
    
    # Reshape for broadcasting: [batch_size] -> [batch_size, 1, 1]
    tau = tau.view(batch_size, 1, 1).to(device)
    
    return tau


def compute_flow_matching_loss(
    action_expert: FlowMatchingActionExpert,
    actions_gt: torch.Tensor,
    vlm_features: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    action_mask: Optional[torch.Tensor] = None,  # [추가됨] 로봇별 유효 관절 마스크 [B, action_dim]
    alpha: float = 0.5,
    beta: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute Flow Matching Loss (Equation 1)
    """
    batch_size = actions_gt.shape[0]
    device = actions_gt.device
    
    # Sample τ from Beta distribution
    tau = sample_beta_distribution(batch_size, alpha, beta, device)
    # [batch_size, 1, 1]
    
    # Sample noise
    epsilon = torch.randn_like(actions_gt)
    # [batch_size, chunk_size, action_dim]
    
    # Compute noisy actions
    noisy_actions = tau * actions_gt + (1 - tau) * epsilon
    # [batch_size, chunk_size, action_dim]
    
    # Target vector field
    target_vector_field = actions_gt - epsilon
    # [batch_size, chunk_size, action_dim]
    
    # Predict vector field
    predicted_vector_field = action_expert(
        noisy_actions=noisy_actions,
        vlm_features=vlm_features,
        attention_mask=attention_mask,
    )
    # [batch_size, chunk_size, action_dim]
    
    # =====================================================================
    # [수정됨] Action Masking이 적용된 MSE Loss 계산
    # =====================================================================
    if action_mask is not None:
        # 1. 차원별 오차를 쪼개서 유지 (reduction='none')
        raw_loss = F.mse_loss(predicted_vector_field, target_vector_field, reduction='none')
        # raw_loss: [B, chunk_size, action_dim]
        
        # 2. 마스크 차원 확장 (시간축 chunk_size에 대해 Broadcasting)
        # [B, action_dim] -> [B, 1, action_dim]
        mask_expanded = action_mask.unsqueeze(1)
        
        # 3. 마스크 곱하기 (0으로 패딩된 차원의 오차는 완전히 소거됨)
        masked_loss = raw_loss * mask_expanded
        
        # 4. 유효한(1.0) 데이터 개수만 카운트 (batch 안의 모든 유효 관절 수 * chunk_size)
        valid_elements = mask_expanded.sum() * actions_gt.size(1) 
        
        # 5. 최종 평균 오차 계산 (0 나누기 에러 방지)
        if valid_elements > 0:
            loss = masked_loss.sum() / valid_elements
        else:
            loss = masked_loss.sum() * 0.0
    else:
        # 마스크가 입력되지 않았을 때의 기본 동작 (기존과 동일)
        loss = F.mse_loss(predicted_vector_field, target_vector_field)
    # =====================================================================
    
    # Additional info for logging
    info = {
        'loss': loss.item(),
        'tau_mean': tau.mean().item(),
        'pred_norm': predicted_vector_field.norm(dim=-1).mean().item(),
        'target_norm': target_vector_field.norm(dim=-1).mean().item(),
    }
    
    return loss, info


if __name__ == "__main__":
    # Test the action expert
    print("Testing FlowMatchingActionExpert...")
    
    # Parameters
    batch_size = 4
    seq_len = 100
    vlm_hidden_dim = 1536
    action_dim = 7
    chunk_size = 50
    
    # Create model
    action_expert = FlowMatchingActionExpert(
        vlm_hidden_dim=vlm_hidden_dim,
        action_dim=action_dim,
        chunk_size=chunk_size,
        num_layers=8,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in action_expert.parameters()) / 1e6:.2f}M")
    
    # Create dummy inputs
    vlm_features = torch.randn(batch_size, seq_len, vlm_hidden_dim)
    actions_gt = torch.randn(batch_size, chunk_size, action_dim)
    attention_mask = torch.ones(batch_size, seq_len)
    
    # [추가됨] 테스트용 더미 action_mask (예: 절반은 실제 관절, 절반은 패딩이라 가정)
    # batch=4, action_dim=7 이라고 할 때, 앞의 4개 차원만 1.0이고 뒤의 3개 차원은 0.0으로 패딩되었다고 가정
    dummy_action_mask = torch.zeros(batch_size, action_dim)
    dummy_action_mask[:, :4] = 1.0  
    
    # Test training loss
    print("\nTesting training loss...")
    loss, info = compute_flow_matching_loss(
        action_expert=action_expert,
        actions_gt=actions_gt,
        vlm_features=vlm_features,
        attention_mask=attention_mask,
        action_mask=dummy_action_mask,  # [추가됨]
    )
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Info: {info}")
    
    # Test inference
    print("\nTesting inference...")
    action_expert.eval()
    predicted_actions = action_expert.predict_action_chunk(
        vlm_features=vlm_features,
        attention_mask=attention_mask,
        num_inference_steps=10,
    )
    
    print(f"Predicted actions shape: {predicted_actions.shape}")
    print(f"Predicted actions mean: {predicted_actions.mean().item():.4f}")
    print(f"Predicted actions std: {predicted_actions.std().item():.4f}")
    
    print("\nFlowMatchingActionExpert test completed!")
