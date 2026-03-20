import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MultiHeadAttention(nn.Module):
    """
    Base Multi-Head Attention Module
    
    This is the foundation for both Cross-Attention and Self-Attention
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.use_bias = bias
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split hidden dim into multiple heads
        
        Args:
            x: [batch_size, seq_len, hidden_dim]
            
        Returns:
            x: [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)
    
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge multiple heads back to hidden dim
        
        Args:
            x: [batch_size, num_heads, seq_len, head_dim]
            
        Returns:
            x: [batch_size, seq_len, hidden_dim]
        """
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.hidden_dim)
    
    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention
        
        Args:
            q: [batch_size, num_heads, q_len, head_dim]
            k: [batch_size, num_heads, k_len, head_dim]
            v: [batch_size, num_heads, v_len, head_dim]
            mask: [batch_size, 1, q_len, k_len] or [q_len, k_len]
            
        Returns:
            output: [batch_size, num_heads, q_len, head_dim]
            attn_weights: [batch_size, num_heads, q_len, k_len]
        """
        # Compute attention scores
        # [B, num_heads, q_len, head_dim] @ [B, num_heads, head_dim, k_len]
        # -> [B, num_heads, q_len, k_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                # [q_len, k_len] -> [1, 1, q_len, k_len]
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                # [batch_size, q_len, k_len] -> [batch_size, 1, q_len, k_len]
                mask = mask.unsqueeze(1)
            
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply dropout
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # [B, num_heads, q_len, k_len] @ [B, num_heads, v_len, head_dim]
        # -> [B, num_heads, q_len, head_dim]
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights


class CrossAttention(MultiHeadAttention):
    """
    Cross-Attention Layer
    
    Based on Table 6: Cross-attention (CA) vs self-attention (SA)
    
    Query: action tokens
    Key/Value: VLM features
    
    This allows action tokens to attend to VLM features,
    enabling the action expert to condition on visual and language information.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__(hidden_dim, num_heads, dropout, bias)
        
        # Query projection (from action tokens)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        
        # Key/Value projections (from VLM features)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for cross-attention
        
        Args:
            query: [batch_size, q_len, hidden_dim] - action tokens
            key_value: [batch_size, kv_len, hidden_dim] - VLM features
            attention_mask: [batch_size, kv_len] or [batch_size, q_len, kv_len]
            return_attention_weights: Whether to return attention weights
            
        Returns:
            output: [batch_size, q_len, hidden_dim]
            attn_weights: [batch_size, num_heads, q_len, kv_len] (optional)
        """
        # Project to Q, K, V
        q = self.q_proj(query)       # [B, q_len, D]
        k = self.k_proj(key_value)   # [B, kv_len, D]
        v = self.v_proj(key_value)   # [B, kv_len, D]
        
        # Split heads
        q = self._split_heads(q)  # [B, num_heads, q_len, head_dim]
        k = self._split_heads(k)  # [B, num_heads, kv_len, head_dim]
        v = self._split_heads(v)  # [B, num_heads, kv_len, head_dim]
        
        # Prepare attention mask
        if attention_mask is not None and attention_mask.dim() == 2:
            # [B, kv_len] -> [B, 1, kv_len]
            # Broadcast to [B, q_len, kv_len]
            batch_size, kv_len = attention_mask.shape
            q_len = query.shape[1]
            attention_mask = attention_mask.unsqueeze(1).expand(-1, q_len, -1)
        
        # Compute attention
        attn_output, attn_weights = self._compute_attention(q, k, v, attention_mask)
        
        # Merge heads
        attn_output = self._merge_heads(attn_output)  # [B, q_len, D]
        
        # Output projection
        output = self.out_proj(attn_output)
        
        if return_attention_weights:
            return output, attn_weights
        else:
            return output, None


class CausalSelfAttention(MultiHeadAttention):
    """
    Causal Self-Attention Layer
    
    Based on Table 7: Causal vs bidirectional attention on action tokens
    
    Each action token can only attend to PAST tokens within the chunk.
    This prevents future action dependencies and improves performance.
    
    Causal mask ensures autoregressive property:
    - Action at timestep t can only see actions at timesteps 0, 1, ..., t-1, t
    - Cannot see actions at timesteps t+1, t+2, ...
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__(hidden_dim, num_heads, dropout, bias)
        
        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        
        # Register causal mask as buffer (not a parameter)
        self.register_buffer("causal_mask", None, persistent=False)
        self.register_buffer("cached_seq_len", torch.tensor(0), persistent=False)
    
    def _get_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Get or create causal mask
        
        Causal mask is a lower triangular matrix:
        [[1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1]]
        
        Args:
            seq_len: Sequence length
            device: Device
            
        Returns:
            mask: [seq_len, seq_len] lower triangular matrix
        """
        # Check if we need to create a new mask
        if (self.causal_mask is None or 
            self.cached_seq_len < seq_len or 
            self.causal_mask.device != device):
            
            # Create lower triangular mask
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
            self.causal_mask = mask
            self.cached_seq_len = torch.tensor(seq_len)
        
        # Return mask of the correct size
        return self.causal_mask[:seq_len, :seq_len]
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for causal self-attention
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim] - action tokens
            attention_mask: [batch_size, seq_len] (optional, additional mask)
            return_attention_weights: Whether to return attention weights
            
        Returns:
            output: [batch_size, seq_len, hidden_dim]
            attn_weights: [batch_size, num_heads, seq_len, seq_len] (optional)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute Q, K, V in one go
        qkv = self.qkv_proj(hidden_states)  # [B, seq_len, 3*D]
        
        # Split into Q, K, V
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, seq_len, head_dim]
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Each: [B, num_heads, seq_len, head_dim]
        
        # Get causal mask
        causal_mask = self._get_causal_mask(seq_len, hidden_states.device)
        # [seq_len, seq_len]
        
        # Combine with additional attention mask if provided
        if attention_mask is not None:
            # [B, seq_len] -> [B, seq_len, seq_len]
            attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1)
            # Combine with causal mask
            combined_mask = causal_mask.unsqueeze(0) * attention_mask
        else:
            combined_mask = causal_mask
        
        # Compute attention
        attn_output, attn_weights = self._compute_attention(q, k, v, combined_mask)
        
        # Merge heads
        attn_output = self._merge_heads(attn_output)  # [B, seq_len, D]
        
        # Output projection
        output = self.out_proj(attn_output)
        
        if return_attention_weights:
            return output, attn_weights
        else:
            return output, None


class InterleavedAttentionBlock(nn.Module):
    """
    Interleaved Attention Block with Layer Norm and Residual
    
    Based on Table 6: Interleaving CA and SA yields best results
    
    Architecture:
    - Block 0: Cross-Attention + LayerNorm + Residual + FFN
    - Block 1: Causal Self-Attention + LayerNorm + Residual + FFN
    - Block 2: Cross-Attention + LayerNorm + Residual + FFN
    - Block 3: Causal Self-Attention + LayerNorm + Residual + FFN
    - ...
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        is_cross_attention: bool = True,
        activation: str = "gelu",
    ):
        super().__init__()
        
        self.is_cross_attention = is_cross_attention
        
        # Attention layer
        if is_cross_attention:
            self.attention = CrossAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
            self.attention = CausalSelfAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
        
        # Layer normalization (pre-norm architecture)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        if ff_dim is None:
            ff_dim = 4 * hidden_dim
        
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim] - action tokens
            encoder_hidden_states: [batch_size, enc_len, hidden_dim] - VLM features
                                   (required for cross-attention)
            attention_mask: Attention mask
            return_attention_weights: Whether to return attention weights
            
        Returns:
            output: [batch_size, seq_len, hidden_dim]
            attn_weights: Attention weights (optional)
        """
        # Attention block with pre-norm and residual
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        
        if self.is_cross_attention:
            assert encoder_hidden_states is not None, \
                "encoder_hidden_states required for cross-attention"
            attn_output, attn_weights = self.attention(
                query=hidden_states,
                key_value=encoder_hidden_states,
                attention_mask=attention_mask,
                return_attention_weights=return_attention_weights,
            )
        else:
            attn_output, attn_weights = self.attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                return_attention_weights=return_attention_weights,
            )
        
        hidden_states = residual + attn_output
        
        # Feed-forward block with pre-norm and residual
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        ff_output = self.ff(hidden_states)
        hidden_states = residual + ff_output
        
        return hidden_states, attn_weights


def create_interleaved_attention_stack(
    num_layers: int,
    hidden_dim: int,
    num_heads: int = 8,
    ff_dim: Optional[int] = None,
    dropout: float = 0.1,
) -> nn.ModuleList:
    """
    Create a stack of interleaved attention blocks
    
    Pattern:
    - Even indices (0, 2, 4, ...): Cross-Attention
    - Odd indices (1, 3, 5, ...): Causal Self-Attention
    
    Args:
        num_layers: Number of attention blocks
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        ff_dim: Feed-forward dimension (default: 4 * hidden_dim)
        dropout: Dropout rate
        
    Returns:
        blocks: ModuleList of InterleavedAttentionBlock
    """
    blocks = nn.ModuleList()
    
    for i in range(num_layers):
        is_cross_attention = (i % 2 == 0)
        
        block = InterleavedAttentionBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            is_cross_attention=is_cross_attention,
        )
        
        blocks.append(block)
    
    return blocks


if __name__ == "__main__":
    # Test attention mechanisms
    print("Testing Attention Mechanisms...")
    
    batch_size = 2
    seq_len = 50
    enc_len = 100
    hidden_dim = 512
    num_heads = 8
    
    # Create dummy tensors
    action_tokens = torch.randn(batch_size, seq_len, hidden_dim)
    vlm_features = torch.randn(batch_size, enc_len, hidden_dim)
    attention_mask = torch.ones(batch_size, enc_len)
    
    print("\n1. Testing CrossAttention...")
    cross_attn = CrossAttention(hidden_dim, num_heads)
    output, attn_weights = cross_attn(
        query=action_tokens,
        key_value=vlm_features,
        attention_mask=attention_mask,
        return_attention_weights=True,
    )
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")
    
    print("\n2. Testing CausalSelfAttention...")
    causal_attn = CausalSelfAttention(hidden_dim, num_heads)
    output, attn_weights = causal_attn(
        hidden_states=action_tokens,
        return_attention_weights=True,
    )
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")
    
    # Verify causal mask
    print(f"   Verifying causal mask...")
    attn_weights_mean = attn_weights.mean(dim=(0, 1))  # [seq_len, seq_len]
    upper_triangle = torch.triu(attn_weights_mean, diagonal=1)
    assert torch.allclose(upper_triangle, torch.zeros_like(upper_triangle), atol=1e-6), \
        "Upper triangle should be zero (causal mask)"
    print(f"   ✓ Causal mask working correctly")
    
    print("\n3. Testing InterleavedAttentionBlock...")
    
    # Cross-attention block
    cross_block = InterleavedAttentionBlock(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        is_cross_attention=True,
    )
    output_ca, _ = cross_block(
        hidden_states=action_tokens,
        encoder_hidden_states=vlm_features,
        attention_mask=attention_mask,
    )
    print(f"   Cross-attention output shape: {output_ca.shape}")
    
    # Self-attention block
    self_block = InterleavedAttentionBlock(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        is_cross_attention=False,
    )
    output_sa, _ = self_block(
        hidden_states=action_tokens,
    )
    print(f"   Self-attention output shape: {output_sa.shape}")
    
    print("\n4. Testing Interleaved Stack...")
    num_layers = 8
    blocks = create_interleaved_attention_stack(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
    )
    
    hidden_states = action_tokens
    for i, block in enumerate(blocks):
        is_cross = (i % 2 == 0)
        if is_cross:
            hidden_states, _ = block(
                hidden_states=hidden_states,
                encoder_hidden_states=vlm_features,
                attention_mask=attention_mask,
            )
        else:
            hidden_states, _ = block(
                hidden_states=hidden_states,
            )
    
    print(f"   Final output shape: {hidden_states.shape}")
    print(f"   Number of blocks: {len(blocks)}")
    
    print("\n✓ All attention mechanisms working correctly!")
