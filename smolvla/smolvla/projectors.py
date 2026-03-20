import torch
import torch.nn as nn
from typing import Optional


class StateProjector(nn.Module):
    """
    State Projector
    
    Projects robot sensorimotor states to match VLM token dimension.
    This allows state information to be concatenated with visual and text tokens.
    
    From paper Section 3.1:
    "Sensorimotor states are projected into a single token via a linear layer
     to match the language model's token dimension."
    
    Args:
        state_dim: Dimension of robot state (e.g., 7 for 6-DOF + gripper)
        vlm_hidden_dim: Hidden dimension of VLM (e.g., 1536 for SmolVLM-2)
        use_layernorm: Whether to apply LayerNorm after projection
        dropout: Dropout rate (default: 0.0)
    """
    
    def __init__(
        self,
        state_dim: int,
        vlm_hidden_dim: int,
        use_layernorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.vlm_hidden_dim = vlm_hidden_dim
        
        # Linear projection
        self.projection = nn.Linear(state_dim, vlm_hidden_dim)
        
        # Optional layer normalization
        self.norm = nn.LayerNorm(vlm_hidden_dim) if use_layernorm else nn.Identity()
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights"""
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)
    
    def forward(self, robot_state: torch.Tensor) -> torch.Tensor:
        """
        Project robot state to VLM dimension
        
        Args:
            robot_state: [batch_size, state_dim] or [batch_size, seq_len, state_dim]
            
        Returns:
            state_tokens: [batch_size, 1, vlm_hidden_dim] or 
                         [batch_size, seq_len, vlm_hidden_dim]
        """
        # Handle both 2D and 3D inputs
        input_shape = robot_state.shape
        
        if robot_state.dim() == 2:
            # [B, state_dim] -> [B, 1, state_dim]
            robot_state = robot_state.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Project to VLM dimension
        # [B, seq_len, state_dim] -> [B, seq_len, vlm_hidden_dim]
        state_tokens = self.projection(robot_state)
        
        # Apply normalization and dropout
        state_tokens = self.norm(state_tokens)
        state_tokens = self.dropout(state_tokens)
        
        # Squeeze back if input was 2D
        if squeeze_output:
            # Keep as [B, 1, D] for consistency with token format
            pass
        
        return state_tokens
    
    def get_output_dim(self) -> int:
        """Get output dimension"""
        return self.vlm_hidden_dim


class FeatureProjector(nn.Module):
    """
    Feature Projector
    
    Projects VLM features to Action Expert dimension.
    Adapts the VLM features to align with the action expert's dimension.
    
    From paper Section 3.1:
    "Project the VLM features to adapt the VLM features to align with 
     the action expert's dimension."
    
    The action expert uses hidden_dim = 0.75 × VLM_hidden_dim for efficiency.
    
    Args:
        vlm_hidden_dim: Hidden dimension of VLM (e.g., 1536)
        expert_hidden_dim: Hidden dimension of Action Expert (e.g., 1152 = 0.75 × 1536)
        use_nonlinearity: Whether to use nonlinearity (GELU)
        use_layernorm: Whether to apply LayerNorm
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        vlm_hidden_dim: int,
        expert_hidden_dim: int,
        use_nonlinearity: bool = False,
        use_layernorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.vlm_hidden_dim = vlm_hidden_dim
        self.expert_hidden_dim = expert_hidden_dim
        
        # Projection layers
        if use_nonlinearity:
            # Use intermediate layer with GELU activation
            self.projection = nn.Sequential(
                nn.Linear(vlm_hidden_dim, vlm_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(vlm_hidden_dim, expert_hidden_dim),
            )
        else:
            # Simple linear projection
            self.projection = nn.Linear(vlm_hidden_dim, expert_hidden_dim)
        
        # Optional layer normalization
        self.norm = nn.LayerNorm(expert_hidden_dim) if use_layernorm else nn.Identity()
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights"""
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, vlm_features: torch.Tensor) -> torch.Tensor:
        """
        Project VLM features to Action Expert dimension
        
        Args:
            vlm_features: [batch_size, seq_len, vlm_hidden_dim]
            
        Returns:
            expert_features: [batch_size, seq_len, expert_hidden_dim]
        """
        # Project to expert dimension
        expert_features = self.projection(vlm_features)
        
        # Apply normalization and dropout
        expert_features = self.norm(expert_features)
        expert_features = self.dropout(expert_features)
        
        return expert_features
    
    def get_output_dim(self) -> int:
        """Get output dimension"""
        return self.expert_hidden_dim


class ActionProjector(nn.Module):
    """
    Action Projector
    
    Projects actions to/from Action Expert dimension.
    Used for both input (action → expert dim) and output (expert dim → action).
    
    From paper Section 3.1:
    "Project the actions to match the action expert dimensions."
    
    Args:
        action_dim: Dimension of robot action (e.g., 7 for 6-DOF + gripper)
        expert_hidden_dim: Hidden dimension of Action Expert
        use_layernorm: Whether to apply LayerNorm
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        action_dim: int,
        expert_hidden_dim: int,
        use_layernorm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.expert_hidden_dim = expert_hidden_dim
        
        # Linear projection
        self.projection = nn.Linear(action_dim, expert_hidden_dim)
        
        # Optional layer normalization
        self.norm = nn.LayerNorm(expert_hidden_dim) if use_layernorm else nn.Identity()
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights"""
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)
    
    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Project actions to Action Expert dimension
        
        Args:
            actions: [batch_size, chunk_size, action_dim]
            
        Returns:
            action_features: [batch_size, chunk_size, expert_hidden_dim]
        """
        # Project to expert dimension
        action_features = self.projection(actions)
        
        # Apply normalization and dropout
        action_features = self.norm(action_features)
        action_features = self.dropout(action_features)
        
        return action_features
    
    def get_output_dim(self) -> int:
        """Get output dimension"""
        return self.expert_hidden_dim


class OutputActionProjector(nn.Module):
    """
    Output Action Projector
    
    Projects Action Expert output back to action space.
    This is the final layer that produces robot actions.
    
    Often uses a more expressive architecture with intermediate layers.
    
    Args:
        expert_hidden_dim: Hidden dimension of Action Expert
        action_dim: Dimension of robot action
        use_intermediate: Whether to use intermediate layers
        intermediate_dim: Dimension of intermediate layer
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        expert_hidden_dim: int,
        action_dim: int,
        use_intermediate: bool = True,
        intermediate_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.expert_hidden_dim = expert_hidden_dim
        self.action_dim = action_dim
        
        if use_intermediate:
            # Use intermediate layer with nonlinearity
            if intermediate_dim is None:
                intermediate_dim = expert_hidden_dim
            
            self.projection = nn.Sequential(
                nn.Linear(expert_hidden_dim, intermediate_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_dim, action_dim),
            )
        else:
            # Simple linear projection
            self.projection = nn.Linear(expert_hidden_dim, action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights"""
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                # Use smaller initialization for output layer
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, expert_features: torch.Tensor) -> torch.Tensor:
        """
        Project Action Expert features to action space
        
        Args:
            expert_features: [batch_size, chunk_size, expert_hidden_dim]
            
        Returns:
            actions: [batch_size, chunk_size, action_dim]
        """
        # Project to action dimension
        actions = self.projection(expert_features)
        
        return actions
    
    def get_output_dim(self) -> int:
        """Get output dimension"""
        return self.action_dim


class ProjectorConfig:
    """
    Configuration for all projectors in SmolVLA
    
    Provides a convenient way to configure all projectors consistently.
    """
    
    def __init__(
        self,
        state_dim: int = 7,                    # 6-DOF + gripper
        action_dim: int = 7,                   # 6-DOF + gripper
        vlm_hidden_dim: int = 1536,           # SmolVLM-2 hidden dim
        expert_hidden_dim_ratio: float = 0.75, # 0.75 × VLM dim
        dropout: float = 0.1,
        use_layernorm: bool = True,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.vlm_hidden_dim = vlm_hidden_dim
        self.expert_hidden_dim = int(vlm_hidden_dim * expert_hidden_dim_ratio)
        self.dropout = dropout
        self.use_layernorm = use_layernorm
    
    def create_state_projector(self) -> StateProjector:
        """Create state projector"""
        return StateProjector(
            state_dim=self.state_dim,
            vlm_hidden_dim=self.vlm_hidden_dim,
            use_layernorm=self.use_layernorm,
            dropout=self.dropout,
        )
    
    def create_feature_projector(self) -> FeatureProjector:
        """Create feature projector"""
        return FeatureProjector(
            vlm_hidden_dim=self.vlm_hidden_dim,
            expert_hidden_dim=self.expert_hidden_dim,
            use_layernorm=self.use_layernorm,
            dropout=self.dropout,
        )
    
    def create_action_projector(self) -> ActionProjector:
        """Create action projector (input)"""
        return ActionProjector(
            action_dim=self.action_dim,
            expert_hidden_dim=self.expert_hidden_dim,
            use_layernorm=False,  # Typically no norm for action input
            dropout=self.dropout,
        )
    
    def create_output_action_projector(self) -> OutputActionProjector:
        """Create output action projector"""
        return OutputActionProjector(
            expert_hidden_dim=self.expert_hidden_dim,
            action_dim=self.action_dim,
            use_intermediate=True,
            dropout=self.dropout,
        )


if __name__ == "__main__":
    # Test all projectors
    print("Testing Projectors...")
    
    # Configuration
    batch_size = 4
    seq_len = 100
    chunk_size = 50
    state_dim = 7
    action_dim = 7
    vlm_hidden_dim = 1536
    expert_hidden_dim = int(vlm_hidden_dim * 0.75)  # 1152
    
    print(f"\nConfiguration:")
    print(f"  VLM hidden dim: {vlm_hidden_dim}")
    print(f"  Expert hidden dim: {expert_hidden_dim}")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    
    # Test StateProjector
    print("\n1. Testing StateProjector...")
    state_proj = StateProjector(
        state_dim=state_dim,
        vlm_hidden_dim=vlm_hidden_dim,
    )
    
    robot_state = torch.randn(batch_size, state_dim)
    state_tokens = state_proj(robot_state)
    print(f"   Input shape: {robot_state.shape}")
    print(f"   Output shape: {state_tokens.shape}")
    assert state_tokens.shape == (batch_size, 1, vlm_hidden_dim)
    print("   ✓ State projection working")
    
    # Test FeatureProjector
    print("\n2. Testing FeatureProjector...")
    feature_proj = FeatureProjector(
        vlm_hidden_dim=vlm_hidden_dim,
        expert_hidden_dim=expert_hidden_dim,
    )
    
    vlm_features = torch.randn(batch_size, seq_len, vlm_hidden_dim)
    expert_features = feature_proj(vlm_features)
    print(f"   Input shape: {vlm_features.shape}")
    print(f"   Output shape: {expert_features.shape}")
    assert expert_features.shape == (batch_size, seq_len, expert_hidden_dim)
    print("   ✓ Feature projection working")
    
    # Test ActionProjector
    print("\n3. Testing ActionProjector...")
    action_proj = ActionProjector(
        action_dim=action_dim,
        expert_hidden_dim=expert_hidden_dim,
    )
    
    actions_input = torch.randn(batch_size, chunk_size, action_dim)
    action_features = action_proj(actions_input)
    print(f"   Input shape: {actions_input.shape}")
    print(f"   Output shape: {action_features.shape}")
    assert action_features.shape == (batch_size, chunk_size, expert_hidden_dim)
    print("   ✓ Action projection working")
    
    # Test OutputActionProjector
    print("\n4. Testing OutputActionProjector...")
    output_proj = OutputActionProjector(
        expert_hidden_dim=expert_hidden_dim,
        action_dim=action_dim,
    )
    
    expert_output = torch.randn(batch_size, chunk_size, expert_hidden_dim)
    actions_output = output_proj(expert_output)
    print(f"   Input shape: {expert_output.shape}")
    print(f"   Output shape: {actions_output.shape}")
    assert actions_output.shape == (batch_size, chunk_size, action_dim)
    print("   ✓ Output projection working")
    
    # Test ProjectorConfig
    print("\n5. Testing ProjectorConfig...")
    config = ProjectorConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        vlm_hidden_dim=vlm_hidden_dim,
        expert_hidden_dim_ratio=0.75,
    )
    
    state_proj = config.create_state_projector()
    feature_proj = config.create_feature_projector()
    action_proj = config.create_action_projector()
    output_proj = config.create_output_action_projector()
    
    print(f"   State projector: {state_dim} -> {state_proj.get_output_dim()}")
    print(f"   Feature projector: {vlm_hidden_dim} -> {feature_proj.get_output_dim()}")
    print(f"   Action projector: {action_dim} -> {action_proj.get_output_dim()}")
    print(f"   Output projector: {expert_hidden_dim} -> {output_proj.get_output_dim()}")
    print("   ✓ Config-based creation working")
    
    # Test full pipeline
    print("\n6. Testing full projection pipeline...")
    
    # State: robot state -> VLM tokens
    robot_state = torch.randn(batch_size, state_dim)
    state_tokens = state_proj(robot_state)
    print(f"   Step 1: State {robot_state.shape} -> {state_tokens.shape}")
    
    # Feature: VLM features -> Expert features
    vlm_features = torch.randn(batch_size, seq_len, vlm_hidden_dim)
    expert_features = feature_proj(vlm_features)
    print(f"   Step 2: VLM {vlm_features.shape} -> {expert_features.shape}")
    
    # Action input: actions -> Expert features
    actions_noisy = torch.randn(batch_size, chunk_size, action_dim)
    action_features = action_proj(actions_noisy)
    print(f"   Step 3: Actions {actions_noisy.shape} -> {action_features.shape}")
    
    # Action output: Expert features -> actions
    expert_output = torch.randn(batch_size, chunk_size, expert_hidden_dim)
    actions_pred = output_proj(expert_output)
    print(f"   Step 4: Expert {expert_output.shape} -> {actions_pred.shape}")
    
    print("\n✓ All projectors working correctly!")
