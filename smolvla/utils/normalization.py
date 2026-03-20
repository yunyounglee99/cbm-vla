"""
Data Normalization Utilities for SmolVLA
Based on standard robotics data preprocessing

Key Features:
- Action normalization/denormalization
- State normalization
- Running statistics for online normalization
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import numpy as np


class Normalizer(nn.Module):
    """
    Base normalizer class with running statistics
    
    Tracks mean and std during training for consistent normalization
    """
    
    def __init__(self, dim: int, mode: str = "min_max", eps: float = 1e-8):
        super().__init__()
        
        assert mode in ["min_max", "gaussian"], f"Unknown mode: {mode}"
        self.mode = mode
        self.eps = eps
        self.dim = dim
        
        # Register buffers (not parameters, but saved with model)
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.register_buffer("min", torch.zeros(dim))
        self.register_buffer("max", torch.ones(dim))
        self.register_buffer("count", torch.zeros(1))
    
    def update_stats(self, data: torch.Tensor):
        """
        Update running statistics
        
        Args:
            data: [batch_size, ..., dim]
        """
        # Flatten all dimensions except last
        data_flat = data.reshape(-1, self.dim)
        batch_size = data_flat.shape[0]
        
        # Update count
        self.count += batch_size
        
        # Update mean and std (Welford's online algorithm)
        batch_mean = data_flat.mean(dim=0)
        batch_var = data_flat.var(dim=0)
        
        # Combine with existing statistics
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * batch_size / self.count
        
        m2 = batch_var * (batch_size - 1)
        self.std = torch.sqrt(
            (m2 + delta ** 2 * batch_size * (self.count - batch_size) / self.count) 
            / (self.count - 1 + self.eps)
        )
        
        # Update min and max
        self.min = torch.minimum(self.min, data_flat.min(dim=0)[0])
        self.max = torch.maximum(self.max, data_flat.max(dim=0)[0])
    
    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalize data
        
        Args:
            data: [..., dim]
            
        Returns:
            normalized_data: [..., dim]
        """
        if self.mode == "gaussian":
            # Gaussian normalization: (x - mean) / std
            return (data - self.mean) / (self.std + self.eps)
        else:
            # Min-max normalization: (x - min) / (max - min) * 2 - 1
            # Maps to [-1, 1]
            return 2 * (data - self.min) / (self.max - self.min + self.eps) - 1
    
    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Denormalize data
        
        Args:
            data: [..., dim] normalized data
            
        Returns:
            original_data: [..., dim]
        """
        if self.mode == "gaussian":
            return data * (self.std + self.eps) + self.mean
        else:
            # Inverse of min-max: (x + 1) / 2 * (max - min) + min
            return (data + 1) / 2 * (self.max - self.min + self.eps) + self.min


class ActionNormalizer(Normalizer):
    """
    Action-specific normalizer
    
    Handles action chunks and supports different normalization per dimension
    (e.g., position vs gripper)
    """
    
    def __init__(
        self,
        action_dim: int,
        mode: str = "min_max",
        per_dim: bool = True,
    ):
        super().__init__(dim=action_dim, mode=mode)
        self.per_dim = per_dim
    
    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Normalize action chunk
        
        Args:
            actions: [batch_size, chunk_size, action_dim]
            
        Returns:
            normalized_actions: [batch_size, chunk_size, action_dim]
        """
        return self.normalize(actions)
    
    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Denormalize action chunk
        
        Args:
            actions: [batch_size, chunk_size, action_dim]
            
        Returns:
            original_actions: [batch_size, chunk_size, action_dim]
        """
        return self.denormalize(actions)


class StateNormalizer(Normalizer):
    """
    State-specific normalizer
    
    Normalizes robot proprioceptive states
    """
    
    def __init__(self, state_dim: int, mode: str = "gaussian"):
        super().__init__(dim=state_dim, mode=mode)
    
    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Normalize robot state
        
        Args:
            state: [batch_size, state_dim] or [state_dim]
            
        Returns:
            normalized_state: same shape as input
        """
        return self.normalize(state)
    
    def denormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Denormalize robot state
        
        Args:
            state: [batch_size, state_dim] or [state_dim]
            
        Returns:
            original_state: same shape as input
        """
        return self.denormalize(state)


class NormalizationManager:
    """
    Manages all normalizers for SmolVLA
    
    Convenient wrapper for action and state normalization
    """
    
    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        action_mode: str = "min_max",
        state_mode: str = "gaussian",
    ):
        self.action_normalizer = ActionNormalizer(action_dim, mode=action_mode)
        self.state_normalizer = StateNormalizer(state_dim, mode=state_mode)
    
    def update_from_batch(
        self,
        actions: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
    ):
        """Update statistics from batch"""
        if actions is not None:
            self.action_normalizer.update_stats(actions)
        if states is not None:
            self.state_normalizer.update_stats(states)
    
    def normalize(
        self,
        actions: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Normalize actions and/or state"""
        norm_actions = self.action_normalizer.normalize_actions(actions) if actions is not None else None
        norm_state = self.state_normalizer.normalize_state(state) if state is not None else None
        return norm_actions, norm_state
    
    def denormalize(
        self,
        actions: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Denormalize actions and/or state"""
        denorm_actions = self.action_normalizer.denormalize_actions(actions) if actions is not None else None
        denorm_state = self.state_normalizer.denormalize_state(state) if state is not None else None
        return denorm_actions, denorm_state
    
    def state_dict(self) -> Dict:
        """Get state dict for saving"""
        return {
            'action_normalizer': self.action_normalizer.state_dict(),
            'state_normalizer': self.state_normalizer.state_dict(),
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load state dict"""
        self.action_normalizer.load_state_dict(state_dict['action_normalizer'])
        self.state_normalizer.load_state_dict(state_dict['state_normalizer'])
