"""
Asynchronous Inference Stack for SmolVLA
Based on SmolVLA paper Section 3.3 and Algorithm 1

Key Features:
- Decouples action execution from observation processing
- Queue-based action management with threshold g
- Observation similarity filtering
- Eliminates idle gaps during inference
"""

import torch
import numpy as np
from typing import Optional, Dict, List, Tuple
from collections import deque
import time
from dataclasses import dataclass


@dataclass
class AsyncInferenceConfig:
    """
    Configuration for asynchronous inference
    
    Based on paper Section 3.3
    """
    # Queue threshold g ∈ [0, 1]
    # When |A_t| / n < g, send new observation
    queue_threshold: float = 0.7  # g=0.7 in paper
    
    # Observation similarity threshold ε
    # Don't send if distance in joint space < ε
    similarity_threshold: float = 0.01  # epsilon
    
    # Action chunk size
    chunk_size: int = 50  # n=50 in paper
    
    # Control cycle time (in seconds)
    delta_t: float = 0.033  # 30 Hz
    
    # Force processing when queue is empty
    force_when_empty: bool = True


class ActionQueue:
    """
    Action queue for async inference
    
    Manages action chunks with overlapping aggregation
    """
    
    def __init__(self, chunk_size: int = 50):
        self.chunk_size = chunk_size
        self.queue = deque()
        self.current_chunk_id = 0
    
    def __len__(self) -> int:
        return len(self.queue)
    
    def is_empty(self) -> bool:
        return len(self.queue) == 0
    
    def add_chunk(
        self,
        actions: np.ndarray,
        aggregate: bool = True,
    ):
        """
        Add new action chunk to queue
        
        Args:
            actions: [chunk_size, action_dim]
            aggregate: Whether to aggregate with existing queue
        """
        if not aggregate or self.is_empty():
            # Simply extend queue
            for action in actions:
                self.queue.append(action)
        else:
            # Aggregate overlapping sections
            # Keep existing actions, blend incoming actions
            overlap_size = min(len(self.queue), len(actions))
            
            if overlap_size > 0:
                # Weighted average on overlapping section
                existing_actions = np.array([self.queue[i] for i in range(overlap_size)])
                incoming_actions = actions[:overlap_size]
                
                # Weight: favor newer predictions
                alpha = 0.7  # Weight for incoming actions
                blended = alpha * incoming_actions + (1 - alpha) * existing_actions
                
                # Replace overlapping section
                for i in range(overlap_size):
                    self.queue[i] = blended[i]
                
                # Add non-overlapping section
                for action in actions[overlap_size:]:
                    self.queue.append(action)
            else:
                # No overlap, just extend
                for action in actions:
                    self.queue.append(action)
    
    def pop_front(self) -> Optional[np.ndarray]:
        """Pop action from front of queue"""
        if self.is_empty():
            return None
        return self.queue.popleft()
    
    def get_fill_ratio(self) -> float:
        """Get current fill ratio |A_t| / n"""
        return len(self.queue) / self.chunk_size
    
    def clear(self):
        """Clear queue"""
        self.queue.clear()


class ObservationBuffer:
    """
    Buffer for tracking observation similarity
    
    Prevents redundant processing of near-duplicate observations
    """
    
    def __init__(self, similarity_threshold: float = 0.01):
        self.similarity_threshold = similarity_threshold
        self.last_observation = None
    
    def is_similar(self, observation: Dict[str, np.ndarray]) -> bool:
        """
        Check if observation is similar to last one
        
        Uses joint-space distance for proprioceptive state
        
        Args:
            observation: Dict with 'state' key
            
        Returns:
            is_similar: True if similar to last observation
        """
        if self.last_observation is None:
            return False
        
        # Compare robot state in joint space
        if 'state' in observation and 'state' in self.last_observation:
            state_diff = np.linalg.norm(
                observation['state'] - self.last_observation['state']
            )
            return state_diff < self.similarity_threshold
        
        return False
    
    def update(self, observation: Dict[str, np.ndarray]):
        """Update last observation"""
        self.last_observation = observation.copy()


class AsyncInferenceManager:
    """
    Asynchronous Inference Manager
    
    Implements Algorithm 1 from paper Section 3.3
    
    Key insight: Decouple action prediction from execution
    - RobotClient executes actions from queue
    - PolicyServer predicts new chunks asynchronously
    - Queue threshold determines when to request new predictions
    """
    
    def __init__(self, config: Optional[AsyncInferenceConfig] = None):
        if config is None:
            config = AsyncInferenceConfig()
        
        self.config = config
        
        # Action queue
        self.action_queue = ActionQueue(chunk_size=config.chunk_size)
        
        # Observation buffer
        self.obs_buffer = ObservationBuffer(
            similarity_threshold=config.similarity_threshold
        )
        
        # Timing statistics
        self.last_request_time = None
        self.inference_times = []
        
        # State
        self.is_processing = False
    
    def should_request_prediction(self, force: bool = False) -> bool:
        """
        Determine if we should request new prediction
        
        Based on queue threshold g and observation similarity
        
        Args:
            force: Force request even if not needed
            
        Returns:
            should_request: True if should request new prediction
        """
        # Force if queue is empty
        if self.config.force_when_empty and self.action_queue.is_empty():
            return True
        
        # Force request
        if force:
            return True
        
        # Check queue threshold
        fill_ratio = self.action_queue.get_fill_ratio()
        return fill_ratio < self.config.queue_threshold
    
    def filter_observation(self, observation: Dict[str, np.ndarray]) -> bool:
        """
        Check if observation should be processed
        
        Returns False if too similar to last observation
        
        Args:
            observation: Current observation
            
        Returns:
            should_process: True if should process this observation
        """
        # Always process if queue is empty
        if self.action_queue.is_empty():
            self.obs_buffer.update(observation)
            return True
        
        # Check similarity
        if self.obs_buffer.is_similar(observation):
            return False
        
        # Update buffer and process
        self.obs_buffer.update(observation)
        return True
    
    def add_predicted_chunk(self, actions: np.ndarray):
        """
        Add newly predicted action chunk to queue
        
        Args:
            actions: [chunk_size, action_dim]
        """
        self.action_queue.add_chunk(actions, aggregate=True)
        self.is_processing = False
        
        # Track inference time
        if self.last_request_time is not None:
            inference_time = time.time() - self.last_request_time
            self.inference_times.append(inference_time)
    
    def get_next_action(self) -> Optional[np.ndarray]:
        """
        Get next action to execute
        
        Returns:
            action: [action_dim] or None if queue is empty
        """
        return self.action_queue.pop_front()
    
    def mark_request_sent(self):
        """Mark that a prediction request was sent"""
        self.is_processing = True
        self.last_request_time = time.time()
    
    def get_statistics(self) -> Dict:
        """Get inference statistics"""
        if not self.inference_times:
            return {
                'mean_inference_time': 0.0,
                'max_inference_time': 0.0,
                'queue_size': len(self.action_queue),
            }
        
        return {
            'mean_inference_time': np.mean(self.inference_times),
            'std_inference_time': np.std(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'queue_size': len(self.action_queue),
            'fill_ratio': self.action_queue.get_fill_ratio(),
        }
    
    def reset(self):
        """Reset manager"""
        self.action_queue.clear()
        self.obs_buffer.last_observation = None
        self.is_processing = False
        self.inference_times.clear()
