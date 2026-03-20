"""
Robot Client for Asynchronous Inference
Based on SmolVLA paper Section 3.3 and Algorithm 1

Executes actions while requesting new predictions asynchronously
"""

import numpy as np
from typing import Dict, Optional, Callable
import time
from .async_inference import AsyncInferenceManager, AsyncInferenceConfig
from .policy_server import PolicyServer


class RobotClient:
    """
    Robot Client for SmolVLA
    
    Implements Algorithm 1 from paper Section 3.3
    
    Main control loop:
    1. Execute action from queue
    2. Check if should request new prediction (based on g threshold)
    3. Send observation if needed (with similarity filtering)
    4. Receive and integrate new action chunks
    """
    
    def __init__(
        self,
        policy_server: PolicyServer,
        robot_interface: Callable,  # Function to execute action on robot
        config: Optional[AsyncInferenceConfig] = None,
    ):
        self.policy_server = policy_server
        self.robot_interface = robot_interface
        
        # Async inference manager
        self.async_manager = AsyncInferenceManager(config)
        
        # Current request tracking
        self.pending_request_id = None
        
        # Statistics
        self.total_steps = 0
        self.idle_steps = 0  # Steps with empty queue
    
    def run_episode(
        self,
        get_observation: Callable,
        episode_length: int,
        instruction: str,
    ) -> Dict:
        """
        Run one episode with async inference
        
        Args:
            get_observation: Function to get current observation
            episode_length: Maximum episode length
            instruction: Task instruction
            
        Returns:
            episode_info: Dictionary with episode statistics
        """
        self.async_manager.reset()
        episode_start_time = time.time()
        
        for step in range(episode_length):
            step_start_time = time.time()
            
            # 1. Get current observation
            observation = get_observation()
            observation['instruction'] = instruction
            
            # 2. Check if we have pending response
            if self.pending_request_id is not None:
                response = self.policy_server.get_response(self.pending_request_id)
                if response is not None:
                    # Add actions to queue
                    self.async_manager.add_predicted_chunk(response.actions)
                    self.pending_request_id = None
            
            # 3. Get next action from queue
            action = self.async_manager.get_next_action()
            
            if action is None:
                # Queue is empty - force request
                print(f"Step {step}: Queue empty, forcing prediction request")
                self.idle_steps += 1
                
                # Send request
                if self.pending_request_id is None:
                    request_id = self.policy_server.send_request(observation)
                    self.pending_request_id = request_id
                    self.async_manager.mark_request_sent()
                
                # Wait for response (blocking in this case)
                while self.pending_request_id is not None:
                    response = self.policy_server.get_response(self.pending_request_id)
                    if response is not None:
                        self.async_manager.add_predicted_chunk(response.actions)
                        self.pending_request_id = None
                        action = self.async_manager.get_next_action()
                        break
                    time.sleep(0.001)  # Brief sleep
            
            # 4. Execute action
            self.robot_interface(action)
            
            # 5. Check if should request new prediction
            if self.async_manager.should_request_prediction():
                # Check observation similarity
                if self.async_manager.filter_observation(observation):
                    # Send request if not already pending
                    if self.pending_request_id is None:
                        request_id = self.policy_server.send_request(observation)
                        self.pending_request_id = request_id
                        self.async_manager.mark_request_sent()
            
            self.total_steps += 1
            
            # 6. Maintain control frequency
            step_time = time.time() - step_start_time
            sleep_time = self.async_manager.config.delta_t - step_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Episode statistics
        episode_time = time.time() - episode_start_time
        
        return {
            'episode_length': episode_length,
            'episode_time': episode_time,
            'idle_steps': self.idle_steps,
            'idle_ratio': self.idle_steps / episode_length,
            'async_stats': self.async_manager.get_statistics(),
            'server_stats': self.policy_server.get_statistics(),
        }
    
    def step(
        self,
        observation: Dict[str, np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Single step (for manual control loop)
        
        Args:
            observation: Current observation
            
        Returns:
            action: Next action to execute, or None if queue empty
        """
        # Check for pending response
        if self.pending_request_id is not None:
            response = self.policy_server.get_response(self.pending_request_id)
            if response is not None:
                self.async_manager.add_predicted_chunk(response.actions)
                self.pending_request_id = None
        
        # Get next action
        action = self.async_manager.get_next_action()
        
        # Request new prediction if needed
        if self.async_manager.should_request_prediction():
            if self.async_manager.filter_observation(observation):
                if self.pending_request_id is None:
                    request_id = self.policy_server.send_request(observation)
                    self.pending_request_id = request_id
                    self.async_manager.mark_request_sent()
        
        return action
