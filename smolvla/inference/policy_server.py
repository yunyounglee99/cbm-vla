"""
Policy Server for Asynchronous Inference
Based on SmolVLA paper Section 3.3

Receives observations from RobotClient and returns action chunks
"""

import torch
import numpy as np
from typing import Dict, Optional
import threading
import queue
from dataclasses import dataclass
import time


@dataclass
class InferenceRequest:
    """Inference request from robot client"""
    request_id: int
    observation: Dict[str, np.ndarray]
    timestamp: float


@dataclass
class InferenceResponse:
    """Inference response to robot client"""
    request_id: int
    actions: np.ndarray  # [chunk_size, action_dim]
    inference_time: float


class PolicyServer:
    """
    Policy Server for SmolVLA
    
    Handles inference requests asynchronously
    Processes observations and returns action chunks
    """
    
    def __init__(
        self,
        model,  # SmolVLA model
        device: str = "cuda",
        max_queue_size: int = 10,
        num_inference_steps: int = 10,
    ):
        self.model = model
        self.device = device
        self.num_inference_steps = num_inference_steps
        
        # Request/response queues
        self.request_queue = queue.Queue(maxsize=max_queue_size)
        self.response_dict = {}
        
        # Threading
        self.is_running = False
        self.worker_thread = None
        
        # Statistics
        self.total_requests = 0
        self.inference_times = []
        
        # Lock for response dict
        self.response_lock = threading.Lock()
    
    def start(self):
        """Start policy server"""
        if self.is_running:
            print("Policy server already running")
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        print(f"Policy server started on {self.device}")
    
    def stop(self):
        """Stop policy server"""
        self.is_running = False
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=5.0)
        print("Policy server stopped")
    
    def _worker_loop(self):
        """Worker loop for processing requests"""
        self.model.eval()
        
        while self.is_running:
            try:
                # Get request with timeout
                request = self.request_queue.get(timeout=0.1)
                
                # Process request
                response = self._process_request(request)
                
                # Store response
                with self.response_lock:
                    self.response_dict[request.request_id] = response
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in worker loop: {e}")
                continue
    
    def _process_request(self, request: InferenceRequest) -> InferenceResponse:
        """
        Process single inference request
        
        Args:
            request: InferenceRequest
            
        Returns:
            response: InferenceResponse
        """
        start_time = time.time()
        
        # Extract observation components
        obs = request.observation
        raw_images = obs['images']  # Numpy array: [Num_Views, C, H, W] or [Num_Views, H, W, C]
        state = torch.from_numpy(obs['state']).to(self.device)
        instruction = obs.get('instruction', "")

        processor = self.model.vision_encoder.processor

        try:
            inputs = processor(
                images=list(raw_images), # Numpy 배열을 리스트로 변환하여 전달
                return_tensors="pt",
                do_resize=True,           # 설정된 크기로 리사이징
                do_rescale=True,          # 1/255 스케일링
                do_normalize=True,        # SigLIP Mean/Std 정규화
            )
            # 전처리된 텐서 (pixel_values) 가져오기
            images = inputs['pixel_values'].to(self.device)
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            # 실패 시 기존 방식대로 하되 float 변환 및 0~1 스케일링이라도 수행
            images = torch.from_numpy(raw_images).to(self.device).float() / 255.0
        
        # Run inference
        with torch.no_grad():
            actions = self.model.predict(
                images=images,
                instruction=instruction,
                robot_state=state,
                num_inference_steps=self.num_inference_steps,
            )
        
        # Convert to numpy
        actions_np = actions.cpu().numpy()
        
        # Compute inference time
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return InferenceResponse(
            request_id=request.request_id,
            actions=actions_np,
            inference_time=inference_time,
        )
    
    def send_request(self, observation: Dict[str, np.ndarray]) -> int:
        """
        Send inference request
        
        Args:
            observation: Dictionary with 'images', 'state', 'instruction'
            
        Returns:
            request_id: ID for this request
        """
        request_id = self.total_requests
        self.total_requests += 1
        
        request = InferenceRequest(
            request_id=request_id,
            observation=observation,
            timestamp=time.time(),
        )
        
        # Add to queue (non-blocking)
        try:
            self.request_queue.put_nowait(request)
        except queue.Full:
            print("Warning: Request queue full, dropping oldest request")
            # Drop oldest and retry
            try:
                self.request_queue.get_nowait()
                self.request_queue.put_nowait(request)
            except:
                pass
        
        return request_id
    
    def get_response(self, request_id: int) -> Optional[InferenceResponse]:
        """
        Get response for request
        
        Args:
            request_id: Request ID
            
        Returns:
            response: InferenceResponse or None if not ready
        """
        with self.response_lock:
            return self.response_dict.pop(request_id, None)
    
    def get_statistics(self) -> Dict:
        """Get server statistics"""
        if not self.inference_times:
            return {}
        
        return {
            'total_requests': self.total_requests,
            'mean_inference_time': np.mean(self.inference_times),
            'std_inference_time': np.std(self.inference_times),
            'queue_size': self.request_queue.qsize(),
        }
