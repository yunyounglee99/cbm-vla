"""
SmolVLA: Small, Efficient, and Capable Vision-Language-Action Model
Based on SmolVLA paper (https://arxiv.org/html/2506.01844v1)

Complete implementation with:
- SmolVLM-2 backbone with layer skipping
- Vision encoder with token reduction
- Flow matching action expert with interleaved attention
- State/Feature/Action projectors
- Normalization support
- Asynchronous inference support

Author: Implementation based on HuggingFace SmolVLA
Paper: SmolVLA: A vision-language-action model for affordable and efficient robotics
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple, Union, Callable
from pathlib import Path
import json
import warnings

# Import model components
from .vlm_backbone import SmolVLMBackbone
from .vision_encoder import VisionEncoder, MultiImageVisionEncoder
from .action_expert import FlowMatchingActionExpert, compute_flow_matching_loss
from .projectors import (
    StateProjector,
    FeatureProjector,
    ActionProjector,
    OutputActionProjector,
    ProjectorConfig,
)

# Import utilities
from utils.normalization import NormalizationManager
from inference.async_inference import AsyncInferenceConfig
from inference.policy_server import PolicyServer
from inference.robot_client import RobotClient


class SmolVLAConfig:
    """
    Configuration for SmolVLA model
    
    Default values from paper Section 4.3:
    - VLM: SmolVLM-2 with L/2 layer skipping
    - Visual tokens: 64 per frame
    - Action chunk: 50 timesteps
    - Expert hidden dim: 0.75 × VLM hidden dim
    - Flow matching: 10 inference steps
    
    Args:
        vlm_model_name: HuggingFace model name for VLM backbone
        vlm_hidden_dim: VLM hidden dimension (auto-detected from model)
        use_vlm_layers: Number of VLM layers to use (None = L/2)
        expert_hidden_dim_ratio: Expert hidden dim = ratio × VLM hidden dim
        
        visual_tokens_per_image: Target visual tokens per frame
        image_size: Input image size
        num_camera_views: Number of camera views (e.g., 3 for top/wrist/side)
        
        state_dim: Robot state dimension (e.g., 7 for 6-DOF + gripper)
        action_dim: Robot action dimension
        
        action_chunk_size: Action chunk size n
        num_expert_layers: Number of action expert layers
        num_attention_heads: Number of attention heads
        
        dropout: Dropout rate
        freeze_vlm: Whether to freeze VLM during training
        freeze_vision_encoder: Whether to freeze vision encoder
        
        beta_alpha: Beta distribution α for flow matching
        beta_beta: Beta distribution β for flow matching
        num_inference_steps: Number of denoising steps
        
        normalize_actions: Whether to normalize actions
        normalize_states: Whether to normalize states
        action_norm_mode: Normalization mode for actions ('min_max' or 'gaussian')
        state_norm_mode: Normalization mode for states
    """
    
    def __init__(
        self,
        # Model architecture
        vlm_model_name: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        vlm_hidden_dim: int = 1536,
        use_vlm_layers: Optional[int] = None,  # None = L/2
        expert_hidden_dim_ratio: float = 0.75,
        
        # Vision encoder
        visual_tokens_per_image: int = 64,
        image_size: int = 512,
        num_camera_views: int = 3,
        
        # Robot configuration
        state_dim: int = 7,  # 6-DOF + gripper
        action_dim: int = 7,
        
        # Action expert
        action_chunk_size: int = 50,
        num_expert_layers: int = 8,
        num_attention_heads: int = 8,
        
        # Training
        dropout: float = 0.1,
        freeze_vlm: bool = True,
        freeze_vision_encoder: bool = True,
        
        # Flow matching
        beta_alpha: float = 0.5,
        beta_beta: float = 0.5,
        num_inference_steps: int = 10,
        
        # Normalization
        normalize_actions: bool = True,
        normalize_states: bool = True,
        action_norm_mode: str = "min_max",
        state_norm_mode: str = "gaussian",
    ):
        # Model architecture
        self.vlm_model_name = vlm_model_name
        self.vlm_hidden_dim = vlm_hidden_dim
        self.use_vlm_layers = use_vlm_layers
        self.expert_hidden_dim = int(vlm_hidden_dim * expert_hidden_dim_ratio)
        
        # Vision
        self.visual_tokens_per_image = visual_tokens_per_image
        self.image_size = image_size
        self.num_camera_views = num_camera_views
        
        # Robot
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Action expert
        self.action_chunk_size = action_chunk_size
        self.num_expert_layers = num_expert_layers
        self.num_attention_heads = num_attention_heads
        
        # Training
        self.dropout = dropout
        self.freeze_vlm = freeze_vlm
        self.freeze_vision_encoder = freeze_vision_encoder
        
        # Flow matching
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        self.num_inference_steps = num_inference_steps
        
        # Normalization
        self.normalize_actions = normalize_actions
        self.normalize_states = normalize_states
        self.action_norm_mode = action_norm_mode
        self.state_norm_mode = state_norm_mode
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'SmolVLAConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def save(self, path: Union[str, Path]):
        """Save config to JSON file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'SmolVLAConfig':
        """Load config from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class SmolVLA(nn.Module):
    """
    SmolVLA: Vision-Language-Action Model
    
    Architecture (from paper Section 3.1):
    ┌─────────────────────────────────────────────────────────────┐
    │ Input: Multi-view Images + Instruction + Robot State       │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ 1. Vision Encoder (SigLIP)                                 │
    │    - Process images → visual tokens (64 per image)         │
    │    - Pixel shuffle token reduction                         │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ 2. State Projector                                         │
    │    - Robot state → VLM token dimension                     │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ 3. VLM Backbone (SmolVLM-2)                                │
    │    - Layer skipping: Use first N=L/2 layers                │
    │    - Concat: [visual | state | text] tokens                │
    │    - Output: VLM features at layer N                       │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ 4. Feature Projector                                       │
    │    - VLM features → Action Expert dimension (0.75x)        │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ 5. Action Expert (Flow Matching)                           │
    │    - Interleaved CA/SA transformer                         │
    │    - Training: Flow matching (Equation 1)                  │
    │    - Inference: 10-step denoising                          │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ Output: Action Chunk (50 timesteps)                        │
    └─────────────────────────────────────────────────────────────┘
    
    Training (Section 3.2):
    - Pretrain on community datasets (~23K episodes)
    - VLM frozen, only Action Expert trained
    - Flow matching objective: L = E[||v_θ(A^τ, o) - u||²]
    
    Inference (Section 3.3):
    - Synchronous: Execute full chunk before next prediction
    - Asynchronous: Decouple prediction and execution
      - Queue threshold g = 0.7
      - Observation similarity filtering
    """
    
    def __init__(self, config: Optional[SmolVLAConfig] = None):
        super().__init__()
        
        if config is None:
            config = SmolVLAConfig()
        
        self.config = config
        
        print("="*70)
        print("Initializing SmolVLA")
        print("="*70)
        print(f"VLM Model: {config.vlm_model_name}")
        print(f"VLM Hidden Dim: {config.vlm_hidden_dim}")
        print(f"Expert Hidden Dim: {config.expert_hidden_dim}")
        print(f"Action Chunk Size: {config.action_chunk_size}")
        print(f"State Dim: {config.state_dim}, Action Dim: {config.action_dim}")
        print("="*70)
        
        # ================================================================
        # 1. Vision Encoder
        # ================================================================
        print("\n[1/6] Loading Vision Encoder...")
        self.vision_encoder = VisionEncoder(
            model_name=config.vlm_model_name,
            visual_tokens_per_image=config.visual_tokens_per_image,
            image_size=config.image_size,
            freeze=config.freeze_vision_encoder,
        )
        print(f"  ✓ Vision encoder loaded")
        print(f"    - Hidden dim: {self.vision_encoder.get_hidden_dim()}")
        print(f"    - Visual tokens per image: {config.visual_tokens_per_image}")
        
        # ================================================================
        # 2. VLM Backbone with Layer Skipping
        # ================================================================
        print("\n[2/6] Loading VLM Backbone...")
        self.vlm_backbone = SmolVLMBackbone(
            model_name=config.vlm_model_name,
            use_layers=config.use_vlm_layers,
            freeze_vlm=config.freeze_vlm,
            visual_tokens_per_image=config.visual_tokens_per_image,
            image_size=config.image_size,
        )
        print(f"  ✓ VLM backbone loaded")
        print(f"    - Total layers: {self.vlm_backbone.total_layers}")
        print(f"    - Using layers: {self.vlm_backbone.get_num_layers_used()}")
        print(f"    - Hidden dim: {self.vlm_backbone.get_hidden_dim()}")
        
        # Get actual hidden dim from VLM
        self.vlm_hidden_dim = self.vlm_backbone.get_hidden_dim()
        
        # ================================================================
        # 3. Projectors
        # ================================================================
        print("\n[3/6] Initializing Projectors...")
        projector_config = ProjectorConfig(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            vlm_hidden_dim=self.vlm_hidden_dim,
            expert_hidden_dim_ratio=config.expert_hidden_dim / self.vlm_hidden_dim,
            dropout=config.dropout,
        )
        
        self.state_projector = projector_config.create_state_projector()
        self.feature_projector = projector_config.create_feature_projector()
        
        print(f"  ✓ Projectors initialized")
        print(f"    - State: {config.state_dim} → {self.vlm_hidden_dim}")
        print(f"    - Feature: {self.vlm_hidden_dim} → {config.expert_hidden_dim}")
        
        # ================================================================
        # 4. Action Expert
        # ================================================================
        print("\n[4/6] Initializing Action Expert...")
        self.action_expert = FlowMatchingActionExpert(
            vlm_hidden_dim=self.vlm_hidden_dim,
            action_dim=config.action_dim,
            chunk_size=config.action_chunk_size,
            num_layers=config.num_expert_layers,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            hidden_dim_ratio=config.expert_hidden_dim / self.vlm_hidden_dim,
        )
        print(f"  ✓ Action expert initialized")
        print(f"    - Layers: {config.num_expert_layers}")
        print(f"    - Attention heads: {config.num_attention_heads}")
        print(f"    - Chunk size: {config.action_chunk_size}")
        
        # ================================================================
        # 5. Multi-view Encoder (Optional)
        # ================================================================
        print("\n[5/6] Setting up Multi-view Support...")
        if config.num_camera_views > 1:
            self.multi_view_encoder = MultiImageVisionEncoder(
                vision_encoder=self.vision_encoder,
                num_views=config.num_camera_views,
            )
            print(f"  ✓ Multi-view encoder initialized ({config.num_camera_views} views)")
        else:
            self.multi_view_encoder = None
            print(f"  ✓ Single-view mode")
        
        # ================================================================
        # 6. Normalization Manager
        # ================================================================
        print("\n[6/6] Initializing Normalization...")
        self.normalization = NormalizationManager(
            action_dim=config.action_dim,
            state_dim=config.state_dim,
            action_mode=config.action_norm_mode,
            state_mode=config.state_norm_mode,
        )
        print(f"  ✓ Normalization initialized")
        print(f"    - Action normalization: {config.action_norm_mode}")
        print(f"    - State normalization: {config.state_norm_mode}")
        
        # ================================================================
        # Model Information
        # ================================================================
        self._print_model_info()
    
    def _print_model_info(self):
        """Print model information and parameter counts"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        vlm_params = sum(p.numel() for p in self.vlm_backbone.parameters())
        vision_params = sum(p.numel() for p in self.vision_encoder.parameters())
        expert_params = sum(p.numel() for p in self.action_expert.parameters())
        projector_params = (
            sum(p.numel() for p in self.state_projector.parameters()) +
            sum(p.numel() for p in self.feature_projector.parameters())
        )
        
        print("\n" + "="*70)
        print("SmolVLA Model Summary")
        print("="*70)
        print(f"Total Parameters:      {total_params / 1e6:>10.2f}M")
        print(f"Trainable Parameters:  {trainable_params / 1e6:>10.2f}M")
        print(f"Frozen Parameters:     {(total_params - trainable_params) / 1e6:>10.2f}M")
        print("-"*70)
        print("Breakdown:")
        print(f"  VLM Backbone:        {vlm_params / 1e6:>10.2f}M  {'(frozen)' if self.config.freeze_vlm else '(trainable)'}")
        print(f"  Vision Encoder:      {vision_params / 1e6:>10.2f}M  {'(frozen)' if self.config.freeze_vision_encoder else '(trainable)'}")
        print(f"  Action Expert:       {expert_params / 1e6:>10.2f}M  (trainable)")
        print(f"  Projectors:          {projector_params / 1e6:>10.2f}M  (trainable)")
        print("="*70 + "\n")
    
    def forward(
        self,
        images: torch.Tensor,
        instruction: Union[str, List[str]],
        robot_state: torch.Tensor,
        actions_gt: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        normalize: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            images: [batch_size, num_views, C, H, W] or [B, C, H, W]
                   Multi-view images from robot cameras
            instruction: String or list of strings
                        Natural language task instruction
            robot_state: [batch_size, state_dim]
                        Robot proprioceptive state (joint positions + gripper)
            actions_gt: [batch_size, chunk_size, action_dim] (optional)
                       Ground truth actions for training
            return_loss: Whether to compute and return loss (training mode)
            normalize: Whether to apply normalization
            
        Returns:
            outputs: Dictionary containing:
                Training mode (return_loss=True):
                    - loss: Flow matching loss (scalar)
                    - info: Dictionary with loss components
                
                Inference mode (return_loss=False):
                    - actions: [batch_size, chunk_size, action_dim]
                    - vlm_features: [batch_size, seq_len, vlm_hidden_dim]
                    - info: Dictionary with additional information
        """
        batch_size = images.shape[0]
        device = images.device
        
        # ================================================================
        # Normalization (if enabled)
        # ================================================================
        if normalize and self.config.normalize_actions:
            actions_gt_norm = None
            if actions_gt is not None:
                actions_gt_norm, _ = self.normalization.normalize(
                    actions=actions_gt, state=None
                )
        else:
            actions_gt_norm = actions_gt
        
        if normalize and self.config.normalize_states:
            _, state_norm = self.normalization.normalize(
                actions=None, state=robot_state
            )
        else:
            state_norm = robot_state
        
        # ================================================================
        # 1. Tokenize instruction
        # ================================================================
        if isinstance(instruction, str):
            instruction = [instruction] * batch_size
            
        # [수정됨] 이미지 개수 파악 (단일 이미지, 다중 뷰, 딕셔너리 형태 고려)
        if isinstance(images, dict):
            num_imgs = len(images)
        elif images.dim() == 5:
            num_imgs = images.shape[1]
        else:
            num_imgs = 1
            
        # Instruct 모델을 위한 Chat Template 적용
        formatted_instructions = []
        for inst in instruction:
            # [수정됨] 텍스트 앞에 이미지 개수만큼 <image> 토큰 플레이스홀더 추가
            content = [{"type": "image"}] * num_imgs
            content.append({"type": "text", "text": inst})
            
            messages = [
                {"role": "user", "content": content}
            ]
            # SmolVLM2 포맷에 맞춰 <|im_start|> 및 <image> 스페셜 토큰 자동 맵핑
            formatted_text = self.vlm_backbone.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted_instructions.append(formatted_text)
        
        text_inputs = self.vlm_backbone.processor.tokenizer(
            formatted_instructions,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        
        # ================================================================
        # 2. Project robot state to VLM tokens
        # ================================================================
        state_tokens = self.state_projector(state_norm)
        # [B, 1, vlm_hidden_dim]
        
        # ================================================================
        # 3. Forward through VLM backbone with layer skipping
        # ================================================================
        # 1. Vision Encoder를 통한 이미지 토큰화 (Multi-view 처리 포함)
        if self.multi_view_encoder is not None and isinstance(images, dict):
            visual_features = self.multi_view_encoder(images)
        else:
            visual_features, _ = self.vision_encoder(images, return_dict=False)

        # 2. VLM Backbone에 토큰 전달 (layer skipping 적용)
        vlm_features, vlm_info = self.vlm_backbone.forward_with_layer_skipping(
            visual_features=visual_features,
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask'],
            state_features=state_tokens,
            return_dict=True
        )
        # [B, seq_len_total, vlm_hidden_dim]
        # seq_len_total = num_visual_tokens + 1 (state) + text_len
        
        # ================================================================
        # 4. Project VLM features to Action Expert dimension
        # ================================================================
        expert_features = self.feature_projector(vlm_features)
        # [B, seq_len_total, expert_hidden_dim]
        
        # ================================================================
        # 5. Training or Inference
        # ================================================================
        if return_loss and actions_gt_norm is not None:
            # ============================================================
            # Training Mode: Compute flow matching loss
            # ============================================================
            loss, loss_info = compute_flow_matching_loss(
                action_expert=self.action_expert,
                actions_gt=actions_gt_norm,
                vlm_features=expert_features,
                attention_mask=text_inputs['attention_mask'],
                alpha=self.config.beta_alpha,
                beta=self.config.beta_beta,
            )
            
            return {
                'loss': loss,
                'info': loss_info,
            }
        else:
            # ============================================================
            # Inference Mode: Sample actions
            # ============================================================
            actions = self.action_expert.predict_action_chunk(
                vlm_features=expert_features,
                attention_mask=text_inputs['attention_mask'],
                num_inference_steps=self.config.num_inference_steps,
            )
            # [B, chunk_size, action_dim]
            
            # Denormalize actions
            if normalize and self.config.normalize_actions:
                actions, _ = self.normalization.denormalize(
                    actions=actions, state=None
                )
            
            return {
                'actions': actions,
                'vlm_features': vlm_features,
                'expert_features': expert_features,
                'info': vlm_info,
            }
    
    @torch.no_grad()
    def predict(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]],
        instruction: str,
        robot_state: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Predict action chunk (inference only)
        
        This is a convenience method for single-sample inference.
        
        Args:
            images: Images in various formats:
                   - Tensor: [num_views, C, H, W] or [C, H, W]
                   - List: List of image tensors
                   - Dict: {'view_name': tensor} for multi-view
            instruction: Natural language instruction
            robot_state: Robot state [state_dim] or [1, state_dim]
            num_inference_steps: Number of denoising steps 
                                (default: config value)
            normalize: Whether to apply normalization
            
        Returns:
            actions: [chunk_size, action_dim]
                    Predicted action chunk
        """
        self.eval()
        
        # ================================================================
        # Handle different input formats
        # ================================================================
        if isinstance(images, dict):
            # Multi-view dictionary: {'top': tensor, 'wrist': tensor, ...}
            if self.multi_view_encoder is None:
                raise ValueError("Multi-view encoder not initialized")
            # Stack in consistent order
            images = torch.stack([images[k] for k in sorted(images.keys())])
        elif isinstance(images, list):
            # List of tensors
            images = torch.stack(images)
        
        # Add batch dimension if needed
        if images.dim() == 3:
            # Single image: [C, H, W] -> [1, C, H, W]
            images = images.unsqueeze(0)
        elif images.dim() == 4:
            # Multi-view: [N, C, H, W] -> [1, N, C, H, W]
            images = images.unsqueeze(0)
        
        if robot_state.dim() == 1:
            # Single state: [state_dim] -> [1, state_dim]
            robot_state = robot_state.unsqueeze(0)
        
        # ================================================================
        # Override inference steps if provided
        # ================================================================
        original_steps = self.config.num_inference_steps
        if num_inference_steps is not None:
            self.config.num_inference_steps = num_inference_steps
        
        # ================================================================
        # Forward pass
        # ================================================================
        outputs = self.forward(
            images=images,
            instruction=instruction,
            robot_state=robot_state,
            return_loss=False,
            normalize=normalize,
        )
        
        # Restore original steps
        self.config.num_inference_steps = original_steps
        
        # Remove batch dimension
        actions = outputs['actions'].squeeze(0)  # [chunk_size, action_dim]
        
        return actions
    
    def train_step(
        self,
        images: torch.Tensor,
        instruction: Union[str, List[str]],
        robot_state: torch.Tensor,
        actions_gt: torch.Tensor,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Single training step
        
        Args:
            images: [batch_size, num_views, C, H, W]
            instruction: Instruction(s)
            robot_state: [batch_size, state_dim]
            actions_gt: [batch_size, chunk_size, action_dim]
            normalize: Whether to apply normalization
            
        Returns:
            loss: Scalar loss
            info: Dictionary with loss components and statistics
        """
        self.train()
        
        # Update normalization statistics during training
        if normalize:
            self.normalization.update_from_batch(
                actions=actions_gt,
                states=robot_state,
            )
        
        outputs = self.forward(
            images=images,
            instruction=instruction,
            robot_state=robot_state,
            actions_gt=actions_gt,
            return_loss=True,
            normalize=normalize,
        )
        
        return outputs['loss'], outputs['info']
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """
        Get trainable parameters
        
        By default, only Action Expert and Projectors are trainable.
        VLM and Vision Encoder are frozen (as per paper Section 3.2).
        
        Returns:
            trainable_params: List of trainable parameters
        """
        trainable_params = []
        
        # Action Expert (always trainable)
        trainable_params.extend(self.action_expert.parameters())
        
        # Projectors (always trainable)
        trainable_params.extend(self.state_projector.parameters())
        trainable_params.extend(self.feature_projector.parameters())
        
        # VLM Backbone (only if not frozen)
        if not self.config.freeze_vlm:
            trainable_params.extend(self.vlm_backbone.parameters())
        
        # Vision Encoder (only if not frozen)
        if not self.config.freeze_vision_encoder:
            trainable_params.extend(self.vision_encoder.parameters())
        
        return trainable_params
    
    def unfreeze_vlm(self):
        """Unfreeze VLM backbone for fine-tuning"""
        self.vlm_backbone.unfreeze_vlm()
        self.config.freeze_vlm = False
        print("✓ VLM backbone unfrozen")
    
    def unfreeze_vision_encoder(self):
        """Unfreeze vision encoder for fine-tuning"""
        self.vision_encoder.unfreeze_vision_encoder()
        self.config.freeze_vision_encoder = False
        print("✓ Vision encoder unfrozen")
    
    def freeze_vlm(self):
        """Freeze VLM backbone"""
        self.vlm_backbone._freeze_vlm()
        self.config.freeze_vlm = True
        print("✓ VLM backbone frozen")
    
    def freeze_vision_encoder(self):
        """Freeze vision encoder"""
        self.vision_encoder._freeze_vision_encoder()
        self.config.freeze_vision_encoder = True
        print("✓ Vision encoder frozen")
    
    # ====================================================================
    # Asynchronous Inference Support
    # ====================================================================
    
    def create_policy_server(
        self,
        device: str = "cuda",
        num_inference_steps: Optional[int] = None,
        max_queue_size: int = 10,
    ) -> PolicyServer:
        """
        Create policy server for asynchronous inference
        
        The policy server runs in a separate thread and processes
        inference requests asynchronously.
        
        Args:
            device: Device for inference
            num_inference_steps: Override config value
            max_queue_size: Maximum request queue size
            
        Returns:
            server: PolicyServer instance
        """
        if num_inference_steps is None:
            num_inference_steps = self.config.num_inference_steps
        
        server = PolicyServer(
            model=self,
            device=device,
            num_inference_steps=num_inference_steps,
            max_queue_size=max_queue_size,
        )
        
        print(f"✓ Policy server created on {device}")
        return server
    
    def create_robot_client(
        self,
        policy_server: PolicyServer,
        robot_interface: Callable,
        config: Optional[AsyncInferenceConfig] = None,
    ) -> RobotClient:
        """
        Create robot client for asynchronous inference
        
        The robot client executes actions while requesting predictions
        asynchronously from the policy server.
        
        Args:
            policy_server: PolicyServer instance
            robot_interface: Function to execute actions on robot
                          Signature: robot_interface(action: np.ndarray) -> None
            config: AsyncInferenceConfig (optional)
            
        Returns:
            client: RobotClient instance
        """
        if config is None:
            config = AsyncInferenceConfig(
                chunk_size=self.config.action_chunk_size,
            )
        
        client = RobotClient(
            policy_server=policy_server,
            robot_interface=robot_interface,
            config=config,
        )
        
        print(f"✓ Robot client created")
        print(f"  Queue threshold g = {config.queue_threshold}")
        print(f"  Similarity threshold ε = {config.similarity_threshold}")
        
        return client
    
    # ====================================================================
    # Save/Load Methods (HuggingFace Style)
    # ====================================================================
    
    def save_pretrained(self, save_directory: Union[str, Path]):
        """
        Save model to directory (HuggingFace style)
        
        Saves:
        - config.json: Model configuration
        - pytorch_model.bin: Model weights
        - normalization.bin: Normalization statistics
        
        Args:
            save_directory: Path to save directory
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save config
        self.config.save(save_directory / "config.json")
        
        # Save model weights
        torch.save(self.state_dict(), save_directory / "pytorch_model.bin")
        
        # Save normalization statistics
        torch.save(
            self.normalization.state_dict(),
            save_directory / "normalization.bin"
        )
        
        print(f"✓ Model saved to {save_directory}")
        print(f"  - config.json")
        print(f"  - pytorch_model.bin")
        print(f"  - normalization.bin")
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        **kwargs,
    ) -> 'SmolVLA':
        """
        Load pretrained model (HuggingFace style)
        
        Args:
            pretrained_model_name_or_path: Path or HuggingFace model name
                Can be:
                - Local path: "./checkpoints/smolvla"
                - HuggingFace Hub: "username/smolvla-model"
            **kwargs: Additional arguments to override config
            
        Returns:
            model: Loaded SmolVLA model
        """
        path = Path(pretrained_model_name_or_path)
        
        if path.is_dir():
            # ============================================================
            # Load from local directory
            # ============================================================
            print(f"Loading model from local directory: {path}")
            
            # Load config
            config = SmolVLAConfig.load(path / "config.json")
            
            # Override config with kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    print(f"  Overriding config.{key} = {value}")
            
            # Initialize model
            model = cls(config)
            
            # Load weights
            state_dict = torch.load(
                path / "pytorch_model.bin",
                map_location='cpu',
            )
            model.load_state_dict(state_dict, strict=False)
            
            # Load normalization statistics if available
            norm_path = path / "normalization.bin"
            if norm_path.exists():
                norm_state_dict = torch.load(norm_path, map_location='cpu')
                model.normalization.load_state_dict(norm_state_dict)
                print("  ✓ Normalization statistics loaded")
            
            print(f"✓ Model loaded successfully from {path}")
            
            return model
        else:
            # ============================================================
            # Try to load from HuggingFace Hub
            # ============================================================
            # ============================================================
            # Try to load from HuggingFace Hub (Modified for safetensors & Pretrained base)
            # ============================================================
            try:
                from huggingface_hub import hf_hub_download
                from safetensors.torch import load_file  # safetensors 모듈 추가
                
                print(f"Loading model from HuggingFace Hub: {pretrained_model_name_or_path}")
                
                try:
                    config_path = hf_hub_download(
                        repo_id=pretrained_model_name_or_path,
                        filename="config.json",
                    )
                    config = SmolVLAConfig.load(config_path)
                except Exception as e:
                    print(f"Warning: Could not load config.json ({e}). Using default SmolVLAConfig.")
                    config = SmolVLAConfig()
                
                # Override with kwargs
                for key, value in kwargs.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                
                # 1. 모델 초기화 (이때 SmolVLM2의 기본 가중치가 먼저 로드되고 Action Expert는 빈 껍데기로 생성됨)
                model = cls(config)
                
                # 2. 사전 학습된 SmolVLA 전체 파라미터 로드 (safetensors 우선 시도)
                try:
                    weights_path = hf_hub_download(
                        repo_id=pretrained_model_name_or_path,
                        filename="model.safetensors",
                    )
                    state_dict = load_file(weights_path)
                    print("  ✓ Loaded full pre-trained weights from model.safetensors")
                except:
                    weights_path = hf_hub_download(
                        repo_id=pretrained_model_name_or_path,
                        filename="pytorch_model.bin",
                    )
                    state_dict = torch.load(weights_path, map_location='cpu')
                    print("  ✓ Loaded full pre-trained weights from pytorch_model.bin")
                
                # 3. Key Mapping 로직 (중요)
                # 공식 저장소의 파라미터 이름에 'model.' 또는 'policy.' 접두사가 붙어있을 경우 
                # 현재 구현된 클래스 변수명과 맞추기 위해 제거합니다.
                cleaned_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("policy."):
                        new_k = k.replace("policy.", "action_expert.", 1)
                    elif k.startswith("model.vision_model."):
                        new_k = k.replace("model.vision_model.", "vision_encoder.vision_model.", 1)
                    elif k.startswith("model.language_model."):
                        new_k = k.replace("model.language_model.", "vlm_backbone.vlm.language_model.", 1)
                    elif k.startswith("model."):
                        new_k = k.replace("model.", "", 1)
                    else:
                        new_k = k
                        
                    cleaned_state_dict[new_k] = v

                # 4. 가중치 로드 및 누락 키 확인
                missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
                
                # [중요] 디버깅을 위해 로드되지 못한 핵심 가중치를 반드시 출력
                if missing:
                    print("\n[WARNING] The following keys were NOT loaded (Randomly Initialized):")
                    for m in missing:
                        if "vlm_backbone" in m or "action_expert" in m or "vision_encoder" in m:
                            print(f"  - {m}")
                    print("If core components are missing, check the key mapping logic.\n")

                # Load normalization if available
                try:
                    norm_path = hf_hub_download(
                        repo_id=pretrained_model_name_or_path,
                        filename="normalization.bin",
                    )
                    norm_state_dict = torch.load(norm_path, map_location='cpu')
                    model.normalization.load_state_dict(norm_state_dict)
                    print("  ✓ Normalization statistics loaded")
                except:
                    print("  - No normalization.bin found in repository. Using uninitialized normalizer.")
                
                print(f"✓ SmolVLA Pre-trained Model successfully loaded and ready for finetuning!")
                
                return model
                
            except Exception as e:
                raise ValueError(
                    f"Could not load model from {pretrained_model_name_or_path}.\n"
                    f"Error: {e}"
                )
    
    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload SmolVLA model",
        private: bool = False,
    ):
        """
        Push model to HuggingFace Hub
        
        Args:
            repo_id: Repository ID (username/model-name)
            commit_message: Commit message
            private: Whether to make repo private
        """
        from huggingface_hub import HfApi, create_repo
        import tempfile
        
        print(f"Pushing model to HuggingFace Hub: {repo_id}")
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model
            self.save_pretrained(tmpdir)
            
            # Create repo
            create_repo(repo_id, private=private, exist_ok=True)
            
            # Upload
            api = HfApi()
            api.upload_folder(
                folder_path=tmpdir,
                repo_id=repo_id,
                commit_message=commit_message,
            )
            
            print(f"✓ Model pushed to https://huggingface.co/{repo_id}")


# ========================================================================
# Convenience Functions
# ========================================================================

def create_smolvla(
    state_dim: int = 7,
    action_dim: int = 7,
    pretrained: bool = False,
    device: str = "cuda",
    **kwargs,
) -> SmolVLA:
    """
    Convenience function to create SmolVLA model
    
    Args:
        state_dim: Robot state dimension
        action_dim: Robot action dimension
        pretrained: Whether to load pretrained weights (not implemented yet)
        device: Device to load on
        **kwargs: Additional config arguments
        
    Returns:
        model: SmolVLA model
        
    Example:
        >>> model = create_smolvla(state_dim=7, action_dim=7, device="cuda")
        >>> actions = model.predict(images, "pick up the cube", robot_state)
    """
    config = SmolVLAConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        **kwargs,
    )
    
    model = SmolVLA(config)
    model = model.to(device)
    
    if pretrained:
        warnings.warn(
            "Pretrained weights loading not yet implemented. "
            "Initialize from scratch."
        )
    
    return model


# ========================================================================
# Test/Demo
# ========================================================================

if __name__ == "__main__":
    print("Testing SmolVLA...")
    
    # ====================================================================
    # 1. Create model
    # ====================================================================
    print("\n" + "="*70)
    print("1. Creating SmolVLA model")
    print("="*70)
    
    model = create_smolvla(
        state_dim=7,
        action_dim=7,
        device="cpu",  # Use CPU for testing
    )
    
    # ====================================================================
    # 2. Create dummy inputs
    # ====================================================================
    print("\n" + "="*70)
    print("2. Creating dummy inputs")
    print("="*70)
    
    batch_size = 2
    num_views = 3
    
    images = torch.randn(batch_size, num_views, 3, 512, 512)
    instruction = ["Pick up the red cube", "Place the cube in the box"]
    robot_state = torch.randn(batch_size, 7)
    actions_gt = torch.randn(batch_size, 50, 7)
    
    print(f"Images shape: {images.shape}")
    print(f"Instructions: {instruction}")
    print(f"Robot state shape: {robot_state.shape}")
    print(f"Ground truth actions shape: {actions_gt.shape}")
    
    # ====================================================================
    # 3. Test training step
    # ====================================================================
    print("\n" + "="*70)
    print("3. Testing training step")
    print("="*70)
    
    loss, info = model.train_step(
        images=images,
        instruction=instruction,
        robot_state=robot_state,
        actions_gt=actions_gt,
    )
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Info: {info}")
    
    # ====================================================================
    # 4. Test inference
    # ====================================================================
    print("\n" + "="*70)
    print("4. Testing inference")
    print("="*70)
    
    model.eval()
    with torch.no_grad():
        outputs = model.forward(
            images=images,
            instruction=instruction,
            robot_state=robot_state,
            return_loss=False,
        )
    
    actions_pred = outputs['actions']
    print(f"Predicted actions shape: {actions_pred.shape}")
    print(f"Actions mean: {actions_pred.mean().item():.4f}")
    print(f"Actions std: {actions_pred.std().item():.4f}")
    
    # ====================================================================
    # 5. Test predict method (single sample)
    # ====================================================================
    print("\n" + "="*70)
    print("5. Testing predict method (single sample)")
    print("="*70)
    
    single_image = torch.randn(num_views, 3, 512, 512)
    single_state = torch.randn(7)
    
    action_chunk = model.predict(
        images=single_image,
        instruction="Pick up the cube",
        robot_state=single_state,
    )
    
    print(f"Action chunk shape: {action_chunk.shape}")
    print(f"Expected shape: [50, 7]")
    
    # ====================================================================
    # 6. Test save/load
    # ====================================================================
    print("\n" + "="*70)
    print("6. Testing save/load")
    print("="*70)
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        model.save_pretrained(tmpdir)
        
        # Load
        loaded_model = SmolVLA.from_pretrained(tmpdir)
        
        print("✓ Model saved and loaded successfully")
        
        # Test that loaded model produces same output
        with torch.no_grad():
            output_original = model.predict(single_image, "test", single_state)
            output_loaded = loaded_model.predict(single_image, "test", single_state)
        
        diff = (output_original - output_loaded).abs().max()
        print(f"Max difference between original and loaded: {diff.item():.6f}")
    
    # ====================================================================
    # 7. Test parameter counts
    # ====================================================================
    print("\n" + "="*70)
    print("7. Testing parameter counts")
    print("="*70)
    
    trainable_params = model.get_trainable_parameters()
    trainable_count = sum(p.numel() for p in trainable_params)
    
    print(f"Trainable parameters: {trainable_count / 1e6:.2f}M")
    print("Expected: ~100M (Action Expert + Projectors)")
    
    print("\n" + "="*70)
    print("✓ All tests completed successfully!")
    print("="*70)
