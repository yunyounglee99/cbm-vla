import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from transformers import AutoModel, AutoProcessor
import math


class PixelShuffleTokenReduction(nn.Module):
    """
    Pixel Shuffle based token reduction
    
    Reduces visual tokens from N to target_tokens using spatial reorganization
    Similar to pixel shuffle in super-resolution but in reverse
    """
    
    def __init__(
        self,
        hidden_dim: int,
        target_tokens: int = 64,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.target_tokens = target_tokens
        
        # Learnable projection for adaptive reduction
        self.reduction_proj = None  # Initialized dynamically
    
    def _init_projection(self, input_tokens: int, device: torch.device):
        """Initialize reduction projection if needed"""
        if self.reduction_proj is None or self.reduction_proj.in_features != input_tokens:
            self.reduction_proj = nn.Linear(
                input_tokens,
                self.target_tokens,
                bias=False
            ).to(device)
            
            # Initialize with averaging weights
            with torch.no_grad():
                # Each output token averages from corresponding input region
                ratio = input_tokens / self.target_tokens
                for i in range(self.target_tokens):
                    start_idx = int(i * ratio)
                    end_idx = int((i + 1) * ratio)
                    self.reduction_proj.weight[i, start_idx:end_idx] = 1.0 / (end_idx - start_idx)
    
    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        Reduce visual tokens
        
        Args:
            visual_tokens: [batch_size, num_tokens, hidden_dim]
            
        Returns:
            reduced_tokens: [batch_size, target_tokens, hidden_dim]
        """
        batch_size, num_tokens, hidden_dim = visual_tokens.shape
        
        # If already at target size, return as is
        if num_tokens == self.target_tokens:
            return visual_tokens
        
        # If smaller than target, use interpolation
        if num_tokens < self.target_tokens:
            return self._upsample_tokens(visual_tokens)
        
        # Try spatial reduction first (for perfect squares)
        spatial_size = int(math.sqrt(num_tokens))
        if spatial_size * spatial_size == num_tokens:
            return self._spatial_reduction(visual_tokens, spatial_size)
        
        # Otherwise use learned projection
        return self._learned_reduction(visual_tokens)
    
    def _spatial_reduction(
        self,
        visual_tokens: torch.Tensor,
        spatial_size: int,
    ) -> torch.Tensor:
        """
        Spatial reduction using pixel shuffle
        
        Args:
            visual_tokens: [batch_size, num_tokens, hidden_dim]
            spatial_size: sqrt(num_tokens)
            
        Returns:
            reduced_tokens: [batch_size, target_tokens, hidden_dim]
        """
        batch_size, num_tokens, hidden_dim = visual_tokens.shape
        
        # Reshape to 2D spatial layout
        # [B, N, D] -> [B, H, W, D]
        tokens_2d = visual_tokens.view(batch_size, spatial_size, spatial_size, hidden_dim)
        
        # Calculate target spatial size
        target_spatial_size = int(math.sqrt(self.target_tokens))
        if target_spatial_size * target_spatial_size != self.target_tokens:
            # Fallback to learned reduction if target is not perfect square
            return self._learned_reduction(visual_tokens)
        
        # Calculate reduction factor
        reduction_factor = spatial_size // target_spatial_size
        
        if reduction_factor == 1:
            return visual_tokens
        
        # Perform spatial pooling
        # [B, H, W, D] -> [B, H//r, W//r, D]
        tokens_reduced = F.avg_pool2d(
            tokens_2d.permute(0, 3, 1, 2),  # [B, D, H, W]
            kernel_size=reduction_factor,
            stride=reduction_factor,
        )
        
        # Reshape back to sequence
        # [B, D, H', W'] -> [B, H'*W', D]
        tokens_reduced = tokens_reduced.flatten(2).transpose(1, 2)
        
        return tokens_reduced
    
    def _learned_reduction(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        Learned linear reduction
        
        Args:
            visual_tokens: [batch_size, num_tokens, hidden_dim]
            
        Returns:
            reduced_tokens: [batch_size, target_tokens, hidden_dim]
        """
        batch_size, num_tokens, hidden_dim = visual_tokens.shape
        
        # Initialize projection if needed
        self._init_projection(num_tokens, visual_tokens.device)
        
        # Apply reduction
        # [B, N, D] -> [B, D, N] -> [B, D, target_N] -> [B, target_N, D]
        tokens_transposed = visual_tokens.transpose(1, 2)  # [B, D, N]
        reduced = self.reduction_proj(tokens_transposed)    # [B, D, target_N]
        reduced = reduced.transpose(1, 2)                   # [B, target_N, D]
        
        return reduced
    
    def _upsample_tokens(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        Upsample tokens if input is smaller than target
        
        Args:
            visual_tokens: [batch_size, num_tokens, hidden_dim]
            
        Returns:
            upsampled_tokens: [batch_size, target_tokens, hidden_dim]
        """
        batch_size, num_tokens, hidden_dim = visual_tokens.shape
        
        # Use linear interpolation
        # [B, N, D] -> [B, D, N] -> [B, D, target_N] -> [B, target_N, D]
        tokens_transposed = visual_tokens.transpose(1, 2).unsqueeze(-1)  # [B, D, N, 1]
        upsampled = F.interpolate(
            tokens_transposed,
            size=(self.target_tokens, 1),
            mode='bilinear',
            align_corners=False,
        )
        upsampled = upsampled.squeeze(-1).transpose(1, 2)  # [B, target_N, D]
        
        return upsampled


class VisionEncoder(nn.Module):
    """
    Vision Encoder for SmolVLA
    
    Uses SigLIP-SO400M from SmolVLM-2 with token reduction
    
    Args:
        model_name: SmolVLM-2 model name
        visual_tokens_per_image: Target number of visual tokens (default: 64)
        image_size: Input image size (default: 512)
        freeze: Whether to freeze vision encoder
    """
    
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        visual_tokens_per_image: int = 64,
        image_size: int = 512,
        freeze: bool = True,
    ):
        super().__init__()
        
        print(f"Loading vision encoder from {model_name}...")
        
        # Load SmolVLM-2 to extract vision encoder
        full_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        
        # Extract vision encoder (SigLIP)
        self.vision_model = full_model.vision_model
        
        # Get vision config
        self.vision_config = full_model.config.vision_config
        self.hidden_dim = self.vision_config.hidden_size
        
        print(f"Vision encoder hidden dim: {self.hidden_dim}")
        
        # Image preprocessing
        self.image_size = image_size
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        
        # Token reduction
        self.visual_tokens_per_image = visual_tokens_per_image
        self.token_reduction = PixelShuffleTokenReduction(
            hidden_dim=self.hidden_dim,
            target_tokens=visual_tokens_per_image,
        )
        
        print(f"Visual tokens per image: {visual_tokens_per_image}")
        
        # Freeze vision encoder if specified
        if freeze:
            self._freeze_vision_encoder()
            print("Vision encoder frozen")
        
        # No image tiling for efficiency (as per paper)
        self.use_tiling = False
    
    def _freeze_vision_encoder(self):
        """Freeze vision encoder parameters"""
        for param in self.vision_model.parameters():
            param.requires_grad = False
    
    def unfreeze_vision_encoder(self):
        """Unfreeze vision encoder parameters"""
        for param in self.vision_model.parameters():
            param.requires_grad = True
        print("Vision encoder unfrozen")
    
    def preprocess_images(
        self,
        images: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Preprocess images for vision encoder
        
        Args:
            images: List of image tensors [C, H, W] or PIL Images
            
        Returns:
            processed_images: [batch_size, num_images, C, H, W]
        """
        # Stack images if needed
        if isinstance(images, list):
            if isinstance(images[0], torch.Tensor):
                # Already tensors
                images = torch.stack(images)
            else:
                # PIL Images - use processor
                processed = self.processor(
                    images=images,
                    return_tensors="pt",
                    size={"height": self.image_size, "width": self.image_size},
                )
                images = processed['pixel_values']
        
        return images
    
    def forward(
        self,
        images: torch.Tensor,
        return_dict: bool = True,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass through vision encoder
        
        Args:
            images: [batch_size, num_images, channels, height, width]
                   or [batch_size, channels, height, width]
            return_dict: Whether to return additional info
            
        Returns:
            visual_features: [batch_size, num_images, num_visual_tokens, hidden_dim]
                            or [batch_size, num_visual_tokens, hidden_dim]
            info_dict: Optional dictionary with additional info
        """
        # Handle both batched and unbatched inputs
        if images.dim() == 4:
            # Single image per batch: [B, C, H, W]
            batch_size = images.shape[0]
            num_images = 1
            images = images.unsqueeze(1)  # [B, 1, C, H, W]
        elif images.dim() == 5:
            # Multiple images per batch: [B, N, C, H, W]
            batch_size, num_images = images.shape[:2]
        else:
            raise ValueError(f"Expected 4D or 5D input, got {images.dim()}D")
        
        # Flatten batch and num_images for processing
        # [B, N, C, H, W] -> [B*N, C, H, W]
        images_flat = images.view(-1, *images.shape[2:])
        
        # Process through vision encoder
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
            vision_outputs = self.vision_model(
                pixel_values=images_flat,
                output_hidden_states=False,
                return_dict=True,
            )
        
        # Get visual features
        visual_features_flat = vision_outputs.last_hidden_state
        # [B*N, num_patches, hidden_dim]
        
        num_original_tokens = visual_features_flat.shape[1]
        
        # Reduce visual tokens
        visual_features_reduced = self.token_reduction(visual_features_flat)
        # [B*N, target_tokens, hidden_dim]
        
        # Reshape back to [B, N, target_tokens, D]
        visual_features = visual_features_reduced.view(
            batch_size, num_images, self.visual_tokens_per_image, self.hidden_dim
        )
        
        if return_dict:
            info = {
                'original_tokens': num_original_tokens,
                'reduced_tokens': self.visual_tokens_per_image,
                'reduction_ratio': num_original_tokens / self.visual_tokens_per_image,
            }
            return visual_features, info
        else:
            return visual_features, None
    
    def encode_images(
        self,
        images: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Convenience method to encode a list of images
        
        Args:
            images: List of image tensors [C, H, W]
            
        Returns:
            visual_features: [batch_size, num_images, num_visual_tokens, hidden_dim]
        """
        # Preprocess
        images_tensor = self.preprocess_images(images)
        
        # Add batch dimension if needed
        if images_tensor.dim() == 4:
            images_tensor = images_tensor.unsqueeze(0)
        
        # Encode
        visual_features, _ = self.forward(images_tensor, return_dict=False)
        
        return visual_features
    
    def get_hidden_dim(self) -> int:
        """Get hidden dimension of vision encoder"""
        return self.hidden_dim
    
    def get_num_visual_tokens(self) -> int:
        """Get number of visual tokens per image"""
        return self.visual_tokens_per_image


class MultiImageVisionEncoder(nn.Module):
    """
    Vision encoder wrapper for handling multiple camera views
    
    SmolVLA typically uses 3 camera views:
    - Top view (OBS_IMAGE_1)
    - Wrist view (OBS_IMAGE_2)
    - Side view (OBS_IMAGE_3)
    """
    
    def __init__(
        self,
        vision_encoder: VisionEncoder,
        num_views: int = 3,
        view_names: Optional[List[str]] = None,
    ):
        super().__init__()
        
        self.vision_encoder = vision_encoder
        self.num_views = num_views
        
        if view_names is None:
            view_names = [f"view_{i}" for i in range(num_views)]
        self.view_names = view_names
        
        # Optional: View-specific embeddings
        self.view_embeddings = nn.Parameter(
            torch.randn(num_views, vision_encoder.get_hidden_dim()) * 0.02
        )
    
    def forward(
        self,
        images: dict,
        add_view_embeddings: bool = True,
    ) -> torch.Tensor:
        """
        Process multi-view images
        
        Args:
            images: Dictionary with view names as keys
                   e.g., {'top': tensor, 'wrist': tensor, 'side': tensor}
                   Each tensor: [batch_size, C, H, W]
            add_view_embeddings: Whether to add view-specific embeddings
            
        Returns:
            visual_features: [batch_size, num_views * num_tokens, hidden_dim]
        """
        batch_size = list(images.values())[0].shape[0]
        all_features = []
        
        # Process each view
        for view_idx, view_name in enumerate(self.view_names):
            if view_name not in images:
                raise ValueError(f"Missing view: {view_name}")
            
            view_images = images[view_name]  # [B, C, H, W]
            
            # Encode
            view_features, _ = self.vision_encoder(
                view_images,
                return_dict=False,
            )
            # [B, num_tokens, D]
            
            # Add view embedding if requested
            if add_view_embeddings:
                view_features = view_features + self.view_embeddings[view_idx]
            
            all_features.append(view_features)
        
        # Concatenate all views
        # [B, num_views * num_tokens, D]
        visual_features = torch.cat(all_features, dim=1)
        
        return visual_features


if __name__ == "__main__":
    # Test vision encoder
    print("Testing VisionEncoder...")
    
    # Initialize
    vision_encoder = VisionEncoder(
        visual_tokens_per_image=64,
        image_size=512,
        freeze=True,
    )
    
    print(f"\nVision encoder config:")
    print(f"  Hidden dim: {vision_encoder.get_hidden_dim()}")
    print(f"  Visual tokens: {vision_encoder.get_num_visual_tokens()}")
    
    # Test with single image
    print("\n1. Testing single image...")
    single_image = torch.randn(2, 3, 512, 512)  # [B, C, H, W]
    features_single, info = vision_encoder(single_image, return_dict=True)
    print(f"   Input shape: {single_image.shape}")
    print(f"   Output shape: {features_single.shape}")
    print(f"   Info: {info}")
    
    # Test with multiple images
    print("\n2. Testing multiple images...")
    multi_images = torch.randn(2, 3, 3, 512, 512)  # [B, N, C, H, W]
    features_multi, info = vision_encoder(multi_images, return_dict=True)
    print(f"   Input shape: {multi_images.shape}")
    print(f"   Output shape: {features_multi.shape}")
    print(f"   Info: {info}")
    
    # Test multi-view wrapper
    print("\n3. Testing MultiImageVisionEncoder...")
    multi_view_encoder = MultiImageVisionEncoder(
        vision_encoder=vision_encoder,
        num_views=3,
        view_names=['top', 'wrist', 'side'],
    )
    
    images_dict = {
        'top': torch.randn(2, 3, 512, 512),
        'wrist': torch.randn(2, 3, 512, 512),
        'side': torch.randn(2, 3, 512, 512),
    }
    
    multi_view_features = multi_view_encoder(images_dict)
    print(f"   Output shape: {multi_view_features.shape}")
    print(f"   Expected: [batch=2, tokens=3*64=192, hidden=1152]")
    
    # Test token reduction
    print("\n4. Testing token reduction ratios...")
    for target_tokens in [16, 32, 64, 128]:
        encoder = VisionEncoder(
            visual_tokens_per_image=target_tokens,
            image_size=512,
        )
        features, info = encoder(single_image, return_dict=True)
        print(f"   Target {target_tokens} tokens: "
              f"reduction ratio {info['reduction_ratio']:.2f}x")
    
    print("\n✓ VisionEncoder test completed!")
