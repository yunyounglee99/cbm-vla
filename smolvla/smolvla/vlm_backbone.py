import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict
from transformers import AutoModel, AutoProcessor, AutoConfig
from transformers.modeling_outputs import BaseModelOutput


class SmolVLMBackbone(nn.Module):
    """
    SmolVLM-2 Backbone with Layer Skipping
    
    Architecture:
    - Vision Encoder: SigLIP-SO400M
    - Language Decoder: SmolLM2-1.7B
    - Layer Skipping: Use first N=L/2 layers only
    
    Args:
        model_name: HuggingFace model name
        use_layers: Number of layers to use (default: L/2)
        freeze_vlm: Whether to freeze VLM parameters
        visual_tokens_per_image: Target number of visual tokens (default: 64)
    """
    
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        use_layers: Optional[int] = None,
        freeze_vlm: bool = True,
        visual_tokens_per_image: int = 64,
        image_size: int = 512,
    ):
        super().__init__()
        
        print(f"Loading SmolVLM-2 from {model_name}...")
        
        # Load config first to get architecture details
        self.config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load pretrained SmolVLM-2
        self.vlm = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.total_layers = self.vlm.language_model.model.layers.__len__()
        
        # Layer skipping: Use only first N layers (N=L/2 by default)
        self.use_layers = use_layers if use_layers is not None else self.total_layers // 2
        assert self.use_layers <= self.total_layers, \
            f"use_layers ({self.use_layers}) must be <= total_layers ({self.total_layers})"
        
        print(f"Layer Skipping: Using {self.use_layers}/{self.total_layers} layers")

        self.vlm.language_model.model.layers = self.vlm.language_model.model.layers[:self.use_layers]
        
        # Visual token reduction settings
        self.visual_tokens_per_image = visual_tokens_per_image
        self.image_size = image_size
        
        # Get hidden dimension from language model
        self.hidden_dim = self.config.text_config.hidden_size
        
        # [FIX] Pre-register adaptive pool as a proper submodule.
        # Previously: created dynamically via hasattr() check in _adaptive_reduce().
        # Problem: Dynamically created nn.Linear was not tracked by Module system,
        #          so weights were silently dropped from state_dict().
        # Fix: Register with a reasonable default size; resize if needed.
        self._adaptive_pool = None  # Will be registered on first use via _get_adaptive_pool()
        
        # Freeze VLM if specified (for efficiency during action expert training)
        if freeze_vlm:
            self._freeze_vlm()
            print("VLM parameters frozen")
        
        # Cache for efficient inference
        self._cache_enabled = False
        self._kv_cache = None
    
    def _freeze_vlm(self):
        """Freeze all VLM parameters"""
        for param in self.vlm.parameters():
            param.requires_grad = False
    
    def unfreeze_vlm(self):
        """Unfreeze VLM parameters for fine-tuning"""
        for param in self.vlm.parameters():
            param.requires_grad = True
        print("VLM parameters unfrozen")
    
    def _reduce_visual_tokens(
        self, 
        visual_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Reduce visual tokens using spatial pooling
        
        Args:
            visual_features: [batch_size, num_tokens, hidden_dim]
            
        Returns:
            reduced_features: [batch_size, target_tokens, hidden_dim]
        """
        batch_size, num_tokens, hidden_dim = visual_features.shape
        
        # If already at target size, return as is
        if num_tokens == self.visual_tokens_per_image:
            return visual_features
        
        # Calculate reduction factor
        reduction_factor = num_tokens // self.visual_tokens_per_image
        
        if reduction_factor <= 1:
            return visual_features
        
        # Reshape for pooling
        # [B, N, D] -> [B, sqrt(N), sqrt(N), D]
        spatial_size = int(num_tokens ** 0.5)
        if spatial_size * spatial_size != num_tokens:
            # If not perfect square, use adaptive pooling
            return self._adaptive_reduce(visual_features)
        
        features_2d = visual_features.view(
            batch_size, spatial_size, spatial_size, hidden_dim
        )
        
        # Spatial reduction via average pooling
        r = int(reduction_factor ** 0.5)
        if r < 1:
            return self._adaptive_reduce(visual_features)
        
        h_new, w_new = spatial_size // r, spatial_size // r
        
        # Use avg_pool2d for clean spatial reduction
        # [B, H, W, D] -> [B, D, H, W]
        features_permuted = features_2d.permute(0, 3, 1, 2)
        features_reduced = torch.nn.functional.avg_pool2d(
            features_permuted, kernel_size=r, stride=r
        )
        # [B, D, H', W'] -> [B, H'*W', D]
        features_reduced = features_reduced.flatten(2).transpose(1, 2)
        
        return features_reduced
    
    def _get_adaptive_pool(
        self,
        num_tokens: int,
        device: torch.device,
    ) -> nn.Linear:
        """
        [FIX] Get or create adaptive pool layer as a registered submodule.
        
        Previously: Used hasattr() to check and dynamically assign nn.Linear.
        Problem: The layer was never registered as a submodule, so:
          1. state_dict() didn't include its weights
          2. .to(device) didn't move it
          3. Save/load silently lost trained weights
        
        Fix: Use register_module() for proper tracking.
        """
        if (self._adaptive_pool is None 
                or self._adaptive_pool.in_features != num_tokens):
            pool = nn.Linear(num_tokens, self.visual_tokens_per_image).to(device)
            # Initialize with averaging weights
            with torch.no_grad():
                pool.weight.zero_()
                ratio = num_tokens / self.visual_tokens_per_image
                for i in range(self.visual_tokens_per_image):
                    start = int(i * ratio)
                    end = min(int((i + 1) * ratio), num_tokens)
                    if end > start:
                        pool.weight[i, start:end] = 1.0 / (end - start)
                if pool.bias is not None:
                    pool.bias.zero_()
            # Register as submodule so it appears in state_dict
            self._adaptive_pool = pool
            # Force re-registration
            self.register_module('_adaptive_pool', pool)
        return self._adaptive_pool
    
    def _adaptive_reduce(
        self, 
        visual_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Adaptive pooling for non-square token arrangements
        
        [FIX] Now uses _get_adaptive_pool() which properly registers the layer.
        
        Args:
            visual_features: [batch_size, num_tokens, hidden_dim]
            
        Returns:
            reduced_features: [batch_size, target_tokens, hidden_dim]
        """
        batch_size, num_tokens, hidden_dim = visual_features.shape
        
        pool = self._get_adaptive_pool(num_tokens, visual_features.device)
        
        # [B, N, D] -> [B, D, N] -> [B, D, target_N] -> [B, target_N, D]
        features_transposed = visual_features.transpose(1, 2)
        reduced = pool(features_transposed)
        reduced = reduced.transpose(1, 2)
        
        return reduced
    
    def prepare_inputs(
        self,
        images: List[torch.Tensor],
        instruction: str,
        robot_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for VLM
        
        Args:
            images: List of image tensors [C, H, W]
            instruction: Natural language instruction
            robot_state: Robot state vector [state_dim]
            
        Returns:
            inputs: Dictionary with processed inputs
        """
        # Stack images
        images_tensor = torch.stack(images).unsqueeze(0)  # [1, N_img, C, H, W]
        
        # Process through processor
        text_inputs = self.processor.tokenizer(
            instruction,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        return {
            'images': images_tensor,
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
            'robot_state': robot_state
        }
    
    def forward_with_layer_skipping(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_features: torch.Tensor,
        state_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with layer skipping
        
        [FIX] Multi-<image> token handling:
        
        Original bug: Only image_positions[0] was used, meaning only the first
        <image> token was replaced with visual features. When the chat template
        generates multiple <image> placeholders (one per camera view), the
        remaining <image> tokens stayed as raw text embeddings in the sequence.
        
        Fix: Replace ALL <image> tokens with the corresponding visual features.
        Strategy: 
          - Find all <image> token positions
          - Remove all <image> embeddings from the text sequence  
          - Insert the full visual feature block at the position of the first
            <image> token (since visual_features is already flattened from
            all views by the caller in smolvla.py)
        
        This ensures the visual features correctly replace the placeholder
        tokens regardless of how many <image> tokens the template generates.
        """
        batch_size = input_ids.shape[0]
        
        # Flatten visual features
        # [B, N_img, N_tokens, D] -> [B, N_img*N_tokens, D]
        if visual_features.dim() == 4:
            num_images, num_visual_tokens = visual_features.shape[1:3]
            visual_features_flat = visual_features.view(
                batch_size, num_images * num_visual_tokens, self.hidden_dim
            )
        else:
            # Already [B, total_visual_tokens, D]
            visual_features_flat = visual_features

        text_embeds = self.vlm.language_model.model.embed_tokens(input_ids)
        
        # Get <image> token ID
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        
        new_inputs_embeds = []
        new_attention_masks = []
        
        for i in range(batch_size):
            # [FIX] Find ALL <image> token positions, not just the first one
            image_mask = (input_ids[i] == image_token_id)
            image_positions = image_mask.nonzero(as_tuple=True)[0]
            
            if len(image_positions) > 0 and visual_features_flat is not None:
                num_vis_tokens = visual_features_flat.shape[1]
                
                # [FIX] Build the merged embedding by:
                # 1. Taking text before the first <image> token
                # 2. Inserting ALL visual features
                # 3. Taking text after the last <image> token
                # This removes ALL <image> placeholder tokens and replaces
                # them with the actual visual feature block.
                
                first_img_pos = image_positions[0].item()
                last_img_pos = image_positions[-1].item()
                
                # Text before first <image>
                pre_embeds = text_embeds[i, :first_img_pos]
                
                # Text after last <image> (skip all <image> tokens)
                post_embeds = text_embeds[i, last_img_pos + 1:]
                
                # Visual features replace all <image> tokens
                vis_embeds = visual_features_flat[i]
                
                merged_embeds = torch.cat([pre_embeds, vis_embeds, post_embeds], dim=0)
                new_inputs_embeds.append(merged_embeds)
                
                # Build matching attention mask
                pre_mask = attention_mask[i, :first_img_pos]
                vis_mask = torch.ones(
                    num_vis_tokens, 
                    dtype=attention_mask.dtype, 
                    device=attention_mask.device
                )
                post_mask = attention_mask[i, last_img_pos + 1:]
                
                merged_mask = torch.cat([pre_mask, vis_mask, post_mask], dim=0)
                new_attention_masks.append(merged_mask)
            else:
                new_inputs_embeds.append(text_embeds[i])
                new_attention_masks.append(attention_mask[i])
        
        # [FIX] Pad sequences to the same length within the batch.
        # After replacing <image> tokens (potentially different counts per sample),
        # sequence lengths may differ. Pad to max length for batched processing.
        max_len = max(e.shape[0] for e in new_inputs_embeds)
        
        padded_embeds = []
        padded_masks = []
        for embeds, mask in zip(new_inputs_embeds, new_attention_masks):
            seq_len = embeds.shape[0]
            if seq_len < max_len:
                pad_len = max_len - seq_len
                embed_pad = torch.zeros(
                    pad_len, embeds.shape[1],
                    dtype=embeds.dtype, device=embeds.device
                )
                mask_pad = torch.zeros(
                    pad_len,
                    dtype=mask.dtype, device=mask.device
                )
                padded_embeds.append(torch.cat([embeds, embed_pad], dim=0))
                padded_masks.append(torch.cat([mask, mask_pad], dim=0))
            else:
                padded_embeds.append(embeds)
                padded_masks.append(mask)
        
        inputs_embeds = torch.stack(padded_embeds)
        extended_attention_mask = torch.stack(padded_masks)
                
        # State Feature는 모델이 일관성을 가지도록 텍스트 임베딩 맨 앞에 부착
        if state_features is not None:
            inputs_embeds = torch.cat([state_features, inputs_embeds], dim=1)
            state_mask = torch.ones(
                batch_size, state_features.shape[1],
                dtype=extended_attention_mask.dtype,
                device=extended_attention_mask.device
            )
            extended_attention_mask = torch.cat([state_mask, extended_attention_mask], dim=1)
        
        outputs = self.vlm.language_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
        )
        
        # 잘려진(Truncated) 마지막 레이어의 출력값 (최종 Norm 자동 적용됨)
        hidden_states = outputs.last_hidden_state
        
        outputs_dict = {
            'last_hidden_state': hidden_states,
            'layer_idx': self.use_layers,
            'attention_mask': extended_attention_mask,  # Return for downstream use
        }
        
        return hidden_states, outputs_dict
    
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state_features: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Main forward pass
        
        [NOTE] This method includes its own vision encoding via forward_vision_encoder().
        However, in the main SmolVLA pipeline (smolvla.py), vision encoding is handled
        by the separate VisionEncoder class, and this method's forward_vision_encoder()
        is NOT called. Use forward_with_layer_skipping() directly when visual_features
        are already computed externally.
        
        Args:
            images: [batch_size, num_images, channels, height, width]
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            state_features: [batch_size, 1, hidden_dim] (pre-projected)
            return_dict: Whether to return dict with additional info
            
        Returns:
            vlm_features: [batch_size, seq_len_total, hidden_dim]
            outputs_dict: Optional dictionary with additional outputs
        """
        # Process images through vision encoder
        visual_features = self._forward_vision_encoder(images)
        
        # Forward with layer skipping
        vlm_features, outputs_dict = self.forward_with_layer_skipping(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_features=visual_features,
            state_features=state_features,
        )
        
        if return_dict:
            return vlm_features, outputs_dict
        else:
            return vlm_features
    
    def _forward_vision_encoder(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process images through the VLM's internal vision encoder.
        
        [FIX] Renamed from forward_vision_encoder() to _forward_vision_encoder()
        to indicate this is an internal/private method. The public VisionEncoder
        class in vision_encoder.py is the primary way to encode images in the
        SmolVLA pipeline. This method exists only for the self-contained
        forward() path and should not be called directly by external code.
        
        Args:
            images: [batch_size, num_images, channels, height, width]
            
        Returns:
            visual_features: [batch_size, num_images, num_visual_tokens, hidden_dim]
        """
        batch_size, num_images = images.shape[:2]
        
        # Flatten batch and num_images for processing
        # [B, N_img, C, H, W] -> [B*N_img, C, H, W]
        images_flat = images.view(-1, *images.shape[2:])
        
        # Process through vision encoder
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            vision_outputs = self.vlm.vision_model(images_flat)
            visual_features_flat = vision_outputs.last_hidden_state
        
        # Reduce visual tokens
        visual_features_flat = self._reduce_visual_tokens(visual_features_flat)
        
        # Reshape back to [B, N_img, N_tokens, D]
        num_tokens = visual_features_flat.shape[1]
        visual_features = visual_features_flat.view(
            batch_size, num_images, num_tokens, self.hidden_dim
        )
        
        return visual_features
    
    def get_hidden_dim(self) -> int:
        """Get hidden dimension of VLM"""
        return self.hidden_dim
    
    def get_num_layers_used(self) -> int:
        """Get number of layers being used"""
        return self.use_layers
    
    @torch.no_grad()
    def extract_features_at_layer(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state_features: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Extract features at specific layer (for analysis)
        
        Args:
            images: [batch_size, num_images, channels, height, width]
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            state_features: [batch_size, 1, hidden_dim]
            layer_idx: Layer to extract from (default: use_layers)
            
        Returns:
            features: Features at specified layer
        """
        if layer_idx is None:
            layer_idx = self.use_layers
        
        # Temporarily change use_layers
        original_use_layers = self.use_layers
        self.use_layers = layer_idx
        
        # Forward pass
        features, _ = self.forward(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            state_features=state_features,
            return_dict=True,
        )
        
        # Restore original
        self.use_layers = original_use_layers
        
        return features
    
    def compute_feature_statistics(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute statistics across all layers (for analysis)
        
        Returns dict with layer-wise statistics
        """
        stats = {}
        
        for layer_idx in range(1, self.total_layers + 1):
            features = self.extract_features_at_layer(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                layer_idx=layer_idx
            )
            
            stats[f'layer_{layer_idx}'] = {
                'mean': features.mean(),
                'std': features.std(),
                'norm': features.norm(dim=-1).mean(),
            }
        
        return stats


# Utility functions for loading
def load_smolvlm_backbone(
    model_name: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    use_layers: Optional[int] = None,
    freeze: bool = True,
    device: str = "cuda",
) -> SmolVLMBackbone:
    """
    Load SmolVLM-2 backbone
    
    Args:
        model_name: HuggingFace model name
        use_layers: Number of layers to use (None = L/2)
        freeze: Whether to freeze VLM
        device: Device to load on
        
    Returns:
        SmolVLMBackbone instance
    """
    backbone = SmolVLMBackbone(
        model_name=model_name,
        use_layers=use_layers,
        freeze_vlm=freeze,
    )
    
    return backbone.to(device)


if __name__ == "__main__":
    # Test the backbone
    print("Testing SmolVLMBackbone...")
    
    # Initialize
    backbone = SmolVLMBackbone(
        use_layers=8,  # Use only 8 layers for testing
        freeze_vlm=True,
    )
    
    print(f"Hidden dim: {backbone.get_hidden_dim()}")
    print(f"Using {backbone.get_num_layers_used()} layers")
    
    # Create dummy inputs
    batch_size = 2
    num_images = 3
    
    images = torch.randn(batch_size, num_images, 3, 512, 512)
    input_ids = torch.randint(0, 1000, (batch_size, 20))
    attention_mask = torch.ones(batch_size, 20)
    state_features = torch.randn(batch_size, 1, backbone.get_hidden_dim())
    
    # Forward pass
    print("\nForward pass...")
    features, outputs = backbone(
        images=images,
        input_ids=input_ids,
        attention_mask=attention_mask,
        state_features=state_features,
        return_dict=True,
    )
    
    print(f"Output shape: {features.shape}")
    print(f"Layer used: {outputs['layer_idx']}")
    
    print("\nSmolVLMBackbone test completed!")