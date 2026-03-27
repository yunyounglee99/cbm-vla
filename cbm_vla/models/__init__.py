"""
CBM-VLA Models
===============

Concept Bottleneck Model modules for Vision-Language-Action.

Modules:
    CBMEncoder:              MLP residual on SigLIP output (Phase 1)
    ConceptScoringModule:    CLG-CBM style concept scoring (Phase 2)
    ConceptCrossAttention:   Cross-attention for concept embedding extraction (Phase 3)
    ConceptOrderAttention:   Causal self-attention for order learning (Phase 3)
    CompletionDetector:      Rule-based or MLP completion detection
    CBMVLA:                  Main model integrating all modules
"""

from .cbm_encoder import CBMEncoder
from .concept_scoring import (
    ConceptScoringModule,
    compute_scoring_loss,
    compute_similarity_loss,
    compute_activation_bce_loss,
    compute_sparsity_loss,
)
from .concept_cross_attention import ConceptCrossAttention
from .concept_order_attention import (
    ConceptOrderAttention,
    compute_order_loss,
)
from .completion_detector import (
    CompletionSignal,
    RuleBasedCompletionDetector,
    MLPCompletionDetector,
)
from .cbmvla import CBMVLA, CBMVLAConfig