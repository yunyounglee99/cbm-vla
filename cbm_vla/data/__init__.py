"""
CBM-VLA Data Collection Pipeline
=================================

Modules:
    AutoSegmenter:              Rule-based episode segmentation (Stage 1)
    GeminiConceptAnnotator:     LLM-based concept annotation (Stage 2)
    ConceptRefiner:             T5-based concept refinement & pool construction (Stage 3)
    SO101ConceptDatasetBuilder: End-to-end pipeline for SO-101 datasets
"""
from .auto_segmentation import AutoSegmenter
from .llm_annotation import GeminiConceptAnnotator
from .t5_refinement import ConceptRefiner
from .so101_dataset_builder import SO101ConceptDatasetBuilder