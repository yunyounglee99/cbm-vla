"""
CBM-VLA Data Collection Pipeline
=================================

SmolVLA 학습에 사용된 정확히 동일한 데이터셋
(HuggingFaceVLA/community_dataset_v1 + v2)을 로드하여
CBM-VLA 학습용 concept 데이터셋을 구축합니다.

Modules:
    AutoSegmenter:                  Rule-based episode segmentation (Stage 1)
    GeminiConceptAnnotator:         LLM-based concept annotation (Stage 2)
    ConceptRefiner:                 T5-based concept refinement & pool (Stage 3)
    SmolVLACommunityDatasetBuilder: End-to-end pipeline (SmolVLA v1+v2 datasets)
"""
from .auto_segmentation import AutoSegmenter
from .llm_annotation import GeminiConceptAnnotator
from .t5_refinement import ConceptRefiner
from .dataset_builder import (
    SmolVLACommunityDatasetBuilder,
    SubDatasetInfo,
    discover_sub_datasets_from_local,
    discover_sub_datasets_from_hub,
    load_episodes_from_sub_dataset,
    estimate_annotation_cost,
    SMOLVLA_COMMUNITY_REPOS,
    MAX_ACTION_DIM,
    MAX_STATE_DIM,
)

__all__ = [
    "AutoSegmenter",
    "GeminiConceptAnnotator",
    "ConceptRefiner",
    "SmolVLACommunityDatasetBuilder",
    "SubDatasetInfo",
    "discover_sub_datasets_from_local",
    "discover_sub_datasets_from_hub",
    "load_episodes_from_sub_dataset",
    "estimate_annotation_cost",
    "SMOLVLA_COMMUNITY_REPOS",
    "MAX_ACTION_DIM",
    "MAX_STATE_DIM",
]