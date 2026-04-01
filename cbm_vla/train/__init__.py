"""
CBM-VLA Training Pipeline
===========================

3-Phase 순차 학습 파이프라인:

  Phase 1: CBM Encoder 학습 (~393K params)
    - Loss: InfoNCE alignment (enhanced features ↔ image description text)
    - 학습 대상: CBM Encoder만
    - 입력: SigLIP features + target text embeddings

  Phase 2: Concept Scoring Module 학습 (~288K params)
    - Loss: similarity + activation BCE + sparsity (CLG-CBM style)
    - 학습 대상: ConceptScoringModule만
    - 입력: backbone output (detached) + reference scores + GT labels

  Phase 3: End-to-End with LoRA (~13M trainable params)
    - Loss: flow matching + order + contrastive embedding
    - 학습 대상: LoRA + CrossAttn + OrderAttn + ActionExpert + Projectors
    - 입력: full pipeline (images → backbone → concept → action)

Usage:
  # Phase별 개별 실행
  python -m cbm_vla.train.train_phase1 --dataset_dir ./data/so101
  python -m cbm_vla.train.train_phase2 --dataset_dir ./data/so101
  python -m cbm_vla.train.train_phase3 --dataset_dir ./data/so101

  # 또는 Python API로:
  from cbm_vla.train import train_phase1, train_phase2, train_phase3
"""

from .trainer import (
    TrainConfig,
    TrainLogger,
    CheckpointManager,
    create_scheduler,
    set_seed,
    get_torch_dtype,
    count_parameters,
)

from .train_phase1 import (
    Phase1Config,
    train_phase1,
    evaluate_phase1,
)

from .train_phase2 import (
    Phase2Config,
    train_phase2,
    evaluate_phase2,
)

from .train_phase3 import (
    Phase3Config,
    train_phase3,
    evaluate_phase3,
    setup_lora,
    build_optimizer,
)