"""
Phase 2 Training: Concept Scoring Module
==========================================

목적: Backbone output에서 어떤 concept이 현재 프레임과 관련 있는지 판별

학습 대상: ConceptScoringModule (~288K params)만 학습
           Backbone + CBM Encoder + 나머지 모두 freeze
Loss: L_scoring = λ₁·L_similarity + λ₂·L_activation_BCE + λ₃·L_sparsity
  - L_similarity: SigLIP reference score와의 MSE (soft teacher)
  - L_activation_BCE: GT active concept과의 BCE (hard supervision)
  - L_sparsity: 전체 concept 활성도 L1 penalty (대부분 비활성이어야 함)

입력 데이터:
  - backbone_output: [B, 139, 960]      — VLM backbone 출력 (detach됨)
  - reference_scores: [B, num_concepts]  — SigLIP image-concept similarity
  - gt_active_concepts: [B, num_concepts] — binary GT (data collection에서 생성)

사전 조건:
  - Phase 1 완료 (CBM Encoder 가중치 로드)
  - Concept pool 설정 완료 (concept_text_embeddings)
  - GT active concept labels 존재 (so101_dataset_builder 출력의 frame_labels)

데이터 흐름:
  Image → SigLIP(freeze) → CBM Encoder(Phase1 freeze) → MLP Projector
  → VLM Backbone(freeze) → backbone_output(detach) → Concept Scoring → loss
"""

import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass

from .trainer import (
    TrainConfig, TrainLogger, CheckpointManager,
    create_scheduler, set_seed, get_torch_dtype, count_parameters,
)


@dataclass
class Phase2Config(TrainConfig):
    """Phase 2 전용 설정"""
    # 모델
    smolvla_model_id: str = "lerobot/smolvla_base"
    phase1_checkpoint: str = "./checkpoints/phase1/final"
    concept_pool_path: str = "./data/so101_concept_dataset/concept_pool.json"
    
    # Phase 2 하이퍼파라미터
    lr: float = 5e-4              # Scoring module도 작은 모듈이므로 lr 높게
    epochs: int = 15
    warmup_steps: int = 300
    
    # Loss weights
    lambda_similarity: float = 0.3
    lambda_activation_bce: float = 1.0
    lambda_sparsity: float = 0.1
    
    # 출력
    output_dir: str = "./checkpoints/phase2"
    
    # Backbone output 사전 계산 여부
    precompute_backbone: bool = True   # True면 backbone output을 미리 계산하여 캐시


# ================================================================
# Phase 2 Trainer
# ================================================================

def train_phase2(
    model,   # CBMVLA instance
    train_dataloader: DataLoader,
    config: Phase2Config,
    eval_dataloader: Optional[DataLoader] = None,
):
    """
    Phase 2 학습 실행
    
    Args:
        model: CBMVLA 모델 (Phase 1 가중치 로드 완료)
        train_dataloader: 학습 데이터 로더
            각 batch는 dict:
              "backbone_output": [B, 139, 960]      — 미리 계산 or forward 중 생성
              "reference_scores": [B, num_concepts]  — SigLIP reference
              "gt_active_concepts": [B, num_concepts] — binary GT
            또는 (precompute_backbone=False인 경우):
              "images": [B, num_views, C, H, W]
              "instruction": List[str]
              "robot_state": [B, state_dim]
              + 위의 reference_scores, gt_active_concepts
        config: Phase2Config
        eval_dataloader: 검증 데이터 로더 (선택)
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    set_seed(config.seed)
    
    # Phase 2 설정
    model.configure_phase2()
    model = model.to(device)
    
    trainable, total = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"Phase 2: Concept Scoring Module Training")
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print(f"  LR: {config.lr}, Epochs: {config.epochs}")
    print(f"  Loss weights: sim={config.lambda_similarity} "
          f"bce={config.lambda_activation_bce} sparsity={config.lambda_sparsity}")
    print(f"{'='*60}\n")
    
    # Optimizer
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    
    total_steps = len(train_dataloader) * config.epochs // config.gradient_accumulation_steps
    scheduler = create_scheduler(optimizer, config, total_steps)
    
    logger = TrainLogger(config, "Phase2")
    ckpt_mgr = CheckpointManager(config.output_dir, config.save_total_limit)
    
    # Resume
    start_step, start_epoch = 0, 0
    if config.resume_from:
        meta = ckpt_mgr.load(model, optimizer, scheduler, config.resume_from)
        start_step, start_epoch = meta["step"], meta["epoch"]
    
    # Training loop
    global_step = start_step
    model.train()
    
    # Scoring module만 train mode, 나머지는 eval mode
    # (BatchNorm, Dropout 등이 backbone에 있을 수 있으므로)
    if model.vision_encoder is not None:
        model.vision_encoder.eval()
    if model.vlm_backbone is not None:
        model.vlm_backbone.eval()
    
    for epoch in range(start_epoch, config.epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            
            # backbone_output 가져오기
            if "backbone_output" in batch:
                # 사전 계산된 경우
                backbone_output = batch["backbone_output"].to(device)
            else:
                # Forward pass로 생성 (느리지만 유연)
                with torch.no_grad():
                    backbone_output = _compute_backbone_output(
                        model, batch, device)
            
            reference_scores = batch.get("reference_scores")
            if reference_scores is not None:
                reference_scores = reference_scores.to(device)
            
            gt_active = batch.get("gt_active_concepts")
            if gt_active is not None:
                gt_active = gt_active.to(device)
            
            # Forward (backbone_output는 내부에서 detach됨)
            result = model.forward_phase2(
                backbone_output=backbone_output,
                reference_scores=reference_scores,
                gt_active_concepts=gt_active,
            )
            
            loss = result["loss"] / config.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                logger.log_step(
                    result["info"], global_step, epoch,
                    optimizer.param_groups[0]["lr"],
                )
                
                if global_step % config.save_interval == 0:
                    ckpt_mgr.save(model, optimizer, scheduler,
                                  global_step, epoch, result["info"])
                
                if eval_dataloader and global_step % config.eval_interval == 0:
                    eval_info = evaluate_phase2(model, eval_dataloader, device)
                    print(f"  [Eval] step={global_step} | "
                          f"loss={eval_info['loss']:.4f} | "
                          f"active={eval_info['num_active']:.1f}")
                    model.train()
    
    ckpt_mgr.save(model, optimizer, scheduler, global_step, config.epochs, tag="final")
    logger.finish()
    
    print(f"\n✓ Phase 2 complete. Steps: {global_step}")
    return model


def _compute_backbone_output(model, batch, device):
    """precompute_backbone=False일 때 backbone output 계산"""
    images = batch["images"].to(device)
    instruction = batch["instruction"]
    robot_state = batch["robot_state"].to(device)
    
    # SigLIP → CBM Encoder
    visual_features = model.vision_encoder(images)
    enhanced = model.cbm_encoder(visual_features)
    
    # VLM Backbone
    state_features = model.state_projector(robot_state).unsqueeze(1)
    backbone_output, _ = model.vlm_backbone.forward_with_layer_skipping(
        visual_features=enhanced,
        state_features=state_features,
        input_ids=None,  # instruction 처리 필요
        attention_mask=None,
    )
    
    return backbone_output


@torch.no_grad()
def evaluate_phase2(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_active = 0
    count = 0
    
    for batch in dataloader:
        backbone_output = batch["backbone_output"].to(device) if "backbone_output" in batch \
            else _compute_backbone_output(model, batch, device)
        
        ref = batch.get("reference_scores")
        gt = batch.get("gt_active_concepts")
        
        result = model.forward_phase2(
            backbone_output=backbone_output,
            reference_scores=ref.to(device) if ref is not None else None,
            gt_active_concepts=gt.to(device) if gt is not None else None,
        )
        total_loss += result["loss"].item()
        total_active += result["info"].get("num_active_concepts", 0)
        count += 1
    
    return {
        "loss": total_loss / max(count, 1),
        "num_active": total_active / max(count, 1),
    }


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="CBM-VLA Phase 2: Concept Scoring Training")
    parser.add_argument("--smolvla_model_id", type=str, default="lerobot/smolvla_base")
    parser.add_argument("--phase1_checkpoint", type=str, default="./checkpoints/phase1/final")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/phase2")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)
    args = parser.parse_args()
    
    config = Phase2Config(
        smolvla_model_id=args.smolvla_model_id,
        phase1_checkpoint=args.phase1_checkpoint,
        output_dir=args.output_dir,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_wandb=args.use_wandb,
        resume_from=args.resume_from,
    )
    
    print("Loading model and Phase 1 weights...")
    print(f"  [TODO] Load SmolVLA → CBMVLA → Phase 1 checkpoint")
    print(f"  [TODO] Connect Phase2Dataset dataloader")
    print(f"  Expected batch format:")
    print(f"    backbone_output: [B, 139, 960]")
    print(f"    reference_scores: [B, num_concepts]")
    print(f"    gt_active_concepts: [B, num_concepts]")


if __name__ == "__main__":
    main()