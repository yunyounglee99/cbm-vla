"""
Phase 1 Training: CBM Encoder
===============================

목적: SigLIP feature 위에 MLP residual을 추가하여
      scene concept 방향의 정보를 보강합니다.

학습 대상: CBM Encoder (~393K params)만 학습, 나머지 모두 freeze
Loss: InfoNCE contrastive alignment (enhanced features ↔ image description text)
입력 데이터:
  - siglip_features: [B, 128, 768]  — SigLIP vision encoder 출력 (freeze)
  - target_text_embeddings: [B, 768] — image description의 SigLIP text embedding

사전 조건: 
  - SmolVLA 모델이 로드되어 CBMVLA.init_from_smolvla() 완료
  - 데이터셋에 image_description 필드 존재 (data collection Stage 2에서 생성)
  - SigLIP text encoder로 description을 사전 인코딩하여 저장해두면 효율적

결과물:
  - checkpoint에 cbm_encoder 가중치 저장
  - Phase 2에서 이 가중치를 로드하여 사용
"""

import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass

from .trainer import (
    TrainConfig, TrainLogger, CheckpointManager,
    create_scheduler, set_seed, get_torch_dtype, count_parameters,
)


@dataclass
class Phase1Config(TrainConfig):
    """Phase 1 전용 설정"""
    # 모델
    smolvla_model_id: str = "lerobot/smolvla_base"
    
    # Phase 1 하이퍼파라미터
    lr: float = 3e-4                   # CBM Encoder는 작은 모듈이므로 lr 높여도 됨
    epochs: int = 20
    warmup_steps: int = 200
    alignment_temperature: float = 0.07
    alignment_pooling: str = "mean"
    
    # 출력
    output_dir: str = "./checkpoints/phase1"


# ================================================================
# Phase 1 Trainer
# ================================================================

def train_phase1(
    model,   # CBMVLA instance (init_from_smolvla 완료)
    train_dataloader: DataLoader,
    config: Phase1Config,
    eval_dataloader: Optional[DataLoader] = None,
):
    """
    Phase 1 학습 실행
    
    Args:
        model: CBMVLA 모델 (SmolVLA backbone 로드 완료)
        train_dataloader: 학습 데이터 로더
            각 batch는 dict:
              "siglip_features": [B, num_tokens, 768]
              "target_text_embeddings": [B, 768]
        config: Phase1Config
        eval_dataloader: 검증 데이터 로더 (선택)
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    set_seed(config.seed)
    
    # Phase 1 설정: CBM Encoder만 학습
    model.configure_phase1()
    model = model.to(device)
    
    trainable, total = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"Phase 1: CBM Encoder Training")
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print(f"  LR: {config.lr}, Epochs: {config.epochs}")
    print(f"  Temperature: {config.alignment_temperature}")
    print(f"{'='*60}\n")
    
    # Optimizer: CBM Encoder params만
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    
    total_steps = len(train_dataloader) * config.epochs // config.gradient_accumulation_steps
    scheduler = create_scheduler(optimizer, config, total_steps)
    
    logger = TrainLogger(config, "Phase1")
    ckpt_mgr = CheckpointManager(config.output_dir, config.save_total_limit)
    
    # Resume
    start_step = 0
    start_epoch = 0
    if config.resume_from:
        meta = ckpt_mgr.load(model, optimizer, scheduler, config.resume_from)
        start_step = meta["step"]
        start_epoch = meta["epoch"]
    
    # Training loop
    global_step = start_step
    model.train()
    
    for epoch in range(start_epoch, config.epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            # 데이터를 device로 이동
            siglip_features = batch["siglip_features"].to(device)
            target_text_embs = batch["target_text_embeddings"].to(device)
            
            # Forward
            result = model.forward_phase1(
                siglip_features=siglip_features,
                target_text_embeddings=target_text_embs,
            )
            
            loss = result["loss"] / config.gradient_accumulation_steps
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging
                logger.log_step(
                    result["info"], global_step, epoch,
                    optimizer.param_groups[0]["lr"],
                )
                
                # Save
                if global_step % config.save_interval == 0:
                    ckpt_mgr.save(model, optimizer, scheduler,
                                  global_step, epoch, result["info"])
                
                # Eval
                if eval_dataloader and global_step % config.eval_interval == 0:
                    eval_loss = evaluate_phase1(model, eval_dataloader, device)
                    print(f"  [Eval] step={global_step} | eval_loss={eval_loss:.4f}")
                    model.train()
    
    # 최종 저장
    ckpt_mgr.save(model, optimizer, scheduler, global_step, config.epochs,
                   tag="final")
    logger.finish()
    
    print(f"\n✓ Phase 1 training complete. Steps: {global_step}")
    return model


@torch.no_grad()
def evaluate_phase1(model, dataloader, device):
    """Phase 1 검증"""
    model.eval()
    total_loss = 0
    count = 0
    
    for batch in dataloader:
        result = model.forward_phase1(
            siglip_features=batch["siglip_features"].to(device),
            target_text_embeddings=batch["target_text_embeddings"].to(device),
        )
        total_loss += result["loss"].item()
        count += 1
    
    return total_loss / max(count, 1)


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="CBM-VLA Phase 1: CBM Encoder Training")
    parser.add_argument("--smolvla_model_id", type=str, default="lerobot/smolvla_base")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to concept dataset (from so101_dataset_builder)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/phase1")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)
    args = parser.parse_args()
    
    config = Phase1Config(
        smolvla_model_id=args.smolvla_model_id,
        output_dir=args.output_dir,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_wandb=args.use_wandb,
        resume_from=args.resume_from,
    )
    
    # 모델 로드
    print("Loading SmolVLA model...")
    from smolvla.smolvla.smolvla import SmolVLA
    from cbm_vla.models.cbmvla import CBMVLA, CBMVLAConfig
    
    smolvla = SmolVLA.from_pretrained(config.smolvla_model_id)
    cbm_config = CBMVLAConfig()
    model = CBMVLA(cbm_config)
    model.init_from_smolvla(smolvla)
    del smolvla
    
    # 데이터 로더 (TODO: 실제 데이터셋 클래스로 교체)
    print(f"Loading dataset from {args.dataset_dir}...")
    print("  [TODO] Connect your Phase1Dataset here")
    print("  Expected batch format:")
    print("    siglip_features: [B, num_tokens, 768]")
    print("    target_text_embeddings: [B, 768]")
    
    # train_phase1(model, train_dataloader, config)


if __name__ == "__main__":
    main()