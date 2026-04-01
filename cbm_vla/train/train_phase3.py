"""
Phase 3 Training: End-to-End with LoRA
========================================

목적: 실제 action 생성 능력 학습
      Concept embedding이 action expert를 올바르게 조향하도록 학습

학습 대상:
  - Backbone VLM: LoRA adapters (~2M params, r=16)
  - ConceptCrossAttention: ~2.7M params
  - ConceptOrderAttention: ~196K params
  - concept_to_expert_proj: Linear 128→720
  - ActionExpert: ~8M params
  - Feature/State Projectors
  
  Freeze:
  - Vision Encoder (SigLIP): 완전 freeze
  - CBM Encoder: Phase 1에서 학습 완료, freeze
  - Concept Scoring Module: Phase 2에서 학습 완료, freeze
  
Loss:
  L = L_flow_matching + λ_order · L_order + λ_contrastive · L_contrastive_embedding
  - L_flow_matching: SmolVLA 원본 flow matching loss (action 생성)
  - L_order: pairwise ranking loss (concept 실행 순서)
  - L_contrastive: InfoNCE (concept embedding 의미 보존)

입력 데이터:
  - images: [B, num_views, C, H, W]
  - instruction: List[str]
  - robot_state: [B, state_dim]
  - actions_gt: [B, chunk_size, action_dim]
  - gt_concept_ids: [B, top_n]         — GT로 선택해야 하는 concept indices
  - gt_concept_order: [B, top_n]       — GT concept 실행 순서

사전 조건:
  - Phase 1 + Phase 2 완료
  - Concept pool 설정 완료
"""

import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass

from .trainer import (
    TrainConfig, TrainLogger, CheckpointManager,
    create_scheduler, set_seed, get_torch_dtype, count_parameters,
)


@dataclass
class Phase3Config(TrainConfig):
    """Phase 3 전용 설정"""
    # 모델
    smolvla_model_id: str = "lerobot/smolvla_base"
    phase1_checkpoint: str = "./checkpoints/phase1/final"
    phase2_checkpoint: str = "./checkpoints/phase2/final"
    concept_pool_path: str = "./data/so101_concept_dataset/concept_pool.json"
    
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    
    # LR (차별적 학습률)
    lr_lora: float = 2e-5             # LoRA adapters (VLM backbone)
    lr_cbm: float = 1e-4              # Cross/Order attention + proj
    lr_expert: float = 1e-4           # Action expert + projectors
    
    # Phase 3 하이퍼파라미터
    epochs: int = 30
    warmup_steps: int = 1000
    
    # Loss weights
    lambda_order: float = 0.1
    lambda_contrastive: float = 0.05
    contrastive_temperature: float = 0.1
    
    # Flow matching
    beta_alpha: float = 0.5
    beta_beta: float = 0.5
    
    # 출력
    output_dir: str = "./checkpoints/phase3"


# ================================================================
# LoRA Setup
# ================================================================

def setup_lora(model, config: Phase3Config):
    """
    VLM Backbone에 LoRA 적용
    
    SmolVLA의 finetuning.py 패턴을 따르되,
    CBM-VLA의 Phase 3에 맞게 조정합니다.
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        print("  [WARNING] peft not installed. Skipping LoRA setup.")
        print("  Install: pip install peft")
        return model
    
    if model.vlm_backbone is None:
        print("  [WARNING] VLM backbone not loaded, skipping LoRA")
        return model
    
    target_modules = [m.strip() for m in config.lora_target_modules.split(",")]
    
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    
    # VLM 내부 모델에 LoRA 적용
    model.vlm_backbone.vlm = get_peft_model(model.vlm_backbone.vlm, lora_config)
    
    print(f"  LoRA applied: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"  Target modules: {target_modules}")
    model.vlm_backbone.vlm.print_trainable_parameters()
    
    return model


def build_optimizer(model, config: Phase3Config) -> AdamW:
    """
    차별적 학습률을 적용한 optimizer 생성
    
    3개 파라미터 그룹:
      1. LoRA params → lr_lora (가장 낮음)
      2. CBM 모듈 params (cross/order attn, proj) → lr_cbm
      3. Action expert + projectors → lr_expert
    """
    lora_params = []
    cbm_params = []
    expert_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if "lora" in name.lower():
            lora_params.append(param)
        elif any(k in name for k in [
            "concept_cross_attn", "concept_order_attn",
            "concept_to_expert_proj", "contrastive_proj",
        ]):
            cbm_params.append(param)
        else:
            expert_params.append(param)
    
    param_groups = []
    if lora_params:
        param_groups.append({"params": lora_params, "lr": config.lr_lora})
    if cbm_params:
        param_groups.append({"params": cbm_params, "lr": config.lr_cbm})
    if expert_params:
        param_groups.append({"params": expert_params, "lr": config.lr_expert})
    
    print(f"  Optimizer groups: LoRA={len(lora_params)}, "
          f"CBM={len(cbm_params)}, Expert={len(expert_params)}")
    
    return AdamW(param_groups, weight_decay=config.weight_decay, betas=config.betas)


# ================================================================
# Phase 3 Trainer
# ================================================================

def train_phase3(
    model,   # CBMVLA instance (Phase 1+2 로드, LoRA 적용 완료)
    train_dataloader: DataLoader,
    config: Phase3Config,
    eval_dataloader: Optional[DataLoader] = None,
):
    """
    Phase 3 학습 실행
    
    Args:
        model: CBMVLA 모델 (Phase 1+2 가중치 로드 + LoRA 적용 완료)
        train_dataloader: 학습 데이터 로더
            각 batch는 dict:
              "backbone_output": [B, 139, 960]
              "vlm_features": [B, 139, 720]
              "attention_mask": [B, 139]
              "actions_gt": [B, chunk_size, action_dim]
              "gt_concept_ids": [B, top_n]
              "gt_concept_order": [B, top_n]
        config: Phase3Config
        eval_dataloader: 검증 데이터 로더 (선택)
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    set_seed(config.seed)
    
    # Phase 3 설정 + LoRA
    model.configure_phase3()
    model = setup_lora(model, config)
    model = model.to(device)
    
    trainable, total = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"Phase 3: End-to-End Training with LoRA")
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print(f"  LR: lora={config.lr_lora}, cbm={config.lr_cbm}, expert={config.lr_expert}")
    print(f"  Loss: flow + {config.lambda_order}*order + "
          f"{config.lambda_contrastive}*contrastive")
    print(f"{'='*60}\n")
    
    # Optimizer (차별적 LR)
    optimizer = build_optimizer(model, config)
    
    total_steps = len(train_dataloader) * config.epochs // config.gradient_accumulation_steps
    scheduler = create_scheduler(optimizer, config, total_steps)
    
    logger = TrainLogger(config, "Phase3")
    ckpt_mgr = CheckpointManager(config.output_dir, config.save_total_limit)
    
    # Resume
    start_step, start_epoch = 0, 0
    if config.resume_from:
        meta = ckpt_mgr.load(model, optimizer, scheduler, config.resume_from)
        start_step, start_epoch = meta["step"], meta["epoch"]
    
    # Mixed precision
    use_amp = config.dtype in ("float16", "bfloat16") and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=(config.dtype == "float16"))
    amp_dtype = get_torch_dtype(config.dtype) if use_amp else torch.float32
    
    # Training loop
    global_step = start_step
    model.train()
    
    # Vision encoder는 항상 eval
    if model.vision_encoder is not None:
        model.vision_encoder.eval()
    
    for epoch in range(start_epoch, config.epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            
            # 데이터 이동
            backbone_output = batch["backbone_output"].to(device)
            vlm_features = batch["vlm_features"].to(device)
            actions_gt = batch["actions_gt"].to(device)
            gt_concept_ids = batch["gt_concept_ids"].to(device)
            gt_concept_order = batch["gt_concept_order"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Forward with AMP
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                result = model.forward_phase3(
                    backbone_output=backbone_output,
                    vlm_features=vlm_features,
                    gt_actions=actions_gt,
                    gt_concept_ids=gt_concept_ids,
                    gt_concept_order=gt_concept_order,
                    attention_mask=attention_mask,
                )
                loss = result["loss"] / config.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # 로깅 (lr은 첫 번째 그룹 = lora)
                current_lr = optimizer.param_groups[0]["lr"]
                logger.log_step(result["info"], global_step, epoch, current_lr)
                
                if global_step % config.save_interval == 0:
                    ckpt_mgr.save(model, optimizer, scheduler,
                                  global_step, epoch, result["info"])
                
                if eval_dataloader and global_step % config.eval_interval == 0:
                    eval_info = evaluate_phase3(model, eval_dataloader, device)
                    print(f"  [Eval] step={global_step} | "
                          f"loss={eval_info['loss']:.4f} | "
                          f"flow={eval_info['flow']:.4f}")
                    model.train()
                    if model.vision_encoder is not None:
                        model.vision_encoder.eval()
    
    ckpt_mgr.save(model, optimizer, scheduler, global_step, config.epochs, tag="final")
    logger.finish()
    
    print(f"\n✓ Phase 3 complete. Steps: {global_step}")
    return model


@torch.no_grad()
def evaluate_phase3(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_flow = 0
    count = 0
    
    for batch in dataloader:
        result = model.forward_phase3(
            backbone_output=batch["backbone_output"].to(device),
            vlm_features=batch["vlm_features"].to(device),
            gt_actions=batch["actions_gt"].to(device),
            gt_concept_ids=batch["gt_concept_ids"].to(device),
            gt_concept_order=batch["gt_concept_order"].to(device),
            attention_mask=batch.get("attention_mask", torch.ones(
                batch["backbone_output"].shape[:2])).to(device),
        )
        total_loss += result["info"].get("loss_total", 0)
        total_flow += result["info"].get("loss_flow", 0)
        count += 1
    
    return {
        "loss": total_loss / max(count, 1),
        "flow": total_flow / max(count, 1),
    }


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="CBM-VLA Phase 3: End-to-End Training")
    parser.add_argument("--smolvla_model_id", type=str, default="lerobot/smolvla_base")
    parser.add_argument("--phase1_checkpoint", type=str, default="./checkpoints/phase1/final")
    parser.add_argument("--phase2_checkpoint", type=str, default="./checkpoints/phase2/final")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/phase3")
    parser.add_argument("--lr_lora", type=float, default=2e-5)
    parser.add_argument("--lr_cbm", type=float, default=1e-4)
    parser.add_argument("--lr_expert", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)
    args = parser.parse_args()
    
    config = Phase3Config(
        smolvla_model_id=args.smolvla_model_id,
        phase1_checkpoint=args.phase1_checkpoint,
        phase2_checkpoint=args.phase2_checkpoint,
        output_dir=args.output_dir,
        lr_lora=args.lr_lora,
        lr_cbm=args.lr_cbm,
        lr_expert=args.lr_expert,
        lora_r=args.lora_r,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_wandb=args.use_wandb,
        resume_from=args.resume_from,
    )
    
    print("Loading model with Phase 1 + Phase 2 weights...")
    print(f"  [TODO] Load SmolVLA → CBMVLA → Phase1 → Phase2 checkpoints")
    print(f"  [TODO] Connect Phase3Dataset dataloader")
    print(f"  Expected batch format:")
    print(f"    backbone_output: [B, 139, 960]")
    print(f"    vlm_features: [B, 139, 720]")
    print(f"    actions_gt: [B, chunk_size, action_dim]")
    print(f"    gt_concept_ids: [B, top_n]")
    print(f"    gt_concept_order: [B, top_n]")


if __name__ == "__main__":
    main()