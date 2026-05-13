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
@dataclass
class Phase3Config(TrainConfig):
    # 모델
    smolvla_model_id: str = "lerobot/smolvla_base"
    phase1_checkpoint: str = "./checkpoints/phase1/final"
    phase2_checkpoint: str = "./checkpoints/phase2/final"
    concept_pool_path: str = "./data/smolvla_concept_dataset/concept_pool.json"
    
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    
    # LR (차별적 학습률)
    lr_lora: float = 2e-5
    lr_scoring: float = 1e-4      # [수정] scoring fine-tune용 lr 추가
    lr_cbm: float = 1e-4
    lr_expert: float = 1e-4
    
    # Phase 3 하이퍼파라미터
    epochs: int = 30
    warmup_steps: int = 1000
    
    # Loss weights (action 계열)
    lambda_order: float = 0.1
    lambda_contrastive: float = 0.05
    contrastive_temperature: float = 0.1
    
    # Loss weights (scoring 계열)            # [수정] Phase 3에서도 scoring loss 사용
    lambda_similarity: float = 0.3
    lambda_activation_bce: float = 1.0
    lambda_sparsity: float = 0.1
    
    # Flow matching
    beta_alpha: float = 0.5
    beta_beta: float = 0.5
    
    output_dir: str = "./checkpoints/phase3"


def build_optimizer(model, config: Phase3Config) -> AdamW:
    lora_params = []
    scoring_params = []                                    # [수정] scoring 그룹 추가
    cbm_params = []
    expert_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if "lora" in name.lower():
            lora_params.append(param)
        elif "concept_scoring" in name:                    # [수정] scoring 분리
            scoring_params.append(param)
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
    if scoring_params:                                     # [수정] scoring 그룹
        param_groups.append({"params": scoring_params, "lr": config.lr_scoring})
    if cbm_params:
        param_groups.append({"params": cbm_params, "lr": config.lr_cbm})
    if expert_params:
        param_groups.append({"params": expert_params, "lr": config.lr_expert})
    
    print(f"  Optimizer groups: LoRA={len(lora_params)}, "
          f"Scoring={len(scoring_params)}, "                # [수정] 로깅 추가
          f"CBM={len(cbm_params)}, Expert={len(expert_params)}")
    
    return AdamW(param_groups, weight_decay=config.weight_decay, betas=config.betas)


def train_phase3(
    model,
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
              "reference_scores": [B, num_concepts]       # [수정] scoring용 추가
              "gt_active_concepts": [B, num_concepts]     # [수정] scoring용 추가
        config: Phase3Config
        eval_dataloader: 검증 데이터 로더 (선택)
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    set_seed(config.seed)
    
    model.configure_phase3()
    model = setup_lora(model, config)
    model = model.to(device)
    
    trainable, total = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"Phase 3: End-to-End Training with LoRA + Scoring Fine-tune")  # [수정]
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print(f"  LR: lora={config.lr_lora}, scoring={config.lr_scoring}, "   # [수정]
          f"cbm={config.lr_cbm}, expert={config.lr_expert}")
    print(f"  Action loss: flow + {config.lambda_order}*order + "
          f"{config.lambda_contrastive}*contrastive")
    print(f"  Scoring loss: {config.lambda_similarity}*sim + "            # [수정]
          f"{config.lambda_activation_bce}*bce + "
          f"{config.lambda_sparsity}*sparsity")
    print(f"{'='*60}\n")
    
    optimizer = build_optimizer(model, config)
    
    total_steps = len(train_dataloader) * config.epochs // config.gradient_accumulation_steps
    scheduler = create_scheduler(optimizer, config, total_steps)
    
    logger = TrainLogger(config, "Phase3")
    ckpt_mgr = CheckpointManager(config.output_dir, config.save_total_limit)
    
    start_step, start_epoch = 0, 0
    if config.resume_from:
        meta = ckpt_mgr.load(model, optimizer, scheduler, config.resume_from)
        start_step, start_epoch = meta["step"], meta["epoch"]
    
    use_amp = config.dtype in ("float16", "bfloat16") and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=(config.dtype == "float16"))
    amp_dtype = get_torch_dtype(config.dtype) if use_amp else torch.float32
    
    global_step = start_step
    model.train()
    
    if model.vision_encoder is not None:
        model.vision_encoder.eval()
    
    for epoch in range(start_epoch, config.epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            
            backbone_output = batch["backbone_output"].to(device)
            vlm_features = batch["vlm_features"].to(device)
            actions_gt = batch["actions_gt"].to(device)
            gt_concept_ids = batch["gt_concept_ids"].to(device)
            gt_concept_order = batch["gt_concept_order"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # [수정] scoring용 데이터 추출
            reference_scores = batch.get("reference_scores")
            if reference_scores is not None:
                reference_scores = reference_scores.to(device)
            gt_active_concepts = batch.get("gt_active_concepts")
            if gt_active_concepts is not None:
                gt_active_concepts = gt_active_concepts.to(device)
            
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                result = model.forward_phase3(
                    backbone_output=backbone_output,
                    vlm_features=vlm_features,
                    gt_actions=actions_gt,
                    gt_concept_ids=gt_concept_ids,
                    gt_concept_order=gt_concept_order,
                    attention_mask=attention_mask,
                    reference_scores=reference_scores,          # [수정]
                    gt_active_concepts=gt_active_concepts,      # [수정]
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
                
                current_lr = optimizer.param_groups[0]["lr"]
                logger.log_step(result["info"], global_step, epoch, current_lr)
                
                if global_step % config.save_interval == 0:
                    ckpt_mgr.save(model, optimizer, scheduler,
                                  global_step, epoch, result["info"])
                
                if eval_dataloader and global_step % config.eval_interval == 0:
                    eval_info = evaluate_phase3(model, eval_dataloader, device)
                    print(f"  [Eval] step={global_step} | "
                          f"loss={eval_info['loss']:.4f} | "
                          f"flow={eval_info['flow']:.4f} | "
                          f"scoring={eval_info['scoring']:.4f}")       # [수정]
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
    total_scoring = 0                                      # [수정]
    count = 0
    
    for batch in dataloader:
        # [수정] scoring 데이터도 전달
        reference_scores = batch.get("reference_scores")
        gt_active_concepts = batch.get("gt_active_concepts")
        
        result = model.forward_phase3(
            backbone_output=batch["backbone_output"].to(device),
            vlm_features=batch["vlm_features"].to(device),
            gt_actions=batch["actions_gt"].to(device),
            gt_concept_ids=batch["gt_concept_ids"].to(device),
            gt_concept_order=batch["gt_concept_order"].to(device),
            attention_mask=batch.get("attention_mask",
                torch.ones(batch["backbone_output"].shape[:2])).to(device),
            reference_scores=reference_scores.to(device) if reference_scores is not None else None,
            gt_active_concepts=gt_active_concepts.to(device) if gt_active_concepts is not None else None,
        )
        total_loss += result["info"].get("loss_total", 0)
        total_flow += result["info"].get("loss_flow", 0)
        total_scoring += result["info"].get("loss_scoring", 0)  # [수정]
        count += 1
    
    return {
        "loss": total_loss / max(count, 1),
        "flow": total_flow / max(count, 1),
        "scoring": total_scoring / max(count, 1),          # [수정]
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