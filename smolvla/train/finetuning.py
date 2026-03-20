"""
SmolVLA LoRA Fine-Tuning Script
Based on SmolVLA paper and standard LeRobot PEFT practices.
"""
import os
import sys
from pathlib import Path

# smolvla 모듈을 임포트하기 위해 부모 디렉토리를 시스템 경로에 추가
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
import argparse

from smolvla import SmolVLA

def setup_lora_model(model: SmolVLA, r: int = 16, alpha: int = 32, dropout: float = 0.05) -> SmolVLA:
    """
    VLM 백본에 LoRA를 적용하고, Action Expert와 Projector는 전체 학습이 가능하도록 설정합니다.
    """
    print("\n[LoRA Setup] Unfreezing VLM backbone for PEFT...")
    # 1. PEFT가 래핑할 수 있도록 VLM의 파라미터 동결을 먼저 해제합니다.
    model.unfreeze_vlm()
    
    # 2. SmolVLM-2(언어 모델 기반)의 Attention 및 MLP 선형 레이어 타겟팅
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ]
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    
    # 3. 내부 Hugging Face VLM 모델을 LoRA 모델로 변환
    model.vlm_backbone.vlm = get_peft_model(model.vlm_backbone.vlm, lora_config)
    
    # 4. Action Expert 및 Projector 파라미터는 Full Fine-Tuning 되도록 강제 설정
    for param in model.action_expert.parameters():
        param.requires_grad = True
    for param in model.state_projector.parameters():
        param.requires_grad = True
    for param in model.feature_projector.parameters():
        param.requires_grad = True
        
    print("[LoRA Setup] VLM Backbone Trainable Parameters:")
    model.vlm_backbone.vlm.print_trainable_parameters()
    
    return model

def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for SmolVLA")
    # 모델 및 학습 하이퍼파라미터 (SmolVLA / LeRobot 논문 기준)
    parser.add_argument("--model_id", type=str, default="lerobot/smolvla_base", help="Pretrained model ID")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Total epochs")
    
    # 학습률: VLM의 LoRA 어댑터는 작게, Action Expert는 상대적으로 크게 설정
    parser.add_argument("--lr_lora", type=float, default=2e-5, help="Learning rate for LoRA adapters")
    parser.add_argument("--lr_expert", type=float, default=1e-4, help="Learning rate for Action Expert & Projectors")
    
    # LoRA 하이퍼파라미터
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA Rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA Alpha")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading base model: {args.model_id} onto {device}")
    model = SmolVLA.from_pretrained(args.model_id)
    
    # LoRA 적용
    model = setup_lora_model(model, r=args.lora_r, alpha=args.lora_alpha)
    model = model.to(device)

    # 파라미터 그룹 분리 (LoRA vs Full Trainable)
    lora_params = []
    expert_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if "lora" in name.lower():
            lora_params.append(param)
        else:
            expert_params.append(param)

    # Optimizer 설정 (서로 다른 학습률 적용)
    optimizer = AdamW([
        {"params": lora_params, "lr": args.lr_lora},
        {"params": expert_params, "lr": args.lr_expert}
    ], weight_decay=1e-4)

    # ----------------------------------------------------------------------
    # TODO: LeRobotDataset 등 사용자의 실제 데이터셋을 Dataloader에 연결하세요.
    # ----------------------------------------------------------------------
    # dummy_dataset = YourRoboticsDataset()
    # dataloader = DataLoader(dummy_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # total_steps = len(dataloader) * args.epochs
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=total_steps)
    
    print("\n[Mock] Starting Training Loop...")
    model.train()
    
    # 실제 학습 루프 예시
    """
    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            # smolvla.py 내부의 train_step 활용하여 Flow Matching Loss 계산
            loss, info = model.train_step(
                images=batch['images'].to(device),
                instruction=batch['instruction'],
                robot_state=batch['state'].to(device),
                actions_gt=batch['action_chunk'].to(device)
            )
            
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            if step % 50 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
                
        # 모델 저장
        # save_path = f"checkpoints/smolvla_lora_epoch_{epoch}"
        # model.save_pretrained(save_path)
    """
    print("Train loop placeholder reached. Connect your dataloader to begin!")

if __name__ == "__main__":
    main()