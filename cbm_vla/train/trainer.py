"""
CBM-VLA Trainer Utilities
==========================

모든 Phase가 공유하는 학습 인프라:
  - TrainConfig: 공통 하이퍼파라미터
  - TrainLogger: 콘솔 + WandB 로깅
  - CheckpointManager: 체크포인트 저장/로드/정리
  - create_scheduler: LR scheduler 팩토리
  - set_seed, get_torch_dtype 등 유틸
"""

import os
import json
import time
import math
import shutil
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict


# ================================================================
# Training Config
# ================================================================

@dataclass
class TrainConfig:
    """Phase-공통 학습 설정. 각 Phase trainer가 이를 상속/확장합니다."""
    
    # 기본
    output_dir: str = "./checkpoints"
    seed: int = 42
    device: str = "cuda"
    dtype: str = "bfloat16"
    
    # 학습 루프
    epochs: int = 10
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Optimizer
    lr: float = 1e-4
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.999)
    
    # Scheduler
    scheduler_type: str = "cosine"     # cosine, linear, constant
    warmup_steps: int = 500
    min_lr_ratio: float = 0.1
    
    # 로깅
    log_interval: int = 50
    eval_interval: int = 500
    save_interval: int = 1000
    use_wandb: bool = False
    wandb_project: str = "cbm-vla"
    wandb_run_name: Optional[str] = None
    
    # 체크포인트
    resume_from: Optional[str] = None
    save_total_limit: int = 3
    
    # 데이터
    num_workers: int = 4
    pin_memory: bool = True


# ================================================================
# Logger
# ================================================================

class TrainLogger:
    """콘솔 + WandB 듀얼 로거"""
    
    def __init__(self, config: TrainConfig, phase_name: str):
        self.config = config
        self.phase_name = phase_name
        self._wandb_run = None
        self._start_time = time.time()
        
        if config.use_wandb:
            try:
                import wandb
                run_name = config.wandb_run_name or f"{phase_name}_{time.strftime('%m%d_%H%M')}"
                self._wandb_run = wandb.init(
                    project=config.wandb_project,
                    name=run_name,
                    config=asdict(config),
                )
            except ImportError:
                print("  [WARNING] wandb not installed")
    
    def log_step(self, info: Dict[str, Any], step: int, epoch: int, lr: float):
        """학습 스텝 로깅"""
        elapsed = time.time() - self._start_time
        
        # 콘솔 출력
        if step % self.config.log_interval == 0:
            loss_val = info.get("loss_total", info.get("loss_encoder_alignment", 0))
            detail = " | ".join(
                f"{k}={v:.4f}" for k, v in sorted(info.items())
                if isinstance(v, float) and k != "loss_total"
            )
            # 너무 길면 잘라냄
            if len(detail) > 120:
                detail = detail[:117] + "..."
            
            print(f"  [{self.phase_name}] ep={epoch} step={step} | "
                  f"loss={loss_val:.4f} | lr={lr:.2e} | {detail} | {elapsed:.0f}s")
        
        # WandB
        if self._wandb_run is not None:
            metrics = {f"{self.phase_name}/{k}": v for k, v in info.items()
                       if isinstance(v, (int, float))}
            metrics[f"{self.phase_name}/lr"] = lr
            self._wandb_run.log(metrics, step=step)
    
    def finish(self):
        if self._wandb_run is not None:
            self._wandb_run.finish()


# ================================================================
# Checkpoint Manager
# ================================================================

class CheckpointManager:
    """체크포인트 저장/로드/자동 정리"""
    
    def __init__(self, output_dir: str, save_total_limit: int = 3):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_total_limit = save_total_limit
        self._saved: List[str] = []
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        step: int,
        epoch: int,
        info: Optional[Dict] = None,
        tag: Optional[str] = None,
    ):
        name = tag or f"step_{step:06d}"
        save_dir = self.output_dir / name
        save_dir.mkdir(exist_ok=True)
        
        torch.save(model.state_dict(), save_dir / "model.pt")
        torch.save(optimizer.state_dict(), save_dir / "optimizer.pt")
        if scheduler is not None:
            torch.save(scheduler.state_dict(), save_dir / "scheduler.pt")
        
        with open(save_dir / "meta.json", 'w') as f:
            json.dump({"step": step, "epoch": epoch, "info": info or {}}, f)
        
        print(f"  ✓ Saved: {save_dir}")
        
        # 오래된 체크포인트 정리 (tag가 없는 step 체크포인트만 대상)
        if tag is None:
            self._saved.append(str(save_dir))
            while len(self._saved) > self.save_total_limit:
                old = self._saved.pop(0)
                if Path(old).exists():
                    shutil.rmtree(old)
    
    def load(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        path: Optional[str] = None,
    ) -> Dict[str, Any]:
        load_dir = Path(path) if path else self._find_latest()
        if load_dir is None or not load_dir.exists():
            print("  No checkpoint found, starting from scratch")
            return {"step": 0, "epoch": 0}
        
        state = torch.load(load_dir / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        
        if optimizer and (load_dir / "optimizer.pt").exists():
            optimizer.load_state_dict(
                torch.load(load_dir / "optimizer.pt", map_location="cpu", weights_only=True))
        if scheduler and (load_dir / "scheduler.pt").exists():
            scheduler.load_state_dict(
                torch.load(load_dir / "scheduler.pt", map_location="cpu", weights_only=True))
        
        meta = {"step": 0, "epoch": 0}
        if (load_dir / "meta.json").exists():
            with open(load_dir / "meta.json") as f:
                meta = json.load(f)
        
        print(f"  ✓ Loaded: {load_dir} (step={meta['step']})")
        return meta
    
    def _find_latest(self) -> Optional[Path]:
        candidates = sorted(self.output_dir.glob("step_*"), reverse=True)
        return candidates[0] if candidates else None


# ================================================================
# Scheduler Factory
# ================================================================

def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = config.warmup_steps
    min_r = config.min_lr_ratio
    
    if config.scheduler_type == "cosine":
        def fn(step):
            if step < warmup:
                return step / max(warmup, 1)
            p = (step - warmup) / max(total_steps - warmup, 1)
            return min_r + (1 - min_r) * 0.5 * (1 + math.cos(math.pi * p))
    elif config.scheduler_type == "linear":
        def fn(step):
            if step < warmup:
                return step / max(warmup, 1)
            return max(min_r, 1 - (step - warmup) / max(total_steps - warmup, 1))
    else:  # constant
        def fn(step):
            return step / max(warmup, 1) if step < warmup else 1.0
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


# ================================================================
# Utilities
# ================================================================

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_torch_dtype(s: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[s]


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total