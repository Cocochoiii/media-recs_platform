"""
Model Training Pipeline

Comprehensive training script for all recommendation models with
experiment tracking, early stopping, and distributed training support.
"""

import os
import sys
import logging
from typing import Dict, List
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    experiment_name: str = "media_recommender"
    model_type: str = "collaborative"
    device: str = "cuda"
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    early_stopping: bool = True
    patience: int = 5
    min_delta: float = 0.001
    use_amp: bool = True
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 5
    keep_n_checkpoints: int = 3
    data_path: str = "data/interactions.csv"
    num_workers: int = 4
    eval_every_n_steps: int = 1000
    k_values: List[int] = field(default_factory=lambda: [5, 10, 20])


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class WarmupScheduler:
    """Learning rate warmup scheduler."""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            scale = self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            scale = 0.5 * (1 + np.cos(np.pi * progress))
        
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = max(self.min_lr, self.base_lrs[i] * scale)


class Trainer:
    """Unified trainer for all recommendation models."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if config.use_amp else None
        self.early_stopping = EarlyStopping(config.patience, config.min_delta) if config.early_stopping else None
        
        self.train_losses: List[float] = []
        self.val_metrics: List[Dict[str, float]] = []
        self.best_model_state = None
        self.best_val_metric = float('inf')
        
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        if MLFLOW_AVAILABLE:
            mlflow.set_experiment(config.experiment_name)
    
    def setup_model(self, model: nn.Module):
        """Initialize model."""
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        logger.info(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int, loss_fn) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()
            
            if self.config.use_amp and self.scaler:
                with autocast():
                    loss = loss_fn(self.model, batch)
                self.scaler.scale(loss).backward()
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = loss_fn(self.model, batch)
                loss.backward()
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, loss_fn) -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()
        total_loss = 0.0
        
        for batch in val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            loss = loss_fn(self.model, batch)
            total_loss += loss.item()
        
        return {"val_loss": total_loss / len(val_loader)}
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, loss_fn):
        """Full training loop."""
        total_steps = len(train_loader) * self.config.epochs
        self.scheduler = WarmupScheduler(self.optimizer, self.config.warmup_steps, total_steps)
        
        if MLFLOW_AVAILABLE:
            mlflow.start_run()
            mlflow.log_params({"epochs": self.config.epochs, "batch_size": self.config.batch_size})
        
        logger.info("Starting training...")
        
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch, loss_fn)
            val_metrics = self.evaluate(val_loader, loss_fn)
            
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_metrics['val_loss']:.4f}")
            
            if val_metrics['val_loss'] < self.best_val_metric:
                self.best_val_metric = val_metrics['val_loss']
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self._save_checkpoint(epoch, is_best=True)
            
            if self.early_stopping and self.early_stopping(val_metrics['val_loss']):
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        if MLFLOW_AVAILABLE:
            mlflow.end_run()
        
        logger.info("Training complete!")
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint."""
        filename = "best_model.pt" if is_best else f"checkpoint_{epoch}.pt"
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        logger.info(f"Saved: {path}")


def collaborative_loss(model, batch):
    """Loss for collaborative filtering."""
    user_ids = batch["user_ids"]
    item_ids = batch["item_ids"]
    
    if "neg_item_ids" in batch:
        pos_scores = model(user_ids, item_ids)
        neg_scores = model(user_ids, batch["neg_item_ids"][:, 0])
        return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
    else:
        predictions = model(user_ids, item_ids)
        return nn.functional.mse_loss(predictions, batch["ratings"])


def main():
    """Main entry point."""
    config = TrainingConfig(model_type="collaborative", epochs=10, batch_size=64)
    trainer = Trainer(config)
    
    # Demo: Create sample model and data
    from models.collaborative_filter import NeuralCollaborativeFiltering, CollaborativeConfig
    from data.dataset import InteractionDataset
    
    model_config = CollaborativeConfig(num_users=1000, num_items=5000, embedding_dim=64)
    model = NeuralCollaborativeFiltering(model_config)
    trainer.setup_model(model)
    
    # Sample data
    sample_data = pd.DataFrame({
        "user_id": np.random.randint(1, 1000, 10000),
        "item_id": np.random.randint(1, 5000, 10000),
        "rating": np.random.uniform(1, 5, 10000)
    })
    
    train_df = sample_data.iloc[:8000]
    val_df = sample_data.iloc[8000:]
    
    train_dataset = InteractionDataset(train_df, 5000, mode="train")
    val_dataset = InteractionDataset(val_df, 5000, mode="val")
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    trainer.train(train_loader, val_loader, collaborative_loss)


if __name__ == "__main__":
    main()
