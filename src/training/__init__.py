"""Training module for Media Recommender."""

from .trainer import Trainer, TrainingConfig, EarlyStopping, WarmupScheduler

__all__ = [
    "Trainer", 
    "TrainingConfig", 
    "EarlyStopping", 
    "WarmupScheduler"
]

# Training scripts are available as:
# python -m src.training.train_collaborative
# python -m src.training.train_sequential
# python scripts/train_all.py
