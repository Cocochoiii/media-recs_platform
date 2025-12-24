#!/usr/bin/env python3
"""
Train Collaborative Filtering Models

This module provides training functionality for collaborative filtering models
including NCF, Matrix Factorization, and BPR-based models.

Usage:
    python -m src.training.train_collaborative
    python -m src.training.train_collaborative --model ncf --epochs 20
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.collaborative_filter import (
    CollaborativeConfig,
    MatrixFactorization,
    NeuralCollaborativeFiltering,
    ImplicitFeedbackNCF,
    CollaborativeFilteringRecommender
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_data(
    num_users: int = 5000,
    num_items: int = 2000,
    num_interactions: int = 50000
) -> Dict[str, Any]:
    """Generate sample training data."""
    np.random.seed(42)
    
    users = np.random.randint(0, num_users, num_interactions)
    items = np.random.randint(0, num_items, num_interactions)
    ratings = np.clip(np.random.normal(3.5, 1.0, num_interactions), 1, 5)
    
    df = pd.DataFrame({
        'user_id': users,
        'item_id': items,
        'rating': ratings
    }).drop_duplicates(['user_id', 'item_id'])
    
    # Split
    n = len(df)
    train_df = df.iloc[:int(0.8*n)]
    val_df = df.iloc[int(0.8*n):]
    
    return {
        'train': train_df,
        'val': val_df,
        'num_users': num_users,
        'num_items': num_items
    }


def train_ncf(
    data: Dict[str, Any],
    config: CollaborativeConfig,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 0.001,
    device: str = 'cpu'
) -> CollaborativeFilteringRecommender:
    """Train NCF model."""
    
    recommender = CollaborativeFilteringRecommender(
        config=config,
        model_type='ncf',
        device=device
    )
    
    optimizer = torch.optim.Adam(recommender.model.parameters(), lr=lr)
    train_df = data['train']
    
    logger.info(f"Training NCF on {len(train_df)} samples...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        
        pbar = tqdm(range(0, len(train_df), batch_size), desc=f"Epoch {epoch+1}")
        for i in pbar:
            batch_df = train_df.iloc[i:i+batch_size]
            
            batch = {
                'user_ids': torch.tensor(batch_df['user_id'].values, dtype=torch.long),
                'item_ids': torch.tensor(batch_df['item_id'].values, dtype=torch.long),
                'ratings': torch.tensor(batch_df['rating'].values / 5.0, dtype=torch.float32),
                'neg_item_ids': torch.randint(0, data['num_items'], (len(batch_df),))
            }
            
            loss = recommender.train_step(batch, optimizer)
            epoch_loss += loss
            n_batches += 1
            
            pbar.set_postfix({'loss': f'{epoch_loss/n_batches:.4f}'})
        
        avg_loss = epoch_loss / n_batches
        logger.info(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    return recommender


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ncf', choices=['mf', 'ncf', 'implicit'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--embedding-dim', type=int, default=64)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save-path', type=str, default='checkpoints/collaborative_model.pt')
    args = parser.parse_args()
    
    # Generate data
    data = generate_sample_data()
    
    # Configure model
    config = CollaborativeConfig(
        num_users=data['num_users'],
        num_items=data['num_items'],
        embedding_dim=args.embedding_dim,
        hidden_layers=[128, 64, 32]
    )
    
    # Train
    recommender = train_ncf(
        data=data,
        config=config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )
    
    # Save
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    recommender.save(args.save_path)
    logger.info(f"Model saved to {args.save_path}")
    
    # Test
    recs = recommender.recommend(user_id=0, n_recommendations=5)
    logger.info(f"Sample recommendations for user 0: {recs}")


if __name__ == '__main__':
    main()
