#!/usr/bin/env python3
"""
Train Sequential Models (LSTM, GRU4Rec, SASRec)

Usage:
    python -m src.training.train_sequential
    python -m src.training.train_sequential --model sasrec --epochs 20
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.lstm_sequential import (
    LSTMConfig,
    LSTMSequentialModel,
    GRU4Rec,
    SASRec,
    SequentialRecommender
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sequence_data(
    num_users: int = 1000,
    num_items: int = 500,
    max_seq_len: int = 50
) -> Dict[str, Any]:
    """Generate sample sequence data."""
    np.random.seed(42)
    
    sequences = []
    for user_id in range(num_users):
        seq_len = np.random.randint(5, max_seq_len)
        items = np.random.randint(0, num_items, seq_len).tolist()
        if len(items) > 1:
            sequences.append({
                'user_id': user_id,
                'sequence': items[:-1],
                'target': items[-1]
            })
    
    # Split
    n = len(sequences)
    train_seqs = sequences[:int(0.8*n)]
    val_seqs = sequences[int(0.8*n):]
    
    return {
        'train': train_seqs,
        'val': val_seqs,
        'num_items': num_items,
        'max_seq_len': max_seq_len
    }


def train_sequential(
    data: Dict[str, Any],
    model_type: str = 'lstm',
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 0.001,
    device: str = 'cpu'
) -> nn.Module:
    """Train sequential model."""
    
    config = LSTMConfig(
        num_items=data['num_items'],
        embedding_dim=64,
        hidden_size=128,
        num_layers=2,
        max_sequence_length=data['max_seq_len']
    )
    
    recommender = SequentialRecommender(config, model_type=model_type, device=device)
    optimizer = torch.optim.Adam(recommender.model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    train_seqs = data['train']
    
    logger.info(f"Training {model_type.upper()} on {len(train_seqs)} sequences...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        np.random.shuffle(train_seqs)
        
        pbar = tqdm(range(0, len(train_seqs), batch_size), desc=f"Epoch {epoch+1}")
        for i in pbar:
            batch_seqs = train_seqs[i:i+batch_size]
            
            max_len = data['max_seq_len']
            padded_seqs = []
            targets = []
            
            for s in batch_seqs:
                seq = s['sequence'][-max_len:]
                padded = [0] * (max_len - len(seq)) + seq
                padded_seqs.append(padded)
                targets.append(s['target'])
            
            seq_tensor = torch.tensor(padded_seqs, dtype=torch.long).to(device)
            target_tensor = torch.tensor(targets, dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            recommender.model.train()
            
            logits = recommender.model(seq_tensor)
            loss = criterion(logits, target_tensor)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': f'{epoch_loss/n_batches:.4f}'})
        
        logger.info(f"Epoch {epoch+1}: Loss = {epoch_loss/n_batches:.4f}")
    
    return recommender.model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'gru4rec', 'sasrec'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save-path', type=str, default=None)
    args = parser.parse_args()
    
    if args.save_path is None:
        args.save_path = f'checkpoints/{args.model}_model.pt'
    
    data = generate_sequence_data()
    
    model = train_sequential(
        data=data,
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )
    
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, args.save_path)
    logger.info(f"Model saved to {args.save_path}")


if __name__ == '__main__':
    main()
