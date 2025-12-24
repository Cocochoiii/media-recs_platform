#!/usr/bin/env python3
"""
Train All Models Script

This script trains all recommendation models in sequence with proper
data loading, validation, and checkpointing.

Usage:
    python scripts/train_all.py
    python scripts/train_all.py --models ncf,bert4rec --epochs 50
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(
    num_users: int = 10000,
    num_items: int = 5000,
    num_interactions: int = 100000,
    seed: int = 42
) -> Dict[str, Any]:
    """Generate synthetic training data for demonstration."""
    np.random.seed(seed)
    
    logger.info(f"Generating synthetic data: {num_users} users, {num_items} items, {num_interactions} interactions")
    
    # Generate user-item interactions with power-law distribution
    # Popular items get more interactions
    item_popularity = np.random.pareto(1.5, num_items) + 1
    item_probs = item_popularity / item_popularity.sum()
    
    user_activity = np.random.pareto(1.2, num_users) + 1
    user_probs = user_activity / user_activity.sum()
    
    users = np.random.choice(num_users, num_interactions, p=user_probs)
    items = np.random.choice(num_items, num_interactions, p=item_probs)
    
    # Generate ratings (implicit to explicit conversion simulation)
    # More interactions = higher implicit score
    ratings = np.clip(np.random.normal(3.5, 1.0, num_interactions), 1, 5)
    
    # Timestamps
    base_time = 1600000000
    timestamps = base_time + np.sort(np.random.randint(0, 100000000, num_interactions))
    
    interactions_df = pd.DataFrame({
        'user_id': users,
        'item_id': items,
        'rating': ratings,
        'timestamp': timestamps
    })
    
    # Remove duplicates (keep last interaction)
    interactions_df = interactions_df.drop_duplicates(
        subset=['user_id', 'item_id'], keep='last'
    ).reset_index(drop=True)
    
    # Generate item features
    genres = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror', 'Romance', 'Documentary', 'Thriller']
    item_features = pd.DataFrame({
        'item_id': range(num_items),
        'title': [f'Item {i}' for i in range(num_items)],
        'genre': np.random.choice(genres, num_items),
        'popularity': item_popularity / item_popularity.max(),
        'release_year': np.random.randint(1990, 2024, num_items)
    })
    
    # Generate user features
    age_groups = ['18-24', '25-34', '35-44', '45-54', '55+']
    user_features = pd.DataFrame({
        'user_id': range(num_users),
        'age_group': np.random.choice(age_groups, num_users),
        'activity_level': user_activity / user_activity.max()
    })
    
    # Create sequences for sequential models
    sequences = []
    for user_id in interactions_df['user_id'].unique()[:1000]:  # Limit for speed
        user_items = interactions_df[
            interactions_df['user_id'] == user_id
        ].sort_values('timestamp')['item_id'].tolist()
        
        if len(user_items) >= 5:
            sequences.append({
                'user_id': user_id,
                'sequence': user_items[:-1][-50:],  # Max 50 items
                'target': user_items[-1]
            })
    
    # Train/val/test split
    n = len(interactions_df)
    train_idx = int(0.8 * n)
    val_idx = int(0.9 * n)
    
    train_df = interactions_df.iloc[:train_idx]
    val_df = interactions_df.iloc[train_idx:val_idx]
    test_df = interactions_df.iloc[val_idx:]
    
    logger.info(f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df,
        'items': item_features,
        'users': user_features,
        'sequences': sequences,
        'num_users': num_users,
        'num_items': num_items
    }


def train_collaborative_filtering(
    data: Dict[str, Any],
    config: Optional[Dict] = None,
    epochs: int = 10,
    device: str = 'cpu'
) -> Dict[str, float]:
    """Train collaborative filtering model (NCF)."""
    from src.models.collaborative_filter import (
        CollaborativeConfig, CollaborativeFilteringRecommender
    )
    
    logger.info("Training Collaborative Filtering (NCF)...")
    
    cfg = CollaborativeConfig(
        num_users=data['num_users'],
        num_items=data['num_items'],
        embedding_dim=config.get('embedding_dim', 64) if config else 64,
        hidden_layers=config.get('hidden_layers', [128, 64]) if config else [128, 64]
    )
    
    recommender = CollaborativeFilteringRecommender(
        config=cfg,
        model_type='ncf',
        device=device
    )
    
    optimizer = torch.optim.Adam(recommender.model.parameters(), lr=0.001)
    
    train_df = data['train']
    batch_size = 256
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        # Shuffle data
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        
        for i in range(0, len(train_df), batch_size):
            batch_df = train_df.iloc[i:i+batch_size]
            
            batch = {
                'user_ids': torch.tensor(batch_df['user_id'].values, dtype=torch.long),
                'item_ids': torch.tensor(batch_df['item_id'].values, dtype=torch.long),
                'ratings': torch.tensor(batch_df['rating'].values, dtype=torch.float32),
                'neg_item_ids': torch.randint(0, data['num_items'], (len(batch_df),))
            }
            
            loss = recommender.train_step(batch, optimizer)
            epoch_loss += loss
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        logger.info(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save model
    save_path = Path('checkpoints/ncf_model.pt')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    recommender.save(str(save_path))
    logger.info(f"  Model saved to {save_path}")
    
    return {'final_loss': losses[-1], 'losses': losses}


def train_sequential_model(
    data: Dict[str, Any],
    model_type: str = 'lstm',
    config: Optional[Dict] = None,
    epochs: int = 10,
    device: str = 'cpu'
) -> Dict[str, float]:
    """Train sequential model (LSTM/GRU4Rec/SASRec)."""
    from src.models.lstm_sequential import LSTMConfig, SequentialRecommender
    
    logger.info(f"Training Sequential Model ({model_type.upper()})...")
    
    cfg = LSTMConfig(
        num_items=data['num_items'],
        embedding_dim=config.get('embedding_dim', 64) if config else 64,
        hidden_size=config.get('hidden_size', 128) if config else 128,
        num_layers=config.get('num_layers', 2) if config else 2,
        max_sequence_length=50
    )
    
    recommender = SequentialRecommender(
        config=cfg,
        model_type=model_type,
        device=device
    )
    
    optimizer = torch.optim.Adam(recommender.model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    sequences = data['sequences']
    batch_size = 64
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        np.random.shuffle(sequences)
        
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            
            # Pad sequences
            max_len = min(50, max(len(s['sequence']) for s in batch_seqs))
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
        
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        logger.info(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save model
    save_path = Path(f'checkpoints/{model_type}_model.pt')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': recommender.model.state_dict(),
        'config': cfg
    }, str(save_path))
    logger.info(f"  Model saved to {save_path}")
    
    return {'final_loss': losses[-1], 'losses': losses}


def train_transformer_model(
    data: Dict[str, Any],
    config: Optional[Dict] = None,
    epochs: int = 10,
    device: str = 'cpu'
) -> Dict[str, float]:
    """Train BERT4Rec transformer model."""
    from src.models.transformer_models import TransformerRecConfig, BERT4Rec
    
    logger.info("Training BERT4Rec...")
    
    cfg = TransformerRecConfig(
        num_items=data['num_items'],
        max_seq_len=50,
        embedding_dim=config.get('embedding_dim', 128) if config else 128,
        num_heads=config.get('num_heads', 4) if config else 4,
        num_layers=config.get('num_layers', 2) if config else 2
    )
    
    model = BERT4Rec(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    sequences = data['sequences']
    batch_size = 32
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        np.random.shuffle(sequences)
        
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            
            max_len = 50
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
            model.train()
            
            logits = model(seq_tensor)
            loss = criterion(logits, target_tensor)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        logger.info(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save model
    save_path = Path('checkpoints/bert4rec_model.pt')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': cfg
    }, str(save_path))
    logger.info(f"  Model saved to {save_path}")
    
    return {'final_loss': losses[-1], 'losses': losses}


def train_deepfm_model(
    data: Dict[str, Any],
    config: Optional[Dict] = None,
    epochs: int = 10,
    device: str = 'cpu'
) -> Dict[str, float]:
    """Train DeepFM model for CTR prediction."""
    from src.models.deep_feature_interaction import FeatureInteractionConfig, DeepFM
    
    logger.info("Training DeepFM...")
    
    cfg = FeatureInteractionConfig(
        sparse_feature_dims={
            'user_id': data['num_users'],
            'item_id': data['num_items'],
            'hour': 24,
            'day': 7
        },
        embedding_dim=config.get('embedding_dim', 16) if config else 16,
        hidden_dims=config.get('hidden_dims', [128, 64]) if config else [128, 64]
    )
    
    model = DeepFM(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    train_df = data['train'].copy()
    # Convert ratings to binary (>3 = positive)
    train_df['label'] = (train_df['rating'] > 3).astype(float)
    train_df['hour'] = (train_df['timestamp'] // 3600) % 24
    train_df['day'] = (train_df['timestamp'] // 86400) % 7
    
    batch_size = 256
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        
        for i in range(0, len(train_df), batch_size):
            batch_df = train_df.iloc[i:i+batch_size]
            
            batch = {
                'user_id': torch.tensor(batch_df['user_id'].values, dtype=torch.long).to(device),
                'item_id': torch.tensor(batch_df['item_id'].values, dtype=torch.long).to(device),
                'hour': torch.tensor(batch_df['hour'].values, dtype=torch.long).to(device),
                'day': torch.tensor(batch_df['day'].values, dtype=torch.long).to(device)
            }
            labels = torch.tensor(batch_df['label'].values, dtype=torch.float32).to(device)
            
            optimizer.zero_grad()
            model.train()
            
            predictions = model(batch)
            loss = criterion(predictions, labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        logger.info(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save model
    save_path = Path('checkpoints/deepfm_model.pt')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': cfg
    }, str(save_path))
    logger.info(f"  Model saved to {save_path}")
    
    return {'final_loss': losses[-1], 'losses': losses}


def train_mmoe_model(
    data: Dict[str, Any],
    config: Optional[Dict] = None,
    epochs: int = 10,
    device: str = 'cpu'
) -> Dict[str, float]:
    """Train MMoE multi-task model."""
    from src.models.multi_task_learning import MTLConfig, MMoE
    
    logger.info("Training MMoE (Multi-Task Learning)...")
    
    cfg = MTLConfig(
        num_users=data['num_users'],
        num_items=data['num_items'],
        task_names=['ctr', 'cvr', 'engagement'],
        num_experts=config.get('num_experts', 6) if config else 6,
        embedding_dim=config.get('embedding_dim', 64) if config else 64
    )
    
    model = MMoE(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_df = data['train'].copy()
    # Generate synthetic multi-task labels
    train_df['ctr'] = (train_df['rating'] > 2).astype(float)  # Click
    train_df['cvr'] = (train_df['rating'] > 4).astype(float)  # Convert
    train_df['engagement'] = train_df['rating'] / 5  # Engagement score
    
    batch_size = 256
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        
        for i in range(0, len(train_df), batch_size):
            batch_df = train_df.iloc[i:i+batch_size]
            
            user_ids = torch.tensor(batch_df['user_id'].values, dtype=torch.long).to(device)
            item_ids = torch.tensor(batch_df['item_id'].values, dtype=torch.long).to(device)
            
            labels = {
                'ctr': torch.tensor(batch_df['ctr'].values, dtype=torch.float32).to(device),
                'cvr': torch.tensor(batch_df['cvr'].values, dtype=torch.float32).to(device),
                'engagement': torch.tensor(batch_df['engagement'].values, dtype=torch.float32).to(device)
            }
            
            optimizer.zero_grad()
            model.train()
            
            outputs = model(user_ids, item_ids)
            
            # Multi-task loss
            loss = 0
            for task in cfg.task_names:
                task_loss = nn.functional.binary_cross_entropy(outputs[task], labels[task])
                loss += task_loss
            loss /= len(cfg.task_names)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        logger.info(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save model
    save_path = Path('checkpoints/mmoe_model.pt')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': cfg
    }, str(save_path))
    logger.info(f"  Model saved to {save_path}")
    
    return {'final_loss': losses[-1], 'losses': losses}


def evaluate_models(data: Dict[str, Any], device: str = 'cpu') -> Dict[str, Dict[str, float]]:
    """Evaluate all trained models."""
    from src.utils.metrics import (
        compute_precision_at_k, compute_recall_at_k, 
        compute_ndcg_at_k, compute_hit_rate
    )
    
    logger.info("Evaluating models...")
    results = {}
    
    # Build ground truth from test set
    test_df = data['test']
    ground_truth = {}
    for user_id in test_df['user_id'].unique()[:100]:
        user_items = test_df[test_df['user_id'] == user_id]['item_id'].tolist()
        if len(user_items) > 0:
            ground_truth[user_id] = set(user_items)
    
    # Evaluate NCF
    try:
        from src.models.collaborative_filter import (
            CollaborativeConfig, CollaborativeFilteringRecommender
        )
        
        cfg = CollaborativeConfig(
            num_users=data['num_users'],
            num_items=data['num_items']
        )
        recommender = CollaborativeFilteringRecommender(cfg, model_type='ncf', device=device)
        recommender.load('checkpoints/ncf_model.pt')
        
        predictions = {}
        for user_id in list(ground_truth.keys())[:50]:
            recs = recommender.recommend(user_id, n_recommendations=10)
            predictions[user_id] = [item_id for item_id, _ in recs]
        
        results['ncf'] = {
            'precision@10': compute_precision_at_k(predictions, ground_truth, k=10),
            'recall@10': compute_recall_at_k(predictions, ground_truth, k=10),
            'ndcg@10': compute_ndcg_at_k(predictions, ground_truth, k=10),
            'hit_rate@10': compute_hit_rate(predictions, ground_truth)
        }
        logger.info(f"  NCF: {results['ncf']}")
    except Exception as e:
        logger.warning(f"  NCF evaluation failed: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train all recommendation models')
    parser.add_argument('--models', type=str, default='all',
                       help='Comma-separated list of models to train (ncf,lstm,gru4rec,sasrec,bert4rec,deepfm,mmoe)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, mps, auto)')
    parser.add_argument('--data-size', type=str, default='medium',
                       help='Data size: small, medium, large')
    parser.add_argument('--eval', action='store_true',
                       help='Run evaluation after training')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Data size configuration
    size_configs = {
        'small': {'num_users': 1000, 'num_items': 500, 'num_interactions': 10000},
        'medium': {'num_users': 10000, 'num_items': 5000, 'num_interactions': 100000},
        'large': {'num_users': 50000, 'num_items': 20000, 'num_interactions': 500000}
    }
    
    data_config = size_configs.get(args.data_size, size_configs['medium'])
    
    # Generate data
    data = generate_synthetic_data(**data_config)
    
    # Parse models to train
    if args.models == 'all':
        models_to_train = ['ncf', 'lstm', 'bert4rec', 'deepfm', 'mmoe']
    else:
        models_to_train = [m.strip() for m in args.models.split(',')]
    
    # Training results
    results = {}
    
    print("\n" + "="*60)
    print(" Media Recommender - Model Training")
    print("="*60 + "\n")
    
    # Train each model
    for model_name in models_to_train:
        try:
            if model_name == 'ncf':
                results['ncf'] = train_collaborative_filtering(data, epochs=args.epochs, device=device)
            elif model_name == 'lstm':
                results['lstm'] = train_sequential_model(data, model_type='lstm', epochs=args.epochs, device=device)
            elif model_name == 'gru4rec':
                results['gru4rec'] = train_sequential_model(data, model_type='gru4rec', epochs=args.epochs, device=device)
            elif model_name == 'sasrec':
                results['sasrec'] = train_sequential_model(data, model_type='sasrec', epochs=args.epochs, device=device)
            elif model_name == 'bert4rec':
                results['bert4rec'] = train_transformer_model(data, epochs=args.epochs, device=device)
            elif model_name == 'deepfm':
                results['deepfm'] = train_deepfm_model(data, epochs=args.epochs, device=device)
            elif model_name == 'mmoe':
                results['mmoe'] = train_mmoe_model(data, epochs=args.epochs, device=device)
            else:
                logger.warning(f"Unknown model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print(" Training Summary")
    print("="*60)
    for model_name, result in results.items():
        print(f"  {model_name}: final_loss = {result['final_loss']:.4f}")
    
    # Evaluation
    if args.eval:
        print("\n" + "="*60)
        print(" Evaluation Results")
        print("="*60)
        eval_results = evaluate_models(data, device=device)
        for model_name, metrics in eval_results.items():
            print(f"\n  {model_name}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")
    
    print("\n✅ Training complete!")
    print(f"   Checkpoints saved to: checkpoints/")


if __name__ == '__main__':
    main()
