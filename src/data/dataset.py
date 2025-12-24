"""
Dataset and Data Loading Module

This module provides dataset classes and data loading utilities
for training recommendation models.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
import random
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data processing."""
    min_user_interactions: int = 3
    min_item_interactions: int = 5
    max_sequence_length: int = 50
    negative_sample_ratio: int = 4
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42


class InteractionDataset(Dataset):
    """
    Dataset for user-item interactions.
    
    Supports both explicit ratings and implicit feedback.
    """
    
    def __init__(
        self,
        interactions: pd.DataFrame,
        num_items: int,
        mode: str = "train",
        negative_samples: int = 4
    ):
        """
        Initialize interaction dataset.
        
        Args:
            interactions: DataFrame with columns [user_id, item_id, rating/label, timestamp]
            num_items: Total number of items
            mode: 'train', 'val', or 'test'
            negative_samples: Number of negative samples per positive
        """
        self.interactions = interactions
        self.num_items = num_items
        self.mode = mode
        self.negative_samples = negative_samples
        
        # Build user-item interaction sets for negative sampling
        self.user_items = defaultdict(set)
        for _, row in interactions.iterrows():
            self.user_items[row["user_id"]].add(row["item_id"])
        
        # Convert to list for indexing
        self.data = interactions.to_dict("records")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        interaction = self.data[idx]
        user_id = interaction["user_id"]
        item_id = interaction["item_id"]
        
        result = {
            "user_ids": torch.tensor(user_id, dtype=torch.long),
            "item_ids": torch.tensor(item_id, dtype=torch.long)
        }
        
        # Add rating if available (explicit feedback)
        if "rating" in interaction:
            result["ratings"] = torch.tensor(interaction["rating"], dtype=torch.float)
        
        # Sample negative items for implicit feedback
        if self.mode == "train" and self.negative_samples > 0:
            neg_items = self._sample_negatives(user_id, self.negative_samples)
            result["neg_item_ids"] = torch.tensor(neg_items, dtype=torch.long)
        
        return result
    
    def _sample_negatives(self, user_id: int, n: int) -> List[int]:
        """Sample negative items not in user's history."""
        positives = self.user_items[user_id]
        negatives = []
        
        while len(negatives) < n:
            item = random.randint(1, self.num_items - 1)
            if item not in positives:
                negatives.append(item)
        
        return negatives


class SequenceDataset(Dataset):
    """
    Dataset for sequential recommendation.
    
    Creates sequences of user interactions for sequential models.
    """
    
    def __init__(
        self,
        interactions: pd.DataFrame,
        max_length: int = 50,
        mode: str = "train"
    ):
        """
        Initialize sequence dataset.
        
        Args:
            interactions: DataFrame sorted by timestamp
            max_length: Maximum sequence length
            mode: 'train', 'val', or 'test'
        """
        self.max_length = max_length
        self.mode = mode
        
        # Group interactions by user
        self.user_sequences = self._build_sequences(interactions)
        
        # Create training samples
        self.samples = self._create_samples()
    
    def _build_sequences(self, interactions: pd.DataFrame) -> Dict[int, List[int]]:
        """Build item sequences for each user."""
        sequences = {}
        
        for user_id, group in interactions.groupby("user_id"):
            items = group.sort_values("timestamp")["item_id"].tolist()
            sequences[user_id] = items
        
        return sequences
    
    def _create_samples(self) -> List[Dict]:
        """Create sequence samples for training/evaluation."""
        samples = []
        
        for user_id, sequence in self.user_sequences.items():
            if len(sequence) < 3:
                continue
            
            if self.mode == "train":
                # Create multiple samples per user
                for i in range(2, len(sequence)):
                    input_seq = sequence[:i][-self.max_length:]
                    target = sequence[i]
                    samples.append({
                        "user_id": user_id,
                        "sequence": input_seq,
                        "target": target
                    })
            else:
                # For eval, use last item as target
                input_seq = sequence[:-1][-self.max_length:]
                target = sequence[-1]
                samples.append({
                    "user_id": user_id,
                    "sequence": input_seq,
                    "target": target
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        sequence = sample["sequence"]
        
        # Pad sequence
        padded_seq = [0] * (self.max_length - len(sequence)) + sequence
        
        return {
            "sequences": torch.tensor(padded_seq, dtype=torch.long),
            "lengths": torch.tensor(len(sequence), dtype=torch.long),
            "targets": torch.tensor(sample["target"], dtype=torch.long),
            "user_ids": torch.tensor(sample["user_id"], dtype=torch.long)
        }


class ContentDataset(Dataset):
    """
    Dataset for content-based recommendation.
    
    Pairs user interactions with item content features.
    """
    
    def __init__(
        self,
        interactions: pd.DataFrame,
        item_content: Dict[int, Dict[str, str]],
        tokenizer,
        max_length: int = 128,
        negative_samples: int = 4
    ):
        """
        Initialize content dataset.
        
        Args:
            interactions: DataFrame with user-item interactions
            item_content: Dictionary mapping item_id to content
            tokenizer: BERT tokenizer
            max_length: Maximum token length
            negative_samples: Number of negatives per positive
        """
        self.interactions = interactions.to_dict("records")
        self.item_content = item_content
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.negative_samples = negative_samples
        
        # Build item pool for negative sampling
        self.all_items = list(item_content.keys())
        self.user_items = defaultdict(set)
        for inter in self.interactions:
            self.user_items[inter["user_id"]].add(inter["item_id"])
    
    def __len__(self) -> int:
        return len(self.interactions)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        interaction = self.interactions[idx]
        user_id = interaction["user_id"]
        item_id = interaction["item_id"]
        
        # Get item content
        content = self.item_content.get(item_id, {})
        text = f"{content.get('title', '')} {content.get('description', '')}"
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        result = {
            "user_ids": torch.tensor(user_id, dtype=torch.long),
            "item_ids": torch.tensor(item_id, dtype=torch.long),
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0)
        }
        
        if "rating" in interaction:
            result["ratings"] = torch.tensor(interaction["rating"], dtype=torch.float)
        
        return result


class ContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning.
    
    Creates positive and negative pairs for contrastive training.
    """
    
    def __init__(
        self,
        interactions: pd.DataFrame,
        item_embeddings: Dict[int, np.ndarray],
        num_negatives: int = 10,
        hard_negative_ratio: float = 0.5
    ):
        """
        Initialize contrastive dataset.
        
        Args:
            interactions: DataFrame with user-item interactions
            item_embeddings: Pre-computed item embeddings
            num_negatives: Number of negative samples
            hard_negative_ratio: Ratio of hard negatives
        """
        self.interactions = interactions.to_dict("records")
        self.item_embeddings = item_embeddings
        self.num_negatives = num_negatives
        self.hard_negative_ratio = hard_negative_ratio
        
        self.all_items = list(item_embeddings.keys())
        self.user_items = defaultdict(set)
        for inter in self.interactions:
            self.user_items[inter["user_id"]].add(inter["item_id"])
        
        # Pre-compute item embedding matrix for hard negative mining
        self.item_ids = list(item_embeddings.keys())
        self.embedding_matrix = np.array([item_embeddings[i] for i in self.item_ids])
    
    def __len__(self) -> int:
        return len(self.interactions)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        interaction = self.interactions[idx]
        user_id = interaction["user_id"]
        pos_item_id = interaction["item_id"]
        
        # Get positive item embedding
        pos_embedding = self.item_embeddings[pos_item_id]
        
        # Sample negatives
        neg_embeddings = self._sample_negatives(user_id, pos_embedding)
        
        return {
            "user_ids": torch.tensor(user_id, dtype=torch.long),
            "positive_indices": torch.tensor(
                self.item_ids.index(pos_item_id) if pos_item_id in self.item_ids else 0,
                dtype=torch.long
            ),
            "positive_embeddings": torch.tensor(pos_embedding, dtype=torch.float),
            "negative_embeddings": torch.tensor(neg_embeddings, dtype=torch.float)
        }
    
    def _sample_negatives(self, user_id: int, pos_embedding: np.ndarray) -> np.ndarray:
        """Sample negative items with optional hard negative mining."""
        positives = self.user_items[user_id]
        
        # Number of hard vs random negatives
        num_hard = int(self.num_negatives * self.hard_negative_ratio)
        num_random = self.num_negatives - num_hard
        
        negatives = []
        
        # Random negatives
        while len(negatives) < num_random:
            item = random.choice(self.all_items)
            if item not in positives:
                negatives.append(self.item_embeddings[item])
        
        # Hard negatives (most similar to positive but not in history)
        if num_hard > 0:
            similarities = np.dot(self.embedding_matrix, pos_embedding)
            sorted_indices = np.argsort(similarities)[::-1]
            
            for idx in sorted_indices:
                item_id = self.item_ids[idx]
                if item_id not in positives and len(negatives) < self.num_negatives:
                    negatives.append(self.item_embeddings[item_id])
                if len(negatives) >= self.num_negatives:
                    break
        
        return np.array(negatives)


class DataProcessor:
    """
    Data processor for preparing recommendation datasets.
    
    Handles data loading, preprocessing, splitting, and batch creation.
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
    
    def load_interactions(self, filepath: str) -> pd.DataFrame:
        """Load interaction data from file."""
        if filepath.endswith(".csv"):
            df = pd.read_csv(filepath)
        elif filepath.endswith(".parquet"):
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        return df
    
    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter users and items with minimum interactions."""
        # Filter users
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= self.config.min_user_interactions].index
        df = df[df["user_id"].isin(valid_users)]
        
        # Filter items
        item_counts = df["item_id"].value_counts()
        valid_items = item_counts[item_counts >= self.config.min_item_interactions].index
        df = df[df["item_id"].isin(valid_items)]
        
        logger.info(f"After filtering: {len(df)} interactions, "
                   f"{df['user_id'].nunique()} users, "
                   f"{df['item_id'].nunique()} items")
        
        return df
    
    def create_id_mappings(
        self, 
        df: pd.DataFrame
    ) -> Tuple[Dict[Any, int], Dict[Any, int]]:
        """Create user and item ID mappings."""
        user_ids = df["user_id"].unique()
        item_ids = df["item_id"].unique()
        
        user_mapping = {uid: idx + 1 for idx, uid in enumerate(user_ids)}
        item_mapping = {iid: idx + 1 for idx, iid in enumerate(item_ids)}
        
        return user_mapping, item_mapping
    
    def split_data(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/val/test sets."""
        # Sort by timestamp if available
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")
        
        n = len(df)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        logger.info(f"Split sizes - Train: {len(train_df)}, "
                   f"Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def create_dataloaders(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        num_items: int,
        batch_size: int = 64,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create DataLoaders for train/val/test."""
        train_dataset = InteractionDataset(train_df, num_items, mode="train")
        val_dataset = InteractionDataset(val_df, num_items, mode="val")
        test_dataset = InteractionDataset(test_df, num_items, mode="test")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        return train_loader, val_loader, test_loader


def collate_sequences(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for sequence data."""
    return {
        "sequences": torch.stack([item["sequences"] for item in batch]),
        "lengths": torch.stack([item["lengths"] for item in batch]),
        "targets": torch.stack([item["targets"] for item in batch]),
        "user_ids": torch.stack([item["user_ids"] for item in batch])
    }


if __name__ == "__main__":
    # Example usage
    config = DataConfig(
        min_user_interactions=3,
        min_item_interactions=5
    )
    
    processor = DataProcessor(config)
    
    # Create sample data
    sample_data = pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
        "item_id": [10, 20, 30, 10, 40, 50, 20, 30, 40, 60],
        "rating": [5, 4, 3, 5, 4, 3, 4, 5, 3, 4],
        "timestamp": range(10)
    })
    
    train_df, val_df, test_df = processor.split_data(sample_data)
    print(f"Train samples: {len(train_df)}")
