"""
Pytest Configuration and Fixtures

Shared fixtures for all tests.
"""

import pytest
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd


@pytest.fixture(scope="session")
def sample_interactions():
    """Generate sample interaction data."""
    np.random.seed(42)
    n_interactions = 10000
    n_users = 500
    n_items = 1000
    
    return pd.DataFrame({
        "user_id": np.random.randint(1, n_users + 1, n_interactions),
        "item_id": np.random.randint(1, n_items + 1, n_interactions),
        "rating": np.random.uniform(1, 5, n_interactions).round(1),
        "timestamp": np.sort(np.random.randint(1600000000, 1700000000, n_interactions)),
        "interaction_type": np.random.choice(
            ["view", "click", "purchase"], n_interactions, p=[0.7, 0.2, 0.1]
        )
    })


@pytest.fixture(scope="session")
def sample_items():
    """Generate sample item catalog."""
    n_items = 1000
    genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Horror", "Romance", "Documentary"]
    tags = ["popular", "trending", "classic", "indie", "award-winning", "family", "mature"]
    
    items = []
    for i in range(1, n_items + 1):
        items.append({
            "id": i,
            "title": f"Item {i}",
            "description": f"Description for item {i}. " * 5,
            "genre": np.random.choice(genres),
            "tags": list(np.random.choice(tags, size=np.random.randint(1, 4), replace=False)),
            "duration": np.random.randint(1800, 10800),  # 30 min to 3 hours
            "release_date": f"20{np.random.randint(10, 24):02d}-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}",
            "popularity_score": np.random.uniform(0, 1)
        })
    
    return items


@pytest.fixture(scope="session")
def sample_users():
    """Generate sample user profiles."""
    n_users = 500
    age_groups = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    genders = ["male", "female", "other"]
    devices = ["mobile", "desktop", "tablet"]
    tiers = ["free", "basic", "premium"]
    
    users = []
    for i in range(1, n_users + 1):
        users.append({
            "id": i,
            "age_group": np.random.choice(age_groups),
            "gender": np.random.choice(genders),
            "device_type": np.random.choice(devices),
            "subscription_tier": np.random.choice(tiers, p=[0.6, 0.3, 0.1])
        })
    
    return users


@pytest.fixture
def sample_sequences(sample_interactions):
    """Generate sequence data from interactions."""
    sequences = []
    
    for user_id in sample_interactions["user_id"].unique()[:50]:
        user_items = sample_interactions[
            sample_interactions["user_id"] == user_id
        ].sort_values("timestamp")["item_id"].tolist()
        
        if len(user_items) >= 5:
            sequences.append({
                "user_id": user_id,
                "sequence": user_items[:-1],
                "target": user_items[-1]
            })
    
    return sequences


@pytest.fixture
def sample_embeddings():
    """Generate random embeddings for testing."""
    n_items = 1000
    embedding_dim = 128
    
    return {
        i: np.random.randn(embedding_dim).astype(np.float32)
        for i in range(1, n_items + 1)
    }


@pytest.fixture
def ground_truth(sample_interactions):
    """Generate ground truth for evaluation."""
    ground_truth = {}
    
    for user_id in sample_interactions["user_id"].unique()[:100]:
        user_items = sample_interactions[
            sample_interactions["user_id"] == user_id
        ]["item_id"].tolist()
        
        # Use last 20% as test items
        split_idx = int(len(user_items) * 0.8)
        ground_truth[user_id] = set(user_items[split_idx:])
    
    return ground_truth


@pytest.fixture(scope="session")
def device():
    """Get device for testing (CPU for CI)."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# Skip markers for different test categories
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
