"""
Unit Tests for Recommendation Models

Comprehensive tests for all model components.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from typing import Dict, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.collaborative_filter import (
    CollaborativeConfig,
    MatrixFactorization,
    NeuralCollaborativeFiltering,
    ImplicitFeedbackNCF,
    CollaborativeFilteringRecommender
)
from models.lstm_sequential import (
    LSTMConfig,
    LSTMSequentialModel,
    GRU4Rec,
    SASRec,
    SequentialRecommender
)
from models.contrastive_learner import (
    ContrastiveConfig,
    ContrastiveEncoder,
    InfoNCELoss,
    ContrastiveLearningRecommender
)


# Fixtures
@pytest.fixture
def collab_config():
    return CollaborativeConfig(
        num_users=100,
        num_items=50,
        embedding_dim=32,
        hidden_layers=[64, 32]
    )


@pytest.fixture
def lstm_config():
    return LSTMConfig(
        num_items=50,
        embedding_dim=32,
        hidden_size=64,
        num_layers=1,
        max_sequence_length=20
    )


@pytest.fixture
def contrastive_config():
    return ContrastiveConfig(
        embedding_dim=64,
        projection_dim=32,
        temperature=0.1,
        num_negatives=5
    )


@pytest.fixture
def sample_batch():
    batch_size = 8
    return {
        "user_ids": torch.randint(1, 100, (batch_size,)),
        "item_ids": torch.randint(1, 50, (batch_size,)),
        "ratings": torch.rand(batch_size) * 4 + 1,
        "neg_item_ids": torch.randint(1, 50, (batch_size, 4))
    }


@pytest.fixture
def sample_sequences():
    batch_size = 8
    seq_len = 15
    return {
        "sequences": torch.randint(1, 50, (batch_size, seq_len)),
        "lengths": torch.randint(5, seq_len + 1, (batch_size,)),
        "targets": torch.randint(1, 50, (batch_size,))
    }


# Collaborative Filtering Tests
class TestMatrixFactorization:
    def test_forward(self, collab_config, sample_batch):
        model = MatrixFactorization(collab_config)
        output = model(sample_batch["user_ids"], sample_batch["item_ids"])
        
        assert output.shape == (8,)
        assert not torch.isnan(output).any()
    
    def test_embedding_shapes(self, collab_config):
        model = MatrixFactorization(collab_config)
        
        assert model.user_embedding.weight.shape == (100, 32)
        assert model.item_embedding.weight.shape == (50, 32)
    
    def test_get_embeddings(self, collab_config):
        model = MatrixFactorization(collab_config)
        
        user_emb = model.get_user_embedding(5)
        item_emb = model.get_item_embedding(10)
        
        assert user_emb.shape == (32,)
        assert item_emb.shape == (32,)


class TestNeuralCollaborativeFiltering:
    def test_forward(self, collab_config, sample_batch):
        model = NeuralCollaborativeFiltering(collab_config)
        output = model(sample_batch["user_ids"], sample_batch["item_ids"])
        
        assert output.shape == (8,)
        assert (output >= 0).all() and (output <= 1).all()  # Sigmoid output
    
    def test_user_representation(self, collab_config):
        model = NeuralCollaborativeFiltering(collab_config)
        rep = model.get_user_representation(5)
        
        # GMF dim + MLP dim
        expected_dim = 32 * 2
        assert rep.shape == (expected_dim,)


class TestImplicitFeedbackNCF:
    def test_bpr_loss(self, collab_config, sample_batch):
        model = ImplicitFeedbackNCF(collab_config)
        
        pos_scores, neg_scores, loss = model.forward_triplet(
            sample_batch["user_ids"],
            sample_batch["item_ids"],
            sample_batch["neg_item_ids"][:, 0]
        )
        
        assert pos_scores.shape == (8,)
        assert neg_scores.shape == (8,)
        assert loss.dim() == 0  # Scalar
        assert loss >= 0


class TestCollaborativeRecommender:
    def test_recommend(self, collab_config):
        recommender = CollaborativeFilteringRecommender(
            config=collab_config,
            model_type="ncf",
            device="cpu"
        )
        
        recs = recommender.recommend(user_id=5, n_recommendations=5)
        
        assert len(recs) == 5
        assert all(isinstance(r, tuple) and len(r) == 2 for r in recs)
        assert all(r[0] > 0 for r in recs)  # Valid item IDs
    
    def test_exclude_items(self, collab_config):
        recommender = CollaborativeFilteringRecommender(
            config=collab_config,
            model_type="ncf",
            device="cpu"
        )
        
        exclude = [1, 2, 3, 4, 5]
        recs = recommender.recommend(
            user_id=5,
            n_recommendations=5,
            exclude_items=exclude
        )
        
        rec_ids = [r[0] for r in recs]
        assert not any(item in rec_ids for item in exclude)


# Sequential Model Tests
class TestLSTMSequentialModel:
    def test_forward(self, lstm_config, sample_sequences):
        model = LSTMSequentialModel(lstm_config)
        output = model(
            sample_sequences["sequences"],
            sample_sequences["lengths"],
            sample_sequences["targets"]
        )
        
        assert "logits" in output
        assert "predictions" in output
        assert "loss" in output
        assert output["logits"].shape == (8, 50)
    
    def test_predict_next(self, lstm_config):
        model = LSTMSequentialModel(lstm_config)
        model.eval()
        
        sequence = [1, 5, 10, 15, 20]
        predictions = model.predict_next(sequence, top_k=5)
        
        assert len(predictions) == 5
        assert all(p[1] >= 0 for p in predictions)  # Valid probabilities


class TestGRU4Rec:
    def test_forward(self, lstm_config, sample_sequences):
        model = GRU4Rec(lstm_config)
        logits = model(sample_sequences["sequences"], sample_sequences["lengths"])
        
        assert logits.shape == (8, 50)


class TestSASRec:
    def test_forward(self, lstm_config, sample_sequences):
        model = SASRec(lstm_config)
        logits = model(sample_sequences["sequences"], sample_sequences["lengths"])
        
        assert logits.shape == (8, 50)


class TestSequentialRecommender:
    @pytest.mark.parametrize("model_type", ["lstm", "gru4rec", "sasrec"])
    def test_predict(self, lstm_config, model_type, sample_sequences):
        recommender = SequentialRecommender(
            config=lstm_config,
            model_type=model_type,
            device="cpu"
        )
        
        results = recommender.predict(
            sample_sequences["sequences"],
            sample_sequences["lengths"],
            top_k=5
        )
        
        assert len(results) == 8
        assert all(len(r) == 5 for r in results)


# Contrastive Learning Tests
class TestContrastiveEncoder:
    def test_forward(self, contrastive_config):
        encoder = ContrastiveEncoder(contrastive_config, input_dim=64)
        x = torch.randn(8, 64)
        
        output = encoder(x, return_projection=True)
        
        assert output.shape == (8, 32)
        # Check normalization
        norms = output.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestInfoNCELoss:
    def test_with_explicit_negatives(self):
        loss_fn = InfoNCELoss(temperature=0.1)
        
        anchor = torch.randn(8, 32)
        positive = torch.randn(8, 32)
        negatives = torch.randn(8, 5, 32)
        
        # Normalize
        anchor = torch.nn.functional.normalize(anchor, dim=1)
        positive = torch.nn.functional.normalize(positive, dim=1)
        negatives = torch.nn.functional.normalize(negatives, dim=2)
        
        loss = loss_fn(anchor, positive, negatives)
        
        assert loss.dim() == 0
        assert loss >= 0
    
    def test_in_batch_negatives(self):
        loss_fn = InfoNCELoss(temperature=0.1)
        
        anchor = torch.randn(8, 32)
        positive = torch.randn(8, 32)
        
        anchor = torch.nn.functional.normalize(anchor, dim=1)
        positive = torch.nn.functional.normalize(positive, dim=1)
        
        loss = loss_fn(anchor, positive)
        
        assert loss.dim() == 0
        assert loss >= 0


class TestContrastiveLearningRecommender:
    def test_forward(self, contrastive_config):
        model = ContrastiveLearningRecommender(
            config=contrastive_config,
            num_users=100,
            num_items=50
        )
        
        user_ids = torch.randint(0, 100, (8,))
        pos_embeddings = torch.randn(8, 300)  # word2vec dim
        
        output = model(user_ids, pos_embeddings)
        
        assert "loss" in output
        assert "user_embeddings" in output
        assert "item_embeddings" in output


# Integration Tests
class TestModelIntegration:
    def test_training_step(self, collab_config, sample_batch):
        recommender = CollaborativeFilteringRecommender(
            config=collab_config,
            model_type="implicit",
            device="cpu"
        )
        
        optimizer = torch.optim.Adam(recommender.model.parameters(), lr=0.001)
        
        loss = recommender.train_step(sample_batch, optimizer)
        
        assert isinstance(loss, float)
        assert loss >= 0
    
    def test_save_load(self, collab_config, tmp_path):
        recommender = CollaborativeFilteringRecommender(
            config=collab_config,
            model_type="ncf",
            device="cpu"
        )
        
        # Get initial recommendations
        initial_recs = recommender.recommend(user_id=5, n_recommendations=5)
        
        # Save
        save_path = tmp_path / "model.pt"
        recommender.save(str(save_path))
        
        # Create new model and load
        new_recommender = CollaborativeFilteringRecommender(
            config=collab_config,
            model_type="ncf",
            device="cpu"
        )
        new_recommender.load(str(save_path))
        
        # Verify same recommendations
        loaded_recs = new_recommender.recommend(user_id=5, n_recommendations=5)
        
        assert initial_recs == loaded_recs


# Performance Tests
class TestPerformance:
    @pytest.mark.slow
    def test_large_batch_collaborative(self):
        config = CollaborativeConfig(
            num_users=50000,
            num_items=10000,
            embedding_dim=128
        )
        model = NeuralCollaborativeFiltering(config)
        
        batch_size = 1024
        user_ids = torch.randint(1, 50000, (batch_size,))
        item_ids = torch.randint(1, 10000, (batch_size,))
        
        import time
        start = time.time()
        
        with torch.no_grad():
            output = model(user_ids, item_ids)
        
        elapsed = time.time() - start
        
        assert elapsed < 0.5  # Should be fast
        assert output.shape == (batch_size,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
