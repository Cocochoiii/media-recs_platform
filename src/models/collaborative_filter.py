"""
Collaborative Filtering Model using Matrix Factorization with Neural Networks

This module implements a neural collaborative filtering approach that combines
matrix factorization with deep learning for personalized recommendations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class CollaborativeConfig:
    """Configuration for Collaborative Filtering model."""
    num_users: int
    num_items: int
    embedding_dim: int = 128
    hidden_layers: List[int] = None
    dropout: float = 0.2
    use_bias: bool = True
    regularization: float = 0.01
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [256, 128, 64]


class MatrixFactorization(nn.Module):
    """
    Classic Matrix Factorization with embeddings.
    
    Predicts user-item affinity as dot product of latent factors.
    """
    
    def __init__(self, config: CollaborativeConfig):
        super().__init__()
        self.config = config
        
        # User and Item embeddings
        self.user_embedding = nn.Embedding(
            config.num_users, 
            config.embedding_dim,
            padding_idx=0
        )
        self.item_embedding = nn.Embedding(
            config.num_items, 
            config.embedding_dim,
            padding_idx=0
        )
        
        # Bias terms
        if config.use_bias:
            self.user_bias = nn.Embedding(config.num_users, 1)
            self.item_bias = nn.Embedding(config.num_items, 1)
            self.global_bias = nn.Parameter(torch.zeros(1))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings with Xavier uniform."""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        if self.config.use_bias:
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)
    
    def forward(
        self, 
        user_ids: torch.Tensor, 
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass computing predicted ratings.
        
        Args:
            user_ids: Tensor of user IDs [batch_size]
            item_ids: Tensor of item IDs [batch_size]
            
        Returns:
            Predicted ratings [batch_size]
        """
        user_emb = self.user_embedding(user_ids)  # [batch, embed_dim]
        item_emb = self.item_embedding(item_ids)  # [batch, embed_dim]
        
        # Dot product
        prediction = (user_emb * item_emb).sum(dim=1)  # [batch]
        
        if self.config.use_bias:
            prediction += self.user_bias(user_ids).squeeze()
            prediction += self.item_bias(item_ids).squeeze()
            prediction += self.global_bias
        
        return prediction
    
    def get_user_embedding(self, user_id: int) -> np.ndarray:
        """Get embedding for a specific user."""
        with torch.no_grad():
            return self.user_embedding(torch.tensor([user_id])).numpy()[0]
    
    def get_item_embedding(self, item_id: int) -> np.ndarray:
        """Get embedding for a specific item."""
        with torch.no_grad():
            return self.item_embedding(torch.tensor([item_id])).numpy()[0]


class NeuralCollaborativeFiltering(nn.Module):
    """
    Neural Collaborative Filtering (NCF) combining GMF and MLP.
    
    Combines Generalized Matrix Factorization with a Multi-Layer Perceptron
    for capturing both linear and non-linear user-item interactions.
    """
    
    def __init__(self, config: CollaborativeConfig):
        super().__init__()
        self.config = config
        
        # GMF embeddings
        self.gmf_user_embedding = nn.Embedding(
            config.num_users, config.embedding_dim
        )
        self.gmf_item_embedding = nn.Embedding(
            config.num_items, config.embedding_dim
        )
        
        # MLP embeddings (separate from GMF)
        self.mlp_user_embedding = nn.Embedding(
            config.num_users, config.embedding_dim
        )
        self.mlp_item_embedding = nn.Embedding(
            config.num_items, config.embedding_dim
        )
        
        # MLP layers
        mlp_input_dim = config.embedding_dim * 2
        layers = []
        
        for hidden_dim in config.hidden_layers:
            layers.extend([
                nn.Linear(mlp_input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(config.dropout)
            ])
            mlp_input_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Final prediction layer (GMF dim + last MLP hidden dim)
        final_dim = config.embedding_dim + config.hidden_layers[-1]
        self.prediction = nn.Linear(final_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize all weights."""
        for embedding in [
            self.gmf_user_embedding, self.gmf_item_embedding,
            self.mlp_user_embedding, self.mlp_item_embedding
        ]:
            nn.init.xavier_uniform_(embedding.weight)
        
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        nn.init.xavier_uniform_(self.prediction.weight)
        nn.init.zeros_(self.prediction.bias)
    
    def forward(
        self, 
        user_ids: torch.Tensor, 
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through NCF.
        
        Args:
            user_ids: User ID tensor [batch_size]
            item_ids: Item ID tensor [batch_size]
            
        Returns:
            Predicted scores [batch_size]
        """
        # GMF path
        gmf_user = self.gmf_user_embedding(user_ids)
        gmf_item = self.gmf_item_embedding(item_ids)
        gmf_output = gmf_user * gmf_item  # Element-wise product
        
        # MLP path
        mlp_user = self.mlp_user_embedding(user_ids)
        mlp_item = self.mlp_item_embedding(item_ids)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=1)
        mlp_output = self.mlp(mlp_input)
        
        # Concatenate and predict
        combined = torch.cat([gmf_output, mlp_output], dim=1)
        prediction = self.prediction(combined).squeeze()
        
        return torch.sigmoid(prediction)
    
    def get_user_representation(self, user_id: int) -> np.ndarray:
        """Get combined user representation."""
        with torch.no_grad():
            user_tensor = torch.tensor([user_id])
            gmf_emb = self.gmf_user_embedding(user_tensor)
            mlp_emb = self.mlp_user_embedding(user_tensor)
            return torch.cat([gmf_emb, mlp_emb], dim=1).numpy()[0]


class ImplicitFeedbackNCF(NeuralCollaborativeFiltering):
    """
    NCF variant optimized for implicit feedback (clicks, views, etc.).
    
    Uses Bayesian Personalized Ranking (BPR) loss for training.
    """
    
    def __init__(self, config: CollaborativeConfig):
        super().__init__(config)
        self.margin = 0.5
    
    def forward_triplet(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for triplet (user, positive item, negative item).
        
        Returns:
            Tuple of (positive scores, negative scores, BPR loss)
        """
        pos_scores = self.forward(user_ids, pos_item_ids)
        neg_scores = self.forward(user_ids, neg_item_ids)
        
        # BPR Loss: -log(sigmoid(pos - neg))
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        return pos_scores, neg_scores, bpr_loss
    
    def compute_bpr_loss(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute BPR loss for a batch."""
        _, _, loss = self.forward_triplet(user_ids, pos_item_ids, neg_item_ids)
        
        # Add L2 regularization
        l2_reg = self.config.regularization * (
            self.gmf_user_embedding.weight.norm(2) +
            self.gmf_item_embedding.weight.norm(2) +
            self.mlp_user_embedding.weight.norm(2) +
            self.mlp_item_embedding.weight.norm(2)
        )
        
        return loss + l2_reg


class CollaborativeFilteringRecommender:
    """
    High-level interface for Collaborative Filtering recommendations.
    
    Provides methods for training, inference, and recommendation generation.
    """
    
    def __init__(
        self,
        config: CollaborativeConfig,
        model_type: str = "ncf",  # "mf", "ncf", "implicit"
        device: str = "cuda"
    ):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        if model_type == "mf":
            self.model = MatrixFactorization(config)
        elif model_type == "ncf":
            self.model = NeuralCollaborativeFiltering(config)
        elif model_type == "implicit":
            self.model = ImplicitFeedbackNCF(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = self.model.to(self.device)
        self.model_type = model_type
        
        # Cache for fast inference
        self._item_embeddings_cache: Optional[torch.Tensor] = None
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> float:
        """Execute one training step."""
        self.model.train()
        optimizer.zero_grad()
        
        user_ids = batch["user_ids"].to(self.device)
        item_ids = batch["item_ids"].to(self.device)
        
        if self.model_type == "implicit":
            neg_item_ids = batch["neg_item_ids"].to(self.device)
            # Handle multi-dimensional neg_item_ids (multiple negatives per positive)
            if neg_item_ids.dim() > 1:
                neg_item_ids = neg_item_ids[:, 0]  # Use first negative sample
            loss = self.model.compute_bpr_loss(user_ids, item_ids, neg_item_ids)
        else:
            ratings = batch["ratings"].to(self.device)
            predictions = self.model(user_ids, item_ids)
            loss = F.mse_loss(predictions, ratings)
            
            # Add regularization
            loss += self.config.regularization * (
                self.model.user_embedding.weight.norm(2) +
                self.model.item_embedding.weight.norm(2)
            )
        
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def recommend(
        self,
        user_id: int,
        n_recommendations: int = 10,
        exclude_items: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Generate top-N recommendations for a user.
        
        Args:
            user_id: User ID to generate recommendations for
            n_recommendations: Number of recommendations to return
            exclude_items: Item IDs to exclude (e.g., already interacted)
            
        Returns:
            List of (item_id, score) tuples sorted by score descending
        """
        self.model.eval()
        
        # Get all item IDs
        all_items = torch.arange(1, self.config.num_items).to(self.device)
        user_tensor = torch.full_like(all_items, user_id)
        
        # Compute scores for all items
        scores = self.model(user_tensor, all_items)
        
        # Exclude specified items
        if exclude_items:
            exclude_mask = torch.zeros(self.config.num_items - 1, dtype=torch.bool)
            for item_id in exclude_items:
                if 0 < item_id < self.config.num_items:
                    exclude_mask[item_id - 1] = True
            scores[exclude_mask] = float("-inf")
        
        # Get top-N
        top_scores, top_indices = torch.topk(scores, n_recommendations)
        
        recommendations = [
            (idx.item() + 1, score.item()) 
            for idx, score in zip(top_indices, top_scores)
        ]
        
        return recommendations
    
    @torch.no_grad()
    def compute_user_similarity(
        self,
        user_id: int,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Find most similar users based on embedding similarity."""
        self.model.eval()
        
        if hasattr(self.model, 'gmf_user_embedding'):
            user_emb = self.model.gmf_user_embedding.weight
        else:
            user_emb = self.model.user_embedding.weight
        
        target_emb = user_emb[user_id].unsqueeze(0)
        
        # Cosine similarity
        similarities = F.cosine_similarity(target_emb, user_emb)
        similarities[user_id] = float("-inf")  # Exclude self
        
        top_scores, top_indices = torch.topk(similarities, top_k)
        
        return [
            (idx.item(), score.item())
            for idx, score in zip(top_indices, top_scores)
        ]
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "model_type": self.model_type
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        # weights_only=False for PyTorch 2.6+ compatibility with custom config classes
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()


if __name__ == "__main__":
    # Example usage
    config = CollaborativeConfig(
        num_users=50000,
        num_items=10000,
        embedding_dim=128
    )
    
    recommender = CollaborativeFilteringRecommender(
        config=config,
        model_type="ncf",
        device="cuda"
    )
    
    # Get recommendations for user
    recs = recommender.recommend(user_id=123, n_recommendations=10)
    print(f"Top 10 recommendations: {recs}")
