"""
Model Ensemble and Stacking for Recommendations

Advanced ensemble methods that combine multiple recommendation models:
- Weighted Average Ensemble
- Learning-to-Rank Stacking
- Cascading Models
- Multi-Armed Bandit Model Selection
- Neural Ensemble with Cross-Stitch Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class EnsembleConfig:
    """Configuration for ensemble models."""
    num_models: int = 4
    num_items: int = 10000
    embedding_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.1
    
    # Bandit parameters
    exploration_rate: float = 0.1
    ucb_constant: float = 2.0
    
    # Stacking parameters
    meta_hidden_dims: List[int] = field(default_factory=lambda: [128, 64])


class BaseRecommender(ABC):
    """Abstract base class for recommenders in ensemble."""
    
    @abstractmethod
    def predict(self, user_id: int, item_ids: List[int]) -> np.ndarray:
        """Predict scores for items."""
        pass
    
    @abstractmethod
    def recommend(self, user_id: int, k: int) -> List[Tuple[int, float]]:
        """Get top-k recommendations."""
        pass


class WeightedEnsemble:
    """
    Weighted average ensemble of recommendation models.
    
    Combines predictions from multiple models using learned or fixed weights.
    """
    
    def __init__(
        self,
        models: List[BaseRecommender],
        weights: Optional[List[float]] = None,
        normalize: bool = True
    ):
        self.models = models
        self.normalize = normalize
        
        if weights is None:
            # Equal weights
            self.weights = np.ones(len(models)) / len(models)
        else:
            self.weights = np.array(weights)
            if normalize:
                self.weights = self.weights / self.weights.sum()
    
    def predict(self, user_id: int, item_ids: List[int]) -> np.ndarray:
        """Get weighted ensemble predictions."""
        predictions = []
        
        for model in self.models:
            pred = model.predict(user_id, item_ids)
            predictions.append(pred)
        
        predictions = np.stack(predictions, axis=0)  # [num_models, num_items]
        
        if self.normalize:
            # Normalize each model's predictions to [0, 1]
            predictions = (predictions - predictions.min(axis=1, keepdims=True)) / (
                predictions.max(axis=1, keepdims=True) - predictions.min(axis=1, keepdims=True) + 1e-8
            )
        
        # Weighted average
        ensemble_pred = np.sum(predictions * self.weights[:, np.newaxis], axis=0)
        
        return ensemble_pred
    
    def recommend(
        self, 
        user_id: int, 
        k: int = 10,
        exclude_items: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """Get top-k recommendations from ensemble."""
        # Get all item IDs
        all_items = list(range(self.models[0].num_items))
        
        if exclude_items:
            all_items = [i for i in all_items if i not in exclude_items]
        
        scores = self.predict(user_id, all_items)
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:k]
        
        return [
            (all_items[idx], scores[idx])
            for idx in top_indices
        ]
    
    def update_weights(self, new_weights: List[float]):
        """Update ensemble weights."""
        self.weights = np.array(new_weights)
        if self.normalize:
            self.weights = self.weights / self.weights.sum()


class StackingMetaLearner(nn.Module):
    """
    Neural meta-learner for model stacking.
    
    Learns to combine base model predictions optimally.
    """
    
    def __init__(self, config: EnsembleConfig):
        super().__init__()
        self.config = config
        
        input_dim = config.num_models + config.embedding_dim  # Model scores + user features
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in config.meta_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(config.dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Attention over models
        self.model_attention = nn.Sequential(
            nn.Linear(config.num_models + config.embedding_dim, config.num_models),
            nn.Softmax(dim=-1)
        )
    
    def forward(
        self,
        model_scores: torch.Tensor,
        user_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            model_scores: Predictions from base models [batch, num_models]
            user_features: User feature vectors [batch, embedding_dim]
            
        Returns:
            Final prediction [batch]
        """
        combined = torch.cat([model_scores, user_features], dim=-1)
        
        # Attention weights for interpretability
        attention = self.model_attention(combined)  # [batch, num_models]
        
        # Weighted model scores
        weighted_scores = (model_scores * attention).sum(dim=-1, keepdim=True)
        
        # Neural combination
        output = self.network(combined)
        
        # Combine attention-weighted and neural output
        final = 0.5 * weighted_scores + 0.5 * output
        
        return torch.sigmoid(final.squeeze(-1)), attention


class LearningToRankStacker(nn.Module):
    """
    Learning-to-Rank based stacking.
    
    Uses pairwise learning to combine model rankings.
    """
    
    def __init__(self, config: EnsembleConfig):
        super().__init__()
        self.config = config
        
        # Score transformer per model
        self.score_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
            for _ in range(config.num_models)
        ])
        
        # Combination layer
        self.combine = nn.Sequential(
            nn.Linear(config.num_models, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, model_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            model_scores: [batch, num_models]
            
        Returns:
            Combined ranking scores [batch]
        """
        transformed = []
        for i, transform in enumerate(self.score_transforms):
            transformed.append(transform(model_scores[:, i:i+1]))
        
        combined = torch.cat(transformed, dim=-1)  # [batch, num_models]
        output = self.combine(combined)
        
        return output.squeeze(-1)
    
    def compute_pairwise_loss(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor
    ) -> torch.Tensor:
        """BPR-style pairwise loss."""
        return -F.logsigmoid(pos_scores - neg_scores).mean()


class CascadeEnsemble:
    """
    Cascading ensemble with early stopping.
    
    Uses simpler/faster models first, then more complex models
    only when needed for difficult cases.
    """
    
    def __init__(
        self,
        models: List[BaseRecommender],
        thresholds: List[float],
        confidence_fn: Optional[Callable] = None
    ):
        """
        Args:
            models: List of models in order of complexity (simple to complex)
            thresholds: Confidence thresholds for each model
            confidence_fn: Function to compute prediction confidence
        """
        self.models = models
        self.thresholds = thresholds
        self.confidence_fn = confidence_fn or self._default_confidence
        
        self.cascade_stats = {
            "model_usage": [0] * len(models),
            "total_queries": 0
        }
    
    def _default_confidence(self, predictions: np.ndarray) -> float:
        """Default confidence based on score distribution."""
        sorted_preds = np.sort(predictions)[::-1]
        if len(sorted_preds) < 2:
            return 1.0
        
        # Confidence based on gap between top predictions
        gap = sorted_preds[0] - sorted_preds[1]
        return min(1.0, gap * 2)
    
    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_items: Optional[List[int]] = None
    ) -> Tuple[List[Tuple[int, float]], int]:
        """
        Get recommendations using cascade.
        
        Returns:
            recommendations: List of (item_id, score) tuples
            model_used: Index of model that produced final recommendations
        """
        self.cascade_stats["total_queries"] += 1
        
        all_items = list(range(self.models[0].num_items))
        if exclude_items:
            all_items = [i for i in all_items if i not in exclude_items]
        
        for i, (model, threshold) in enumerate(zip(self.models, self.thresholds)):
            predictions = model.predict(user_id, all_items)
            confidence = self.confidence_fn(predictions)
            
            self.cascade_stats["model_usage"][i] += 1
            
            if confidence >= threshold or i == len(self.models) - 1:
                # Use this model's predictions
                top_indices = np.argsort(predictions)[::-1][:k]
                
                return [
                    (all_items[idx], predictions[idx])
                    for idx in top_indices
                ], i
        
        # Fallback (shouldn't reach here)
        return [], -1
    
    def get_cascade_stats(self) -> Dict:
        """Get statistics on cascade usage."""
        total = self.cascade_stats["total_queries"]
        if total == 0:
            return {"model_usage_ratio": [0] * len(self.models)}
        
        return {
            "model_usage_ratio": [
                count / total for count in self.cascade_stats["model_usage"]
            ],
            "total_queries": total
        }


class MultiArmedBanditSelector:
    """
    Multi-Armed Bandit for dynamic model selection.
    
    Learns which model works best for different user contexts.
    """
    
    def __init__(
        self,
        models: List[BaseRecommender],
        exploration_rate: float = 0.1,
        ucb_constant: float = 2.0,
        strategy: str = "ucb"  # ucb, epsilon_greedy, thompson
    ):
        self.models = models
        self.num_models = len(models)
        self.exploration_rate = exploration_rate
        self.ucb_constant = ucb_constant
        self.strategy = strategy
        
        # Statistics per model
        self.counts = np.zeros(self.num_models)
        self.rewards = np.zeros(self.num_models)
        self.total_pulls = 0
        
        # For Thompson Sampling
        self.alpha = np.ones(self.num_models)  # Successes
        self.beta = np.ones(self.num_models)   # Failures
    
    def select_model(self, user_context: Optional[np.ndarray] = None) -> int:
        """Select which model to use."""
        self.total_pulls += 1
        
        if self.strategy == "epsilon_greedy":
            if np.random.random() < self.exploration_rate:
                return np.random.randint(self.num_models)
            else:
                return np.argmax(self.rewards / (self.counts + 1e-8))
        
        elif self.strategy == "ucb":
            if self.total_pulls <= self.num_models:
                return self.total_pulls - 1
            
            avg_rewards = self.rewards / (self.counts + 1e-8)
            exploration_bonus = self.ucb_constant * np.sqrt(
                np.log(self.total_pulls) / (self.counts + 1e-8)
            )
            ucb_values = avg_rewards + exploration_bonus
            return np.argmax(ucb_values)
        
        elif self.strategy == "thompson":
            samples = np.random.beta(self.alpha, self.beta)
            return np.argmax(samples)
        
        else:
            return np.random.randint(self.num_models)
    
    def update(self, model_idx: int, reward: float):
        """Update model statistics based on reward."""
        self.counts[model_idx] += 1
        self.rewards[model_idx] += reward
        
        # Update Thompson sampling parameters
        if reward > 0:
            self.alpha[model_idx] += reward
        else:
            self.beta[model_idx] += (1 - reward)
    
    def recommend(
        self,
        user_id: int,
        k: int = 10,
        user_context: Optional[np.ndarray] = None
    ) -> Tuple[List[Tuple[int, float]], int]:
        """Get recommendations using bandit-selected model."""
        model_idx = self.select_model(user_context)
        recs = self.models[model_idx].recommend(user_id, k)
        return recs, model_idx
    
    def get_model_stats(self) -> Dict:
        """Get model selection statistics."""
        return {
            "counts": self.counts.tolist(),
            "avg_rewards": (self.rewards / (self.counts + 1e-8)).tolist(),
            "selection_ratio": (self.counts / (self.total_pulls + 1e-8)).tolist()
        }


class CrossStitchEnsemble(nn.Module):
    """
    Cross-Stitch Networks for multi-task ensemble.
    
    Learns soft parameter sharing between models through
    learnable cross-stitch units.
    """
    
    def __init__(self, config: EnsembleConfig):
        super().__init__()
        self.config = config
        self.num_models = config.num_models
        
        # Cross-stitch matrices (one per layer)
        self.cross_stitch = nn.ParameterList([
            nn.Parameter(torch.eye(config.num_models))
            for _ in range(3)  # 3 layers
        ])
        
        # Individual model towers
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.embedding_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.embedding_dim)
            )
            for _ in range(config.num_models)
        ])
        
        # Final output layer
        self.output = nn.Linear(config.embedding_dim * config.num_models, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [batch, embedding_dim]
            
        Returns:
            Ensemble prediction [batch]
        """
        # Initialize hidden states for each model
        hidden_states = [x] * self.num_models
        
        # Process through layers with cross-stitch
        for layer_idx, cross_matrix in enumerate(self.cross_stitch):
            # Stack hidden states
            stacked = torch.stack(hidden_states, dim=1)  # [batch, num_models, dim]
            
            # Apply cross-stitch (mix between models)
            cross_matrix_normalized = F.softmax(cross_matrix, dim=1)
            mixed = torch.einsum('bmd,mn->bnd', stacked, cross_matrix_normalized)
            
            # Apply individual tower layers
            new_hidden = []
            for i, tower in enumerate(self.towers):
                if layer_idx == 0:
                    h = tower[0:2](mixed[:, i])  # First layer
                elif layer_idx == 1:
                    h = tower[2:4](mixed[:, i])  # Second layer
                else:
                    h = tower[4:](mixed[:, i])   # Third layer
                new_hidden.append(h)
            
            hidden_states = new_hidden
        
        # Combine all model outputs
        combined = torch.cat(hidden_states, dim=-1)
        output = self.output(combined)
        
        return torch.sigmoid(output.squeeze(-1))
    
    def get_cross_stitch_weights(self) -> List[np.ndarray]:
        """Get cross-stitch weights for interpretability."""
        return [
            F.softmax(cs, dim=1).detach().cpu().numpy()
            for cs in self.cross_stitch
        ]


class DiversityAwareEnsemble:
    """
    Ensemble that optimizes for both accuracy and diversity.
    
    Uses Maximum Marginal Relevance (MMR) to combine model outputs.
    """
    
    def __init__(
        self,
        models: List[BaseRecommender],
        item_embeddings: np.ndarray,
        lambda_diversity: float = 0.5
    ):
        self.models = models
        self.item_embeddings = item_embeddings  # [num_items, dim]
        self.lambda_diversity = lambda_diversity
    
    def _compute_similarity(self, item1: int, item2: int) -> float:
        """Compute cosine similarity between items."""
        emb1 = self.item_embeddings[item1]
        emb2 = self.item_embeddings[item2]
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
    
    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_items: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """Get diverse recommendations using MMR."""
        # Aggregate scores from all models
        all_items = list(range(len(self.item_embeddings)))
        if exclude_items:
            all_items = [i for i in all_items if i not in exclude_items]
        
        combined_scores = np.zeros(len(all_items))
        for model in self.models:
            scores = model.predict(user_id, all_items)
            # Normalize to [0, 1]
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            combined_scores += scores
        combined_scores /= len(self.models)
        
        # MMR selection
        selected = []
        remaining = list(range(len(all_items)))
        
        while len(selected) < k and remaining:
            if not selected:
                # First item: highest score
                idx = np.argmax(combined_scores[remaining])
                best = remaining[idx]
            else:
                # MMR: balance relevance and diversity
                best = None
                best_mmr = float('-inf')
                
                for idx in remaining:
                    relevance = combined_scores[idx]
                    
                    # Max similarity to already selected
                    max_sim = max(
                        self._compute_similarity(all_items[idx], all_items[s])
                        for s in selected
                    )
                    
                    mmr = self.lambda_diversity * relevance - (1 - self.lambda_diversity) * max_sim
                    
                    if mmr > best_mmr:
                        best_mmr = mmr
                        best = idx
            
            selected.append(best)
            remaining.remove(best)
        
        return [
            (all_items[idx], combined_scores[idx])
            for idx in selected
        ]


if __name__ == "__main__":
    # Example usage
    config = EnsembleConfig(
        num_models=4,
        num_items=1000,
        embedding_dim=64
    )
    
    # Test Cross-Stitch Ensemble
    cross_stitch = CrossStitchEnsemble(config)
    x = torch.randn(32, 64)
    output = cross_stitch(x)
    print(f"Cross-stitch output shape: {output.shape}")
    
    # Get cross-stitch weights
    weights = cross_stitch.get_cross_stitch_weights()
    print(f"Cross-stitch weights shape: {[w.shape for w in weights]}")
    
    # Test Meta Learner
    meta_learner = StackingMetaLearner(config)
    model_scores = torch.randn(32, 4)
    user_features = torch.randn(32, 64)
    pred, attention = meta_learner(model_scores, user_features)
    print(f"Meta-learner output: {pred.shape}, attention: {attention.shape}")
