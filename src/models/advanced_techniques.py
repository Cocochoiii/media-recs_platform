"""
Advanced Recommendation Techniques

1. User Interest Evolution Modeling - Captures how user preferences change over time
2. Fairness-Aware Recommendations - Ensures fair exposure across items/providers
3. Diversity Optimization - Balances relevance with diversity
4. Novelty and Serendipity Enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import math


@dataclass
class AdvancedRecConfig:
    """Configuration for advanced recommendation techniques."""
    num_users: int = 50000
    num_items: int = 10000
    num_providers: int = 1000
    embedding_dim: int = 64
    hidden_dim: int = 128
    
    # Time modeling
    num_time_slots: int = 24  # Hours in a day
    time_embedding_dim: int = 16
    
    # Fairness
    fairness_lambda: float = 0.1
    min_exposure_ratio: float = 0.1
    
    # Diversity
    diversity_lambda: float = 0.3
    
    dropout: float = 0.1


# ============================================
# User Interest Evolution Modeling
# ============================================

class TemporalAttention(nn.Module):
    """Attention mechanism that considers time decay."""
    
    def __init__(self, dim: int, time_dim: int):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim + time_dim, dim)
        self.value = nn.Linear(dim, dim)
        self.time_encoder = nn.Linear(1, time_dim)
        
        self.scale = math.sqrt(dim)
    
    def forward(
        self,
        query: torch.Tensor,
        items: torch.Tensor,
        timestamps: torch.Tensor,
        current_time: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, dim]
            items: [batch, seq_len, dim]
            timestamps: [batch, seq_len]
            current_time: [batch]
        """
        # Time decay features
        time_diff = current_time.unsqueeze(1) - timestamps  # [batch, seq_len]
        time_features = self.time_encoder(time_diff.unsqueeze(-1))  # [batch, seq_len, time_dim]
        
        # Keys include time information
        keys = self.key(torch.cat([items, time_features], dim=-1))
        
        # Standard attention
        q = self.query(query).unsqueeze(1)  # [batch, 1, dim]
        
        attn_scores = torch.bmm(q, keys.transpose(1, 2)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        values = self.value(items)
        output = torch.bmm(attn_weights, values).squeeze(1)
        
        return output, attn_weights.squeeze(1)


class InterestEvolutionModel(nn.Module):
    """
    Models how user interests evolve over time.
    
    Uses:
    - Long-term stable preferences (user embedding)
    - Short-term dynamic interests (recent behavior)
    - Temporal patterns (time-of-day, day-of-week)
    """
    
    def __init__(self, config: AdvancedRecConfig):
        super().__init__()
        self.config = config
        
        # Long-term user embedding
        self.user_embedding = nn.Embedding(config.num_users, config.embedding_dim)
        self.item_embedding = nn.Embedding(config.num_items, config.embedding_dim)
        
        # Temporal embeddings
        self.hour_embedding = nn.Embedding(24, config.time_embedding_dim)
        self.day_embedding = nn.Embedding(7, config.time_embedding_dim)
        
        # Interest evolution with GRU
        self.interest_gru = nn.GRU(
            config.embedding_dim + config.time_embedding_dim * 2,
            config.hidden_dim,
            batch_first=True
        )
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(
            config.embedding_dim,
            config.time_embedding_dim
        )
        
        # Interest combination
        self.interest_combiner = nn.Sequential(
            nn.Linear(config.embedding_dim + config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )
        
        # Prediction
        self.predictor = nn.Linear(config.embedding_dim * 2, 1)
    
    def get_evolved_interest(
        self,
        user_ids: torch.Tensor,
        history_items: torch.Tensor,
        history_timestamps: torch.Tensor,
        history_hours: torch.Tensor,
        history_days: torch.Tensor,
        current_time: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute user's evolved interest representation.
        
        Args:
            history_items: [batch, seq_len]
            history_timestamps: [batch, seq_len] - Unix timestamps
            history_hours: [batch, seq_len] - Hour of day (0-23)
            history_days: [batch, seq_len] - Day of week (0-6)
            current_time: [batch] - Current Unix timestamp
        """
        batch_size = user_ids.size(0)
        
        # Long-term interest
        long_term = self.user_embedding(user_ids)  # [batch, dim]
        
        # Get item embeddings
        item_embs = self.item_embedding(history_items)  # [batch, seq_len, dim]
        
        # Temporal features
        hour_embs = self.hour_embedding(history_hours)
        day_embs = self.day_embedding(history_days)
        
        # Combine item and temporal features
        combined = torch.cat([item_embs, hour_embs, day_embs], dim=-1)
        
        # GRU for sequential modeling
        gru_out, _ = self.interest_gru(combined)
        short_term = gru_out[:, -1, :]  # Last hidden state
        
        # Temporal attention for recent relevant items
        attended, _ = self.temporal_attention(
            long_term, item_embs, history_timestamps, current_time
        )
        
        # Combine long-term and short-term
        combined_interest = self.interest_combiner(
            torch.cat([long_term + attended, short_term], dim=-1)
        )
        
        return combined_interest
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        history_items: torch.Tensor,
        history_timestamps: torch.Tensor,
        history_hours: torch.Tensor,
        history_days: torch.Tensor,
        current_time: torch.Tensor
    ) -> torch.Tensor:
        """Predict scores for user-item pairs."""
        user_interest = self.get_evolved_interest(
            user_ids, history_items, history_timestamps,
            history_hours, history_days, current_time
        )
        
        item_emb = self.item_embedding(item_ids)
        
        combined = torch.cat([user_interest, item_emb], dim=-1)
        score = self.predictor(combined).squeeze(-1)
        
        return torch.sigmoid(score)


# ============================================
# Fairness-Aware Recommendations
# ============================================

class FairnessConstraint:
    """Tracks and enforces fairness constraints."""
    
    def __init__(self, config: AdvancedRecConfig):
        self.config = config
        
        # Exposure tracking per provider
        self.provider_exposure = defaultdict(float)
        self.total_recommendations = 0
        
        # Item to provider mapping
        self.item_provider: Dict[int, int] = {}
    
    def set_item_providers(self, item_provider: Dict[int, int]):
        """Set mapping from items to their providers."""
        self.item_provider = item_provider
    
    def update_exposure(self, recommended_items: List[int]):
        """Update exposure counts."""
        for item in recommended_items:
            provider = self.item_provider.get(item, 0)
            self.provider_exposure[provider] += 1
        self.total_recommendations += len(recommended_items)
    
    def get_fairness_scores(self, candidate_items: List[int]) -> np.ndarray:
        """
        Compute fairness boost scores for candidate items.
        Items from under-exposed providers get higher scores.
        """
        if self.total_recommendations == 0:
            return np.ones(len(candidate_items))
        
        scores = []
        avg_exposure = self.total_recommendations / max(1, len(self.provider_exposure))
        
        for item in candidate_items:
            provider = self.item_provider.get(item, 0)
            provider_exp = self.provider_exposure.get(provider, 0)
            
            # Boost items from under-exposed providers
            if provider_exp < avg_exposure * self.config.min_exposure_ratio:
                boost = 2.0
            elif provider_exp < avg_exposure:
                boost = 1.5
            else:
                boost = 1.0
            
            scores.append(boost)
        
        return np.array(scores)
    
    def get_exposure_stats(self) -> Dict:
        """Get exposure statistics."""
        exposures = list(self.provider_exposure.values())
        return {
            "gini_coefficient": self._gini(exposures),
            "min_exposure": min(exposures) if exposures else 0,
            "max_exposure": max(exposures) if exposures else 0,
            "mean_exposure": np.mean(exposures) if exposures else 0,
            "num_providers": len(self.provider_exposure)
        }
    
    def _gini(self, values: List[float]) -> float:
        """Compute Gini coefficient (0 = perfect equality, 1 = perfect inequality)."""
        if not values:
            return 0
        sorted_values = np.sort(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0


class FairnessAwareRecommender(nn.Module):
    """
    Recommender that balances relevance with fairness.
    
    Incorporates:
    - Provider fairness (equal exposure across content creators)
    - Consumer fairness (similar quality for all users)
    - Item fairness (popular vs long-tail items)
    """
    
    def __init__(self, config: AdvancedRecConfig):
        super().__init__()
        self.config = config
        
        # Base recommender
        self.user_embedding = nn.Embedding(config.num_users, config.embedding_dim)
        self.item_embedding = nn.Embedding(config.num_items, config.embedding_dim)
        
        # Provider embeddings for fairness modeling
        self.provider_embedding = nn.Embedding(config.num_providers, config.embedding_dim // 2)
        
        self.predictor = nn.Sequential(
            nn.Linear(config.embedding_dim * 2 + config.embedding_dim // 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1)
        )
        
        # Fairness constraint tracker
        self.fairness_constraint = FairnessConstraint(config)
        
        # Item to provider mapping
        self.item_provider: Dict[int, int] = {}
    
    def set_item_providers(self, item_provider: Dict[int, int]):
        """Set item to provider mapping."""
        self.item_provider = item_provider
        self.fairness_constraint.set_item_providers(item_provider)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        provider_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute relevance scores."""
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        if provider_ids is None:
            provider_ids = torch.tensor([
                self.item_provider.get(i.item(), 0) for i in item_ids
            ], device=item_ids.device)
        
        provider_emb = self.provider_embedding(provider_ids)
        
        combined = torch.cat([user_emb, item_emb, provider_emb], dim=-1)
        return self.predictor(combined).squeeze(-1)
    
    def compute_fair_loss(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        labels: torch.Tensor,
        provider_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss with fairness regularization.
        """
        # Relevance loss
        predictions = self.forward(user_ids, item_ids, provider_ids)
        relevance_loss = F.binary_cross_entropy_with_logits(predictions, labels)
        
        # Provider fairness loss
        # Encourage similar prediction distributions across providers
        unique_providers = provider_ids.unique()
        provider_means = []
        
        for provider in unique_providers:
            mask = provider_ids == provider
            if mask.sum() > 0:
                provider_means.append(predictions[mask].mean())
        
        if len(provider_means) > 1:
            provider_means = torch.stack(provider_means)
            fairness_loss = provider_means.var()
        else:
            fairness_loss = torch.tensor(0.0, device=predictions.device)
        
        total_loss = relevance_loss + self.config.fairness_lambda * fairness_loss
        
        return total_loss, {
            "relevance_loss": relevance_loss.item(),
            "fairness_loss": fairness_loss.item()
        }
    
    @torch.no_grad()
    def recommend_fair(
        self,
        user_id: int,
        candidate_items: List[int],
        k: int = 10,
        fairness_weight: float = 0.3
    ) -> List[Tuple[int, float]]:
        """
        Generate fair recommendations.
        
        Combines relevance scores with fairness boosts.
        """
        user_tensor = torch.tensor([user_id] * len(candidate_items))
        item_tensor = torch.tensor(candidate_items)
        
        # Get relevance scores
        relevance_scores = torch.sigmoid(self.forward(user_tensor, item_tensor))
        relevance_np = relevance_scores.cpu().numpy()
        
        # Get fairness scores
        fairness_scores = self.fairness_constraint.get_fairness_scores(candidate_items)
        
        # Combine scores
        final_scores = (1 - fairness_weight) * relevance_np + fairness_weight * fairness_scores
        
        # Get top-k
        top_indices = np.argsort(final_scores)[::-1][:k]
        
        results = [
            (candidate_items[idx], final_scores[idx])
            for idx in top_indices
        ]
        
        # Update exposure tracking
        self.fairness_constraint.update_exposure([r[0] for r in results])
        
        return results


# ============================================
# Diversity Optimization
# ============================================

class DiversityOptimizer:
    """
    Optimizes recommendation diversity using various strategies.
    """
    
    def __init__(self, config: AdvancedRecConfig):
        self.config = config
        self.item_embeddings: Optional[np.ndarray] = None
        self.item_categories: Dict[int, int] = {}
    
    def set_item_embeddings(self, embeddings: np.ndarray):
        """Set item embeddings for similarity computation."""
        self.item_embeddings = embeddings
    
    def set_item_categories(self, categories: Dict[int, int]):
        """Set item category mapping."""
        self.item_categories = categories
    
    def compute_similarity(self, item1: int, item2: int) -> float:
        """Compute cosine similarity between items."""
        if self.item_embeddings is None:
            return 0.0
        
        emb1 = self.item_embeddings[item1]
        emb2 = self.item_embeddings[item2]
        
        return float(np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
        ))
    
    def mmr_rerank(
        self,
        candidates: List[Tuple[int, float]],
        k: int,
        lambda_param: float = 0.5
    ) -> List[Tuple[int, float]]:
        """
        Maximal Marginal Relevance re-ranking.
        
        MMR = λ * Relevance - (1-λ) * max_similarity_to_selected
        """
        if not candidates:
            return []
        
        selected = []
        remaining = list(candidates)
        
        # First item: highest relevance
        remaining.sort(key=lambda x: x[1], reverse=True)
        selected.append(remaining.pop(0))
        
        while len(selected) < k and remaining:
            best_mmr = float('-inf')
            best_idx = 0
            
            for idx, (item_id, relevance) in enumerate(remaining):
                # Max similarity to already selected items
                max_sim = max(
                    self.compute_similarity(item_id, sel_item)
                    for sel_item, _ in selected
                )
                
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx
            
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    def category_diversify(
        self,
        candidates: List[Tuple[int, float]],
        k: int,
        max_per_category: int = 2
    ) -> List[Tuple[int, float]]:
        """Ensure category diversity in recommendations."""
        selected = []
        category_counts: Dict[int, int] = defaultdict(int)
        
        # Sort by relevance
        sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        
        for item_id, score in sorted_candidates:
            if len(selected) >= k:
                break
            
            category = self.item_categories.get(item_id, 0)
            
            if category_counts[category] < max_per_category:
                selected.append((item_id, score))
                category_counts[category] += 1
        
        return selected
    
    def compute_diversity_metrics(
        self,
        recommendations: List[int]
    ) -> Dict[str, float]:
        """Compute diversity metrics for a recommendation list."""
        if len(recommendations) < 2:
            return {"avg_distance": 0, "category_coverage": 0}
        
        # Average pairwise distance
        distances = []
        for i, item1 in enumerate(recommendations):
            for item2 in recommendations[i+1:]:
                sim = self.compute_similarity(item1, item2)
                distances.append(1 - sim)
        
        avg_distance = np.mean(distances) if distances else 0
        
        # Category coverage
        categories = set(
            self.item_categories.get(item, 0) 
            for item in recommendations
        )
        num_categories = len(self.item_categories)
        coverage = len(categories) / max(1, num_categories)
        
        return {
            "avg_distance": avg_distance,
            "category_coverage": coverage,
            "num_categories": len(categories)
        }


# ============================================
# Novelty and Serendipity
# ============================================

class NoveltyEnhancer:
    """
    Enhances recommendations with novelty and serendipity.
    """
    
    def __init__(self, config: AdvancedRecConfig):
        self.config = config
        self.item_popularity: Dict[int, float] = {}
        self.user_history: Dict[int, Set[int]] = defaultdict(set)
    
    def set_item_popularity(self, popularity: Dict[int, float]):
        """Set item popularity scores (0-1, higher = more popular)."""
        self.item_popularity = popularity
    
    def update_user_history(self, user_id: int, item_ids: List[int]):
        """Update user's interaction history."""
        self.user_history[user_id].update(item_ids)
    
    def compute_novelty(self, item_id: int) -> float:
        """
        Compute novelty score (inverse of popularity).
        Novel items are less popular.
        """
        popularity = self.item_popularity.get(item_id, 0.5)
        return 1 - popularity
    
    def compute_serendipity(
        self,
        user_id: int,
        item_id: int,
        item_embeddings: np.ndarray
    ) -> float:
        """
        Compute serendipity (unexpected but relevant).
        Items that are dissimilar to user's history but still relevant.
        """
        history = self.user_history.get(user_id, set())
        if not history:
            return 0.5
        
        item_emb = item_embeddings[item_id]
        
        # Average similarity to user's history
        similarities = []
        for hist_item in history:
            if hist_item < len(item_embeddings):
                hist_emb = item_embeddings[hist_item]
                sim = np.dot(item_emb, hist_emb) / (
                    np.linalg.norm(item_emb) * np.linalg.norm(hist_emb) + 1e-8
                )
                similarities.append(sim)
        
        avg_sim = np.mean(similarities) if similarities else 0.5
        
        # Serendipity = dissimilarity to history
        return 1 - avg_sim
    
    def enhance_recommendations(
        self,
        candidates: List[Tuple[int, float]],
        user_id: int,
        item_embeddings: np.ndarray,
        novelty_weight: float = 0.2,
        serendipity_weight: float = 0.1
    ) -> List[Tuple[int, float]]:
        """
        Re-score candidates with novelty and serendipity.
        """
        enhanced = []
        
        for item_id, relevance in candidates:
            novelty = self.compute_novelty(item_id)
            serendipity = self.compute_serendipity(user_id, item_id, item_embeddings)
            
            # Combined score
            score = (
                (1 - novelty_weight - serendipity_weight) * relevance +
                novelty_weight * novelty +
                serendipity_weight * serendipity
            )
            
            enhanced.append((item_id, score))
        
        return sorted(enhanced, key=lambda x: x[1], reverse=True)


class AdvancedRecommenderSystem:
    """
    Complete advanced recommender combining all techniques.
    """
    
    def __init__(self, config: AdvancedRecConfig, device: str = "cuda"):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Models
        self.interest_model = InterestEvolutionModel(config).to(self.device)
        self.fairness_model = FairnessAwareRecommender(config).to(self.device)
        
        # Optimizers
        self.diversity_optimizer = DiversityOptimizer(config)
        self.novelty_enhancer = NoveltyEnhancer(config)
    
    def recommend(
        self,
        user_id: int,
        candidate_items: List[int],
        user_history: Dict,  # Contains history items, timestamps, etc.
        k: int = 10,
        diversity_lambda: float = 0.3,
        fairness_weight: float = 0.2,
        novelty_weight: float = 0.1
    ) -> List[Tuple[int, float, Dict]]:
        """
        Generate recommendations with all enhancements.
        
        Returns:
            List of (item_id, score, metadata) tuples
        """
        # Get base relevance scores from interest evolution model
        # ... (implementation details)
        
        # Apply fairness
        fair_recs = self.fairness_model.recommend_fair(
            user_id, candidate_items, k=k*2, fairness_weight=fairness_weight
        )
        
        # Apply diversity via MMR
        diverse_recs = self.diversity_optimizer.mmr_rerank(
            fair_recs, k=k, lambda_param=1-diversity_lambda
        )
        
        # Enhance with novelty
        item_embeddings = self.fairness_model.item_embedding.weight.detach().cpu().numpy()
        final_recs = self.novelty_enhancer.enhance_recommendations(
            diverse_recs, user_id, item_embeddings,
            novelty_weight=novelty_weight
        )
        
        # Add metadata
        results = []
        diversity_metrics = self.diversity_optimizer.compute_diversity_metrics(
            [r[0] for r in final_recs[:k]]
        )
        
        for item_id, score in final_recs[:k]:
            results.append((item_id, score, {
                "novelty": self.novelty_enhancer.compute_novelty(item_id),
                "diversity_metrics": diversity_metrics
            }))
        
        return results


if __name__ == "__main__":
    config = AdvancedRecConfig(
        num_users=1000,
        num_items=5000,
        num_providers=100
    )
    
    # Test Interest Evolution
    model = InterestEvolutionModel(config)
    
    batch_size = 4
    seq_len = 20
    
    user_ids = torch.randint(0, 1000, (batch_size,))
    item_ids = torch.randint(0, 5000, (batch_size,))
    history_items = torch.randint(0, 5000, (batch_size, seq_len))
    history_ts = torch.randint(0, 1000000, (batch_size, seq_len)).float()
    history_hours = torch.randint(0, 24, (batch_size, seq_len))
    history_days = torch.randint(0, 7, (batch_size, seq_len))
    current_time = torch.randint(1000000, 2000000, (batch_size,)).float()
    
    scores = model(
        user_ids, item_ids, history_items, history_ts,
        history_hours, history_days, current_time
    )
    print(f"Interest Evolution scores: {scores.shape}")
    
    # Test Fairness-Aware Recommender
    fair_model = FairnessAwareRecommender(config)
    fair_model.set_item_providers({i: i % 100 for i in range(5000)})
    
    recs = fair_model.recommend_fair(user_id=0, candidate_items=list(range(100)), k=10)
    print(f"Fair recommendations: {len(recs)}")
    print(f"Fairness stats: {fair_model.fairness_constraint.get_exposure_stats()}")
