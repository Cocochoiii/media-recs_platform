"""
Hybrid Recommender System

This module implements a hybrid recommendation system that combines multiple
approaches (collaborative filtering, content-based, sequential, contrastive)
to provide robust personalized recommendations and solve the cold start problem.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import logging

from .collaborative_filter import CollaborativeFilteringRecommender, CollaborativeConfig
from .bert_embedder import BertContentEmbedder, ContentBasedRecommender, BertEmbedderConfig
from .lstm_sequential import SequentialRecommender, LSTMConfig
from .contrastive_learner import ContrastiveLearningRecommender, ContrastiveConfig

logger = logging.getLogger(__name__)


class RecommenderType(Enum):
    """Types of recommender models."""
    COLLABORATIVE = "collaborative"
    CONTENT_BASED = "content_based"
    SEQUENTIAL = "sequential"
    CONTRASTIVE = "contrastive"


@dataclass
class HybridConfig:
    """Configuration for Hybrid Recommender."""
    # Model weights for ensemble
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        "collaborative": 0.30,
        "content_based": 0.25,
        "sequential": 0.25,
        "contrastive": 0.20
    })
    
    # General settings
    num_recommendations: int = 10
    diversity_factor: float = 0.3  # Balance relevance vs diversity
    cold_start_threshold: int = 5  # Min interactions before using collaborative
    
    # Model configurations
    collaborative_config: Optional[CollaborativeConfig] = None
    bert_config: Optional[BertEmbedderConfig] = None
    lstm_config: Optional[LSTMConfig] = None
    contrastive_config: Optional[ContrastiveConfig] = None
    
    # User/item counts
    num_users: int = 50000
    num_items: int = 10000


class UserProfile:
    """
    User profile for recommendation context.
    
    Stores user information needed for personalized recommendations.
    """
    
    def __init__(
        self,
        user_id: int,
        interaction_history: List[int] = None,
        content_preferences: List[str] = None,
        demographics: Dict[str, Any] = None
    ):
        self.user_id = user_id
        self.interaction_history = interaction_history or []
        self.content_preferences = content_preferences or []
        self.demographics = demographics or {}
    
    @property
    def num_interactions(self) -> int:
        return len(self.interaction_history)
    
    @property
    def is_cold_start(self) -> bool:
        return self.num_interactions < 5
    
    def get_recent_history(self, n: int = 50) -> List[int]:
        return self.interaction_history[-n:]


class ColdStartHandler:
    """
    Handler for cold start scenarios (new users/items).
    
    Uses content-based and demographic approaches when
    collaborative signals are unavailable.
    """
    
    def __init__(
        self,
        content_embedder: BertContentEmbedder,
        default_recommendations: List[int] = None
    ):
        self.content_embedder = content_embedder
        self.default_recommendations = default_recommendations or []
        
        # Popular items cache
        self.popular_items: List[Tuple[int, float]] = []
        
        # Demographic-based recommendations
        self.demographic_preferences: Dict[str, List[int]] = {}
    
    def handle_new_user(
        self,
        user_profile: UserProfile,
        item_catalog: Dict[int, Dict[str, str]],
        n_recommendations: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a new user.
        
        Uses content preferences, demographics, and popularity.
        """
        recommendations = []
        
        # If user has content preferences, use content-based approach
        if user_profile.content_preferences:
            pref_embeddings = self.content_embedder.encode(
                user_profile.content_preferences
            )
            user_profile_vec = pref_embeddings.mean(axis=0)
            
            # Score all items
            item_scores = []
            for item_id, item_data in item_catalog.items():
                item_text = f"{item_data.get('title', '')} {item_data.get('description', '')}"
                item_emb = self.content_embedder.encode(item_text)[0]
                
                similarity = np.dot(user_profile_vec, item_emb) / (
                    np.linalg.norm(user_profile_vec) * np.linalg.norm(item_emb) + 1e-8
                )
                item_scores.append((item_id, float(similarity)))
            
            item_scores.sort(key=lambda x: x[1], reverse=True)
            recommendations = item_scores[:n_recommendations]
        
        # If user has demographic info, use demographic-based recommendations
        elif user_profile.demographics:
            demo_key = self._get_demographic_key(user_profile.demographics)
            if demo_key in self.demographic_preferences:
                demo_items = self.demographic_preferences[demo_key]
                recommendations = [(item, 1.0) for item in demo_items[:n_recommendations]]
        
        # Fall back to popular items
        if not recommendations:
            recommendations = self.popular_items[:n_recommendations]
        
        return recommendations
    
    def handle_new_item(
        self,
        item_id: int,
        item_content: Dict[str, str],
        existing_item_embeddings: np.ndarray,
        existing_item_ids: List[int]
    ) -> List[Tuple[int, float]]:
        """
        Find similar items for a new item (item cold start).
        """
        # Encode new item
        item_text = f"{item_content.get('title', '')} {item_content.get('description', '')}"
        item_embedding = self.content_embedder.encode(item_text)[0]
        
        # Find similar items
        similarities = np.dot(existing_item_embeddings, item_embedding) / (
            np.linalg.norm(existing_item_embeddings, axis=1) * 
            np.linalg.norm(item_embedding) + 1e-8
        )
        
        top_indices = np.argsort(similarities)[::-1][:10]
        similar_items = [
            (existing_item_ids[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return similar_items
    
    def _get_demographic_key(self, demographics: Dict[str, Any]) -> str:
        """Create key from demographic attributes."""
        return f"{demographics.get('age_group', 'unknown')}_{demographics.get('gender', 'unknown')}"
    
    def update_popular_items(self, item_popularity: List[Tuple[int, float]]):
        """Update popular items cache."""
        self.popular_items = sorted(item_popularity, key=lambda x: x[1], reverse=True)


class DiversityReranker:
    """
    Re-rank recommendations to improve diversity.
    
    Uses Maximal Marginal Relevance (MMR) to balance
    relevance with diversity.
    """
    
    def __init__(self, lambda_param: float = 0.5):
        self.lambda_param = lambda_param
    
    def rerank(
        self,
        candidates: List[Tuple[int, float]],
        item_embeddings: Dict[int, np.ndarray],
        n_recommendations: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Apply MMR re-ranking for diversity.
        
        Args:
            candidates: List of (item_id, relevance_score) tuples
            item_embeddings: Dictionary mapping item_id to embedding
            n_recommendations: Number of items to return
            
        Returns:
            Re-ranked list of (item_id, score) tuples
        """
        if len(candidates) <= n_recommendations:
            return candidates
        
        selected = []
        remaining = list(candidates)
        
        while len(selected) < n_recommendations and remaining:
            best_score = float('-inf')
            best_idx = 0
            
            for i, (item_id, relevance) in enumerate(remaining):
                if item_id not in item_embeddings:
                    continue
                
                item_emb = item_embeddings[item_id]
                
                # Calculate max similarity to already selected items
                if selected:
                    similarities = []
                    for sel_item_id, _ in selected:
                        if sel_item_id in item_embeddings:
                            sel_emb = item_embeddings[sel_item_id]
                            sim = np.dot(item_emb, sel_emb) / (
                                np.linalg.norm(item_emb) * np.linalg.norm(sel_emb) + 1e-8
                            )
                            similarities.append(sim)
                    max_sim = max(similarities) if similarities else 0
                else:
                    max_sim = 0
                
                # MMR score
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            selected.append(remaining[best_idx])
            remaining.pop(best_idx)
        
        return selected


class HybridRecommender:
    """
    Hybrid Recommender combining multiple recommendation approaches.
    
    Dynamically adjusts weights based on user context and handles
    cold start scenarios.
    """
    
    def __init__(
        self,
        config: HybridConfig,
        device: str = "cuda"
    ):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Initialize sub-models
        self._init_models()
        
        # Cold start handler
        self.cold_start_handler = ColdStartHandler(
            content_embedder=self.content_embedder
        )
        
        # Diversity re-ranker
        self.diversity_reranker = DiversityReranker(
            lambda_param=1 - config.diversity_factor
        )
        
        # Item embeddings cache
        self.item_embeddings_cache: Dict[int, np.ndarray] = {}
        self.item_catalog: Dict[int, Dict[str, str]] = {}
    
    def _init_models(self):
        """Initialize all sub-models."""
        config = self.config
        
        # Collaborative filtering
        collab_config = config.collaborative_config or CollaborativeConfig(
            num_users=config.num_users,
            num_items=config.num_items,
            embedding_dim=128
        )
        self.collaborative = CollaborativeFilteringRecommender(
            config=collab_config,
            model_type="ncf",
            device=str(self.device)
        )
        
        # Content-based (BERT)
        bert_config = config.bert_config or BertEmbedderConfig(
            model_name="bert-base-uncased",
            projection_dim=256
        )
        self.content_embedder = BertContentEmbedder(bert_config)
        
        # Sequential (LSTM)
        lstm_config = config.lstm_config or LSTMConfig(
            num_items=config.num_items,
            embedding_dim=128,
            hidden_size=256
        )
        self.sequential = SequentialRecommender(
            config=lstm_config,
            model_type="lstm",
            device=str(self.device)
        )
        
        # Contrastive
        contrastive_config = config.contrastive_config or ContrastiveConfig(
            embedding_dim=256,
            projection_dim=128
        )
        self.contrastive = ContrastiveLearningRecommender(
            config=contrastive_config,
            num_users=config.num_users,
            num_items=config.num_items
        ).to(self.device)
    
    def _adjust_weights(self, user_profile: UserProfile) -> Dict[str, float]:
        """
        Dynamically adjust model weights based on user context.
        
        Cold start users get more weight on content-based,
        active users get more weight on collaborative.
        """
        weights = self.config.ensemble_weights.copy()
        
        if user_profile.is_cold_start:
            # Boost content-based and contrastive for cold start
            weights["collaborative"] = 0.05
            weights["content_based"] = 0.45
            weights["sequential"] = 0.10
            weights["contrastive"] = 0.40
        
        elif user_profile.num_interactions < 20:
            # Moderate adjustment for sparse users
            weights["collaborative"] = 0.20
            weights["content_based"] = 0.30
            weights["sequential"] = 0.20
            weights["contrastive"] = 0.30
        
        # Normalize weights
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def _get_collaborative_scores(
        self,
        user_id: int,
        exclude_items: List[int]
    ) -> Dict[int, float]:
        """Get collaborative filtering scores."""
        try:
            recs = self.collaborative.recommend(
                user_id=user_id,
                n_recommendations=100,  # Get more candidates
                exclude_items=exclude_items
            )
            return {item_id: score for item_id, score in recs}
        except Exception as e:
            logger.warning(f"Collaborative scoring failed: {e}")
            return {}
    
    def _get_content_scores(
        self,
        user_profile: UserProfile,
        candidate_items: List[int]
    ) -> Dict[int, float]:
        """Get content-based scores."""
        if not user_profile.content_preferences:
            return {}
        
        try:
            # Get user preference embedding
            pref_embeddings = self.content_embedder.encode(user_profile.content_preferences)
            user_vec = pref_embeddings.mean(axis=0)
            
            scores = {}
            for item_id in candidate_items:
                if item_id in self.item_embeddings_cache:
                    item_vec = self.item_embeddings_cache[item_id]
                    sim = np.dot(user_vec, item_vec) / (
                        np.linalg.norm(user_vec) * np.linalg.norm(item_vec) + 1e-8
                    )
                    scores[item_id] = float(sim)
            
            return scores
        except Exception as e:
            logger.warning(f"Content scoring failed: {e}")
            return {}
    
    def _get_sequential_scores(
        self,
        user_profile: UserProfile
    ) -> Dict[int, float]:
        """Get sequential recommendation scores."""
        if not user_profile.interaction_history:
            return {}
        
        try:
            history = user_profile.get_recent_history(50)
            
            if hasattr(self.sequential.model, 'predict_next'):
                recs = self.sequential.model.predict_next(history, top_k=100)
                return {item_id: score for item_id, score in recs}
            
            # Fallback: use the recommender's predict method
            seq_tensor = torch.tensor([history], device=self.device)
            length_tensor = torch.tensor([len(history)], device=self.device)
            
            results = self.sequential.predict(seq_tensor, length_tensor, top_k=100)
            if results:
                return {item_id: score for item_id, score in results[0]}
            return {}
        except Exception as e:
            logger.warning(f"Sequential scoring failed: {e}")
            return {}
    
    def _get_contrastive_scores(
        self,
        user_id: int,
        candidate_embeddings: torch.Tensor
    ) -> Dict[int, float]:
        """Get contrastive learning scores."""
        try:
            recs = self.contrastive.recommend(
                user_id=user_id,
                candidate_item_embeddings=candidate_embeddings,
                n_recommendations=100
            )
            return {item_id: score for item_id, score in recs}
        except Exception as e:
            logger.warning(f"Contrastive scoring failed: {e}")
            return {}
    
    def recommend(
        self,
        user_profile: UserProfile,
        n_recommendations: int = None,
        exclude_items: List[int] = None,
        apply_diversity: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Generate hybrid recommendations for a user.
        
        Args:
            user_profile: User profile with history and preferences
            n_recommendations: Number of recommendations (default from config)
            exclude_items: Items to exclude
            apply_diversity: Whether to apply diversity re-ranking
            
        Returns:
            List of (item_id, score) tuples
        """
        n_recommendations = n_recommendations or self.config.num_recommendations
        exclude_items = exclude_items or user_profile.interaction_history
        
        # Handle cold start
        if user_profile.is_cold_start:
            return self.cold_start_handler.handle_new_user(
                user_profile=user_profile,
                item_catalog=self.item_catalog,
                n_recommendations=n_recommendations
            )
        
        # Get dynamic weights
        weights = self._adjust_weights(user_profile)
        logger.debug(f"Using weights: {weights}")
        
        # Get all candidate items
        all_items = set(self.item_catalog.keys()) - set(exclude_items)
        candidate_items = list(all_items)
        
        # Collect scores from each model
        all_scores: Dict[int, Dict[str, float]] = {
            item_id: {} for item_id in candidate_items
        }
        
        # Collaborative filtering scores
        collab_scores = self._get_collaborative_scores(
            user_profile.user_id, exclude_items
        )
        for item_id, score in collab_scores.items():
            if item_id in all_scores:
                all_scores[item_id]["collaborative"] = score
        
        # Content-based scores
        content_scores = self._get_content_scores(user_profile, candidate_items)
        for item_id, score in content_scores.items():
            if item_id in all_scores:
                all_scores[item_id]["content_based"] = score
        
        # Sequential scores
        seq_scores = self._get_sequential_scores(user_profile)
        for item_id, score in seq_scores.items():
            if item_id in all_scores:
                all_scores[item_id]["sequential"] = score
        
        # Contrastive scores (if embeddings available)
        if self.item_embeddings_cache:
            emb_list = [self.item_embeddings_cache.get(i, np.zeros(256)) for i in candidate_items]
            candidate_emb_tensor = torch.tensor(np.array(emb_list), dtype=torch.float32)
            contrastive_scores = self._get_contrastive_scores(
                user_profile.user_id, candidate_emb_tensor
            )
            for idx, item_id in enumerate(candidate_items):
                if idx in contrastive_scores:
                    all_scores[item_id]["contrastive"] = contrastive_scores[idx]
        
        # Compute final weighted scores
        final_scores = []
        for item_id, model_scores in all_scores.items():
            if not model_scores:
                continue
            
            weighted_score = 0.0
            total_weight = 0.0
            
            for model_name, score in model_scores.items():
                if model_name in weights:
                    weighted_score += weights[model_name] * score
                    total_weight += weights[model_name]
            
            if total_weight > 0:
                final_score = weighted_score / total_weight
                final_scores.append((item_id, final_score))
        
        # Sort by score
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Apply diversity re-ranking if requested
        if apply_diversity and self.item_embeddings_cache:
            final_scores = self.diversity_reranker.rerank(
                candidates=final_scores[:n_recommendations * 3],
                item_embeddings=self.item_embeddings_cache,
                n_recommendations=n_recommendations
            )
        
        return final_scores[:n_recommendations]
    
    def update_item_catalog(
        self,
        items: Dict[int, Dict[str, str]],
        compute_embeddings: bool = True
    ):
        """
        Update the item catalog and optionally compute embeddings.
        
        Args:
            items: Dictionary mapping item_id to item data
            compute_embeddings: Whether to compute content embeddings
        """
        self.item_catalog = items
        
        if compute_embeddings:
            logger.info(f"Computing embeddings for {len(items)} items...")
            
            item_ids = list(items.keys())
            item_texts = [
                f"{items[i].get('title', '')} {items[i].get('description', '')}"
                for i in item_ids
            ]
            
            embeddings = self.content_embedder.encode(item_texts)
            
            for item_id, emb in zip(item_ids, embeddings):
                self.item_embeddings_cache[item_id] = emb
            
            logger.info("Item embeddings computed and cached.")
    
    def save(self, path: str):
        """Save all model components."""
        import os
        os.makedirs(path, exist_ok=True)
        
        self.collaborative.save(os.path.join(path, "collaborative.pt"))
        self.sequential.save(os.path.join(path, "sequential.pt"))
        torch.save(self.contrastive.state_dict(), os.path.join(path, "contrastive.pt"))
        
        # Save item embeddings
        np.savez(
            os.path.join(path, "item_embeddings.npz"),
            **{str(k): v for k, v in self.item_embeddings_cache.items()}
        )
    
    def load(self, path: str):
        """Load all model components."""
        import os
        
        self.collaborative.load(os.path.join(path, "collaborative.pt"))
        self.sequential.load(os.path.join(path, "sequential.pt"))
        self.contrastive.load_state_dict(
            torch.load(os.path.join(path, "contrastive.pt"), map_location=self.device)
        )
        
        # Load item embeddings
        data = np.load(os.path.join(path, "item_embeddings.npz"))
        self.item_embeddings_cache = {int(k): v for k, v in data.items()}


if __name__ == "__main__":
    # Example usage
    config = HybridConfig(
        num_users=50000,
        num_items=10000,
        ensemble_weights={
            "collaborative": 0.30,
            "content_based": 0.25,
            "sequential": 0.25,
            "contrastive": 0.20
        },
        diversity_factor=0.3
    )
    
    recommender = HybridRecommender(config=config, device="cuda")
    
    # Create a test user profile
    user = UserProfile(
        user_id=123,
        interaction_history=[1, 5, 10, 25, 100, 150],
        content_preferences=["sci-fi movies", "action thrillers"],
        demographics={"age_group": "25-34", "gender": "male"}
    )
    
    print(f"User is cold start: {user.is_cold_start}")
    print("Hybrid Recommender initialized successfully!")
