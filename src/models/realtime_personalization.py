"""
Real-time Personalization and Online Learning

Implements techniques for real-time recommendation updates:
- Online Learning with incremental updates
- Contextual Bandits for real-time personalization
- Feature caching and fast inference
- A/B Testing framework
- Session-based real-time recommendations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import time
import hashlib
from abc import ABC, abstractmethod
from threading import Lock
import logging

logger = logging.getLogger(__name__)


@dataclass
class RealTimeConfig:
    """Configuration for real-time personalization."""
    # Model
    num_items: int = 10000
    embedding_dim: int = 64
    context_dim: int = 32
    
    # Online learning
    learning_rate: float = 0.01
    batch_size: int = 32
    update_frequency: int = 100  # Update every N interactions
    
    # Context
    max_context_features: int = 50
    session_timeout: int = 1800  # 30 minutes
    
    # Caching
    cache_ttl: int = 300  # 5 minutes
    max_cache_size: int = 10000


class OnlineModel(nn.Module):
    """
    Neural network with online learning capabilities.
    
    Supports incremental updates without full retraining.
    """
    
    def __init__(self, config: RealTimeConfig):
        super().__init__()
        self.config = config
        
        # Item embeddings
        self.item_embedding = nn.Embedding(
            config.num_items, 
            config.embedding_dim
        )
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(config.context_dim, config.embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(config.embedding_dim)
        )
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def forward(
        self,
        item_ids: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            item_ids: [batch]
            context: [batch, context_dim]
            
        Returns:
            Predictions [batch]
        """
        item_emb = self.item_embedding(item_ids)
        context_emb = self.context_encoder(context)
        
        combined = torch.cat([item_emb, context_emb], dim=-1)
        output = self.predictor(combined)
        
        return torch.sigmoid(output.squeeze(-1))
    
    def get_item_scores(
        self,
        context: torch.Tensor,
        candidate_items: Optional[List[int]] = None
    ) -> torch.Tensor:
        """Score all or candidate items."""
        if candidate_items is None:
            item_ids = torch.arange(self.config.num_items, device=context.device)
        else:
            item_ids = torch.tensor(candidate_items, device=context.device)
        
        # Expand context for all items
        context_expanded = context.unsqueeze(0).expand(len(item_ids), -1)
        
        return self.forward(item_ids, context_expanded)


class OnlineLearner:
    """
    Online learning wrapper for incremental model updates.
    """
    
    def __init__(
        self,
        model: OnlineModel,
        config: RealTimeConfig,
        device: str = "cuda"
    ):
        self.model = model
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Optimizer with small learning rate for stability
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Experience buffer
        self.buffer = deque(maxlen=config.batch_size * 10)
        self.update_counter = 0
        
        # Lock for thread safety
        self.lock = Lock()
        
        # Performance tracking
        self.loss_history = deque(maxlen=1000)
    
    def log_interaction(
        self,
        item_id: int,
        context: np.ndarray,
        reward: float
    ):
        """Log an interaction for online learning."""
        with self.lock:
            self.buffer.append({
                "item_id": item_id,
                "context": context,
                "reward": reward
            })
            self.update_counter += 1
            
            if self.update_counter >= self.config.update_frequency:
                self._update()
                self.update_counter = 0
    
    def _update(self):
        """Perform incremental model update."""
        if len(self.buffer) < self.config.batch_size:
            return
        
        # Sample batch
        indices = np.random.choice(
            len(self.buffer), 
            self.config.batch_size, 
            replace=False
        )
        
        batch = [self.buffer[i] for i in indices]
        
        # Prepare tensors
        item_ids = torch.tensor(
            [b["item_id"] for b in batch],
            device=self.device
        )
        contexts = torch.tensor(
            np.stack([b["context"] for b in batch]),
            device=self.device,
            dtype=torch.float32
        )
        rewards = torch.tensor(
            [b["reward"] for b in batch],
            device=self.device,
            dtype=torch.float32
        )
        
        # Forward pass
        self.model.train()
        predictions = self.model(item_ids, contexts)
        
        # Loss
        loss = F.binary_cross_entropy(predictions, rewards)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self.loss_history.append(loss.item())
        
        logger.debug(f"Online update: loss={loss.item():.4f}")
    
    @torch.no_grad()
    def recommend(
        self,
        context: np.ndarray,
        k: int = 10,
        exclude_items: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """Get real-time recommendations."""
        self.model.eval()
        
        context_tensor = torch.tensor(
            context, device=self.device, dtype=torch.float32
        )
        
        scores = self.model.get_item_scores(context_tensor)
        
        if exclude_items:
            scores[exclude_items] = float('-inf')
        
        top_scores, top_indices = torch.topk(scores, k)
        
        return [
            (idx.item(), score.item())
            for idx, score in zip(top_indices, top_scores)
        ]


class ContextualBandit:
    """
    Contextual bandit for personalized exploration-exploitation.
    
    Balances recommending known good items with exploring new ones.
    """
    
    def __init__(
        self,
        num_items: int,
        context_dim: int,
        alpha: float = 0.1  # Exploration parameter
    ):
        self.num_items = num_items
        self.context_dim = context_dim
        self.alpha = alpha
        
        # Linear model parameters for each item (arm)
        self.A = [np.eye(context_dim) for _ in range(num_items)]
        self.b = [np.zeros(context_dim) for _ in range(num_items)]
        
        # Cached theta (coefficients)
        self.theta = [None] * num_items
        self._update_theta()
    
    def _update_theta(self):
        """Update theta for all arms."""
        for i in range(self.num_items):
            try:
                A_inv = np.linalg.inv(self.A[i])
                self.theta[i] = A_inv @ self.b[i]
            except np.linalg.LinAlgError:
                self.theta[i] = np.zeros(self.context_dim)
    
    def select_item(
        self,
        context: np.ndarray,
        candidate_items: Optional[List[int]] = None
    ) -> int:
        """Select item using LinUCB."""
        if candidate_items is None:
            candidate_items = list(range(self.num_items))
        
        best_item = None
        best_ucb = float('-inf')
        
        for item in candidate_items:
            if self.theta[item] is None:
                continue
            
            # Compute UCB
            try:
                A_inv = np.linalg.inv(self.A[item])
                mean = np.dot(self.theta[item], context)
                std = np.sqrt(context @ A_inv @ context)
                ucb = mean + self.alpha * std
            except np.linalg.LinAlgError:
                ucb = 0
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_item = item
        
        return best_item if best_item is not None else np.random.choice(candidate_items)
    
    def update(self, item: int, context: np.ndarray, reward: float):
        """Update model based on observed reward."""
        self.A[item] += np.outer(context, context)
        self.b[item] += reward * context
        
        # Update theta for this item
        try:
            A_inv = np.linalg.inv(self.A[item])
            self.theta[item] = A_inv @ self.b[item]
        except np.linalg.LinAlgError:
            pass
    
    def recommend(
        self,
        context: np.ndarray,
        k: int = 10,
        exclude_items: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """Get top-k recommendations with UCB scores."""
        candidates = list(range(self.num_items))
        if exclude_items:
            candidates = [i for i in candidates if i not in exclude_items]
        
        ucb_scores = []
        for item in candidates:
            if self.theta[item] is None:
                continue
            
            try:
                A_inv = np.linalg.inv(self.A[item])
                mean = np.dot(self.theta[item], context)
                std = np.sqrt(context @ A_inv @ context)
                ucb = mean + self.alpha * std
                ucb_scores.append((item, ucb))
            except np.linalg.LinAlgError:
                ucb_scores.append((item, 0))
        
        # Sort by UCB
        ucb_scores.sort(key=lambda x: x[1], reverse=True)
        
        return ucb_scores[:k]


class SessionManager:
    """
    Manage user sessions for real-time recommendations.
    """
    
    def __init__(self, config: RealTimeConfig):
        self.config = config
        self.sessions: Dict[str, Dict] = {}
        self.lock = Lock()
    
    def get_or_create_session(self, user_id: str) -> Dict:
        """Get existing session or create new one."""
        with self.lock:
            current_time = time.time()
            
            if user_id in self.sessions:
                session = self.sessions[user_id]
                
                # Check timeout
                if current_time - session["last_active"] > self.config.session_timeout:
                    # Session expired, create new
                    session = self._create_session(user_id)
                else:
                    session["last_active"] = current_time
            else:
                session = self._create_session(user_id)
            
            self.sessions[user_id] = session
            return session
    
    def _create_session(self, user_id: str) -> Dict:
        """Create new session."""
        return {
            "user_id": user_id,
            "created_at": time.time(),
            "last_active": time.time(),
            "interactions": [],
            "context": {}
        }
    
    def add_interaction(
        self,
        user_id: str,
        item_id: int,
        interaction_type: str,
        metadata: Optional[Dict] = None
    ):
        """Add interaction to session."""
        session = self.get_or_create_session(user_id)
        
        session["interactions"].append({
            "item_id": item_id,
            "type": interaction_type,
            "timestamp": time.time(),
            "metadata": metadata or {}
        })
        
        # Keep only recent interactions
        max_interactions = 100
        if len(session["interactions"]) > max_interactions:
            session["interactions"] = session["interactions"][-max_interactions:]
    
    def get_session_context(self, user_id: str) -> np.ndarray:
        """Extract context features from session."""
        session = self.get_or_create_session(user_id)
        
        interactions = session["interactions"]
        
        # Build context vector
        context = np.zeros(self.config.context_dim)
        
        if interactions:
            # Recent items (one-hot style or embedding average)
            recent_items = [i["item_id"] for i in interactions[-10:]]
            context[:min(10, len(recent_items))] = 1.0
            
            # Time features
            session_duration = time.time() - session["created_at"]
            context[10] = min(1.0, session_duration / 3600)  # Hours
            
            # Interaction count
            context[11] = min(1.0, len(interactions) / 50)
            
            # Interaction type distribution
            types = [i["type"] for i in interactions]
            for i, t in enumerate(["click", "view", "purchase"]):
                if i + 12 < len(context):
                    context[12 + i] = types.count(t) / len(types)
        
        return context
    
    def cleanup_expired(self):
        """Remove expired sessions."""
        with self.lock:
            current_time = time.time()
            expired = [
                uid for uid, session in self.sessions.items()
                if current_time - session["last_active"] > self.config.session_timeout
            ]
            for uid in expired:
                del self.sessions[uid]
            
            logger.info(f"Cleaned up {len(expired)} expired sessions")


class ABTestFramework:
    """
    A/B Testing framework for recommendation models.
    """
    
    def __init__(self):
        self.experiments: Dict[str, Dict] = {}
        self.assignments: Dict[str, str] = {}  # user_id -> variant
        self.lock = Lock()
    
    def create_experiment(
        self,
        name: str,
        variants: Dict[str, float],  # variant_name -> traffic_weight
        metrics: List[str]
    ):
        """Create a new A/B test experiment."""
        with self.lock:
            total_weight = sum(variants.values())
            normalized_variants = {
                k: v / total_weight for k, v in variants.items()
            }
            
            self.experiments[name] = {
                "variants": normalized_variants,
                "metrics": {m: {v: [] for v in variants} for m in metrics},
                "created_at": time.time(),
                "status": "running"
            }
    
    def get_variant(self, experiment_name: str, user_id: str) -> str:
        """Get variant assignment for user."""
        key = f"{experiment_name}:{user_id}"
        
        if key not in self.assignments:
            # Deterministic assignment based on hash
            hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
            rand_val = (hash_val % 1000) / 1000.0
            
            cumulative = 0
            variants = self.experiments[experiment_name]["variants"]
            
            for variant, weight in variants.items():
                cumulative += weight
                if rand_val < cumulative:
                    self.assignments[key] = variant
                    break
            else:
                self.assignments[key] = list(variants.keys())[0]
        
        return self.assignments[key]
    
    def log_metric(
        self,
        experiment_name: str,
        user_id: str,
        metric_name: str,
        value: float
    ):
        """Log a metric for a user in an experiment."""
        with self.lock:
            if experiment_name not in self.experiments:
                return
            
            variant = self.get_variant(experiment_name, user_id)
            experiment = self.experiments[experiment_name]
            
            if metric_name in experiment["metrics"]:
                experiment["metrics"][metric_name][variant].append(value)
    
    def get_results(self, experiment_name: str) -> Dict:
        """Get current experiment results with statistical analysis."""
        if experiment_name not in self.experiments:
            return {}
        
        experiment = self.experiments[experiment_name]
        results = {}
        
        for metric_name, variant_data in experiment["metrics"].items():
            metric_results = {}
            
            for variant, values in variant_data.items():
                if values:
                    metric_results[variant] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "count": len(values),
                        "ci_95": 1.96 * np.std(values) / np.sqrt(len(values)) if len(values) > 1 else 0
                    }
            
            results[metric_name] = metric_results
        
        return results
    
    def compute_significance(
        self,
        experiment_name: str,
        metric_name: str,
        control: str,
        treatment: str
    ) -> Dict:
        """Compute statistical significance between variants."""
        from scipy import stats
        
        experiment = self.experiments.get(experiment_name)
        if not experiment:
            return {}
        
        control_data = experiment["metrics"][metric_name][control]
        treatment_data = experiment["metrics"][metric_name][treatment]
        
        if len(control_data) < 2 or len(treatment_data) < 2:
            return {"error": "Insufficient data"}
        
        t_stat, p_value = stats.ttest_ind(control_data, treatment_data)
        
        lift = (np.mean(treatment_data) - np.mean(control_data)) / np.mean(control_data)
        
        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "lift": lift,
            "control_mean": np.mean(control_data),
            "treatment_mean": np.mean(treatment_data)
        }


class RealTimeRecommender:
    """
    Complete real-time recommendation system.
    
    Combines online learning, session management, and A/B testing.
    """
    
    def __init__(self, config: RealTimeConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        
        # Components
        self.model = OnlineModel(config)
        self.learner = OnlineLearner(self.model, config, device)
        self.session_manager = SessionManager(config)
        self.ab_framework = ABTestFramework()
        self.bandit = ContextualBandit(
            config.num_items,
            config.context_dim,
            alpha=0.1
        )
        
        # Cache
        self.cache: Dict[str, Tuple[List, float]] = {}
        self.cache_lock = Lock()
    
    def recommend(
        self,
        user_id: str,
        k: int = 10,
        exclude_items: Optional[List[int]] = None,
        use_bandit: bool = False
    ) -> List[Tuple[int, float]]:
        """Get personalized recommendations."""
        # Check cache
        cache_key = f"{user_id}:{k}:{exclude_items}"
        with self.cache_lock:
            if cache_key in self.cache:
                recs, timestamp = self.cache[cache_key]
                if time.time() - timestamp < self.config.cache_ttl:
                    return recs
        
        # Get session context
        context = self.session_manager.get_session_context(user_id)
        
        if use_bandit:
            recs = self.bandit.recommend(context, k, exclude_items)
        else:
            recs = self.learner.recommend(context, k, exclude_items)
        
        # Update cache
        with self.cache_lock:
            self.cache[cache_key] = (recs, time.time())
            
            # Cleanup cache if too large
            if len(self.cache) > self.config.max_cache_size:
                # Remove oldest entries
                sorted_keys = sorted(
                    self.cache.keys(),
                    key=lambda k: self.cache[k][1]
                )
                for key in sorted_keys[:len(self.cache) // 2]:
                    del self.cache[key]
        
        return recs
    
    def log_interaction(
        self,
        user_id: str,
        item_id: int,
        interaction_type: str,
        reward: float = 1.0
    ):
        """Log user interaction for learning."""
        # Update session
        self.session_manager.add_interaction(
            user_id, item_id, interaction_type
        )
        
        # Get context for learning
        context = self.session_manager.get_session_context(user_id)
        
        # Update online learner
        self.learner.log_interaction(item_id, context, reward)
        
        # Update bandit
        self.bandit.update(item_id, context, reward)
        
        # Invalidate cache for this user
        with self.cache_lock:
            keys_to_remove = [k for k in self.cache if k.startswith(f"{user_id}:")]
            for k in keys_to_remove:
                del self.cache[k]
    
    def get_ab_variant(self, experiment: str, user_id: str) -> str:
        """Get A/B test variant for user."""
        return self.ab_framework.get_variant(experiment, user_id)
    
    def log_ab_metric(
        self,
        experiment: str,
        user_id: str,
        metric: str,
        value: float
    ):
        """Log A/B test metric."""
        self.ab_framework.log_metric(experiment, user_id, metric, value)


if __name__ == "__main__":
    config = RealTimeConfig(
        num_items=1000,
        context_dim=32
    )
    
    recommender = RealTimeRecommender(config, device="cpu")
    
    # Create A/B test
    recommender.ab_framework.create_experiment(
        "model_comparison",
        {"control": 0.5, "treatment": 0.5},
        ["ctr", "conversion"]
    )
    
    # Simulate interactions
    for i in range(100):
        user_id = f"user_{i % 10}"
        
        # Get recommendations
        recs = recommender.recommend(user_id, k=5)
        
        # Simulate interaction
        if recs:
            item_id = recs[0][0]
            recommender.log_interaction(
                user_id, item_id, "click", 
                reward=np.random.random()
            )
            
            # Log A/B metric
            variant = recommender.get_ab_variant("model_comparison", user_id)
            recommender.log_ab_metric(
                "model_comparison", user_id, "ctr", 
                np.random.random()
            )
    
    # Get A/B test results
    results = recommender.ab_framework.get_results("model_comparison")
    print(f"A/B Test Results: {results}")
