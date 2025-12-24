"""
Causal Inference for Debiased Recommendations

Implements causal methods to address various biases in recommender systems:
- Inverse Propensity Scoring (IPS)
- Doubly Robust Estimation
- Causal Embedding
- Counterfactual Learning
- Treatment Effect Estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class CausalConfig:
    """Configuration for causal recommendation models."""
    num_users: int = 50000
    num_items: int = 10000
    embedding_dim: int = 64
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    
    # Propensity estimation
    propensity_hidden_dim: int = 128
    propensity_clip: float = 0.01  # Clip propensity scores
    
    # Causal
    num_treatments: int = 10  # For multi-treatment setting
    
    dropout: float = 0.1


class PropensityNet(nn.Module):
    """
    Propensity score estimation network.
    
    Estimates P(treatment | features) for IPS weighting.
    """
    
    def __init__(self, config: CausalConfig):
        super().__init__()
        
        self.user_embedding = nn.Embedding(config.num_users, config.embedding_dim)
        self.item_embedding = nn.Embedding(config.num_items, config.embedding_dim)
        
        self.network = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.propensity_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.propensity_hidden_dim, config.propensity_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.propensity_hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.clip_min = config.propensity_clip
        self.clip_max = 1 - config.propensity_clip
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        """Estimate propensity scores."""
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        x = torch.cat([user_emb, item_emb], dim=-1)
        propensity = self.network(x).squeeze(-1)
        
        # Clip for numerical stability
        propensity = torch.clamp(propensity, self.clip_min, self.clip_max)
        
        return propensity


class IPSRecommender(nn.Module):
    """
    Inverse Propensity Scoring Recommender
    
    Uses propensity weighting to debias recommendations.
    
    Loss: sum(1/p(exposure) * loss(y, y_hat))
    """
    
    def __init__(self, config: CausalConfig):
        super().__init__()
        self.config = config
        
        # Prediction model
        self.user_embedding = nn.Embedding(config.num_users, config.embedding_dim)
        self.item_embedding = nn.Embedding(config.num_items, config.embedding_dim)
        
        self.predictor = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[1], 1)
        )
        
        # Propensity model (can be pretrained)
        self.propensity_net = PropensityNet(config)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        """Predict ratings/scores."""
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        x = torch.cat([user_emb, item_emb], dim=-1)
        return self.predictor(x).squeeze(-1)
    
    def compute_ips_loss(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        labels: torch.Tensor,
        use_snips: bool = True  # Self-normalized IPS
    ) -> torch.Tensor:
        """Compute IPS-weighted loss."""
        predictions = self.forward(user_ids, item_ids)
        
        with torch.no_grad():
            propensities = self.propensity_net(user_ids, item_ids)
        
        # IPS weights
        weights = 1.0 / propensities
        
        if use_snips:
            # Self-normalized IPS for variance reduction
            weights = weights / weights.sum()
        
        # Weighted loss
        loss = weights * F.mse_loss(predictions, labels, reduction='none')
        
        return loss.mean()


class DoublyRobustEstimator(nn.Module):
    """
    Doubly Robust Estimation for Recommendations
    
    Combines direct method with IPS for robustness.
    
    DR = direct_estimate + IPS_correction
    """
    
    def __init__(self, config: CausalConfig):
        super().__init__()
        self.config = config
        
        # Direct model (imputation model)
        self.direct_model = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[0], 1)
        )
        
        # Outcome model
        self.outcome_model = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[1], 1)
        )
        
        # Propensity model
        self.propensity_net = PropensityNet(config)
        
        # Embeddings
        self.user_embedding = nn.Embedding(config.num_users, config.embedding_dim)
        self.item_embedding = nn.Embedding(config.num_items, config.embedding_dim)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        """Predict using outcome model."""
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        x = torch.cat([user_emb, item_emb], dim=-1)
        
        return self.outcome_model(x).squeeze(-1)
    
    def compute_dr_loss(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        labels: torch.Tensor,
        observed_mask: torch.Tensor  # 1 if observed, 0 otherwise
    ) -> torch.Tensor:
        """Compute doubly robust loss."""
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        x = torch.cat([user_emb, item_emb], dim=-1)
        
        # Direct estimate (imputation)
        direct_estimate = self.direct_model(x).squeeze(-1)
        
        # Outcome prediction
        outcome_pred = self.outcome_model(x).squeeze(-1)
        
        # Propensity
        with torch.no_grad():
            propensity = self.propensity_net(user_ids, item_ids)
        
        # Doubly robust estimate
        # DR = E[direct] + (observed - direct) * (treatment / propensity)
        ips_correction = observed_mask * (labels - direct_estimate) / propensity
        dr_estimate = direct_estimate + ips_correction
        
        # Loss: MSE between DR estimate and outcome prediction
        loss = F.mse_loss(outcome_pred, dr_estimate.detach())
        
        return loss


class CausalEmbedding(nn.Module):
    """
    Causal Embedding for Recommendations
    
    Learns embeddings that separate causal factors from confounders.
    """
    
    def __init__(self, config: CausalConfig):
        super().__init__()
        self.config = config
        
        # Split embedding into causal and confounder parts
        causal_dim = config.embedding_dim // 2
        confounder_dim = config.embedding_dim - causal_dim
        
        # User embeddings
        self.user_causal = nn.Embedding(config.num_users, causal_dim)
        self.user_confounder = nn.Embedding(config.num_users, confounder_dim)
        
        # Item embeddings
        self.item_causal = nn.Embedding(config.num_items, causal_dim)
        self.item_confounder = nn.Embedding(config.num_items, confounder_dim)
        
        # Prediction heads
        self.causal_predictor = nn.Sequential(
            nn.Linear(causal_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.full_predictor = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Independence regularization
        self.mi_estimator = nn.Sequential(
            nn.Linear(causal_dim + confounder_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for emb in [self.user_causal, self.user_confounder, 
                    self.item_causal, self.item_confounder]:
            nn.init.xavier_uniform_(emb.weight)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        use_causal_only: bool = False
    ) -> torch.Tensor:
        """
        Predict with option to use only causal embeddings.
        
        Args:
            use_causal_only: If True, use only causal factors (for unbiased inference)
        """
        user_c = self.user_causal(user_ids)
        item_c = self.item_causal(item_ids)
        
        if use_causal_only:
            x = torch.cat([user_c, item_c], dim=-1)
            return self.causal_predictor(x).squeeze(-1)
        
        user_conf = self.user_confounder(user_ids)
        item_conf = self.item_confounder(item_ids)
        
        user_full = torch.cat([user_c, user_conf], dim=-1)
        item_full = torch.cat([item_c, item_conf], dim=-1)
        x = torch.cat([user_full, item_full], dim=-1)
        
        return self.full_predictor(x).squeeze(-1)
    
    def compute_loss(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        labels: torch.Tensor,
        independence_weight: float = 0.1
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss with independence regularization."""
        # Prediction loss
        pred = self.forward(user_ids, item_ids, use_causal_only=False)
        pred_loss = F.mse_loss(pred, labels)
        
        # Independence regularization
        # Encourage causal and confounder embeddings to be independent
        user_c = self.user_causal(user_ids)
        user_conf = self.user_confounder(user_ids)
        item_c = self.item_causal(item_ids)
        item_conf = self.item_confounder(item_ids)
        
        # Estimate mutual information (simplified)
        combined_user = torch.cat([user_c, user_conf], dim=-1)
        combined_item = torch.cat([item_c, item_conf], dim=-1)
        
        mi_loss = (
            self.mi_estimator(combined_user).mean() +
            self.mi_estimator(combined_item).mean()
        )
        
        total_loss = pred_loss + independence_weight * mi_loss
        
        return total_loss, {
            "pred_loss": pred_loss.item(),
            "mi_loss": mi_loss.item()
        }


class CounterfactualRanking(nn.Module):
    """
    Counterfactual Learning to Rank
    
    Uses counterfactual reasoning for unbiased ranking.
    """
    
    def __init__(self, config: CausalConfig):
        super().__init__()
        self.config = config
        
        self.user_embedding = nn.Embedding(config.num_users, config.embedding_dim)
        self.item_embedding = nn.Embedding(config.num_items, config.embedding_dim)
        
        # Relevance model (what we want to learn)
        self.relevance_model = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[0], 1)
        )
        
        # Examination model (position bias)
        self.examination_model = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Click model: P(click) = P(examine) * P(relevant | examine)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute relevance and examination probabilities.
        
        Returns:
            relevance: P(relevant)
            examination: P(examine | position)
        """
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        x = torch.cat([user_emb, item_emb], dim=-1)
        relevance = torch.sigmoid(self.relevance_model(x).squeeze(-1))
        
        if positions is not None:
            pos_feature = positions.float().unsqueeze(-1)
            examination = self.examination_model(pos_feature).squeeze(-1)
        else:
            examination = torch.ones_like(relevance)
        
        return relevance, examination
    
    def compute_click_probability(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """Compute P(click) = P(examine) * P(relevant)."""
        relevance, examination = self.forward(user_ids, item_ids, positions)
        return examination * relevance
    
    def compute_counterfactual_loss(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        positions: torch.Tensor,
        clicks: torch.Tensor
    ) -> torch.Tensor:
        """Counterfactual loss for position-debiased learning."""
        relevance, examination = self.forward(user_ids, item_ids, positions)
        
        click_prob = examination * relevance
        
        # Binary cross-entropy
        loss = F.binary_cross_entropy(click_prob, clicks.float())
        
        # Regularize examination model (should be monotonically decreasing)
        positions_sorted = torch.arange(10, device=positions.device).float().unsqueeze(-1)
        exam_pred = self.examination_model(positions_sorted).squeeze()
        monotonic_loss = F.relu(exam_pred[1:] - exam_pred[:-1]).mean()
        
        return loss + 0.1 * monotonic_loss
    
    @torch.no_grad()
    def rank_items(
        self,
        user_id: int,
        candidate_items: List[int]
    ) -> List[Tuple[int, float]]:
        """Rank items by relevance (position-debiased)."""
        user_ids = torch.tensor([user_id] * len(candidate_items))
        item_ids = torch.tensor(candidate_items)
        
        # Use only relevance for ranking (not examination)
        relevance, _ = self.forward(user_ids, item_ids)
        
        scores = relevance.cpu().numpy()
        ranked = sorted(zip(candidate_items, scores), key=lambda x: x[1], reverse=True)
        
        return ranked


class TreatmentEffectEstimator(nn.Module):
    """
    Treatment Effect Estimation for Recommendations
    
    Estimates the causal effect of recommendations on user behavior.
    """
    
    def __init__(self, config: CausalConfig):
        super().__init__()
        self.config = config
        
        # Feature encoder
        self.user_embedding = nn.Embedding(config.num_users, config.embedding_dim)
        self.item_embedding = nn.Embedding(config.num_items, config.embedding_dim)
        
        # Outcome networks (potential outcomes framework)
        # Y(0): outcome without treatment (recommendation)
        self.y0_network = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[0], 1)
        )
        
        # Y(1): outcome with treatment
        self.y1_network = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[0], 1)
        )
        
        # Propensity network
        self.propensity_net = PropensityNet(config)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Estimate potential outcomes and treatment effect.
        
        Returns:
            y0: Outcome without recommendation
            y1: Outcome with recommendation
            ate: Average Treatment Effect = E[Y(1) - Y(0)]
        """
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        x = torch.cat([user_emb, item_emb], dim=-1)
        
        y0 = self.y0_network(x).squeeze(-1)
        y1 = self.y1_network(x).squeeze(-1)
        
        ate = y1 - y0  # Individual Treatment Effect
        
        return y0, y1, ate
    
    def compute_tarnet_loss(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        treatments: torch.Tensor,  # 1 if recommended, 0 otherwise
        outcomes: torch.Tensor
    ) -> torch.Tensor:
        """
        TARNet loss for treatment effect estimation.
        """
        y0, y1, _ = self.forward(user_ids, item_ids)
        
        # Factual loss
        y_pred = treatments * y1 + (1 - treatments) * y0
        factual_loss = F.mse_loss(y_pred, outcomes)
        
        return factual_loss
    
    @torch.no_grad()
    def estimate_uplift(
        self,
        user_id: int,
        candidate_items: List[int]
    ) -> List[Tuple[int, float]]:
        """
        Estimate uplift (treatment effect) for candidate items.
        
        Returns items ranked by expected uplift.
        """
        user_ids = torch.tensor([user_id] * len(candidate_items))
        item_ids = torch.tensor(candidate_items)
        
        _, _, ate = self.forward(user_ids, item_ids)
        
        uplifts = ate.cpu().numpy()
        ranked = sorted(zip(candidate_items, uplifts), key=lambda x: x[1], reverse=True)
        
        return ranked


class CausalRecommender:
    """High-level interface for causal recommendation methods."""
    
    def __init__(
        self,
        config: CausalConfig,
        model_type: str = "ips",  # ips, dr, causal_emb, counterfactual, treatment
        device: str = "cuda"
    ):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        if model_type == "ips":
            self.model = IPSRecommender(config)
        elif model_type == "dr":
            self.model = DoublyRobustEstimator(config)
        elif model_type == "causal_emb":
            self.model = CausalEmbedding(config)
        elif model_type == "counterfactual":
            self.model = CounterfactualRanking(config)
        elif model_type == "treatment":
            self.model = TreatmentEffectEstimator(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = self.model.to(self.device)
        self.model_type = model_type
    
    def save(self, path: str):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "model_type": self.model_type
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])


if __name__ == "__main__":
    config = CausalConfig(
        num_users=1000,
        num_items=5000,
        embedding_dim=64
    )
    
    # Test IPS Recommender
    ips = IPSRecommender(config)
    users = torch.randint(0, 1000, (32,))
    items = torch.randint(0, 5000, (32,))
    labels = torch.rand(32)
    
    pred = ips(users, items)
    loss = ips.compute_ips_loss(users, items, labels)
    print(f"IPS prediction shape: {pred.shape}, loss: {loss.item():.4f}")
    
    # Test Causal Embedding
    causal_emb = CausalEmbedding(config)
    pred_causal = causal_emb(users, items, use_causal_only=True)
    pred_full = causal_emb(users, items, use_causal_only=False)
    print(f"Causal-only pred: {pred_causal.shape}, Full pred: {pred_full.shape}")
