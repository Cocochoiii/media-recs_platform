"""
Explainable Recommendations

Implements methods for generating human-readable explanations:
- Attention-based Explanations
- Path-based Explanations (knowledge graph)
- Counterfactual Explanations
- Feature Attribution (SHAP-style)
- Template-based Natural Language Explanations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
from enum import Enum


class ExplanationType(Enum):
    """Types of explanations."""
    FEATURE_BASED = "feature_based"
    COLLABORATIVE = "collaborative"
    CONTENT_BASED = "content_based"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    COUNTERFACTUAL = "counterfactual"


@dataclass
class ExplainableConfig:
    """Configuration for explainable recommendations."""
    num_users: int = 50000
    num_items: int = 10000
    num_features: int = 100
    embedding_dim: int = 64
    hidden_dim: int = 128
    
    # Attention
    num_attention_heads: int = 4
    
    # Explanations
    max_explanation_features: int = 5
    min_attention_threshold: float = 0.1


@dataclass
class Explanation:
    """Container for recommendation explanations."""
    item_id: int
    score: float
    explanation_type: ExplanationType
    
    # Feature-based
    important_features: List[Tuple[str, float]] = field(default_factory=list)
    
    # Collaborative
    similar_users: List[int] = field(default_factory=list)
    similar_items: List[int] = field(default_factory=list)
    
    # Knowledge graph
    paths: List[List[str]] = field(default_factory=list)
    
    # Natural language
    text_explanation: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "item_id": self.item_id,
            "score": self.score,
            "type": self.explanation_type.value,
            "features": self.important_features,
            "similar_users": self.similar_users,
            "similar_items": self.similar_items,
            "paths": self.paths,
            "explanation": self.text_explanation
        }


class AttentionExplainer(nn.Module):
    """
    Attention-based Explainable Recommendations
    
    Uses attention weights to identify important features.
    """
    
    def __init__(self, config: ExplainableConfig):
        super().__init__()
        self.config = config
        
        # Feature embeddings
        self.user_embedding = nn.Embedding(config.num_users, config.embedding_dim)
        self.item_embedding = nn.Embedding(config.num_items, config.embedding_dim)
        self.feature_embedding = nn.Embedding(config.num_features, config.embedding_dim)
        
        # Multi-head attention for feature importance
        self.feature_attention = nn.MultiheadAttention(
            config.embedding_dim,
            config.num_attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        item_features: torch.Tensor,  # [batch, num_features]
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional attention weights.
        
        Args:
            item_features: Feature IDs for each item [batch, max_features]
            
        Returns:
            predictions: [batch]
            attention_weights: [batch, num_features] if return_attention
        """
        user_emb = self.user_embedding(user_ids)  # [batch, dim]
        item_emb = self.item_embedding(item_ids)  # [batch, dim]
        feature_emb = self.feature_embedding(item_features)  # [batch, num_features, dim]
        
        # User as query, features as key/value
        query = user_emb.unsqueeze(1)  # [batch, 1, dim]
        
        attn_output, attn_weights = self.feature_attention(
            query, feature_emb, feature_emb
        )
        attn_output = attn_output.squeeze(1)  # [batch, dim]
        
        # Combine user preference with attended features
        combined = torch.cat([item_emb, attn_output], dim=-1)
        prediction = self.predictor(combined).squeeze(-1)
        
        if return_attention:
            return prediction, attn_weights.squeeze(1)  # [batch, num_features]
        
        return prediction, None
    
    def explain(
        self,
        user_id: int,
        item_id: int,
        item_features: List[int],
        feature_names: Dict[int, str]
    ) -> Explanation:
        """Generate feature-based explanation."""
        user_tensor = torch.tensor([user_id])
        item_tensor = torch.tensor([item_id])
        feature_tensor = torch.tensor([item_features])
        
        with torch.no_grad():
            score, attention = self.forward(
                user_tensor, item_tensor, feature_tensor, return_attention=True
            )
        
        # Get important features
        attention_np = attention[0].cpu().numpy()
        important_features = []
        
        for idx, (feat_id, attn_weight) in enumerate(zip(item_features, attention_np)):
            if attn_weight > self.config.min_attention_threshold:
                feat_name = feature_names.get(feat_id, f"feature_{feat_id}")
                important_features.append((feat_name, float(attn_weight)))
        
        # Sort by attention weight
        important_features.sort(key=lambda x: x[1], reverse=True)
        important_features = important_features[:self.config.max_explanation_features]
        
        # Generate text explanation
        if important_features:
            feature_text = ", ".join([f"{name} ({weight:.1%})" for name, weight in important_features])
            text = f"Recommended because you showed interest in: {feature_text}"
        else:
            text = "Recommended based on your overall preferences."
        
        return Explanation(
            item_id=item_id,
            score=score.item(),
            explanation_type=ExplanationType.FEATURE_BASED,
            important_features=important_features,
            text_explanation=text
        )


class PathExplainer:
    """
    Path-based Explanations using Knowledge Graph
    
    Finds paths between user and recommended item through KG.
    """
    
    def __init__(self, config: ExplainableConfig):
        self.config = config
        
        # Knowledge graph storage
        self.kg: Dict[int, List[Tuple[str, int]]] = defaultdict(list)  # entity -> [(relation, entity)]
        self.entity_names: Dict[int, str] = {}
        self.relation_names: Dict[str, str] = {}
        
        # User history
        self.user_history: Dict[int, List[int]] = defaultdict(list)
    
    def add_triple(self, head: int, relation: str, tail: int):
        """Add a triple to the knowledge graph."""
        self.kg[head].append((relation, tail))
        self.kg[tail].append((f"inverse_{relation}", head))
    
    def set_user_history(self, user_id: int, item_ids: List[int]):
        """Set user's interaction history."""
        self.user_history[user_id] = item_ids
    
    def find_paths(
        self,
        user_id: int,
        target_item: int,
        max_depth: int = 3,
        max_paths: int = 5
    ) -> List[List[Tuple[int, str]]]:
        """
        Find paths from user's history to target item.
        
        Returns:
            List of paths, each path is [(entity, relation), ...]
        """
        user_items = set(self.user_history.get(user_id, []))
        if not user_items:
            return []
        
        paths = []
        
        def dfs(current: int, path: List[Tuple[int, str]], visited: set):
            if len(path) > max_depth:
                return
            
            if current == target_item and len(path) > 0:
                paths.append(path.copy())
                return
            
            if len(paths) >= max_paths:
                return
            
            for relation, neighbor in self.kg.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append((neighbor, relation))
                    dfs(neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)
        
        # Start DFS from each item in user history
        for start_item in user_items:
            if len(paths) >= max_paths:
                break
            
            visited = {start_item}
            dfs(start_item, [(start_item, "user_liked")], visited)
        
        return paths
    
    def explain(self, user_id: int, item_id: int) -> Explanation:
        """Generate path-based explanation."""
        paths = self.find_paths(user_id, item_id)
        
        # Convert paths to readable format
        path_strings = []
        for path in paths:
            path_str = []
            for entity_id, relation in path:
                entity_name = self.entity_names.get(entity_id, f"item_{entity_id}")
                relation_name = self.relation_names.get(relation, relation)
                path_str.append(f"{entity_name} --[{relation_name}]-->")
            path_strings.append(path_str)
        
        # Generate text
        if path_strings:
            text = "Recommended because: "
            text += " | ".join([" ".join(p) for p in path_strings[:2]])
        else:
            text = "Recommended based on overall popularity."
        
        return Explanation(
            item_id=item_id,
            score=0.0,  # Score should be computed separately
            explanation_type=ExplanationType.KNOWLEDGE_GRAPH,
            paths=path_strings,
            text_explanation=text
        )


class CounterfactualExplainer(nn.Module):
    """
    Counterfactual Explanations
    
    Explains what minimal changes would change the recommendation.
    """
    
    def __init__(self, config: ExplainableConfig, base_model: nn.Module):
        super().__init__()
        self.config = config
        self.base_model = base_model  # The model to explain
        
        # Feature perturbation network
        self.perturbation_net = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim),
            nn.Tanh()
        )
    
    def find_counterfactual(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        target_change: str = "decrease",  # "increase" or "decrease"
        max_iterations: int = 100,
        learning_rate: float = 0.01
    ) -> Tuple[torch.Tensor, float]:
        """
        Find minimal perturbation that changes the prediction.
        
        Returns:
            perturbation: The feature perturbation
            new_score: Score after perturbation
        """
        self.base_model.eval()
        
        # Get original prediction
        with torch.no_grad():
            original_pred = self.base_model(user_id, item_id)
        
        # Initialize perturbation
        perturbation = torch.zeros(1, self.config.embedding_dim, requires_grad=True)
        optimizer = torch.optim.Adam([perturbation], lr=learning_rate)
        
        target_direction = -1 if target_change == "decrease" else 1
        
        for _ in range(max_iterations):
            optimizer.zero_grad()
            
            # Apply perturbation (simplified - assumes access to embeddings)
            perturbed_user = self.base_model.user_embedding(user_id) + perturbation
            perturbed_item = self.base_model.item_embedding(item_id)
            
            # Compute new score (simplified forward pass)
            new_pred = (perturbed_user * perturbed_item).sum()
            
            # Loss: change prediction in target direction + minimize perturbation
            pred_loss = -target_direction * new_pred
            sparsity_loss = perturbation.abs().mean()
            
            loss = pred_loss + 0.1 * sparsity_loss
            loss.backward()
            optimizer.step()
        
        return perturbation.detach(), new_pred.item()
    
    def explain(
        self,
        user_id: int,
        item_id: int,
        feature_names: List[str]
    ) -> Explanation:
        """Generate counterfactual explanation."""
        user_tensor = torch.tensor([user_id])
        item_tensor = torch.tensor([item_id])
        
        perturbation, new_score = self.find_counterfactual(
            user_tensor, item_tensor, target_change="decrease"
        )
        
        # Find most important perturbations
        pert_np = perturbation[0].cpu().numpy()
        important_dims = np.argsort(np.abs(pert_np))[::-1][:5]
        
        important_features = []
        for dim in important_dims:
            if dim < len(feature_names):
                direction = "more" if pert_np[dim] > 0 else "less"
                important_features.append((feature_names[dim], pert_np[dim]))
        
        text = "This recommendation would change if you showed "
        changes = [f"{direction} interest in {name}" 
                   for name, val in important_features[:3]]
        text += ", ".join(changes)
        
        return Explanation(
            item_id=item_id,
            score=new_score,
            explanation_type=ExplanationType.COUNTERFACTUAL,
            important_features=important_features,
            text_explanation=text
        )


class TemplateExplainer:
    """
    Template-based Natural Language Explanations
    
    Generates human-readable explanations using templates.
    """
    
    def __init__(self):
        self.templates = {
            "collaborative": [
                "Users similar to you also enjoyed {item_name}",
                "Recommended because users with similar taste rated this highly",
                "Based on preferences of {num_users} users like you",
            ],
            "content_based": [
                "Recommended because you liked {similar_item} which has similar {features}",
                "Matches your preference for {features}",
                "Similar to {similar_item} in terms of {features}",
            ],
            "trending": [
                "Trending in {category} right now",
                "Popular among users in your area",
                "Rising popularity with {growth}% increase this week",
            ],
            "personalized": [
                "Based on your interest in {interest}",
                "Matches your preference for {preference}",
                "Recommended for your taste in {genre}",
            ],
            "knowledge_graph": [
                "Connected to {liked_item} through {relation}",
                "Related to your favorite {entity_type}: {entity_name}",
                "Shares {attribute} with items you enjoyed",
            ]
        }
    
    def generate_explanation(
        self,
        template_type: str,
        **kwargs
    ) -> str:
        """Generate explanation from template."""
        templates = self.templates.get(template_type, self.templates["personalized"])
        template = np.random.choice(templates)
        
        try:
            return template.format(**kwargs)
        except KeyError:
            return templates[0].format(**{k: "items you liked" for k in kwargs})
    
    def explain_collaborative(
        self,
        item_name: str,
        num_similar_users: int
    ) -> str:
        """Generate collaborative filtering explanation."""
        return self.generate_explanation(
            "collaborative",
            item_name=item_name,
            num_users=num_similar_users
        )
    
    def explain_content(
        self,
        similar_item: str,
        shared_features: List[str]
    ) -> str:
        """Generate content-based explanation."""
        features = ", ".join(shared_features[:3])
        return self.generate_explanation(
            "content_based",
            similar_item=similar_item,
            features=features
        )
    
    def explain_knowledge_graph(
        self,
        liked_item: str,
        relation: str
    ) -> str:
        """Generate knowledge graph explanation."""
        return self.generate_explanation(
            "knowledge_graph",
            liked_item=liked_item,
            relation=relation
        )


class ExplainableRecommender(nn.Module):
    """
    Complete Explainable Recommendation System
    
    Combines multiple explanation methods.
    """
    
    def __init__(self, config: ExplainableConfig):
        super().__init__()
        self.config = config
        
        # Core recommendation model with attention
        self.attention_explainer = AttentionExplainer(config)
        
        # Path explainer (non-neural)
        self.path_explainer = PathExplainer(config)
        
        # Template generator
        self.template_explainer = TemplateExplainer()
        
        # Feature names mapping
        self.feature_names: Dict[int, str] = {}
        self.item_names: Dict[int, str] = {}
    
    def set_metadata(
        self,
        feature_names: Dict[int, str],
        item_names: Dict[int, str]
    ):
        """Set feature and item name mappings."""
        self.feature_names = feature_names
        self.item_names = item_names
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        item_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute recommendation scores."""
        scores, _ = self.attention_explainer(user_ids, item_ids, item_features)
        return scores
    
    def recommend_with_explanations(
        self,
        user_id: int,
        candidate_items: List[int],
        item_features: Dict[int, List[int]],
        k: int = 10,
        explanation_types: List[ExplanationType] = None
    ) -> List[Tuple[int, float, Explanation]]:
        """
        Generate recommendations with explanations.
        
        Args:
            user_id: User ID
            candidate_items: List of candidate item IDs
            item_features: Mapping from item ID to feature IDs
            k: Number of recommendations
            explanation_types: Types of explanations to generate
            
        Returns:
            List of (item_id, score, explanation) tuples
        """
        if explanation_types is None:
            explanation_types = [ExplanationType.FEATURE_BASED]
        
        # Compute scores
        user_tensor = torch.tensor([user_id] * len(candidate_items))
        item_tensor = torch.tensor(candidate_items)
        
        # Prepare features
        max_features = max(len(f) for f in item_features.values())
        features_padded = []
        for item in candidate_items:
            feats = item_features.get(item, [0])
            feats = feats + [0] * (max_features - len(feats))
            features_padded.append(feats)
        feature_tensor = torch.tensor(features_padded)
        
        with torch.no_grad():
            scores, attention = self.attention_explainer(
                user_tensor, item_tensor, feature_tensor, return_attention=True
            )
        
        # Get top-k
        scores_np = scores.cpu().numpy()
        attention_np = attention.cpu().numpy()
        
        top_indices = np.argsort(scores_np)[::-1][:k]
        
        results = []
        for idx in top_indices:
            item_id = candidate_items[idx]
            score = scores_np[idx]
            
            # Generate explanation
            if ExplanationType.FEATURE_BASED in explanation_types:
                explanation = self._generate_feature_explanation(
                    item_id, score, item_features[item_id], attention_np[idx]
                )
            elif ExplanationType.KNOWLEDGE_GRAPH in explanation_types:
                explanation = self.path_explainer.explain(user_id, item_id)
                explanation.score = score
            else:
                explanation = Explanation(
                    item_id=item_id,
                    score=score,
                    explanation_type=ExplanationType.FEATURE_BASED,
                    text_explanation="Recommended based on your preferences."
                )
            
            results.append((item_id, score, explanation))
        
        return results
    
    def _generate_feature_explanation(
        self,
        item_id: int,
        score: float,
        features: List[int],
        attention: np.ndarray
    ) -> Explanation:
        """Generate feature-based explanation."""
        important_features = []
        
        for feat_idx, (feat_id, attn_weight) in enumerate(zip(features, attention)):
            if attn_weight > self.config.min_attention_threshold:
                feat_name = self.feature_names.get(feat_id, f"feature_{feat_id}")
                important_features.append((feat_name, float(attn_weight)))
        
        important_features.sort(key=lambda x: x[1], reverse=True)
        important_features = important_features[:self.config.max_explanation_features]
        
        # Generate natural language
        item_name = self.item_names.get(item_id, f"Item {item_id}")
        
        if important_features:
            feature_text = ", ".join([f[0] for f in important_features[:3]])
            text = f"We recommend '{item_name}' because you showed interest in: {feature_text}"
        else:
            text = f"'{item_name}' matches your overall preferences."
        
        return Explanation(
            item_id=item_id,
            score=score,
            explanation_type=ExplanationType.FEATURE_BASED,
            important_features=important_features,
            text_explanation=text
        )


if __name__ == "__main__":
    config = ExplainableConfig(
        num_users=1000,
        num_items=5000,
        num_features=100
    )
    
    model = ExplainableRecommender(config)
    
    # Set metadata
    feature_names = {i: f"genre_{i}" for i in range(100)}
    item_names = {i: f"Movie_{i}" for i in range(5000)}
    model.set_metadata(feature_names, item_names)
    
    # Test recommendation with explanation
    item_features = {
        i: list(np.random.randint(0, 100, size=5))
        for i in range(100)
    }
    
    results = model.recommend_with_explanations(
        user_id=42,
        candidate_items=list(range(100)),
        item_features=item_features,
        k=5
    )
    
    for item_id, score, explanation in results:
        print(f"\nItem {item_id} (score: {score:.3f})")
        print(f"  Explanation: {explanation.text_explanation}")
        print(f"  Features: {explanation.important_features[:3]}")
