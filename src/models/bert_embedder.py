"""
BERT-based Content Embedder for Media Recommendations

This module implements content-based recommendations using BERT embeddings
to understand media content semantics. Specifically designed to solve the
cold start problem by generating rich representations of new items.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BertModel, 
    BertTokenizer, 
    AutoModel, 
    AutoTokenizer
)
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


@dataclass
class BertEmbedderConfig:
    """Configuration for BERT Embedder."""
    model_name: str = "bert-base-uncased"
    embedding_dim: int = 768
    max_length: int = 512
    pooling_strategy: str = "mean"  # mean, cls, max, attention
    projection_dim: Optional[int] = 256
    fine_tune_layers: int = 2  # Number of top layers to fine-tune
    dropout: float = 0.1
    normalize_embeddings: bool = True


class AttentionPooling(nn.Module):
    """Attention-based pooling for sequence representations."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply attention pooling.
        
        Args:
            hidden_states: BERT outputs [batch, seq_len, hidden]
            attention_mask: Mask for valid tokens [batch, seq_len]
            
        Returns:
            Pooled representation [batch, hidden]
        """
        # Compute attention weights
        weights = self.attention(hidden_states)  # [batch, seq_len, 1]
        
        # Apply mask
        mask = attention_mask.unsqueeze(-1).float()
        weights = weights * mask
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Weighted sum
        pooled = (hidden_states * weights).sum(dim=1)
        
        return pooled


class BertContentEmbedder(nn.Module):
    """
    BERT-based content embedder for media items.
    
    Generates semantic embeddings for text content (titles, descriptions, etc.)
    using pre-trained BERT with optional fine-tuning and projection.
    """
    
    def __init__(self, config: BertEmbedderConfig):
        super().__init__()
        self.config = config
        
        # Load pre-trained BERT
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.bert = AutoModel.from_pretrained(config.model_name)
        
        # Freeze lower layers
        self._freeze_layers(config.fine_tune_layers)
        
        # Pooling strategy
        if config.pooling_strategy == "attention":
            self.pooler = AttentionPooling(config.embedding_dim)
        else:
            self.pooler = None
        
        # Optional projection layer
        if config.projection_dim:
            self.projection = nn.Sequential(
                nn.Linear(config.embedding_dim, config.projection_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.projection_dim, config.projection_dim)
            )
        else:
            self.projection = None
    
    def _freeze_layers(self, num_trainable_layers: int):
        """Freeze all but top N layers of BERT."""
        # Freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze lower encoder layers
        num_layers = len(self.bert.encoder.layer)
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < num_layers - num_trainable_layers:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def _pool(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply pooling strategy to get fixed-size representation."""
        if self.config.pooling_strategy == "cls":
            return hidden_states[:, 0, :]
        
        elif self.config.pooling_strategy == "max":
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            hidden_states[mask == 0] = float("-inf")
            return hidden_states.max(dim=1)[0]
        
        elif self.config.pooling_strategy == "attention":
            return self.pooler(hidden_states, attention_mask)
        
        else:  # mean pooling (default)
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = (hidden_states * mask).sum(dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-8)
            return sum_hidden / sum_mask
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate embeddings for tokenized input.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            token_type_ids: Optional token type IDs [batch, seq_len]
            
        Returns:
            Content embeddings [batch, embedding_dim or projection_dim]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Pool to fixed-size representation
        pooled = self._pool(outputs.last_hidden_state, attention_mask)
        
        # Apply projection if configured
        if self.projection:
            pooled = self.projection(pooled)
        
        # Normalize if configured
        if self.config.normalize_embeddings:
            pooled = F.normalize(pooled, p=2, dim=1)
        
        return pooled
    
    def encode(
        self, 
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode text(s) to embeddings.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            Embeddings as numpy array [num_texts, embedding_dim]
        """
        if isinstance(texts, str):
            texts = [texts]
        
        self.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt"
                )
                
                # Move to device
                device = next(self.parameters()).device
                encoded = {k: v.to(device) for k, v in encoded.items()}
                
                # Encode
                embeddings = self.forward(**encoded)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)


class ContentBasedRecommender(nn.Module):
    """
    Content-based recommender using BERT embeddings.
    
    Combines content embeddings with user preference learning
    for personalized content recommendations.
    """
    
    def __init__(
        self,
        embedder_config: BertEmbedderConfig,
        num_users: int,
        user_embedding_dim: int = 128
    ):
        super().__init__()
        
        self.content_embedder = BertContentEmbedder(embedder_config)
        
        # Content embedding dimension
        content_dim = (
            embedder_config.projection_dim 
            if embedder_config.projection_dim 
            else embedder_config.embedding_dim
        )
        
        # User preference model
        self.user_embedding = nn.Embedding(num_users, user_embedding_dim)
        
        # Transform user embedding to content space
        self.user_transform = nn.Sequential(
            nn.Linear(user_embedding_dim, content_dim),
            nn.ReLU(),
            nn.Dropout(embedder_config.dropout),
            nn.Linear(content_dim, content_dim)
        )
        
        # Scoring MLP
        self.scorer = nn.Sequential(
            nn.Linear(content_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(embedder_config.dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Cache for precomputed item embeddings
        self.item_embedding_cache: Dict[int, torch.Tensor] = {}
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_input_ids: torch.Tensor,
        item_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute user-item affinity scores.
        
        Args:
            user_ids: User ID tensor [batch]
            item_input_ids: Item text token IDs [batch, seq_len]
            item_attention_mask: Item text attention mask [batch, seq_len]
            
        Returns:
            Affinity scores [batch]
        """
        # Get user representations
        user_emb = self.user_embedding(user_ids)
        user_rep = self.user_transform(user_emb)
        user_rep = F.normalize(user_rep, p=2, dim=1)
        
        # Get content representations
        content_rep = self.content_embedder(item_input_ids, item_attention_mask)
        
        # Combine and score
        combined = torch.cat([user_rep, content_rep], dim=1)
        scores = self.scorer(combined).squeeze(-1)
        
        return scores
    
    def compute_content_similarity(
        self,
        query_embedding: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find most similar items by content.
        
        Args:
            query_embedding: Query item embedding [embed_dim]
            candidate_embeddings: Candidate embeddings [num_items, embed_dim]
            top_k: Number of similar items to return
            
        Returns:
            Tuple of (similarity scores, indices)
        """
        similarities = F.cosine_similarity(
            query_embedding.unsqueeze(0), 
            candidate_embeddings
        )
        return torch.topk(similarities, top_k)
    
    @torch.no_grad()
    def recommend_for_new_user(
        self,
        user_content_history: List[str],
        candidate_items: List[Dict[str, str]],
        n_recommendations: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Cold start: Recommend for new user based on content preferences.
        
        Args:
            user_content_history: List of content user has engaged with
            candidate_items: List of candidate items with 'title' and 'description'
            n_recommendations: Number of recommendations
            
        Returns:
            List of (item_index, score) tuples
        """
        self.eval()
        
        # Create user profile from history
        if user_content_history:
            history_embeddings = self.content_embedder.encode(user_content_history)
            user_profile = torch.tensor(history_embeddings.mean(axis=0))
        else:
            # Default to zero vector for completely new users
            content_dim = self.content_embedder.config.projection_dim or \
                         self.content_embedder.config.embedding_dim
            user_profile = torch.zeros(content_dim)
        
        user_profile = user_profile.to(next(self.parameters()).device)
        
        # Encode all candidates
        candidate_texts = [
            f"{item.get('title', '')} {item.get('description', '')}"
            for item in candidate_items
        ]
        candidate_embeddings = torch.tensor(
            self.content_embedder.encode(candidate_texts)
        ).to(user_profile.device)
        
        # Compute similarities
        similarities = F.cosine_similarity(
            user_profile.unsqueeze(0),
            candidate_embeddings
        )
        
        # Get top recommendations
        scores, indices = torch.topk(similarities, n_recommendations)
        
        return [
            (idx.item(), score.item())
            for idx, score in zip(indices, scores)
        ]


class DualEncoderModel(nn.Module):
    """
    Dual Encoder model for efficient similarity search.
    
    Separately encodes users and items for fast approximate
    nearest neighbor search in production.
    """
    
    def __init__(
        self,
        embedder_config: BertEmbedderConfig,
        num_users: int,
        user_features_dim: int = 64
    ):
        super().__init__()
        
        # Content encoder
        self.content_encoder = BertContentEmbedder(embedder_config)
        
        content_dim = (
            embedder_config.projection_dim 
            if embedder_config.projection_dim 
            else embedder_config.embedding_dim
        )
        
        # User encoder (based on user features + learned embedding)
        self.user_encoder = nn.Sequential(
            nn.Linear(user_features_dim + 128, 256),
            nn.ReLU(),
            nn.Dropout(embedder_config.dropout),
            nn.Linear(256, content_dim)
        )
        
        self.user_embedding = nn.Embedding(num_users, 128)
        self.content_dim = content_dim
        
        # Temperature for contrastive learning
        self.temperature = nn.Parameter(torch.tensor(0.07))
    
    def encode_user(
        self,
        user_id: torch.Tensor,
        user_features: torch.Tensor
    ) -> torch.Tensor:
        """Encode user to embedding space."""
        user_emb = self.user_embedding(user_id)
        combined = torch.cat([user_emb, user_features], dim=1)
        encoded = self.user_encoder(combined)
        return F.normalize(encoded, p=2, dim=1)
    
    def encode_item(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode item content to embedding space."""
        return self.content_encoder(input_ids, attention_mask)
    
    def forward(
        self,
        user_id: torch.Tensor,
        user_features: torch.Tensor,
        item_input_ids: torch.Tensor,
        item_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute similarity scores."""
        user_emb = self.encode_user(user_id, user_features)
        item_emb = self.encode_item(item_input_ids, item_attention_mask)
        
        # Dot product similarity
        return (user_emb * item_emb).sum(dim=1)
    
    def compute_contrastive_loss(
        self,
        user_embeddings: torch.Tensor,
        pos_item_embeddings: torch.Tensor,
        neg_item_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        InfoNCE contrastive loss.
        
        Args:
            user_embeddings: User representations [batch, dim]
            pos_item_embeddings: Positive item representations [batch, dim]
            neg_item_embeddings: Negative item representations [batch, num_neg, dim]
            
        Returns:
            Contrastive loss
        """
        batch_size = user_embeddings.size(0)
        
        # Positive similarities
        pos_sim = (user_embeddings * pos_item_embeddings).sum(dim=1)
        pos_sim = pos_sim / self.temperature
        
        # Negative similarities
        neg_sim = torch.bmm(
            neg_item_embeddings, 
            user_embeddings.unsqueeze(-1)
        ).squeeze(-1)
        neg_sim = neg_sim / self.temperature
        
        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        
        return F.cross_entropy(logits, labels)


if __name__ == "__main__":
    # Example usage
    config = BertEmbedderConfig(
        model_name="bert-base-uncased",
        projection_dim=256,
        pooling_strategy="mean"
    )
    
    embedder = BertContentEmbedder(config)
    
    # Encode some sample content
    texts = [
        "Breaking Bad: A chemistry teacher turns to cooking meth",
        "Game of Thrones: Epic fantasy with dragons and politics",
        "The Office: A mockumentary about a paper company"
    ]
    
    embeddings = embedder.encode(texts)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Compute similarities
    from scipy.spatial.distance import cosine
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = 1 - cosine(embeddings[i], embeddings[j])
            print(f"Similarity({i}, {j}): {sim:.4f}")
