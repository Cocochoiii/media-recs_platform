"""
Contrastive Learning Model for Media Recommendations

This module implements contrastive learning approaches for recommendation,
combining Word2Vec embeddings with modern contrastive objectives for
robust representation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import logging

# Optional imports
try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    Word2Vec = None

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

logger = logging.getLogger(__name__)


@dataclass
class ContrastiveConfig:
    """Configuration for Contrastive Learning model."""
    embedding_dim: int = 256
    projection_dim: int = 128
    temperature: float = 0.07
    num_negatives: int = 10
    use_hard_negatives: bool = True
    word2vec_dim: int = 300
    dropout: float = 0.1
    word2vec_window: int = 5
    word2vec_min_count: int = 5
    word2vec_epochs: int = 10


class Word2VecEmbedder:
    """Word2Vec-based text embedder using gensim and spaCy (optional)."""
    
    def __init__(self, config: ContrastiveConfig):
        self.config = config
        self.nlp = None  # Lazy load
        self.word2vec = None
        self.idf_weights: Dict[str, float] = {}
        
        if not GENSIM_AVAILABLE:
            logger.warning("gensim not available. Word2VecEmbedder will use random embeddings.")
        if not SPACY_AVAILABLE:
            logger.warning("spacy not available. Using simple tokenization.")
    
    def _load_nlp(self):
        if self.nlp is None and SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            except OSError:
                logger.warning("spacy model not found. Using simple tokenization.")
                self.nlp = None
    
    def _tokenize(self, text: str) -> List[str]:
        if SPACY_AVAILABLE and self.nlp is not None:
            self._load_nlp()
            if self.nlp:
                doc = self.nlp(text.lower())
                tokens = [
                    token.lemma_ for token in doc
                    if not token.is_stop and not token.is_punct and token.is_alpha
                ]
                return tokens
        
        # Fallback: simple tokenization
        import re
        tokens = re.findall(r'\b[a-z]+\b', text.lower())
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'and', 
                     'but', 'or', 'nor', 'for', 'yet', 'so', 'in', 'on', 'at',
                     'to', 'of', 'it', 'its', 'this', 'that', 'these', 'those'}
        return [t for t in tokens if t not in stop_words and len(t) > 2]
    
    def train(self, corpus: List[str], compute_idf: bool = True):
        logger.info(f"Tokenizing {len(corpus)} documents...")
        tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        
        if GENSIM_AVAILABLE and Word2Vec is not None:
            logger.info("Training Word2Vec model...")
            self.word2vec = Word2Vec(
                sentences=tokenized_corpus,
                vector_size=self.config.word2vec_dim,
                window=self.config.word2vec_window,
                min_count=self.config.word2vec_min_count,
                workers=4,
                epochs=self.config.word2vec_epochs
            )
            
            if compute_idf:
                self._compute_idf(tokenized_corpus)
            
            logger.info(f"Vocabulary size: {len(self.word2vec.wv)}")
        else:
            # Fallback: create random embeddings vocabulary
            logger.info("Using random embeddings (gensim not available)")
            self._vocab = {}
            for doc in tokenized_corpus:
                for token in doc:
                    if token not in self._vocab:
                        self._vocab[token] = np.random.randn(self.config.word2vec_dim).astype(np.float32)
            
            if compute_idf:
                self._compute_idf(tokenized_corpus)
            
            logger.info(f"Vocabulary size: {len(self._vocab)}")
    
    def _compute_idf(self, tokenized_corpus: List[List[str]]):
        doc_freq = defaultdict(int)
        num_docs = len(tokenized_corpus)
        
        for doc in tokenized_corpus:
            for token in set(doc):
                doc_freq[token] += 1
        
        for token, freq in doc_freq.items():
            self.idf_weights[token] = np.log(num_docs / (freq + 1))
    
    def encode(self, texts: Union[str, List[str]], use_idf: bool = True) -> np.ndarray:
        if self.word2vec is None and not hasattr(self, '_vocab'):
            raise RuntimeError("Word2Vec model not trained. Call train() first.")
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            tokens = self._tokenize(text)
            
            if not tokens:
                embeddings.append(np.zeros(self.config.word2vec_dim))
                continue
            
            word_vectors, weights = [], []
            
            for token in tokens:
                # Try gensim word2vec first
                if self.word2vec is not None and token in self.word2vec.wv:
                    word_vectors.append(self.word2vec.wv[token])
                    weights.append(self.idf_weights.get(token, 1.0) if use_idf else 1.0)
                # Fallback to random vocab
                elif hasattr(self, '_vocab') and token in self._vocab:
                    word_vectors.append(self._vocab[token])
                    weights.append(self.idf_weights.get(token, 1.0) if use_idf else 1.0)
            
            if word_vectors:
                word_vectors = np.array(word_vectors)
                weights = np.array(weights)
                weights = weights / weights.sum()
                embedding = (word_vectors * weights[:, np.newaxis]).sum(axis=0)
            else:
                embedding = np.zeros(self.config.word2vec_dim)
            
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def save(self, path: str):
        if self.word2vec:
            self.word2vec.save(path)
    
    def load(self, path: str):
        self.word2vec = Word2Vec.load(path)


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=1)


class ContrastiveEncoder(nn.Module):
    """Contrastive encoder for user/item representations."""
    
    def __init__(self, config: ContrastiveConfig, input_dim: int):
        super().__init__()
        self.config = config
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, config.embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim * 2, config.embedding_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        self.projection = ProjectionHead(
            config.embedding_dim, config.embedding_dim, 
            config.projection_dim, config.dropout
        )
        
        self.temperature = nn.Parameter(torch.tensor(config.temperature))
    
    def forward(self, x: torch.Tensor, return_projection: bool = True) -> torch.Tensor:
        encoded = self.encoder(x)
        if return_projection:
            return self.projection(encoded)
        return encoded


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = anchor.size(0)
        pos_sim = (anchor * positive).sum(dim=1) / self.temperature
        
        if negatives is not None:
            neg_sim = torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1) / self.temperature
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        else:
            sim_matrix = torch.mm(anchor, positive.T) / self.temperature
            mask = torch.eye(batch_size, device=anchor.device).bool()
            sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
            logits = torch.cat([pos_sim.unsqueeze(1), sim_matrix], dim=1)
        
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
        return F.cross_entropy(logits, labels)


class HardNegativeMiner:
    """Mining hard negatives for contrastive learning."""
    
    def __init__(self, num_negatives: int = 10, strategy: str = "semi-hard"):
        self.num_negatives = num_negatives
        self.strategy = strategy
    
    def mine(
        self,
        anchor_embeddings: torch.Tensor,
        all_embeddings: torch.Tensor,
        positive_indices: torch.Tensor,
        margin: float = 0.2
    ) -> torch.Tensor:
        batch_size = anchor_embeddings.size(0)
        num_items = all_embeddings.size(0)
        device = anchor_embeddings.device
        
        similarities = torch.mm(anchor_embeddings, all_embeddings.T)
        pos_mask = torch.zeros(batch_size, num_items, device=device)
        pos_mask.scatter_(1, positive_indices.unsqueeze(1), 1)
        
        if self.strategy == "hard":
            similarities = similarities - pos_mask * 1e9
            _, neg_indices = torch.topk(similarities, self.num_negatives, dim=1)
        else:
            neg_mask = 1 - pos_mask
            neg_indices = torch.multinomial(neg_mask, self.num_negatives, replacement=False)
        
        return neg_indices


class ContrastiveLearningRecommender(nn.Module):
    """Complete contrastive learning recommender system."""
    
    def __init__(self, config: ContrastiveConfig, num_users: int, num_items: int):
        super().__init__()
        self.config = config
        
        self.user_encoder = ContrastiveEncoder(config, config.embedding_dim)
        self.item_encoder = ContrastiveEncoder(config, config.word2vec_dim)
        self.user_embedding = nn.Embedding(num_users, config.embedding_dim)
        self.contrastive_loss = InfoNCELoss(config.temperature)
        self.negative_miner = HardNegativeMiner(
            config.num_negatives,
            strategy="semi-hard" if config.use_hard_negatives else "random"
        )
        self.word2vec = Word2VecEmbedder(config)
        self.item_embeddings_cache: Optional[torch.Tensor] = None
        
        nn.init.xavier_uniform_(self.user_embedding.weight)
    
    def encode_users(
        self,
        user_ids: torch.Tensor,
        user_history_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        user_emb = self.user_embedding(user_ids)
        if user_history_embeddings is not None:
            user_emb = user_emb + user_history_embeddings
        return self.user_encoder(user_emb)
    
    def encode_items(self, item_content_embeddings: torch.Tensor) -> torch.Tensor:
        return self.item_encoder(item_content_embeddings)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        positive_item_embeddings: torch.Tensor,
        negative_item_embeddings: Optional[torch.Tensor] = None,
        user_history_embeddings: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        user_rep = self.encode_users(user_ids, user_history_embeddings)
        pos_item_rep = self.encode_items(positive_item_embeddings)
        
        if negative_item_embeddings is not None:
            batch_size, num_neg, _ = negative_item_embeddings.shape
            neg_flat = negative_item_embeddings.view(-1, negative_item_embeddings.size(-1))
            neg_rep = self.encode_items(neg_flat)
            neg_rep = neg_rep.view(batch_size, num_neg, -1)
            loss = self.contrastive_loss(user_rep, pos_item_rep, neg_rep)
        else:
            loss = self.contrastive_loss(user_rep, pos_item_rep)
        
        return {"loss": loss, "user_embeddings": user_rep, "item_embeddings": pos_item_rep}
    
    @torch.no_grad()
    def recommend(
        self,
        user_id: int,
        candidate_item_embeddings: torch.Tensor,
        user_history_embedding: Optional[torch.Tensor] = None,
        n_recommendations: int = 10,
        exclude_indices: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        self.eval()
        device = next(self.parameters()).device
        
        user_tensor = torch.tensor([user_id], device=device)
        if user_history_embedding is not None:
            user_history_embedding = user_history_embedding.unsqueeze(0).to(device)
        
        user_rep = self.encode_users(user_tensor, user_history_embedding)
        item_reps = self.encode_items(candidate_item_embeddings.to(device))
        similarities = torch.mm(user_rep, item_reps.T).squeeze(0)
        
        if exclude_indices:
            for idx in exclude_indices:
                similarities[idx] = float('-inf')
        
        scores, indices = torch.topk(similarities, n_recommendations)
        return [(idx.item(), score.item()) for idx, score in zip(indices, scores)]


class SimCLRLoss(nn.Module):
    """SimCLR-style contrastive loss."""
    
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.size(0)
        device = z_i.device
        
        z = torch.cat([z_i, z_j], dim=0)
        sim = torch.mm(z, z.T) / self.temperature
        
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ]).to(device)
        
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim = sim.masked_fill(mask, float('-inf'))
        
        return F.cross_entropy(sim, labels)


if __name__ == "__main__":
    config = ContrastiveConfig(
        embedding_dim=256,
        projection_dim=128,
        temperature=0.07,
        num_negatives=10
    )
    
    model = ContrastiveLearningRecommender(config=config, num_users=50000, num_items=10000)
    print("Contrastive Learning Recommender initialized successfully!")
