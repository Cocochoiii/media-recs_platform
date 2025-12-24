"""
Knowledge Graph Enhanced Recommendations

Implements knowledge graph embedding methods for capturing complex relationships:
- TransE, TransR: Translation-based embeddings
- RotatE: Rotation-based embeddings
- KGAT: Knowledge Graph Attention Network
- RippleNet: Propagating user preferences over knowledge graph
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict


@dataclass
class KGConfig:
    """Configuration for Knowledge Graph models."""
    num_entities: int = 50000
    num_relations: int = 100
    num_users: int = 10000
    num_items: int = 5000
    embedding_dim: int = 64
    
    # Model specific
    margin: float = 1.0  # TransE margin
    num_hops: int = 2  # RippleNet hops
    num_neighbors: int = 8  # Neighbors per hop
    
    # Attention
    num_heads: int = 4
    dropout: float = 0.1
    
    # Training
    negative_samples: int = 5


class TransE(nn.Module):
    """
    TransE: Translating Embeddings for Modeling Multi-relational Data
    
    Paper: https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data
    
    Score: ||h + r - t||
    """
    
    def __init__(self, config: KGConfig):
        super().__init__()
        self.config = config
        
        self.entity_embedding = nn.Embedding(config.num_entities, config.embedding_dim)
        self.relation_embedding = nn.Embedding(config.num_relations, config.embedding_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        
        # Normalize relation embeddings
        with torch.no_grad():
            self.relation_embedding.weight.data = F.normalize(
                self.relation_embedding.weight.data, p=2, dim=1
            )
    
    def forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute TransE scores.
        
        Args:
            heads: Head entity IDs [batch]
            relations: Relation IDs [batch]
            tails: Tail entity IDs [batch]
            
        Returns:
            Scores (lower is better for positive triples) [batch]
        """
        h = self.entity_embedding(heads)
        r = self.relation_embedding(relations)
        t = self.entity_embedding(tails)
        
        # Normalize entities
        h = F.normalize(h, p=2, dim=-1)
        t = F.normalize(t, p=2, dim=-1)
        
        # Score: ||h + r - t||
        score = torch.norm(h + r - t, p=2, dim=-1)
        
        return score
    
    def compute_loss(
        self,
        pos_heads: torch.Tensor,
        pos_relations: torch.Tensor,
        pos_tails: torch.Tensor,
        neg_tails: torch.Tensor
    ) -> torch.Tensor:
        """Compute margin ranking loss."""
        pos_score = self.forward(pos_heads, pos_relations, pos_tails)
        neg_score = self.forward(
            pos_heads.unsqueeze(1).expand_as(neg_tails),
            pos_relations.unsqueeze(1).expand_as(neg_tails),
            neg_tails
        )
        
        # Margin loss
        loss = torch.relu(self.config.margin + pos_score.unsqueeze(1) - neg_score)
        return loss.mean()
    
    def get_entity_embedding(self, entity_id: int) -> np.ndarray:
        """Get embedding for an entity."""
        with torch.no_grad():
            emb = self.entity_embedding.weight[entity_id]
            return F.normalize(emb, p=2, dim=-1).cpu().numpy()


class RotatE(nn.Module):
    """
    RotatE: Knowledge Graph Embedding by Relational Rotation
    
    Paper: https://arxiv.org/abs/1902.10197
    
    Models relations as rotations in complex space.
    """
    
    def __init__(self, config: KGConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        
        # Complex embeddings (real + imaginary)
        self.entity_embedding = nn.Embedding(
            config.num_entities, config.embedding_dim * 2
        )
        self.relation_embedding = nn.Embedding(
            config.num_relations, config.embedding_dim
        )
        
        self.epsilon = 2.0
        self.gamma = nn.Parameter(torch.Tensor([config.margin]))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.uniform_(
            self.entity_embedding.weight,
            -self.epsilon / self.embedding_dim,
            self.epsilon / self.embedding_dim
        )
        nn.init.uniform_(
            self.relation_embedding.weight,
            -self.epsilon / self.embedding_dim,
            self.epsilon / self.embedding_dim
        )
    
    def forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor
    ) -> torch.Tensor:
        """Compute RotatE scores."""
        # Get embeddings
        head_emb = self.entity_embedding(heads)
        tail_emb = self.entity_embedding(tails)
        rel_emb = self.relation_embedding(relations)
        
        # Split into real and imaginary parts
        h_re, h_im = head_emb.chunk(2, dim=-1)
        t_re, t_im = tail_emb.chunk(2, dim=-1)
        
        # Relation as rotation (phase)
        phase = rel_emb / (self.embedding_dim / np.pi)
        r_re = torch.cos(phase)
        r_im = torch.sin(phase)
        
        # Rotation: h * r
        rot_re = h_re * r_re - h_im * r_im
        rot_im = h_re * r_im + h_im * r_re
        
        # Score: ||h * r - t||
        score_re = rot_re - t_re
        score_im = rot_im - t_im
        
        score = torch.sqrt(score_re ** 2 + score_im ** 2 + 1e-8).sum(dim=-1)
        
        return self.gamma - score


class KGATLayer(nn.Module):
    """
    Knowledge Graph Attention Layer
    
    Aggregates neighbor information with attention.
    """
    
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.W_r = nn.Linear(in_dim, out_dim, bias=False)
        self.attention = nn.Linear(3 * self.head_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(
        self,
        entity_emb: torch.Tensor,
        relation_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            entity_emb: [num_entities, in_dim]
            relation_emb: [num_relations, in_dim]
            edge_index: [2, num_edges]
            edge_type: [num_edges]
            
        Returns:
            Updated entity embeddings [num_entities, out_dim]
        """
        num_entities = entity_emb.size(0)
        
        # Transform
        h = self.W(entity_emb)  # [num_entities, out_dim]
        r = self.W_r(relation_emb)  # [num_relations, out_dim]
        
        # Reshape for multi-head
        h = h.view(num_entities, self.num_heads, self.head_dim)
        
        # Get source and target embeddings
        src, dst = edge_index
        h_src = h[src]  # [num_edges, num_heads, head_dim]
        h_dst = h[dst]
        r_edge = r[edge_type].view(-1, self.num_heads, self.head_dim)
        
        # Attention
        attn_input = torch.cat([h_src, r_edge, h_dst], dim=-1)
        attn_scores = self.leaky_relu(self.attention(attn_input)).squeeze(-1)
        
        # Softmax over neighbors
        attn_scores = torch.exp(attn_scores - attn_scores.max())
        attn_sum = torch.zeros(num_entities, self.num_heads, device=h.device)
        attn_sum.scatter_add_(0, dst.unsqueeze(1).expand(-1, self.num_heads), attn_scores)
        attn_weights = attn_scores / (attn_sum[dst] + 1e-8)
        attn_weights = self.dropout(attn_weights)
        
        # Aggregate
        msg = attn_weights.unsqueeze(-1) * h_src
        out = torch.zeros_like(h)
        out.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, self.head_dim), msg)
        
        return out.view(num_entities, -1)


class KGAT(nn.Module):
    """
    KGAT: Knowledge Graph Attention Network
    
    Paper: https://arxiv.org/abs/1905.07854
    
    Combines GNN with knowledge graph for recommendations.
    """
    
    def __init__(self, config: KGConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.entity_embedding = nn.Embedding(config.num_entities, config.embedding_dim)
        self.relation_embedding = nn.Embedding(config.num_relations, config.embedding_dim)
        self.user_embedding = nn.Embedding(config.num_users, config.embedding_dim)
        
        # KGAT layers
        self.layers = nn.ModuleList([
            KGATLayer(config.embedding_dim, config.embedding_dim, config.num_heads, config.dropout)
            for _ in range(2)
        ])
        
        self.fc = nn.Linear(config.embedding_dim * 3, config.embedding_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        nn.init.xavier_uniform_(self.user_embedding.weight)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor
    ) -> torch.Tensor:
        """Compute user-item scores."""
        entity_emb = self.entity_embedding.weight
        relation_emb = self.relation_embedding.weight
        
        # KGAT propagation
        emb_list = [entity_emb]
        for layer in self.layers:
            entity_emb = layer(entity_emb, relation_emb, edge_index, edge_type)
            entity_emb = F.relu(entity_emb)
            emb_list.append(entity_emb)
        
        # Concatenate layer outputs
        entity_emb = torch.cat(emb_list, dim=-1)
        entity_emb = self.fc(entity_emb)
        
        # Get user and item embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = entity_emb[item_ids]  # Items are entities
        
        # Score
        return (user_emb * item_emb).sum(dim=-1)


class RippleNet(nn.Module):
    """
    RippleNet: Propagating User Preferences over Knowledge Graph
    
    Paper: https://arxiv.org/abs/1803.03467
    
    Propagates user preferences through KG connections.
    """
    
    def __init__(self, config: KGConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.entity_embedding = nn.Embedding(config.num_entities, config.embedding_dim)
        self.relation_embedding = nn.Embedding(config.num_relations, config.embedding_dim)
        
        # Transform matrices for each hop
        self.transforms = nn.ModuleList([
            nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
            for _ in range(config.num_hops)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
    
    def get_ripple_set(
        self,
        user_history: List[int],
        kg_dict: Dict[int, List[Tuple[int, int]]],  # entity -> [(relation, tail)]
        num_hops: int
    ) -> List[List[Tuple[int, int, int]]]:
        """
        Get ripple set (h, r, t) for each hop.
        
        Args:
            user_history: List of item IDs user interacted with
            kg_dict: Knowledge graph adjacency dict
            num_hops: Number of propagation hops
            
        Returns:
            List of ripple sets for each hop
        """
        ripple_sets = []
        seeds = set(user_history)
        
        for hop in range(num_hops):
            hop_set = []
            
            for entity in seeds:
                if entity in kg_dict:
                    neighbors = kg_dict[entity][:self.config.num_neighbors]
                    for relation, tail in neighbors:
                        hop_set.append((entity, relation, tail))
            
            if not hop_set:
                # Pad with dummy values
                hop_set = [(0, 0, 0)] * self.config.num_neighbors
            
            ripple_sets.append(hop_set)
            
            # Update seeds for next hop
            seeds = set(t for _, _, t in hop_set)
        
        return ripple_sets
    
    def forward(
        self,
        item_ids: torch.Tensor,
        ripple_sets: List[torch.Tensor]  # [(batch, num_triples, 3)]
    ) -> torch.Tensor:
        """
        Compute preference scores.
        
        Args:
            item_ids: [batch]
            ripple_sets: Ripple set for each hop
            
        Returns:
            Scores [batch]
        """
        item_emb = self.entity_embedding(item_ids)  # [batch, dim]
        
        for hop, (ripple_h, ripple_r, ripple_t) in enumerate(ripple_sets):
            # Get embeddings
            h_emb = self.entity_embedding(ripple_h)  # [batch, num_triples, dim]
            r_emb = self.relation_embedding(ripple_r)
            t_emb = self.entity_embedding(ripple_t)
            
            # Compute attention: softmax(v^T * R * h)
            # Simplified: attention = softmax(item * (h * r))
            hr = h_emb * r_emb  # [batch, num_triples, dim]
            attention = torch.bmm(hr, item_emb.unsqueeze(-1)).squeeze(-1)  # [batch, num_triples]
            attention = F.softmax(attention, dim=-1)
            
            # Aggregate
            o = torch.bmm(attention.unsqueeze(1), t_emb).squeeze(1)  # [batch, dim]
            
            # Update item embedding
            item_emb = self.transforms[hop](item_emb + o)
        
        # Final score
        return item_emb.sum(dim=-1)


class KGEnhancedRecommender:
    """
    High-level interface for KG-enhanced recommendations.
    """
    
    def __init__(
        self,
        config: KGConfig,
        model_type: str = "kgat",  # transe, rotate, kgat, ripplenet
        device: str = "cuda"
    ):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        if model_type == "transe":
            self.model = TransE(config)
        elif model_type == "rotate":
            self.model = RotatE(config)
        elif model_type == "kgat":
            self.model = KGAT(config)
        elif model_type == "ripplenet":
            self.model = RippleNet(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = self.model.to(self.device)
        self.model_type = model_type
        
        # Knowledge graph storage
        self.kg_dict: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    
    def load_kg(self, triples: List[Tuple[int, int, int]]):
        """Load knowledge graph triples (head, relation, tail)."""
        self.kg_dict.clear()
        for h, r, t in triples:
            self.kg_dict[h].append((r, t))
    
    def get_item_embedding(self, item_id: int) -> np.ndarray:
        """Get embedding for an item."""
        with torch.no_grad():
            if hasattr(self.model, 'entity_embedding'):
                emb = self.model.entity_embedding.weight[item_id]
            else:
                emb = self.model.get_entity_embedding(item_id)
            return emb.cpu().numpy()
    
    def find_similar_items(
        self,
        item_id: int,
        k: int = 10,
        exclude_ids: Optional[Set[int]] = None
    ) -> List[Tuple[int, float]]:
        """Find similar items based on KG embeddings."""
        item_emb = self.get_item_embedding(item_id)
        
        similarities = []
        with torch.no_grad():
            all_embs = self.model.entity_embedding.weight.cpu().numpy()
            
            for i in range(min(self.config.num_items, len(all_embs))):
                if i == item_id or (exclude_ids and i in exclude_ids):
                    continue
                
                sim = np.dot(item_emb, all_embs[i]) / (
                    np.linalg.norm(item_emb) * np.linalg.norm(all_embs[i]) + 1e-8
                )
                similarities.append((i, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
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
    config = KGConfig(
        num_entities=10000,
        num_relations=50,
        num_users=1000,
        num_items=5000,
        embedding_dim=64
    )
    
    # Test TransE
    transe = TransE(config)
    heads = torch.randint(0, 10000, (32,))
    relations = torch.randint(0, 50, (32,))
    tails = torch.randint(0, 10000, (32,))
    
    scores = transe(heads, relations, tails)
    print(f"TransE scores shape: {scores.shape}")
    
    # Test RotatE
    rotate = RotatE(config)
    scores = rotate(heads, relations, tails)
    print(f"RotatE scores shape: {scores.shape}")
