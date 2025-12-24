"""
Graph Neural Network Models for Recommendations

Implements state-of-the-art GNN-based recommendation models including:
- LightGCN: Simplified graph convolution for collaborative filtering
- NGCF: Neural Graph Collaborative Filtering
- GraphSAGE-based recommendations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import scipy.sparse as sp


@dataclass
class GNNConfig:
    """Configuration for GNN-based recommender."""
    num_users: int
    num_items: int
    embedding_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.1
    aggregation: str = "mean"  # mean, sum, concat
    normalize: bool = True
    add_self_loops: bool = False


class LightGCN(nn.Module):
    """
    LightGCN: Simplifying and Powering Graph Convolution Network
    
    Paper: https://arxiv.org/abs/2002.02126
    
    Key insight: Linear propagation without feature transformation
    achieves better performance for collaborative filtering.
    """
    
    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config
        self.num_users = config.num_users
        self.num_items = config.num_items
        self.num_layers = config.num_layers
        
        # Embedding layers
        self.user_embedding = nn.Embedding(config.num_users, config.embedding_dim)
        self.item_embedding = nn.Embedding(config.num_items, config.embedding_dim)
        
        # Layer combination weights (learnable)
        self.layer_weights = nn.Parameter(
            torch.ones(config.num_layers + 1) / (config.num_layers + 1)
        )
        
        self._init_weights()
        
        # Graph will be set during training
        self.norm_adj: Optional[SparseTensor] = None
    
    def _init_weights(self):
        """Initialize embeddings."""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def build_graph(self, interactions: List[Tuple[int, int]]):
        """
        Build normalized adjacency matrix from interactions.
        
        Args:
            interactions: List of (user_id, item_id) tuples
        """
        n_users = self.num_users
        n_items = self.num_items
        
        # Build adjacency matrix
        rows, cols = zip(*interactions)
        rows = torch.tensor(rows, dtype=torch.long)
        cols = torch.tensor(cols, dtype=torch.long) + n_users  # Offset item IDs
        
        # Create bipartite graph edges (both directions)
        edge_index = torch.stack([
            torch.cat([rows, cols]),
            torch.cat([cols, rows])
        ])
        
        # Compute degree for normalization
        deg = torch.zeros(n_users + n_items)
        deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1)))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Normalize: D^(-1/2) * A * D^(-1/2)
        edge_weight = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]
        
        self.norm_adj = SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            value=edge_weight,
            sparse_sizes=(n_users + n_items, n_users + n_items)
        )
    
    def propagate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform light graph convolution.
        
        Returns:
            Tuple of (user_embeddings, item_embeddings) after propagation
        """
        # Concatenate user and item embeddings
        all_embeddings = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)
        
        embeddings_list = [all_embeddings]
        
        # Layer-wise propagation
        for layer in range(self.num_layers):
            all_embeddings = self.norm_adj @ all_embeddings
            embeddings_list.append(all_embeddings)
        
        # Combine layers with learned weights
        weights = F.softmax(self.layer_weights, dim=0)
        final_embeddings = sum(
            w * emb for w, emb in zip(weights, embeddings_list)
        )
        
        user_emb = final_embeddings[:self.num_users]
        item_emb = final_embeddings[self.num_users:]
        
        return user_emb, item_emb
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute prediction scores.
        
        Args:
            user_ids: User ID tensor [batch_size]
            item_ids: Item ID tensor [batch_size]
            
        Returns:
            Prediction scores [batch_size]
        """
        user_emb, item_emb = self.propagate()
        
        user_vectors = user_emb[user_ids]
        item_vectors = item_emb[item_ids]
        
        scores = (user_vectors * item_vectors).sum(dim=1)
        return scores
    
    def compute_bpr_loss(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
        reg_weight: float = 1e-4
    ) -> torch.Tensor:
        """
        Compute BPR loss with L2 regularization.
        """
        user_emb, item_emb = self.propagate()
        
        user_vectors = user_emb[user_ids]
        pos_vectors = item_emb[pos_item_ids]
        neg_vectors = item_emb[neg_item_ids]
        
        pos_scores = (user_vectors * pos_vectors).sum(dim=1)
        neg_scores = (user_vectors * neg_vectors).sum(dim=1)
        
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        # L2 regularization on embeddings
        reg_loss = reg_weight * (
            self.user_embedding.weight[user_ids].norm(2).pow(2) +
            self.item_embedding.weight[pos_item_ids].norm(2).pow(2) +
            self.item_embedding.weight[neg_item_ids].norm(2).pow(2)
        ) / len(user_ids)
        
        return bpr_loss + reg_loss
    
    @torch.no_grad()
    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_items: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """Generate top-k recommendations."""
        self.eval()
        
        user_emb, item_emb = self.propagate()
        user_vector = user_emb[user_id]
        
        scores = torch.matmul(item_emb, user_vector)
        
        if exclude_items:
            scores[exclude_items] = float('-inf')
        
        top_scores, top_indices = torch.topk(scores, k)
        
        return [
            (idx.item(), score.item())
            for idx, score in zip(top_indices, top_scores)
        ]


class NGCFConv(nn.Module):
    """
    NGCF Convolution Layer
    
    Implements message passing with feature transformation.
    """
    
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.W1 = nn.Linear(in_dim, out_dim, bias=True)
        self.W2 = nn.Linear(in_dim, out_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(
        self,
        embeddings: torch.Tensor,
        adj: SparseTensor
    ) -> torch.Tensor:
        """
        NGCF message passing.
        
        Args:
            embeddings: Node embeddings [num_nodes, in_dim]
            adj: Normalized adjacency matrix
            
        Returns:
            Updated embeddings [num_nodes, out_dim]
        """
        # Self transformation
        ego_emb = self.W1(embeddings)
        
        # Neighbor aggregation with interaction
        neighbor_emb = adj @ embeddings
        interaction_emb = neighbor_emb * embeddings
        side_emb = self.W2(neighbor_emb + interaction_emb)
        
        output = ego_emb + side_emb
        output = self.leaky_relu(output)
        output = self.dropout(output)
        
        return F.normalize(output, p=2, dim=1)


class NGCF(nn.Module):
    """
    Neural Graph Collaborative Filtering
    
    Paper: https://arxiv.org/abs/1905.08108
    """
    
    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config
        
        # Initial embeddings
        self.user_embedding = nn.Embedding(config.num_users, config.embedding_dim)
        self.item_embedding = nn.Embedding(config.num_items, config.embedding_dim)
        
        # NGCF layers
        self.layers = nn.ModuleList()
        dims = [config.embedding_dim] * (config.num_layers + 1)
        
        for i in range(config.num_layers):
            self.layers.append(NGCFConv(dims[i], dims[i+1], config.dropout))
        
        self.norm_adj: Optional[SparseTensor] = None
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def build_graph(self, interactions: List[Tuple[int, int]]):
        """Build adjacency matrix."""
        # Same as LightGCN
        n_users = self.config.num_users
        n_items = self.config.num_items
        
        rows, cols = zip(*interactions)
        rows = torch.tensor(rows, dtype=torch.long)
        cols = torch.tensor(cols, dtype=torch.long) + n_users
        
        edge_index = torch.stack([
            torch.cat([rows, cols]),
            torch.cat([cols, rows])
        ])
        
        deg = torch.zeros(n_users + n_items)
        deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1)))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        edge_weight = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]
        
        self.norm_adj = SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            value=edge_weight,
            sparse_sizes=(n_users + n_items, n_users + n_items)
        )
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute prediction scores."""
        all_emb = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)
        
        emb_list = [all_emb]
        
        for layer in self.layers:
            all_emb = layer(all_emb, self.norm_adj)
            emb_list.append(all_emb)
        
        # Concatenate all layer outputs
        final_emb = torch.cat(emb_list, dim=1)
        
        user_emb = final_emb[:self.config.num_users]
        item_emb = final_emb[self.config.num_users:]
        
        user_vectors = user_emb[user_ids]
        item_vectors = item_emb[item_ids]
        
        return (user_vectors * item_vectors).sum(dim=1)


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer for user-item interactions.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.concat = concat
        
        self.W = nn.Linear(in_features, out_features * num_heads, bias=False)
        self.a = nn.Parameter(torch.zeros(num_heads, 2 * out_features))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with attention.
        
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Updated features [num_nodes, out_features * num_heads] or [num_nodes, out_features]
        """
        num_nodes = x.size(0)
        
        # Linear transformation
        h = self.W(x).view(num_nodes, self.num_heads, self.out_features)
        
        # Compute attention coefficients
        src, dst = edge_index
        
        h_src = h[src]  # [num_edges, num_heads, out_features]
        h_dst = h[dst]
        
        # Attention mechanism
        edge_features = torch.cat([h_src, h_dst], dim=-1)  # [num_edges, num_heads, 2*out_features]
        attention = (edge_features * self.a).sum(dim=-1)  # [num_edges, num_heads]
        attention = self.leaky_relu(attention)
        
        # Softmax over neighbors
        attention = torch.exp(attention - attention.max())
        attention_sum = torch.zeros(num_nodes, self.num_heads, device=x.device)
        attention_sum.scatter_add_(0, dst.unsqueeze(1).expand(-1, self.num_heads), attention)
        attention = attention / (attention_sum[dst] + 1e-8)
        attention = self.dropout(attention)
        
        # Aggregate
        out = torch.zeros(num_nodes, self.num_heads, self.out_features, device=x.device)
        out.scatter_add_(
            0,
            dst.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, self.out_features),
            attention.unsqueeze(-1) * h_src
        )
        
        if self.concat:
            return out.view(num_nodes, -1)
        else:
            return out.mean(dim=1)


class GATRecommender(nn.Module):
    """
    Graph Attention Network for Recommendations
    
    Uses attention mechanism to learn importance of different neighbors.
    """
    
    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config
        
        self.user_embedding = nn.Embedding(config.num_users, config.embedding_dim)
        self.item_embedding = nn.Embedding(config.num_items, config.embedding_dim)
        
        # GAT layers
        self.gat1 = GraphAttentionLayer(
            config.embedding_dim, config.embedding_dim, 
            num_heads=4, dropout=config.dropout, concat=True
        )
        self.gat2 = GraphAttentionLayer(
            config.embedding_dim * 4, config.embedding_dim,
            num_heads=1, dropout=config.dropout, concat=False
        )
        
        self.edge_index: Optional[torch.Tensor] = None
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def build_graph(self, interactions: List[Tuple[int, int]]):
        """Build edge index."""
        n_users = self.config.num_users
        
        rows, cols = zip(*interactions)
        rows = torch.tensor(rows, dtype=torch.long)
        cols = torch.tensor(cols, dtype=torch.long) + n_users
        
        self.edge_index = torch.stack([
            torch.cat([rows, cols]),
            torch.cat([cols, rows])
        ])
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute scores."""
        all_emb = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)
        
        # GAT propagation
        h = self.gat1(all_emb, self.edge_index)
        h = F.elu(h)
        h = self.gat2(h, self.edge_index)
        
        # Residual connection
        h = h + all_emb
        
        user_emb = h[:self.config.num_users]
        item_emb = h[self.config.num_users:]
        
        return (user_emb[user_ids] * item_emb[item_ids]).sum(dim=1)


class GNNRecommender:
    """
    High-level interface for GNN-based recommendations.
    """
    
    def __init__(
        self,
        config: GNNConfig,
        model_type: str = "lightgcn",  # lightgcn, ngcf, gat
        device: str = "cuda"
    ):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        if model_type == "lightgcn":
            self.model = LightGCN(config)
        elif model_type == "ngcf":
            self.model = NGCF(config)
        elif model_type == "gat":
            self.model = GATRecommender(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = self.model.to(self.device)
        self.model_type = model_type
    
    def fit(
        self,
        interactions: List[Tuple[int, int]],
        epochs: int = 100,
        batch_size: int = 2048,
        lr: float = 0.001,
        neg_samples: int = 1
    ):
        """Train the GNN model."""
        self.model.build_graph(interactions)
        
        if hasattr(self.model, 'norm_adj') and self.model.norm_adj is not None:
            self.model.norm_adj = self.model.norm_adj.to(self.device)
        if hasattr(self.model, 'edge_index') and self.model.edge_index is not None:
            self.model.edge_index = self.model.edge_index.to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Build item set per user for negative sampling
        user_items = {}
        for u, i in interactions:
            if u not in user_items:
                user_items[u] = set()
            user_items[u].add(i)
        
        all_items = set(range(self.config.num_items))
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            # Shuffle interactions
            np.random.shuffle(interactions)
            
            for i in range(0, len(interactions), batch_size):
                batch = interactions[i:i+batch_size]
                
                users = torch.tensor([x[0] for x in batch], device=self.device)
                pos_items = torch.tensor([x[1] for x in batch], device=self.device)
                
                # Negative sampling
                neg_items = []
                for u, _ in batch:
                    neg_pool = list(all_items - user_items.get(u, set()))
                    neg_items.append(np.random.choice(neg_pool))
                neg_items = torch.tensor(neg_items, device=self.device)
                
                optimizer.zero_grad()
                
                if hasattr(self.model, 'compute_bpr_loss'):
                    loss = self.model.compute_bpr_loss(users, pos_items, neg_items)
                else:
                    pos_scores = self.model(users, pos_items)
                    neg_scores = self.model(users, neg_items)
                    loss = -F.logsigmoid(pos_scores - neg_scores).mean()
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
    
    @torch.no_grad()
    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_items: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """Generate recommendations."""
        return self.model.recommend(user_id, k, exclude_items)
    
    def save(self, path: str):
        """Save model."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "model_type": self.model_type
        }, path)
    
    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])


if __name__ == "__main__":
    # Example usage
    config = GNNConfig(
        num_users=1000,
        num_items=5000,
        embedding_dim=64,
        num_layers=3
    )
    
    recommender = GNNRecommender(config, model_type="lightgcn", device="cpu")
    
    # Sample interactions
    interactions = [(i % 1000, i % 5000) for i in range(10000)]
    
    print("Training LightGCN...")
    recommender.fit(interactions, epochs=20)
    
    # Get recommendations
    recs = recommender.recommend(user_id=0, k=5)
    print(f"Recommendations for user 0: {recs}")
