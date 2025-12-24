"""
Deep Feature Interaction Models for CTR Prediction

Implements state-of-the-art deep learning models for feature interactions:
- DeepFM: Combines FM with DNN for explicit and implicit feature interactions
- DCN v2: Deep & Cross Network with mixture of experts
- AutoInt: Automatic Feature Interaction Learning via Self-Attention
- FiBiNET: Feature Importance and Bilinear feature Interaction Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np


@dataclass
class FeatureInteractionConfig:
    """Configuration for feature interaction models."""
    # Feature dimensions
    sparse_feature_dims: Dict[str, int] = field(default_factory=dict)  # feature_name -> vocab_size
    dense_feature_dim: int = 0
    embedding_dim: int = 16
    
    # Model architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    num_cross_layers: int = 3
    num_attention_heads: int = 4
    attention_dim: int = 64
    
    # Training
    dropout: float = 0.2
    use_batch_norm: bool = True
    
    # User/Item
    num_users: int = 50000
    num_items: int = 10000


class FMLayer(nn.Module):
    """
    Factorization Machine Layer
    
    Computes second-order feature interactions efficiently:
    y = sum_{i<j} <v_i, v_j> * x_i * x_j
    """
    
    def __init__(self, reduce_sum: bool = True):
        super().__init__()
        self.reduce_sum = reduce_sum
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Embedding tensor [batch, num_fields, embed_dim]
            
        Returns:
            FM output [batch, 1] or [batch, embed_dim]
        """
        # Sum of squares vs square of sums
        square_of_sum = torch.pow(x.sum(dim=1), 2)  # [batch, embed_dim]
        sum_of_square = torch.pow(x, 2).sum(dim=1)  # [batch, embed_dim]
        
        cross = 0.5 * (square_of_sum - sum_of_square)  # [batch, embed_dim]
        
        if self.reduce_sum:
            return cross.sum(dim=1, keepdim=True)  # [batch, 1]
        return cross


class DNNLayer(nn.Module):
    """Deep Neural Network Layer."""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int], 
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        output_activation: Optional[str] = None
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.dnn = nn.Sequential(*layers)
        self.output_activation = output_activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.dnn(x)
        if self.output_activation == "sigmoid":
            output = torch.sigmoid(output)
        elif self.output_activation == "relu":
            output = F.relu(output)
        return output


class DeepFM(nn.Module):
    """
    DeepFM: A Factorization-Machine based Neural Network
    
    Paper: https://arxiv.org/abs/1703.04247
    
    Combines:
    - FM for explicit feature interactions (low-order)
    - DNN for implicit feature interactions (high-order)
    """
    
    def __init__(self, config: FeatureInteractionConfig):
        super().__init__()
        self.config = config
        
        # Sparse feature embeddings
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size, config.embedding_dim)
            for name, vocab_size in config.sparse_feature_dims.items()
        })
        
        # First-order (linear) weights for sparse features
        self.linear_embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size, 1)
            for name, vocab_size in config.sparse_feature_dims.items()
        })
        
        # Linear layer for dense features
        if config.dense_feature_dim > 0:
            self.dense_linear = nn.Linear(config.dense_feature_dim, 1)
            self.dense_embedding = nn.Linear(config.dense_feature_dim, config.embedding_dim)
        
        # FM Layer
        self.fm = FMLayer(reduce_sum=True)
        
        # DNN
        num_fields = len(config.sparse_feature_dims)
        if config.dense_feature_dim > 0:
            num_fields += 1
        
        dnn_input_dim = num_fields * config.embedding_dim
        self.dnn = DNNLayer(
            dnn_input_dim, 
            config.hidden_dims, 
            config.dropout,
            config.use_batch_norm
        )
        
        # Final prediction layer
        self.output_layer = nn.Linear(config.hidden_dims[-1], 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for embedding in self.embeddings.values():
            nn.init.xavier_uniform_(embedding.weight)
        for embedding in self.linear_embeddings.values():
            nn.init.zeros_(embedding.weight)
    
    def forward(
        self,
        sparse_features: Dict[str, torch.Tensor],
        dense_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            sparse_features: Dict of feature_name -> tensor of indices
            dense_features: Optional dense feature tensor [batch, dense_dim]
            
        Returns:
            Prediction scores [batch]
        """
        # First-order term
        linear_output = sum(
            self.linear_embeddings[name](sparse_features[name])
            for name in sparse_features
        )  # [batch, 1]
        
        if dense_features is not None and hasattr(self, 'dense_linear'):
            linear_output = linear_output + self.dense_linear(dense_features)
        
        # Embedding lookup
        embeddings = [
            self.embeddings[name](sparse_features[name])  # [batch, embed_dim]
            for name in sparse_features
        ]
        
        if dense_features is not None and hasattr(self, 'dense_embedding'):
            embeddings.append(self.dense_embedding(dense_features))
        
        stacked_embeddings = torch.stack(embeddings, dim=1)  # [batch, num_fields, embed_dim]
        
        # FM term (second-order interactions)
        fm_output = self.fm(stacked_embeddings)  # [batch, 1]
        
        # DNN term (high-order interactions)
        dnn_input = stacked_embeddings.view(stacked_embeddings.size(0), -1)
        dnn_output = self.output_layer(self.dnn(dnn_input))  # [batch, 1]
        
        # Combine
        output = linear_output + fm_output + dnn_output
        return torch.sigmoid(output.squeeze(-1))


class CrossNetV2(nn.Module):
    """
    Cross Network V2 with Mixture of Experts
    
    More expressive cross layers with low-rank decomposition.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        num_layers: int = 3,
        low_rank: int = 64,
        num_experts: int = 4
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_experts = num_experts
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, low_rank, bias=False),
                    nn.ReLU(),
                    nn.Linear(low_rank, input_dim, bias=False)
                )
                for _ in range(num_experts)
            ])
            for _ in range(num_layers)
        ])
        
        # Gating networks
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, num_experts)
            for _ in range(num_layers)
        ])
        
        # Bias terms
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim))
            for _ in range(num_layers)
        ])
    
    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x0: Input features [batch, input_dim]
            
        Returns:
            Cross features [batch, input_dim]
        """
        x = x0
        
        for layer_idx in range(self.num_layers):
            # Expert outputs
            expert_outputs = torch.stack([
                expert(x) for expert in self.experts[layer_idx]
            ], dim=1)  # [batch, num_experts, input_dim]
            
            # Gating
            gate_weights = F.softmax(self.gates[layer_idx](x), dim=-1)  # [batch, num_experts]
            
            # Mixture
            mixed = torch.bmm(
                gate_weights.unsqueeze(1),
                expert_outputs
            ).squeeze(1)  # [batch, input_dim]
            
            # Cross interaction
            x = x0 * mixed + self.biases[layer_idx] + x
        
        return x


class DCNv2(nn.Module):
    """
    DCN V2: Improved Deep & Cross Network
    
    Paper: https://arxiv.org/abs/2008.13535
    
    Uses stacked structure with mixture of low-rank experts.
    """
    
    def __init__(self, config: FeatureInteractionConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size, config.embedding_dim)
            for name, vocab_size in config.sparse_feature_dims.items()
        })
        
        num_fields = len(config.sparse_feature_dims)
        if config.dense_feature_dim > 0:
            num_fields += 1
            self.dense_embedding = nn.Linear(config.dense_feature_dim, config.embedding_dim)
        
        input_dim = num_fields * config.embedding_dim
        
        # Cross network
        self.cross_net = CrossNetV2(
            input_dim,
            num_layers=config.num_cross_layers,
            low_rank=config.embedding_dim,
            num_experts=4
        )
        
        # Deep network
        self.deep_net = DNNLayer(
            input_dim,
            config.hidden_dims,
            config.dropout,
            config.use_batch_norm
        )
        
        # Output layer (stacked: cross + deep)
        self.output_layer = nn.Linear(input_dim + config.hidden_dims[-1], 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for embedding in self.embeddings.values():
            nn.init.xavier_uniform_(embedding.weight)
    
    def forward(
        self,
        sparse_features: Dict[str, torch.Tensor],
        dense_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with stacked cross and deep networks."""
        # Embeddings
        embeddings = [
            self.embeddings[name](sparse_features[name])
            for name in sparse_features
        ]
        
        if dense_features is not None and hasattr(self, 'dense_embedding'):
            embeddings.append(self.dense_embedding(dense_features))
        
        x = torch.cat(embeddings, dim=-1)  # [batch, input_dim]
        
        # Cross network
        cross_out = self.cross_net(x)  # [batch, input_dim]
        
        # Deep network
        deep_out = self.deep_net(x)  # [batch, hidden_dim]
        
        # Stacked combination
        stacked = torch.cat([cross_out, deep_out], dim=-1)
        output = self.output_layer(stacked)
        
        return torch.sigmoid(output.squeeze(-1))


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention for feature interactions."""
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int = 4,
        attention_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.W_q = nn.Linear(embed_dim, attention_dim)
        self.W_k = nn.Linear(embed_dim, attention_dim)
        self.W_v = nn.Linear(embed_dim, attention_dim)
        self.W_o = nn.Linear(attention_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_fields, embed_dim]
            
        Returns:
            Attention output [batch, num_fields, embed_dim]
        """
        batch_size, num_fields, _ = x.size()
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, num_fields, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, num_fields, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, num_fields, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Attention output
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_fields, -1)
        out = self.W_o(out)
        
        # Residual + LayerNorm
        return self.layer_norm(x + out)


class AutoInt(nn.Module):
    """
    AutoInt: Automatic Feature Interaction Learning via Self-Attention
    
    Paper: https://arxiv.org/abs/1810.11921
    
    Uses multi-head self-attention to automatically learn feature interactions.
    """
    
    def __init__(self, config: FeatureInteractionConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size, config.embedding_dim)
            for name, vocab_size in config.sparse_feature_dims.items()
        })
        
        if config.dense_feature_dim > 0:
            self.dense_embedding = nn.Linear(config.dense_feature_dim, config.embedding_dim)
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadSelfAttention(
                config.embedding_dim,
                config.num_attention_heads,
                config.attention_dim,
                config.dropout
            )
            for _ in range(3)
        ])
        
        # Output
        num_fields = len(config.sparse_feature_dims)
        if config.dense_feature_dim > 0:
            num_fields += 1
        
        self.output_layer = nn.Linear(num_fields * config.embedding_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for embedding in self.embeddings.values():
            nn.init.xavier_uniform_(embedding.weight)
    
    def forward(
        self,
        sparse_features: Dict[str, torch.Tensor],
        dense_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward with self-attention feature interactions."""
        # Embeddings
        embeddings = [
            self.embeddings[name](sparse_features[name])
            for name in sparse_features
        ]
        
        if dense_features is not None and hasattr(self, 'dense_embedding'):
            embeddings.append(self.dense_embedding(dense_features))
        
        x = torch.stack(embeddings, dim=1)  # [batch, num_fields, embed_dim]
        
        # Self-attention layers
        for attn_layer in self.attention_layers:
            x = attn_layer(x)
        
        # Flatten and output
        x = x.view(x.size(0), -1)
        output = self.output_layer(x)
        
        return torch.sigmoid(output.squeeze(-1))


class SENetLayer(nn.Module):
    """Squeeze-and-Excitation layer for feature importance."""
    
    def __init__(self, num_fields: int, reduction_ratio: int = 4):
        super().__init__()
        reduced_dim = max(1, num_fields // reduction_ratio)
        
        self.fc = nn.Sequential(
            nn.Linear(num_fields, reduced_dim),
            nn.ReLU(),
            nn.Linear(reduced_dim, num_fields),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_fields, embed_dim]
            
        Returns:
            Reweighted features [batch, num_fields, embed_dim]
        """
        # Squeeze: mean pooling over embedding dim
        z = x.mean(dim=-1)  # [batch, num_fields]
        
        # Excitation: learn importance weights
        weights = self.fc(z).unsqueeze(-1)  # [batch, num_fields, 1]
        
        return x * weights


class BilinearInteraction(nn.Module):
    """Bilinear feature interaction layer."""
    
    def __init__(self, num_fields: int, embed_dim: int, bilinear_type: str = "field_all"):
        super().__init__()
        self.bilinear_type = bilinear_type
        
        if bilinear_type == "field_all":
            self.W = nn.Parameter(torch.zeros(embed_dim, embed_dim))
        elif bilinear_type == "field_each":
            self.W = nn.Parameter(torch.zeros(num_fields, embed_dim, embed_dim))
        elif bilinear_type == "field_interaction":
            num_interactions = num_fields * (num_fields - 1) // 2
            self.W = nn.Parameter(torch.zeros(num_interactions, embed_dim, embed_dim))
        
        nn.init.xavier_uniform_(self.W)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_fields, embed_dim]
            
        Returns:
            Bilinear interactions [batch, num_interactions, embed_dim]
        """
        batch_size, num_fields, embed_dim = x.size()
        interactions = []
        
        idx = 0
        for i in range(num_fields):
            for j in range(i + 1, num_fields):
                if self.bilinear_type == "field_all":
                    W = self.W
                elif self.bilinear_type == "field_each":
                    W = self.W[i]
                else:
                    W = self.W[idx]
                    idx += 1
                
                # x_i * W * x_j
                vi = x[:, i, :]  # [batch, embed_dim]
                vj = x[:, j, :]
                
                interaction = vi.unsqueeze(1) @ W @ vj.unsqueeze(-1)
                interactions.append(interaction.squeeze())
        
        return torch.stack(interactions, dim=1)  # [batch, num_interactions]


class FiBiNET(nn.Module):
    """
    FiBiNET: Feature Importance and Bilinear Feature Interaction
    
    Paper: https://arxiv.org/abs/1905.09433
    
    Combines SE-Net for feature importance with bilinear interactions.
    """
    
    def __init__(self, config: FeatureInteractionConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size, config.embedding_dim)
            for name, vocab_size in config.sparse_feature_dims.items()
        })
        
        num_fields = len(config.sparse_feature_dims)
        
        # SE-Net for feature importance
        self.senet = SENetLayer(num_fields)
        
        # Bilinear interactions
        self.bilinear = BilinearInteraction(num_fields, config.embedding_dim, "field_all")
        self.bilinear_senet = BilinearInteraction(num_fields, config.embedding_dim, "field_all")
        
        # DNN
        num_interactions = num_fields * (num_fields - 1) // 2
        dnn_input = num_interactions * 2  # Original + SE-Net weighted
        
        self.dnn = DNNLayer(
            dnn_input,
            config.hidden_dims,
            config.dropout,
            config.use_batch_norm
        )
        
        self.output_layer = nn.Linear(config.hidden_dims[-1], 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for embedding in self.embeddings.values():
            nn.init.xavier_uniform_(embedding.weight)
    
    def forward(
        self,
        sparse_features: Dict[str, torch.Tensor],
        dense_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward with SE-Net and bilinear interactions."""
        # Embeddings
        embeddings = torch.stack([
            self.embeddings[name](sparse_features[name])
            for name in sparse_features
        ], dim=1)  # [batch, num_fields, embed_dim]
        
        # SE-Net reweighting
        embeddings_senet = self.senet(embeddings)
        
        # Bilinear interactions
        bilinear_out = self.bilinear(embeddings)  # [batch, num_interactions]
        bilinear_senet_out = self.bilinear_senet(embeddings_senet)
        
        # Concatenate
        combined = torch.cat([bilinear_out, bilinear_senet_out], dim=-1)
        
        # DNN
        dnn_out = self.dnn(combined)
        output = self.output_layer(dnn_out)
        
        return torch.sigmoid(output.squeeze(-1))


class FeatureInteractionRecommender:
    """High-level interface for feature interaction models."""
    
    def __init__(
        self,
        config: FeatureInteractionConfig,
        model_type: str = "deepfm",  # deepfm, dcnv2, autoint, fibinet
        device: str = "cuda"
    ):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        if model_type == "deepfm":
            self.model = DeepFM(config)
        elif model_type == "dcnv2":
            self.model = DCNv2(config)
        elif model_type == "autoint":
            self.model = AutoInt(config)
        elif model_type == "fibinet":
            self.model = FiBiNET(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = self.model.to(self.device)
        self.model_type = model_type
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module = nn.BCELoss()
    ) -> float:
        """Execute one training step."""
        self.model.train()
        optimizer.zero_grad()
        
        sparse_features = {
            k: v.to(self.device) for k, v in batch.items()
            if k not in ["label", "dense_features"]
        }
        
        dense_features = batch.get("dense_features")
        if dense_features is not None:
            dense_features = dense_features.to(self.device)
        
        labels = batch["label"].to(self.device).float()
        
        predictions = self.model(sparse_features, dense_features)
        loss = criterion(predictions, labels)
        
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def predict(
        self,
        sparse_features: Dict[str, torch.Tensor],
        dense_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get predictions."""
        self.model.eval()
        
        sparse_features = {k: v.to(self.device) for k, v in sparse_features.items()}
        if dense_features is not None:
            dense_features = dense_features.to(self.device)
        
        return self.model(sparse_features, dense_features)
    
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
    # Example usage
    config = FeatureInteractionConfig(
        sparse_feature_dims={
            "user_id": 10000,
            "item_id": 5000,
            "category": 100,
            "hour": 24
        },
        dense_feature_dim=10,
        embedding_dim=16,
        hidden_dims=[128, 64]
    )
    
    model = DeepFM(config)
    
    # Sample forward pass
    batch = {
        "user_id": torch.randint(0, 10000, (32,)),
        "item_id": torch.randint(0, 5000, (32,)),
        "category": torch.randint(0, 100, (32,)),
        "hour": torch.randint(0, 24, (32,))
    }
    dense = torch.randn(32, 10)
    
    output = model(batch, dense)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
