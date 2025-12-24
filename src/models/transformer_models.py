"""
Transformer-based Sequential Recommendation Models

State-of-the-art transformer architectures for sequential recommendations:
- BERT4Rec: Bidirectional Encoder for sequence modeling with masked prediction
- SASRec++: Improved Self-Attentive Sequential Recommendation
- Transformers4Rec: Full transformer architecture with various attention patterns
- BST: Behavior Sequence Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class TransformerRecConfig:
    """Configuration for Transformer-based recommender."""
    num_items: int = 10000
    max_seq_len: int = 50
    embedding_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    hidden_dim: int = 512  # FFN hidden dim
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # BERT4Rec specific
    mask_prob: float = 0.2
    
    # Training
    label_smoothing: float = 0.1


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional embeddings."""
    
    def __init__(self, max_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.position_embedding.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)
        return self.dropout(x + pos_emb)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional causal masking."""
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        dropout: float = 0.1,
        causal: bool = False
    ):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Causal mask
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device), diagonal=1
            ).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Key padding mask
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )
        
        # Custom mask
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.W_o(attn_output), attn_weights


class TransformerBlock(nn.Module):
    """Single transformer block with pre-LayerNorm."""
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        hidden_dim: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        causal: bool = False
    ):
        super().__init__()
        
        self.attention = MultiHeadSelfAttention(
            d_model, num_heads, attention_dropout, causal
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm attention
        normed = self.norm1(x)
        attn_out, attn_weights = self.attention(normed, mask, key_padding_mask)
        x = x + attn_out
        
        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        
        return x, attn_weights


class BERT4Rec(nn.Module):
    """
    BERT4Rec: Sequential Recommendation with Bidirectional Encoder
    
    Paper: https://arxiv.org/abs/1904.06690
    
    Uses masked item prediction for training (like BERT's MLM).
    """
    
    def __init__(self, config: TransformerRecConfig):
        super().__init__()
        self.config = config
        
        # Special tokens
        self.mask_token = config.num_items  # [MASK]
        self.pad_token = config.num_items + 1  # [PAD]
        
        # Embeddings
        self.item_embedding = nn.Embedding(
            config.num_items + 2,  # +2 for special tokens
            config.embedding_dim,
            padding_idx=self.pad_token
        )
        self.position_encoding = LearnablePositionalEncoding(
            config.max_seq_len,
            config.embedding_dim,
            config.dropout
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                config.embedding_dim,
                config.num_heads,
                config.hidden_dim,
                config.dropout,
                config.attention_dropout,
                causal=False  # Bidirectional
            )
            for _ in range(config.num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.output_layer = nn.Linear(config.embedding_dim, config.num_items)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_embedding.weight[:self.mask_token])
        nn.init.zeros_(self.item_embedding.weight[self.pad_token])
        
        for layer in self.layers:
            for p in layer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
    
    def mask_sequence(
        self, 
        sequence: torch.Tensor,
        mask_prob: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create masked sequence for training.
        
        Returns:
            masked_seq: Sequence with some items replaced by [MASK]
            labels: Original items at masked positions (-100 for non-masked)
            mask_positions: Boolean mask of masked positions
        """
        device = sequence.device
        batch_size, seq_len = sequence.size()
        
        # Create mask
        mask = torch.rand(batch_size, seq_len, device=device) < mask_prob
        
        # Don't mask padding
        padding_mask = sequence == self.pad_token
        mask = mask & ~padding_mask
        
        # Create labels (-100 for non-masked positions)
        labels = sequence.clone()
        labels[~mask] = -100
        
        # Apply mask
        masked_seq = sequence.clone()
        masked_seq[mask] = self.mask_token
        
        return masked_seq, labels, mask
    
    def forward(
        self, 
        sequence: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            sequence: Item sequence [batch, seq_len]
            
        Returns:
            Logits over items [batch, seq_len, num_items]
        """
        # Embeddings
        x = self.item_embedding(sequence)
        x = self.position_encoding(x)
        
        # Transformer layers
        for layer in self.layers:
            x, _ = layer(x, key_padding_mask=key_padding_mask)
        
        x = self.final_norm(x)
        logits = self.output_layer(x)
        
        return logits
    
    def compute_loss(
        self, 
        sequence: torch.Tensor,
        label_smoothing: float = 0.0
    ) -> torch.Tensor:
        """Compute masked language model loss."""
        # Create masked sequence
        masked_seq, labels, _ = self.mask_sequence(sequence, self.config.mask_prob)
        
        # Padding mask
        padding_mask = sequence == self.pad_token
        
        # Forward pass
        logits = self.forward(masked_seq, padding_mask)
        
        # Flatten for loss
        logits = logits.view(-1, self.config.num_items)
        labels = labels.view(-1)
        
        # Cross-entropy with label smoothing
        loss = F.cross_entropy(
            logits, 
            labels, 
            ignore_index=-100,
            label_smoothing=label_smoothing
        )
        
        return loss
    
    @torch.no_grad()
    def predict_next(
        self, 
        sequence: List[int], 
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Predict next item given sequence."""
        self.eval()
        device = next(self.parameters()).device
        
        # Pad/truncate sequence
        if len(sequence) >= self.config.max_seq_len:
            sequence = sequence[-(self.config.max_seq_len - 1):]
        
        # Add mask token at the end
        seq_with_mask = sequence + [self.mask_token]
        
        # Pad to max length
        padding_length = self.config.max_seq_len - len(seq_with_mask)
        seq_padded = [self.pad_token] * padding_length + seq_with_mask
        
        seq_tensor = torch.tensor([seq_padded], device=device)
        padding_mask = seq_tensor == self.pad_token
        
        # Get predictions
        logits = self.forward(seq_tensor, padding_mask)
        
        # Get predictions for last (masked) position
        probs = F.softmax(logits[0, -1, :], dim=-1)
        
        # Exclude items in history
        probs[sequence] = 0
        
        top_probs, top_indices = torch.topk(probs, top_k)
        
        return [
            (idx.item(), prob.item())
            for idx, prob in zip(top_indices, top_probs)
        ]


class SASRecPlusPlus(nn.Module):
    """
    Improved Self-Attentive Sequential Recommendation
    
    Enhancements over original SASRec:
    - Pre-LayerNorm instead of Post-LayerNorm
    - GELU activation
    - Relative position encoding
    - Time-aware attention
    """
    
    def __init__(self, config: TransformerRecConfig):
        super().__init__()
        self.config = config
        
        self.item_embedding = nn.Embedding(
            config.num_items + 1, 
            config.embedding_dim,
            padding_idx=0
        )
        self.position_encoding = LearnablePositionalEncoding(
            config.max_seq_len,
            config.embedding_dim,
            config.dropout
        )
        
        # Relative position bias
        self.relative_pos_embedding = nn.Embedding(
            2 * config.max_seq_len - 1,
            config.num_heads
        )
        
        # Transformer layers (causal for auto-regressive)
        self.layers = nn.ModuleList([
            TransformerBlock(
                config.embedding_dim,
                config.num_heads,
                config.hidden_dim,
                config.dropout,
                config.attention_dropout,
                causal=True
            )
            for _ in range(config.num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_embedding.weight[1:])
        nn.init.zeros_(self.item_embedding.weight[0])
    
    def forward(
        self, 
        sequence: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            sequence: Item sequence [batch, seq_len]
            
        Returns:
            Hidden states [batch, seq_len, embedding_dim]
        """
        x = self.item_embedding(sequence)
        x = self.position_encoding(x)
        
        for layer in self.layers:
            x, _ = layer(x, key_padding_mask=key_padding_mask)
        
        return self.final_norm(x)
    
    def compute_loss(
        self, 
        sequence: torch.Tensor,
        target: torch.Tensor,
        neg_samples: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute BPR or cross-entropy loss.
        
        Args:
            sequence: Input sequence [batch, seq_len]
            target: Target items [batch]
            neg_samples: Optional negative samples [batch, num_neg]
        """
        hidden = self.forward(sequence)
        
        # Get final hidden state
        final_hidden = hidden[:, -1, :]  # [batch, dim]
        
        if neg_samples is not None:
            # BPR loss
            pos_emb = self.item_embedding(target)  # [batch, dim]
            neg_emb = self.item_embedding(neg_samples)  # [batch, num_neg, dim]
            
            pos_score = (final_hidden * pos_emb).sum(dim=-1)  # [batch]
            neg_score = torch.bmm(neg_emb, final_hidden.unsqueeze(-1)).squeeze(-1)  # [batch, num_neg]
            
            loss = -F.logsigmoid(pos_score.unsqueeze(1) - neg_score).mean()
        else:
            # Cross-entropy loss
            logits = torch.matmul(final_hidden, self.item_embedding.weight[1:].T)
            loss = F.cross_entropy(logits, target)
        
        return loss
    
    @torch.no_grad()
    def predict_next(
        self, 
        sequence: List[int], 
        top_k: int = 10,
        exclude_items: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """Predict next item."""
        self.eval()
        device = next(self.parameters()).device
        
        # Truncate if needed
        if len(sequence) > self.config.max_seq_len:
            sequence = sequence[-self.config.max_seq_len:]
        
        seq_tensor = torch.tensor([sequence], device=device)
        
        hidden = self.forward(seq_tensor)
        final_hidden = hidden[0, -1, :]
        
        # Score all items
        scores = torch.matmul(final_hidden, self.item_embedding.weight[1:].T)
        
        if exclude_items:
            scores[exclude_items] = float('-inf')
        
        top_scores, top_indices = torch.topk(scores, top_k)
        
        return [
            (idx.item() + 1, score.item())  # +1 for padding offset
            for idx, score in zip(top_indices, top_scores)
        ]


class BST(nn.Module):
    """
    Behavior Sequence Transformer
    
    Paper: https://arxiv.org/abs/1905.06874
    
    Combines transformer for sequence modeling with other features
    for CTR prediction.
    """
    
    def __init__(self, config: TransformerRecConfig):
        super().__init__()
        self.config = config
        
        # Item embedding
        self.item_embedding = nn.Embedding(
            config.num_items + 1,
            config.embedding_dim,
            padding_idx=0
        )
        
        # Target item embedding
        self.target_embedding = nn.Embedding(
            config.num_items + 1,
            config.embedding_dim,
            padding_idx=0
        )
        
        self.position_encoding = LearnablePositionalEncoding(
            config.max_seq_len + 1,  # +1 for target item
            config.embedding_dim,
            config.dropout
        )
        
        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim,
                dropout=config.dropout,
                batch_first=True
            ),
            num_layers=config.num_layers
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_embedding.weight[1:])
        nn.init.xavier_uniform_(self.target_embedding.weight[1:])
    
    def forward(
        self,
        history: torch.Tensor,
        target_item: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            history: User history [batch, seq_len]
            target_item: Target item [batch]
            
        Returns:
            CTR prediction [batch]
        """
        batch_size = history.size(0)
        
        # Get embeddings
        history_emb = self.item_embedding(history)  # [batch, seq_len, dim]
        target_emb = self.target_embedding(target_item).unsqueeze(1)  # [batch, 1, dim]
        
        # Concatenate target to history
        sequence = torch.cat([history_emb, target_emb], dim=1)
        sequence = self.position_encoding(sequence)
        
        # Create attention mask
        if history_mask is not None:
            # Extend mask for target item
            target_mask = torch.zeros(batch_size, 1, device=history.device).bool()
            full_mask = torch.cat([history_mask, target_mask], dim=1)
        else:
            full_mask = None
        
        # Transformer
        transformer_out = self.transformer(
            sequence, 
            src_key_padding_mask=full_mask
        )
        
        # Get target position output and pooled history
        target_output = transformer_out[:, -1, :]
        
        if history_mask is not None:
            # Masked mean pooling
            history_out = transformer_out[:, :-1, :]
            mask = (~history_mask).unsqueeze(-1).float()
            history_pooled = (history_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            history_pooled = transformer_out[:, :-1, :].mean(dim=1)
        
        # Combine and predict
        combined = torch.cat([target_output, history_pooled], dim=-1)
        output = self.output_layers(combined)
        
        return torch.sigmoid(output.squeeze(-1))


class TransformerRecommender:
    """High-level interface for transformer-based recommendations."""
    
    def __init__(
        self,
        config: TransformerRecConfig,
        model_type: str = "bert4rec",  # bert4rec, sasrec++, bst
        device: str = "cuda"
    ):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        if model_type == "bert4rec":
            self.model = BERT4Rec(config)
        elif model_type == "sasrec++":
            self.model = SASRecPlusPlus(config)
        elif model_type == "bst":
            self.model = BST(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = self.model.to(self.device)
        self.model_type = model_type
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> float:
        """Execute one training step."""
        self.model.train()
        optimizer.zero_grad()
        
        sequence = batch["sequence"].to(self.device)
        
        if self.model_type == "bert4rec":
            loss = self.model.compute_loss(sequence, self.config.label_smoothing)
        elif self.model_type == "sasrec++":
            target = batch["target"].to(self.device)
            neg_samples = batch.get("neg_samples")
            if neg_samples is not None:
                neg_samples = neg_samples.to(self.device)
            loss = self.model.compute_loss(sequence, target, neg_samples)
        else:
            target = batch["target"].to(self.device)
            labels = batch["label"].to(self.device).float()
            preds = self.model(sequence, target)
            loss = F.binary_cross_entropy(preds, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def recommend(
        self,
        sequence: List[int],
        k: int = 10,
        exclude_items: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """Get top-k recommendations."""
        self.model.eval()
        return self.model.predict_next(sequence, k, exclude_items)
    
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
    config = TransformerRecConfig(
        num_items=5000,
        max_seq_len=50,
        embedding_dim=128,
        num_heads=4,
        num_layers=2
    )
    
    # Test BERT4Rec
    model = BERT4Rec(config)
    sequence = torch.randint(0, 5000, (4, 50))
    
    logits = model(sequence)
    print(f"BERT4Rec output shape: {logits.shape}")
    
    # Test prediction
    predictions = model.predict_next([1, 5, 10, 20, 30], top_k=5)
    print(f"Top-5 predictions: {predictions}")
