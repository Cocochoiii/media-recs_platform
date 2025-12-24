"""
LSTM Sequential Recommendation Model

This module implements sequential recommendation using LSTM neural networks
to capture temporal patterns in user behavior and predict next item preferences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import math


@dataclass
class LSTMConfig:
    """Configuration for LSTM Sequential model."""
    num_items: int
    embedding_dim: int = 128
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    max_sequence_length: int = 50
    use_attention: bool = True
    num_attention_heads: int = 4


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position awareness."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalAttention(nn.Module):
    """Multi-head attention for capturing temporal dependencies."""
    
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int = 4, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, \
            "hidden_size must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply multi-head attention.
        
        Args:
            hidden_states: Input sequence [batch, seq_len, hidden]
            attention_mask: Mask for valid positions [batch, seq_len]
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project to Q, K, V
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Causal mask to prevent looking at future
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=scores.device), 
            diagonal=1
        ).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax and apply to values
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, -1)
        
        output = self.out(context)
        
        return output, attention_weights


class LSTMSequentialModel(nn.Module):
    """
    LSTM-based Sequential Recommender.
    
    Captures sequential patterns in user behavior to predict
    the next item(s) a user is likely to interact with.
    """
    
    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config
        
        # Item embedding
        self.item_embedding = nn.Embedding(
            config.num_items + 1,  # +1 for padding
            config.embedding_dim,
            padding_idx=0
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            config.embedding_dim,
            config.max_sequence_length,
            config.dropout
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional
        )
        
        # Output dimension after LSTM
        lstm_output_dim = config.hidden_size * (2 if config.bidirectional else 1)
        
        # Temporal attention
        if config.use_attention:
            self.attention = TemporalAttention(
                lstm_output_dim,
                config.num_attention_heads,
                config.dropout
            )
        else:
            self.attention = None
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(lstm_output_dim)
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(lstm_output_dim, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_items)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.item_embedding.weight[1:])  # Skip padding
        nn.init.zeros_(self.item_embedding.weight[0])  # Zero for padding
        
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
    
    def forward(
        self,
        item_sequences: torch.Tensor,
        sequence_lengths: torch.Tensor,
        target_items: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for sequential prediction.
        
        Args:
            item_sequences: Sequence of item IDs [batch, max_seq_len]
            sequence_lengths: Actual lengths of each sequence [batch]
            target_items: Target item IDs for training [batch]
            
        Returns:
            Dictionary with logits, predictions, and optional loss
        """
        batch_size = item_sequences.size(0)
        
        # Embed items
        embedded = self.item_embedding(item_sequences)  # [batch, seq, embed]
        embedded = self.positional_encoding(embedded)
        
        # Create attention mask from sequence lengths
        max_len = item_sequences.size(1)
        attention_mask = torch.arange(max_len, device=item_sequences.device)
        attention_mask = attention_mask.unsqueeze(0) < sequence_lengths.unsqueeze(1)
        
        # Pack for efficient LSTM processing
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            sequence_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=max_len
        )
        
        # Apply attention if enabled
        if self.attention:
            attended, attn_weights = self.attention(lstm_out, attention_mask)
            lstm_out = lstm_out + attended  # Residual connection
        else:
            attn_weights = None
        
        # Layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Get representation at last valid position for each sequence
        idx = (sequence_lengths - 1).unsqueeze(1).unsqueeze(2)
        idx = idx.expand(-1, -1, lstm_out.size(2))
        last_hidden = lstm_out.gather(1, idx).squeeze(1)  # [batch, hidden]
        
        # Project to item space
        logits = self.output_projection(last_hidden)  # [batch, num_items]
        
        result = {
            "logits": logits,
            "predictions": torch.argmax(logits, dim=1),
            "attention_weights": attn_weights,
            "sequence_representation": last_hidden
        }
        
        # Compute loss if targets provided
        if target_items is not None:
            loss = F.cross_entropy(logits, target_items)
            result["loss"] = loss
        
        return result
    
    @torch.no_grad()
    def predict_next(
        self,
        item_sequence: List[int],
        top_k: int = 10,
        exclude_items: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Predict next items for a given sequence.
        
        Args:
            item_sequence: List of item IDs in order
            top_k: Number of predictions to return
            exclude_items: Items to exclude from predictions
            
        Returns:
            List of (item_id, probability) tuples
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Prepare input
        seq_tensor = torch.tensor([item_sequence], device=device)
        length_tensor = torch.tensor([len(item_sequence)], device=device)
        
        # Get predictions
        output = self.forward(seq_tensor, length_tensor)
        probs = F.softmax(output["logits"], dim=1)[0]
        
        # Exclude specified items
        if exclude_items:
            for item_id in exclude_items:
                if 0 < item_id < len(probs):
                    probs[item_id] = 0
        
        # Get top-k
        top_probs, top_indices = torch.topk(probs, top_k)
        
        return [
            (idx.item(), prob.item())
            for idx, prob in zip(top_indices, top_probs)
        ]


class GRU4Rec(nn.Module):
    """
    GRU-based Session Recommendation (GRU4Rec).
    
    Simplified version using GRU for faster training,
    popular for session-based recommendations.
    """
    
    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config
        
        self.item_embedding = nn.Embedding(
            config.num_items + 1,
            config.embedding_dim,
            padding_idx=0
        )
        
        self.gru = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )
        
        self.output = nn.Linear(config.hidden_size, config.num_items)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_embedding.weight[1:])
        nn.init.zeros_(self.item_embedding.weight[0])
    
    def forward(
        self,
        item_sequences: torch.Tensor,
        sequence_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass returning logits."""
        embedded = self.item_embedding(item_sequences)
        
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            sequence_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        gru_out, hidden = self.gru(packed)
        
        # Use last hidden state
        last_hidden = hidden[-1]  # [batch, hidden]
        
        logits = self.output(last_hidden)
        return logits


class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation (SASRec).
    
    Uses self-attention instead of RNN for capturing
    long-range dependencies in user sequences.
    """
    
    def __init__(self, config: LSTMConfig, num_attention_layers: int = 2):
        super().__init__()
        self.config = config
        
        self.item_embedding = nn.Embedding(
            config.num_items + 1,
            config.hidden_size,
            padding_idx=0
        )
        
        self.positional_embedding = nn.Embedding(
            config.max_sequence_length,
            config.hidden_size
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Self-attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_attention_layers
        )
        
        self.output = nn.Linear(config.hidden_size, config.num_items)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_embedding.weight[1:])
        nn.init.xavier_uniform_(self.positional_embedding.weight)
    
    def forward(
        self,
        item_sequences: torch.Tensor,
        sequence_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with self-attention."""
        batch_size, seq_len = item_sequences.size()
        device = item_sequences.device
        
        # Item embeddings
        item_emb = self.item_embedding(item_sequences)
        
        # Add positional embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_emb = self.positional_embedding(positions)
        
        hidden = self.dropout(item_emb + pos_emb)
        
        # Create attention mask
        attention_mask = torch.arange(seq_len, device=device).unsqueeze(0)
        attention_mask = attention_mask >= sequence_lengths.unsqueeze(1)
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), 
            diagonal=1
        ).bool()
        
        # Transformer forward
        output = self.transformer(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=attention_mask
        )
        
        # Get last valid position
        idx = (sequence_lengths - 1).unsqueeze(1).unsqueeze(2)
        idx = idx.expand(-1, -1, output.size(2))
        last_hidden = output.gather(1, idx).squeeze(1)
        
        logits = self.output(last_hidden)
        return logits


class SequentialRecommender:
    """
    High-level interface for sequential recommendations.
    
    Supports multiple model architectures and provides
    training and inference utilities.
    """
    
    def __init__(
        self,
        config: LSTMConfig,
        model_type: str = "lstm",  # lstm, gru4rec, sasrec
        device: str = "cuda"
    ):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        if model_type == "lstm":
            self.model = LSTMSequentialModel(config)
        elif model_type == "gru4rec":
            self.model = GRU4Rec(config)
        elif model_type == "sasrec":
            self.model = SASRec(config)
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
        
        sequences = batch["sequences"].to(self.device)
        lengths = batch["lengths"].to(self.device)
        targets = batch["targets"].to(self.device)
        
        if self.model_type == "lstm":
            output = self.model(sequences, lengths, targets)
            loss = output["loss"]
        else:
            logits = self.model(sequences, lengths)
            loss = F.cross_entropy(logits, targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def predict(
        self,
        sequences: torch.Tensor,
        lengths: torch.Tensor,
        top_k: int = 10
    ) -> List[List[Tuple[int, float]]]:
        """Predict top-k next items for each sequence."""
        self.model.eval()
        
        sequences = sequences.to(self.device)
        lengths = lengths.to(self.device)
        
        if self.model_type == "lstm":
            output = self.model(sequences, lengths)
            logits = output["logits"]
        else:
            logits = self.model(sequences, lengths)
        
        probs = F.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probs, top_k, dim=1)
        
        results = []
        for i in range(len(sequences)):
            results.append([
                (idx.item(), prob.item())
                for idx, prob in zip(top_indices[i], top_probs[i])
            ])
        
        return results
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Returns:
            Dictionary with HR@k and NDCG@k metrics
        """
        self.model.eval()
        
        hits = {k: 0 for k in k_values}
        ndcgs = {k: 0.0 for k in k_values}
        total = 0
        
        for batch in dataloader:
            sequences = batch["sequences"].to(self.device)
            lengths = batch["lengths"].to(self.device)
            targets = batch["targets"].to(self.device)
            
            if self.model_type == "lstm":
                output = self.model(sequences, lengths)
                logits = output["logits"]
            else:
                logits = self.model(sequences, lengths)
            
            # Get rankings
            _, rankings = torch.topk(logits, max(k_values), dim=1)
            
            for i, target in enumerate(targets):
                for k in k_values:
                    top_k_items = rankings[i, :k]
                    if target in top_k_items:
                        hits[k] += 1
                        rank = (top_k_items == target).nonzero()[0].item()
                        ndcgs[k] += 1.0 / math.log2(rank + 2)
                
                total += 1
        
        metrics = {}
        for k in k_values:
            metrics[f"HR@{k}"] = hits[k] / total
            metrics[f"NDCG@{k}"] = ndcgs[k] / total
        
        return metrics
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "model_type": self.model_type
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()


if __name__ == "__main__":
    # Example usage
    config = LSTMConfig(
        num_items=10000,
        embedding_dim=128,
        hidden_size=256,
        num_layers=2,
        bidirectional=True,
        use_attention=True
    )
    
    recommender = SequentialRecommender(
        config=config,
        model_type="lstm",
        device="cuda"
    )
    
    # Example prediction
    if hasattr(recommender.model, 'predict_next'):
        sequence = [42, 156, 789, 234]
        predictions = recommender.model.predict_next(sequence, top_k=5)
        print(f"Next item predictions: {predictions}")
